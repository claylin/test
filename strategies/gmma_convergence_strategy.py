import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from ..strategy_interface import Strategy, register_strategy
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QSlider
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg

@register_strategy("GMMA Convergence")
class GMMAConvergenceStrategy(Strategy):
    def __init__(self, prominence=0.1):
        self.prominence = prominence
        self.short_emas = [3, 5, 8, 10, 12, 15]
        self.long_emas = [30, 35, 40, 45, 50, 60]
        self.name = "GMMA Convergence"

    def evaluate(self, data, prominence=None):
        prominence = self.prominence if prominence is None else prominence
        df = data.copy()
        for ema in self.short_emas + self.long_emas:
            df[f'EMA_{ema}'] = df['Close'].ewm(span=ema, adjust=False).mean()
        long_ema_keys = [f'EMA_{ema}' for ema in self.long_emas]
        spread_array = df[long_ema_keys].std(axis=1).values
        peak_indices, _ = find_peaks(spread_array, prominence=prominence)
        trough_indices, _ = find_peaks(-spread_array, prominence=prominence)
        df["contraction_signal"] = np.nan
        df["expansion_signal"] = np.nan
        if long_ema_keys:
            contraction_y_values = df.loc[df.index[peak_indices], long_ema_keys].mean(axis=1)
            df.loc[df.index[peak_indices], "contraction_signal"] = contraction_y_values
            expansion_y_values = df.loc[df.index[trough_indices], long_ema_keys].mean(axis=1)
            df.loc[df.index[trough_indices], "expansion_signal"] = expansion_y_values
        signals = []
        for idx in peak_indices:
            signals.append({"action": "contract", "index": df.index[idx], "value": df.loc[df.index[idx], "contraction_signal"]})
        for idx in trough_indices:
            signals.append({"action": "expand", "index": df.index[idx], "value": df.loc[df.index[idx], "expansion_signal"]})
        return signals, spread_array, long_ema_keys, df["contraction_signal"], df["expansion_signal"], df

    def describe(self):
        return f"Detects GMMA convergence/divergence points using spread prominence={self.prominence}."

    def create_config_widget(self):
        class ConfigWidget(QWidget):
            valueChanged = Signal()
        self.widget = widget = ConfigWidget()
        layout = QVBoxLayout(widget)
        prom_layout = QHBoxLayout()
        prom_layout.addWidget(QLabel("Prominence:"))
        prom_slider = QSlider(Qt.Horizontal)
        prom_slider.setRange(1, 100)
        prom_slider.setValue(int(self.prominence * 100))
        prom_spin = QDoubleSpinBox()
        prom_spin.setRange(0.01, 1.0)
        prom_spin.setValue(self.prominence)
        prom_spin.setSingleStep(0.01)
        prom_label = QLabel(f"{self.prominence:.2f}")
        prom_layout.addWidget(prom_slider)
        prom_layout.addWidget(prom_spin)
        prom_layout.addWidget(prom_label)
        layout.addLayout(prom_layout)
        prom_slider.valueChanged.connect(lambda v: prom_spin.setValue(v/100))
        prom_spin.valueChanged.connect(lambda v: prom_slider.setValue(int(v*100)))
        prom_spin.valueChanged.connect(lambda v: prom_label.setText(f"{v:.2f}"))
        prom_slider.valueChanged.connect(lambda _: widget.valueChanged.emit())
        prom_spin.valueChanged.connect(lambda _: widget.valueChanged.emit())
        widget.prom_spin = prom_spin
        widget.prom_slider = prom_slider
        widget.prom_label = prom_label
        return widget

    def get_config(self, widget):
        return {"prominence": widget.prom_spin.value()}

    def get_parameter_info(self):
        return {
            "prominence": {"type": float, "min": 0.01, "max": 1.0, "step": 0.01, "default": 0.1}
        }

    def set_config(self, widget, config):
        widget.prom_spin.setValue(config.get("prominence", self.prominence)) 

    def plot(self, main_window, df):
        params = self.get_config(self.widget)
        signals, spread_array, long_ema_keys, contraction_signal, expansion_signal, df_with_emas = self.evaluate(df, **params)
        # Store signals for alert system
        self.last_signals = signals
        extra_data = (spread_array, long_ema_keys, contraction_signal, expansion_signal, df_with_emas)
        if extra_data and len(extra_data) >= 5:
            spread_array, long_ema_keys, contraction_signal, expansion_signal, df_with_emas = extra_data
        else:
            spread_array, long_ema_keys, contraction_signal, expansion_signal = extra_data if extra_data else (None, None, None, None)
            df_with_emas = df  # Fallback to original df if no EMAs available
        x = df.index.astype(np.int64) // 10**9
        
        # Update short-term EMAs (blue lines)
        for i, ema in enumerate(self.short_emas):
            ema_key = f'EMA_{ema}'
            if ema_key in df_with_emas.columns:
                main_window.gmma_short_line_items[i].setData(x, df_with_emas[ema_key].values)
                # main_window.gmma_short_line_items[i].setName(f'EMA {ema}')
            else:
                main_window.gmma_short_line_items[i].setData([], [])
        
        # Update long-term EMAs (red lines)
        for i, ema in enumerate(self.long_emas):
            ema_key = f'EMA_{ema}'
            if ema_key in df_with_emas.columns:
                main_window.gmma_long_line_items[i].setData(x, df_with_emas[ema_key].values)
                # main_window.gmma_long_line_items[i].setName(f'EMA {ema}')
            else:
                main_window.gmma_long_line_items[i].setData([], [])
        
        # Update signals
        contract_signals = [s for s in signals if s["action"] == "contract"]
        expand_signals = [s for s in signals if s["action"] == "expand"]
        
        if contract_signals:
            contract_idx = [df.index.get_loc(s["index"]) for s in contract_signals]
            main_window.buy_plot_item.setData(x[contract_idx], contraction_signal.iloc[contract_idx].to_numpy())
        else:
            main_window.buy_plot_item.setData([], [])
            
        if expand_signals:
            expand_idx = [df.index.get_loc(s["index"]) for s in expand_signals]
            main_window.sell_plot_item.setData(x[expand_idx], expansion_signal.iloc[expand_idx].to_numpy())
        else:
            main_window.sell_plot_item.setData([], [])
            
        # Plot spread on bottom chart
        if spread_array is not None:
            main_window.indicator_plot_item1.setData(x, spread_array)
            main_window.indicator_plot_item1.setPen(pg.mkPen('m', width=1))
            # main_window.indicator_plot_item1.setName('GMMA Spread')
        else:
            main_window.indicator_plot_item1.setData([], [])
        main_window.indicator_plot_item2.setData([], []) # Clear second indicator
        
        main_window.status.setText(f"Updated {main_window.current_symbol} with {len(contract_signals)} contractions and {len(expand_signals)} expansions.")
    
    def get_last_signals(self):
        """Return the last generated signals for alert system."""
        return getattr(self, 'last_signals', []) 