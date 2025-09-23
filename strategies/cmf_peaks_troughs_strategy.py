import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .strategy_interface import Strategy, register_strategy
# from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QSlider
# from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg

@register_strategy("CMF Peaks & Troughs")
class ConfigurableCMFPeaksTroughsStrategy(Strategy):
    def __init__(self, period=20, wma_period=10, prominence=0.05):
        self.period = period
        self.wma_period = wma_period
        self.prominence = prominence
        self.name = "CMF Peaks & Troughs"

    def evaluate(self, data, period=20, wma_period=10, prominence=0.05):
        print(f"Evaluating CMF Peaks & Troughs with period={period}, wma_period={wma_period}, prominence={prominence}")
        df = data.copy()
        mfm_denom = (df['High'] - df['Low'])
        mfm_denom = mfm_denom.replace(0, 1)
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / mfm_denom
        mfv = mfm * df['Volume']
        df['CMF'] = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
        weights = np.arange(1, wma_period + 1)
        df['CMF_WMA'] = df['CMF'].rolling(wma_period).apply(lambda x: (weights * x).sum() / weights.sum(), raw=True)
        cmf_wma_values = df['CMF_WMA'].fillna(0).values
        peak_indices, _ = find_peaks(cmf_wma_values, prominence=prominence)
        trough_indices, _ = find_peaks(-cmf_wma_values, prominence=prominence)
        signals = []
        for idx in peak_indices:
            signals.append({"action": "sell", "index": df.index[idx], "value": cmf_wma_values[idx]})
        for idx in trough_indices:
            signals.append({"action": "buy", "index": df.index[idx], "value": cmf_wma_values[idx]})
        signals.sort(key=lambda s: s["index"])
        return signals, df['CMF'], df['CMF_WMA']

    def describe(self):
        return (
            f"Buy at CMF troughs and sell at CMF peaks. "
            f"Configurable: period, wma_period, prominence. "
            "Requires DataFrame with columns: High, Low, Close, Volume."
        )

    def create_config_widget(self):
        """Dummy implementation to satisfy abstract base class."""
        return None

    def get_config(self, widget):
        return {
            "period": widget.period_spin.value(),
            "wma_period": widget.wma_spin.value(),
            "prominence": widget.prom_spin.value(),
        }

    def get_parameter_info(self):
        return {
            "period": {"type": int, "min": 20, "max": 20, "step": 1, "default": 20},
            "wma_period": {"type": int, "min": 5, "max": 10, "step": 1, "default": 5},
            "prominence": {"type": float, "min": 0.01, "max": 0.10, "step": 0.01, "default": 0.05}
        }

    def set_config(self, widget, config):
        widget.period_spin.setValue(config.get("period", 20))
        widget.wma_spin.setValue(config.get("wma_period", 10))
        widget.prom_spin.setValue(config.get("prominence", 0.05)) 

    def plot(self, main_window, df):
        params = self.get_config(self.widget)
        signals, cmf, cmf_wma = self.evaluate(df, **params)
        # Store signals for alert system
        self.last_signals = signals
        # extra_data = (cmf, cmf_wma)
        # cmf, cmf_wma = extra_data if extra_data else (None, None)
        x = df.index.astype(np.int64) // 10**9

        buy_signals = [s for s in signals if s["action"] == "buy"]
        sell_signals = [s for s in signals if s["action"] == "sell"]

        # Update buy/sell signals on price plot
        if buy_signals:
            buy_idx = [df.index.get_loc(s["index"]) for s in buy_signals]
            main_window.buy_plot_item.setData(x[buy_idx], df["Close"].iloc[buy_idx].to_numpy())
        else:
            main_window.buy_plot_item.setData([], []) # Clear data

        if sell_signals:
            sell_idx = [df.index.get_loc(s["index"]) for s in sell_signals]
            main_window.sell_plot_item.setData(x[sell_idx], df["Close"].iloc[sell_idx].to_numpy())
        else:
            main_window.sell_plot_item.setData([], []) # Clear data

        # Update CMF and CMF_WMA on CMF plot
        if cmf is not None:
            main_window.indicator_plot_item1.setData(x.to_numpy(), cmf.to_numpy())
            main_window.indicator_plot_item1.setPen(pg.mkPen('b', width=1))
            # main_window.indicator_plot_item1.setName('CMF')
        else:
            main_window.indicator_plot_item1.setData([], [])

        if cmf_wma is not None:
            main_window.indicator_plot_item2.setData(x.to_numpy(), cmf_wma.to_numpy())
            main_window.indicator_plot_item2.setPen(pg.mkPen('y', width=1))
            # main_window.indicator_plot_item2.setName('CMF WMA')
        else:
            main_window.indicator_plot_item2.setData([], [])

        main_window.status.setText(f"Updated {main_window.current_symbol} with {len(buy_signals)} buys and {len(sell_signals)} sells.")
    
    def get_last_signals(self):
        """Return the last generated signals for alert system."""
        return getattr(self, 'last_signals', [])

    def scan_signals_in_period(self, data, recent_period=5, period=None, wma_period=None, prominence=None):
        """
        Return True if any buy/sell signal occurs in the last `recent_period` bars.
        Optionally override period, wma_period, prominence.
        """
        params = {
            'period': period if period is not None else self.period,
            'wma_period': wma_period if wma_period is not None else self.wma_period,
            'prominence': prominence if prominence is not None else self.prominence,
        }
        signals, _, _ = self.evaluate(data, **params)
        if not signals:
            return False, []
        # Only consider signals in the last `recent_period` bars
        last_indices = set(data.index[-recent_period:])
        recent_signals = [s for s in signals if s['index'] in last_indices]
        return bool(recent_signals), recent_signals