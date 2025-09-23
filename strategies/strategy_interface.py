"""
Strategy Interface and Registration Mechanism for User-Defined Strategies

Users can implement their own strategies by subclassing `Strategy` and registering them with the `@register_strategy` decorator.

Example:

    from condition_panel.strategy_interface import Strategy, register_strategy

    @register_strategy("my_strategy")
    class MyStrategy(Strategy):
        def evaluate(self, data):
            # User logic here
            return ...
        def describe(self):
            return "My custom strategy"

"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Callable
from PySide6.QtWidgets import QWidget

# Registry for all user strategies
STRATEGY_REGISTRY: Dict[str, Type['Strategy']] = {}

def register_strategy(name: str) -> Callable[[Type['Strategy']], Type['Strategy']]:
    """
    Decorator to register a user-defined strategy class.
    Usage:
        @register_strategy("my_strategy")
        class MyStrategy(Strategy):
            ...
    """
    def decorator(cls: Type['Strategy']) -> Type['Strategy']:
        # Enforce required interface
        for method in ['create_config_widget', 'get_config']:
            if not hasattr(cls, method):
                raise TypeError(f"Strategy '{name}' must implement '{method}'")
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

class Strategy(ABC):
    """
    Abstract base class for user strategies.
    Users must implement `evaluate` and `describe`.
    """
    @abstractmethod
    def evaluate(self, data, **params):
        """Evaluate the strategy on the given data. Returns result."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of the strategy."""
        pass

    @abstractmethod
    def create_config_widget(self, parent=None) -> QWidget:
        """Return a QWidget for configuring this strategy."""
        pass

    @abstractmethod
    def get_config(self, widget: QWidget) -> dict:
        """Extract config parameters from the widget."""
        pass

    def set_config(self, widget: QWidget, config: dict):
        """(Optional) Set config parameters to the widget."""
        pass

    @abstractmethod
    def plot(self, main_window, df):
        """Plot the strategy results on the main window."""
        pass 
