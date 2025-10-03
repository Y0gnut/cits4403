"""
AUD Exchange Rate Agent-Based Model
Core model implementation package
"""

from .forex_abm import (
    ForexMarket,
    Trader,
    Speculator,
    Hedger,
    Fundamentalist,
    CentralBank,
    Order,
    OrderBook,
    OrderType,
    TraderType
)

from .data_collector import ForexDataCollector

__version__ = '0.1.0'
__author__ = 'CITS4403 Project Team'

__all__ = [
    'ForexMarket',
    'Trader',
    'Speculator',
    'Hedger',
    'Fundamentalist',
    'CentralBank',
    'Order',
    'OrderBook',
    'OrderType',
    'TraderType',
    'ForexDataCollector'
]