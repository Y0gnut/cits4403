"""
Agent-Based Model for Australian Dollar Exchange Rate Dynamics
Core framework for multi-agent forex market simulation
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from enum import Enum


class OrderType(Enum):
    """Types of orders in the market"""
    BUY = 1
    SELL = -1


class TraderType(Enum):
    """Types of traders in the market"""
    SPECULATOR = "Speculator"
    HEDGER = "Hedger"
    FUNDAMENTALIST = "Fundamentalist"
    CENTRAL_BANK = "CentralBank"


class Order:
    """Represents a trading order"""

    def __init__(self, trader_id: int, order_type: OrderType,
                 quantity: float, price: float = None):
        """
        Initialize an order

        Args:
            trader_id: ID of the trader placing the order
            order_type: BUY or SELL
            quantity: Amount to trade
            price: Limit price (None for market order)
        """
        self.trader_id = trader_id
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.timestamp = None

    def __repr__(self):
        return f"Order({self.trader_id}, {self.order_type.name}, {self.quantity:.2f}, {self.price})"


class OrderBook:
    """Manages buy and sell orders and executes trades"""

    def __init__(self):
        self.buy_orders: List[Order] = []
        self.sell_orders: List[Order] = []
        self.trade_history: List[Dict] = []

    def add_order(self, order: Order):
        """Add an order to the order book"""
        if order.order_type == OrderType.BUY:
            self.buy_orders.append(order)
            # Sort by price descending (highest bid first), market orders (price=None) go to the top (inf)
            self.buy_orders.sort(key=lambda x: x.price if x.price else float('inf'), reverse=True)
        else:
            self.sell_orders.append(order)
            # Sort by price ascending (lowest ask first), market orders (price=None) go to the top (0)
            self.sell_orders.sort(key=lambda x: x.price if x.price else 0)

    def match_orders(self) -> float:
        """
        Match buy and sell orders and execute trades

        Returns:
            Clearing price (weighted average of executed trades)
        """
        executed_trades = []

        while self.buy_orders and self.sell_orders:
            buy_order = self.buy_orders[0]
            sell_order = self.sell_orders[0]

            # Use effective price limits for matching
            buy_limit = buy_order.price if buy_order.price else float('inf')
            sell_limit = sell_order.price if sell_order.price else 0

            if buy_limit >= sell_limit:
                # A match occurred (Bid >= Ask)
                trade_quantity = min(buy_order.quantity, sell_order.quantity)

                # --- CORRECT Trade Price Determination ---
                if buy_order.price is None and sell_order.price is None:
                    # Two market orders cannot be matched without a reference price. Break.
                    break
                elif buy_order.price is None:
                    # Market Buy hits Limit Sell: Trade at the passive (Sell) price
                    trade_price = sell_order.price
                elif sell_order.price is None:
                    # Market Sell hits Limit Buy: Trade at the passive (Buy) price
                    trade_price = buy_order.price
                else:
                    # Two limit orders crossing: Use the midpoint
                    trade_price = (buy_order.price + sell_order.price) / 2

                if trade_price is None:
                    break
                # --- END CORRECT ---

                executed_trades.append({
                    'price': trade_price,
                    'quantity': trade_quantity,
                    'buyer': buy_order.trader_id,
                    'seller': sell_order.trader_id
                })

                # Reduce quantities
                buy_order.quantity -= trade_quantity
                sell_order.quantity -= trade_quantity

                # Remove filled orders
                if buy_order.quantity <= 0:
                    self.buy_orders.pop(0)
                if sell_order.quantity <= 0:
                    self.sell_orders.pop(0)
            else:
                # Prices don't cross, stop matching
                break

        if executed_trades:
            # Recompute total_value using all executed trades (price is guaranteed non-None)
            total_value = sum(t['price'] * t['quantity'] for t in executed_trades)
            total_quantity = sum(t['quantity'] for t in executed_trades)
            clearing_price = total_value / total_quantity if total_quantity > 0 else None

            self.trade_history.extend(executed_trades)
            return clearing_price

        return None

    def clear_orders(self):
        """Clear all unfilled orders (end of trading period)"""
        self.buy_orders = []
        self.sell_orders = []

    def get_market_depth(self) -> Tuple[int, int]:
        """Return number of buy and sell orders"""
        return len(self.buy_orders), len(self.sell_orders)


class Trader(ABC):
    """Abstract base class for all trader types"""

    def __init__(self, trader_id: int, initial_capital: float,
                 initial_position: float = 0):
        """
        Initialize a trader
        """
        self.trader_id = trader_id
        self.capital = initial_capital
        self.position = initial_position
        self.trader_type = TraderType.SPECULATOR

        self.wealth_history = [initial_capital]
        self.trade_count = 0

    @abstractmethod
    def decide_action(self, market_state: Dict) -> Order:
        """
        Decide trading action based on market state
        """
        pass

    def update_position(self, trade_price: float, quantity: float, is_buy: bool):
        """Update trader's position after a trade"""
        if is_buy:
            self.capital -= trade_price * quantity
            self.position += quantity
        else:
            self.capital += trade_price * quantity
            self.position -= quantity

        self.trade_count += 1

    def get_wealth(self, current_price: float) -> float:
        """Calculate total wealth at current market price"""
        return self.capital + self.position * current_price

    def __repr__(self):
        return f"{self.trader_type.value}_{self.trader_id}"


class Speculator(Trader):
    """Speculator trader - uses technical analysis and momentum strategies"""

    def __init__(self, trader_id: int, initial_capital: float,
                 momentum_window: int = 5, sentiment_factor: float = 0.1):
        super().__init__(trader_id, initial_capital)
        self.trader_type = TraderType.SPECULATOR
        self.momentum_window = momentum_window
        self.sentiment_factor = sentiment_factor
        self.price_history = []

    def decide_action(self, market_state: Dict) -> Order:
        """Trading based on price momentum and market sentiment"""
        current_price = market_state.get('current_price')
        sentiment = market_state.get('sentiment', 0)

        if current_price is None:
            return None

        self.price_history.append(current_price)

        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)

        # Modification: Increase trading probability during the initial phase
        if len(self.price_history) < self.momentum_window:
            # Increase probability to 80% and use limit orders
            if np.random.rand() < 0.8:
                trade_size_value = self.capital * 0.05  # Increase to 5%
                quantity = trade_size_value / current_price

                spread = 0.002  # Increase spread to 0.2%

                if np.random.rand() < 0.5:
                    limit_price = current_price * (1 + spread)
                    if self.capital > trade_size_value:
                        return Order(self.trader_id, OrderType.BUY, quantity, limit_price)
                else:
                    limit_price = current_price * (1 - spread)
                    if self.position > 0:
                        return Order(self.trader_id, OrderType.SELL, min(quantity, self.position), limit_price)
            return None

        # Standard momentum trading
        momentum = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        signal = momentum + sentiment * self.sentiment_factor

        trade_size = min(abs(signal) * self.capital * 0.1, self.capital * 0.2)

        # Key: Lower the threshold to almost zero
        threshold = 0.00001

        # CRITICAL FIX: Switch from Market Orders to Aggressive Limit Orders ---
        aggressive_spread = 0.00005 # 0.5 basis point (0.005%) - ensures match

        if signal > threshold and self.capital > trade_size * current_price:
            quantity = trade_size / current_price
            # Aggressive Buy Order: Price slightly above current price to ensure match
            limit_price = current_price * (1 + aggressive_spread)
            return Order(self.trader_id, OrderType.BUY, quantity, limit_price)

        elif signal < -threshold and self.position > 0:
            quantity = min(trade_size / current_price, self.position)
            # Aggressive Sell Order: Price slightly below current price to ensure match
            limit_price = current_price * (1 - aggressive_spread)
            return Order(self.trader_id, OrderType.SELL, quantity, limit_price)
        # End of Fix ---

        return None


class Hedger(Trader):
    """Hedger trader - trades to manage risk, provides liquidity"""

    def __init__(self, trader_id: int, initial_capital: float,
                 target_hedge_ratio: float = 0.5):
        super().__init__(trader_id, initial_capital)
        self.trader_type = TraderType.HEDGER
        self.target_hedge_ratio = target_hedge_ratio

    def decide_action(self, market_state: Dict) -> Order:
        """Trading to maintain hedge ratio"""
        current_price = market_state.get('current_price')

        if current_price is None:
            return None

        total_wealth = self.get_wealth(current_price)
        current_ratio = (self.position * current_price) / total_wealth if total_wealth > 0 else 0

        deviation = current_ratio - self.target_hedge_ratio

        # Lower the threshold from 1% to 0.1%
        if abs(deviation) > 0.001:  # Changed to 0.1%
            target_position_value = self.target_hedge_ratio * total_wealth
            target_position = target_position_value / current_price

            quantity = abs(target_position - self.position) * 0.5  # Only trade half each time

            spread = 0.0005  # Tight spread

            if deviation > 0 and self.position > quantity:
                return Order(self.trader_id, OrderType.SELL, quantity, current_price * (1 - spread))
            else:
                if self.capital > quantity * current_price:
                    return Order(self.trader_id, OrderType.BUY, quantity, current_price * (1 + spread))

        return None


class Fundamentalist(Trader):
    """Fundamentalist trader - trades based on fundamental value"""

    def __init__(self, trader_id: int, initial_capital: float,
                 fundamental_value: float, confidence: float = 0.1):
        super().__init__(trader_id, initial_capital)
        self.trader_type = TraderType.FUNDAMENTALIST
        self.fundamental_value = fundamental_value
        self.confidence = confidence

    def update_fundamental_value(self, economic_indicators: Dict):
        """Update fundamental value based on economic indicators"""
        interest_diff = economic_indicators.get('interest_diff', 0)
        commodity_index = economic_indicators.get('commodity_index', 1.0)

        # Scale factor 0.01 is applied here as a sensitivity parameter
        self.fundamental_value *= (1 + interest_diff * 0.01)
        # Note: If commodity_index is 1.0 (default), this does nothing.
        # It's assumed the commodity_index passed here is already scaled or normalized.
        self.fundamental_value *= (commodity_index / 100)

    def decide_action(self, market_state: Dict) -> Order:
        """Trading based on deviation from fundamental value"""
        current_price = market_state.get('current_price')

        if current_price is None:
            return None

        deviation = (self.fundamental_value - current_price) / current_price

        # Lower the threshold from 0.2% to 0.01%
        if abs(deviation) > 0.0001:  # Changed to 0.01%
            # Increase trade size
            trade_size = min(abs(deviation) * self.confidence * self.capital * 2,
                             self.capital * 0.2)

            spread = 0.0008

            if deviation > 0 and self.capital > trade_size:
                quantity = trade_size / current_price
                return Order(self.trader_id, OrderType.BUY, quantity, current_price * (1 + spread))
            elif deviation < 0 and self.position > 0:
                quantity = min(trade_size / current_price, self.position * 0.5)
                return Order(self.trader_id, OrderType.SELL, quantity, current_price * (1 - spread))

        return None


class CentralBank(Trader):
    """Central Bank - intervenes to stabilize exchange rate"""

    def __init__(self, trader_id: int, initial_capital: float,
                 target_rate: float, intervention_threshold: float = 0.05):
        super().__init__(trader_id, initial_capital * 10)
        self.trader_type = TraderType.CENTRAL_BANK
        self.target_rate = target_rate
        self.intervention_threshold = intervention_threshold
        self.intervention_count = 0

    def decide_action(self, market_state: Dict) -> Order:
        """Intervene when exchange rate deviates significantly from target"""
        current_price = market_state.get('current_price')
        volatility = market_state.get('volatility', 0)

        if current_price is None:
            return None

        deviation = (current_price - self.target_rate) / self.target_rate

        should_intervene = (abs(deviation) > self.intervention_threshold or
                          volatility > 0.03)

        if should_intervene:
            intervention_size = self.capital * 0.05
            quantity = intervention_size / current_price

            self.intervention_count += 1

            # Central Bank uses aggressive Limit Orders (to ensure matching and move price)
            if deviation > 0:
                # Price is too high, SELL to push it down
                return Order(self.trader_id, OrderType.SELL, quantity, current_price * 0.99)
            else:
                # Price is too low, BUY to push it up
                return Order(self.trader_id, OrderType.BUY, quantity, current_price * 1.01)

        return None


class ForexMarket:
    """Main forex market environment"""

    def __init__(self, initial_price: float,
                 data_path: str = 'data/merged_data.csv',
                 num_speculators: int = 50,
                 num_hedgers: int = 20,
                 num_fundamentalists: int = 10,
                 include_central_bank: bool = True):
        """Initialize forex market with agents"""

        self.time_step = 0

        self.current_price = initial_price
        self.initial_price = initial_price

        self.traders: List[Trader] = []
        self.order_book = OrderBook()

        self.price_history = [initial_price]
        self.volume_history = []
        self.sentiment = 0.0

        try:
            # Load historical data (for fundamentalist updates)
            self.historical_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

            if self.historical_data.empty:
                raise ValueError("Data file is empty.")

            self.max_steps = len(self.historical_data)

        except (FileNotFoundError, ValueError) as e:
            # If data load fails, fundamentalists will use default logic (constant fundamental value)
            print(f"Error loading data for fundamentalists: {e}. Fundamentalists will use initial price as constant fundamental value.")
            self.historical_data = pd.DataFrame()
            self.max_steps = 100 # Default simulation length for fallback

        self._initialize_traders(num_speculators, num_hedgers,
                                 num_fundamentalists, include_central_bank, initial_price)

    def _initialize_traders(self, num_spec, num_hedge, num_fund, include_cb, initial_price):
        """Create initial population of traders"""
        trader_id = 0

        # Speculators - Give half of them an initial position
        for i in range(num_spec):
            trader = Speculator(trader_id, initial_capital=10000)
            if i % 2 == 0:  # Half the speculators have initial AUD position
                trader.position = np.random.uniform(500, 2000)
            self.traders.append(trader)
            trader_id += 1

        # Hedgers - All have an initial position for rebalancing
        for _ in range(num_hedge):
            trader = Hedger(trader_id, initial_capital=50000)
            trader.position = np.random.uniform(5000, 15000)  # All have a position
            self.traders.append(trader)
            trader_id += 1

        # Fundamentalists - All have an initial position
        for _ in range(num_fund):
            trader = Fundamentalist(trader_id, initial_capital=100000,
                                    fundamental_value=initial_price * np.random.uniform(0.99, 1.01))
            trader.position = np.random.uniform(10000, 30000)  # All have a position
            self.traders.append(trader)
            trader_id += 1

        if include_cb:
            cb = CentralBank(trader_id, initial_capital=1000000,
                             target_rate=initial_price)
            self.traders.append(cb)

    def step(self, external_shock=0.0):
        """Execute one time step of the simulation."""

        # 1. Get current economic indicators from historical data
        current_data = {} # Default to no economic indicators

        # Check if we have historical data and if we are still within its bounds
        if not self.historical_data.empty and self.time_step < self.max_steps:
            current_data = self.historical_data.iloc[self.time_step]

        economic_indicators = {
            'interest_diff': current_data.get('interest_diff', 0),
            'commodity_index': current_data.get('IronOre_Price', 1.0)
        }

        self._update_sentiment()
        volatility = self._calculate_volatility()

        market_state = {
            'current_price': self.current_price,
            'sentiment': self.sentiment,
            'volatility': volatility
        }

        # 2. Update Fundamentalist's fundamental value and collect orders
        for trader in self.traders:
            if isinstance(trader, Fundamentalist):
                # Fundamentalists continue to operate, but indicators are 0/1 after max_steps
                trader.update_fundamental_value(economic_indicators)

            order = trader.decide_action(market_state)
            if order:
                self.order_book.add_order(order)

        # 3. Match orders and update price
        clearing_price = self.order_book.match_orders()

        if clearing_price:
            self.current_price = clearing_price

        # --- Apply External Shock (as a percentage change) ---
        if external_shock != 0.0:
            self.current_price *= (1 + external_shock)


        self.price_history.append(self.current_price)
        self.order_book.clear_orders()

        self.time_step += 1
        return True


    def _update_sentiment(self, window: int = 5):
        """Update market sentiment based on recent price changes"""
        if len(self.price_history) >= window:
            recent_return = (self.price_history[-1] - self.price_history[-window]) / self.price_history[-window]
            self.sentiment = np.clip(recent_return * 10, -1, 1)
        else:
            self.sentiment = 0.0

    def _calculate_volatility(self, window: int = 20) -> float:
        """Calculate rolling volatility"""
        if len(self.price_history) < window:
            return 0.0

        recent_prices = self.price_history[-window:]
        returns = np.diff(np.log(recent_prices))
        return np.std(returns)

    def get_market_statistics(self) -> Dict:
        """Return current market statistics"""
        return {
            'current_price': self.current_price,
            'price_change': (self.current_price - self.initial_price) / self.initial_price,
            'volatility': self._calculate_volatility(),
            'sentiment': self.sentiment,
            'num_traders': len(self.traders),
            'total_trades': len(self.order_book.trade_history)
        }


if __name__ == "__main__":
    print("Testing Forex ABM Framework...")

    data_file_path = 'data/merged_data.csv'

    try:
        # Initial price logic must be outside the model if using the modified __init__
        try:
            temp_data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
            initial_price_test = temp_data['AUD_USD_Close'].iloc[0]
        except Exception:
            initial_price_test = 0.70 # Fallback

        market = ForexMarket(initial_price=initial_price_test, # Pass the price
                             data_path=data_file_path,
                             num_speculators=20,
                             num_hedgers=10,
                             num_fundamentalists=5,
                             include_central_bank=False)

        print(f"Initialized market with {len(market.traders)} traders")
        print(f"Initial price (from test setup): ${market.current_price:.4f}")

        # Use a dynamic step count or a fixed number, depending on data availability
        sim_steps = min(100, market.max_steps if market.max_steps > 0 else 100)

        for t in range(sim_steps):
            market.step() # No external shock in this basic test

            if t % 20 == 0:
                stats = market.get_market_statistics()
                print(f"Step {t}: Price=${stats['current_price']:.4f}, "
                      f"Change={(stats['current_price'] - market.initial_price) / market.initial_price * 100:.2f}%, "
                      f"Vol={stats['volatility']:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during market initialization or simulation: {e}")
        print("Ensure 'data/merged_data.csv' exists and is correctly formatted.")

    print("\nSimulation completed!")