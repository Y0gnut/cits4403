"""
Visualization utilities for AUD Exchange Rate ABM
Provides plotting functions for market dynamics and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_price_history(price_history: List[float],
                       real_data: pd.Series = None,
                       title: str = "AUD/USD Exchange Rate"):
    """
    Plot simulated price history with optional comparison to real data

    Args:
        price_history: List of simulated prices
        real_data: Optional pandas Series with real historical data
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot simulated data
    ax.plot(price_history, label='Simulated', linewidth=2, alpha=0.8)

    # Plot real data if provided
    if real_data is not None:
        ax.plot(real_data.values, label='Historical',
                linewidth=2, alpha=0.6, linestyle='--')

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Exchange Rate (USD)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_returns_distribution(price_history: List[float],
                              real_data: pd.Series = None):
    """
    Plot distribution of returns with comparison to normal distribution
    """
    # Calculate returns
    prices = np.array(price_history)
    returns = np.diff(np.log(prices))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(returns, bins=50, density=True, alpha=0.7,
                 label='Simulated Returns', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0].plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 'r-', linewidth=2, label='Normal Distribution')

    axes[0].set_xlabel('Log Returns', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Distribution of Returns', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_volatility_analysis(price_history: List[float], window: int = 20):
    """
    Plot rolling volatility and volatility clustering
    """
    prices = np.array(price_history)
    returns = np.diff(np.log(prices))

    # Calculate rolling volatility
    volatility = pd.Series(returns).rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Returns plot
    axes[0].plot(returns, linewidth=1, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Returns', fontsize=11)
    axes[0].set_title('Log Returns and Volatility Clustering',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Volatility plot
    axes[1].plot(volatility, linewidth=2, color='orange')
    axes[1].fill_between(range(len(volatility)), volatility, alpha=0.3, color='orange')
    axes[1].set_xlabel('Time Steps', fontsize=11)
    axes[1].set_ylabel('Volatility', fontsize=11)
    axes[1].set_title(f'Rolling Volatility (window={window})',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_agent_composition(traders: List, title: str = "Market Composition"):
    """
    Plot composition of different trader types
    """
    from collections import Counter

    # Count trader types
    trader_types = [trader.trader_type.value for trader in traders]
    type_counts = Counter(trader_types)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    wedges, texts, autotexts = ax.pie(type_counts.values(),
                                      labels=type_counts.keys(),
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90,
                                      textprops={'fontsize': 11})

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def plot_trader_wealth_evolution(traders: List, price_history: List[float]):
    """
    Plot wealth evolution of different trader types
    """
    from collections import defaultdict

    # Group traders by type
    wealth_by_type = defaultdict(list)

    for t in range(len(price_history)):
        for trader in traders:
            if t < len(trader.wealth_history):
                wealth_by_type[trader.trader_type.value].append(
                    trader.wealth_history[t]
                )

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {'Speculator': '#FF6B6B', 'Hedger': '#4ECDC4',
              'Fundamentalist': '#45B7D1', 'CentralBank': '#FFA07A'}

    for trader_type, wealth_list in wealth_by_type.items():
        if wealth_list:
            # Average wealth for each type
            avg_wealth = [sum(wealth_list[i::len(traders)]) /
                          wealth_list[i::len(traders)].__len__()
                          for i in range(min(len(wealth_list), len(price_history)))]
            ax.plot(avg_wealth, label=trader_type,
                    linewidth=2, color=colors.get(trader_type))

    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Average Wealth', fontsize=12)
    ax.set_title('Trader Wealth Evolution by Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_order_book_depth(buy_orders: List, sell_orders: List):
    """
    Visualize order book depth at a point in time
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract prices and quantities
    buy_prices = [o.price for o in buy_orders if o.price]
    buy_quantities = [o.quantity for o in buy_orders if o.price]
    sell_prices = [o.price for o in sell_orders if o.price]
    sell_quantities = [o.quantity for o in sell_orders if o.price]

    # Plot buy orders
    if buy_prices:
        ax.bar(buy_prices, buy_quantities, width=0.0001,
               color='green', alpha=0.6, label='Buy Orders')

    # Plot sell orders
    if sell_prices:
        ax.bar(sell_prices, sell_quantities, width=0.0001,
               color='red', alpha=0.6, label='Sell Orders')

    ax.set_xlabel('Price', fontsize=12)
    ax.set_ylabel('Quantity', fontsize=12)
    ax.set_title('Order Book Depth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(param_values: List,
                              results: List[Dict],
                              param_name: str,
                              metric_name: str = 'volatility'):
    """
    Plot parameter sensitivity analysis results
    """
    metric_values = [r[metric_name] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(param_values, metric_values, marker='o',
            linewidth=2, markersize=8, color='#4ECDC4')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(f'Sensitivity Analysis: {param_name} vs {metric_name}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_interactive_dashboard(price_history: List[float],
                                 volume_history: List[float] = None,
                                 sentiment_history: List[float] = None):
    """
    Create interactive Plotly dashboard
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Exchange Rate', 'Trading Volume', 'Market Sentiment'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )

    # Price chart
    fig.add_trace(
        go.Scatter(y=price_history, mode='lines', name='AUD/USD',
                   line=dict(color='#4ECDC4', width=2)),
        row=1, col=1
    )

    # Volume chart
    if volume_history:
        fig.add_trace(
            go.Bar(y=volume_history, name='Volume',
                   marker=dict(color='#FF6B6B', opacity=0.6)),
            row=2, col=1
        )

    # Sentiment chart
    if sentiment_history:
        colors = ['red' if s < 0 else 'green' for s in sentiment_history]
        fig.add_trace(
            go.Bar(y=sentiment_history, name='Sentiment',
                   marker=dict(color=colors, opacity=0.6)),
            row=3, col=1
        )

    # Update layout
    fig.update_xaxes(title_text="Time Steps", row=3, col=1)
    fig.update_yaxes(title_text="Rate (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="AUD/USD Market Dynamics Dashboard",
        title_font_size=16
    )

    return fig


def plot_comparison_grid(simulations: Dict[str, List[float]],
                         title: str = "Scenario Comparison"):
    """
    Plot multiple simulation scenarios in a grid
    """
    n_scenarios = len(simulations)
    n_cols = 2
    n_rows = (n_scenarios + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_scenarios > 1 else [axes]

    for idx, (scenario_name, prices) in enumerate(simulations.items()):
        ax = axes[idx]
        ax.plot(prices, linewidth=2)
        ax.set_title(scenario_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Exchange Rate', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


def plot_autocorrelation(price_history: List[float], max_lags: int = 50):
    """
    Plot autocorrelation function of returns and squared returns
    """
    prices = np.array(price_history)
    returns = np.diff(np.log(prices))
    squared_returns = returns ** 2

    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ACF of returns
    plot_acf(returns, lags=max_lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation: Returns', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lag', fontsize=11)

    # ACF of squared returns (volatility clustering)
    plot_acf(squared_returns, lags=max_lags, ax=axes[1], alpha=0.05)
    axes[1].set_title('Autocorrelation: Squared Returns',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lag', fontsize=11)

    plt.tight_layout()
    return fig


def save_all_plots(figures: List, prefix: str = "figure",
                   output_dir: str = "outputs"):
    """
    Save all figures to files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for idx, fig in enumerate(figures):
        filename = f"{output_dir}/{prefix}_{idx + 1}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Generate sample data
    np.random.seed(42)
    price_history = [0.70]
    for _ in range(500):
        change = np.random.normal(0, 0.005)
        price_history.append(price_history[-1] * (1 + change))

    # Test plots
    fig1 = plot_price_history(price_history)
    fig2 = plot_returns_distribution(price_history)
    fig3 = plot_volatility_analysis(price_history)

    plt.show()
    print("Visualization tests completed!")