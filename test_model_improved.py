"""
Improved test script with market initialization
This version adds noise, shocks, and advanced dynamics to trigger trading
Save this file as: test_model_improved.py in the PROJECT ROOT directory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats # Import stats explicitly for probplot
from src.forex_abm import ForexMarket, Fundamentalist
from utils.reproducibility import set_seed, env_info, file_sha256, save_run_metadata

# Ensure the 'results/figures' directory exists before saving figures later
os.makedirs('results/figures', exist_ok=True)

# Set a deterministic seed early (can override via env var SEED)
SEED = int(os.getenv('SEED', '42'))
set_seed(SEED)

print("="*60)
print("Testing AUD Forex ABM Model (Advanced Dynamics)")
print("="*60)

# ============================================================
# Load real data
# ============================================================
print("\n[1] Loading real data...")
merged_data = None
data_available = False
initial_price = 0.70 # Default fallback price

try:
    # Attempt to load data
    merged_data = pd.read_csv('data/merged_data.csv', index_col=0, parse_dates=True)
    # Fix 1: Adjust column names based on the actual file content
    required_cols = ['AUD_USD_Close', 'interest_diff', 'IronOre_Price']
    if all(col in merged_data.columns for col in required_cols):
        print(f"Data loaded: {merged_data.shape[0]} days")
        # Use the closing price of the first day as the starting price
        initial_price = merged_data['AUD_USD_Close'].iloc[0]
        data_available = True
    else:
        raise ValueError(f"Data is missing required columns: {required_cols}")

except Exception as e:
    # Warning: File not found or data format error. Fall back to default price.
    print(f"Could not load data: {e}. Falling back to default initial price {initial_price:.4f}.")


# ============================================================
# Initialize market with better parameters
# ============================================================
print("\n[2] Initializing market with active agents...")

market = ForexMarket(
    initial_price=initial_price,
    data_path='data/merged_data.csv', # Pass path again in case ABM logic needs it
    num_speculators=50,
    num_hedgers=20,
    num_fundamentalists=10,
    include_central_bank=True
)

print(f"Market created with {len(market.traders)} agents")
print(f"  Initial price: ${market.current_price:.4f}")

# Give traders initial positions and capital variations
print("  Initializing agent states...")
for i, trader in enumerate(market.traders):
    if i % 3 == 0:
        trader.position = np.random.uniform(100, 1000)
    trader.capital *= np.random.uniform(0.8, 1.2)

# **Fix 2: Initialize Fundamentalist's data based on the loaded data**
fundamentalists = [t for t in market.traders if isinstance(t, Fundamentalist)]
fund_data = pd.DataFrame() # Initialize an empty DataFrame
if data_available:
    # Extract the data needed by the Fundamentalists, and ensure its length is sufficient
    fund_data = merged_data[required_cols]
    if len(fund_data) < 1000:
        print("Warning: Data less than 1000 steps. Using available data length.")

    # Set the Fundamentalist's initial fundamental value to the first value in the data
    for f in fundamentalists:
        f.fundamental_value = fund_data['AUD_USD_Close'].iloc[0]

print("  Agents initialized with varied positions")


# ============================================================
# Run simulation with realistic dynamics
# ============================================================
print("\n[3] Running 1000-step simulation with market dynamics...")

SIM_STEPS = min(1000, len(fund_data)) if data_available and len(fund_data) > 0 else 1000

price_changes = []
current_noise_std = 0.0008 # Lower baseline noise to let Speculators drive the price (first modification)
volatility_shock_timer = 0
VOLATILITY_SHOCK_DURATION = 15 # Number of steps the noise persists after the shock
 

for t in range(SIM_STEPS):
    # Volatility Clustering Mechanism
    if volatility_shock_timer > 0:
        # Shock duration: increase noise level (reduced multiplier from 5 -> 2)
        noise_std = current_noise_std * 2
        volatility_shock_timer -= 1
    else:
        # Normal period: baseline noise
        noise_std = current_noise_std

    noise = np.random.normal(0, noise_std)
    external_shock = 0 # Reset external shock each loop
    # A. Long-term Fundamental Update (Fundamentalist Dynamics)
    if data_available and t < len(fund_data):
        # Extract economic indicators for the current time step
        indicators = {
            # Final Fix 3: Use the correct column name 'interest_diff'
            'interest_diff': fund_data['interest_diff'].iloc[t],
            # Final Fix 4: Use the correct column name 'IronOre_Price' (as Commodity Index)
            'commodity_index': fund_data['IronOre_Price'].iloc[t]
        }
        # Update fundamental value for all Fundamentalists
        for f in fundamentalists:
            # Note: The update logic is inside the Fundamentalist class
            f.update_fundamental_value(indicators)
    elif not data_available and t % 50 == 0 and t > 0:
        # If no data, randomly slightly adjust Fundamentalist's fundamental value to prevent market stagnation at $0.7007
        for f in fundamentalists:
            f.fundamental_value *= (1 + np.random.normal(0, 0.0001))

    # B. External Shock (News Shocks)
    if t % 100 == 0 and t > 0:
        # Large, random news shock (reduced amplitude to 0.5%)
        shock = np.random.choice([-0.005, 0.005])
        print(f"\n  News shock at step {t}: {shock*100:+.1f}%")
        external_shock = shock
        # Activate volatility shock timer
        volatility_shock_timer = VOLATILITY_SHOCK_DURATION

    # C. Market Step
    # Pass both noise and shock simultaneously
    market.step(external_shock=external_shock + noise)

    if t > 0:
        price_change = (market.price_history[-1] - market.price_history[-2]) / market.price_history[-2]
        price_changes.append(price_change)

    if (t + 1) % 200 == 0:
        # FIX: Rename the local variable from 'stats' to 'market_stats_periodic' to avoid shadowing scipy.stats
        market_stats_periodic = market.get_market_statistics()
        recent_vol = np.std(price_changes[-100:]) * 100 if len(price_changes) >= 100 else 0
        print(f"  Step {t+1:4d}: Price=${market_stats_periodic['current_price']:.4f}, "
              f"Return={market_stats_periodic['price_change']*100:+.2f}%, "
              f"Recent Vol={recent_vol:.4f}%") # Print percentage

final_stats = market.get_market_statistics()
print(f"\nSimulation completed! ({SIM_STEPS} steps)")
print(f"\n  Final Results:")
print(f"    Final price:   ${final_stats['current_price']:.4f}")
print(f"    Total return:  {final_stats['price_change']*100:+.2f}%")
print(f"    Total trades:  {final_stats['total_trades']}")
print(f"    Avg volatility: {np.std(price_changes):.4f}")

# ============================================================
# Analyze results
# ============================================================
print("\n[4] Analyzing simulation results...")

prices = np.array(market.price_history)
returns = np.diff(np.log(prices))

print(f"  Price statistics:")
print(f"    Min: ${prices.min():.4f}")
print(f"    Max: ${prices.max():.4f}")
print(f"    Mean: ${prices.mean():.4f}")
print(f"    Std: {prices.std():.4f}")

print(f"\n  Return statistics:")
print(f"    Mean return: {returns.mean()*100:.4f}%")
print(f"    Volatility: {returns.std()*100:.2f}%")
print(f"    Skewness: {pd.Series(returns).skew():.3f}")
print(f"    Kurtosis: {pd.Series(returns).kurtosis():.3f}")

# ============================================================
# Create visualizations
# ============================================================
print("\n[5] Creating visualizations...")

# Plot 1: Price evolution, Returns, and Rolling Volatility
fig1, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(market.price_history, linewidth=1.5, color='blue', alpha=0.7)
axes[0].set_title('Simulated AUD/USD Price', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (USD)', fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(returns, linewidth=1, color='orange', alpha=0.6)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('Log Returns', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Log Returns', fontsize=11)
axes[1].grid(True, alpha=0.3)

rolling_vol = pd.Series(returns).rolling(20).std()
axes[2].plot(rolling_vol, linewidth=2, color='red', alpha=0.7)
axes[2].fill_between(range(len(rolling_vol)), rolling_vol, alpha=0.3, color='red')
axes[2].set_title('Rolling Volatility (20-day window)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time Steps', fontsize=11)
axes[2].set_ylabel('Volatility', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('results/figures/improved_simulation.png', dpi=300, bbox_inches='tight')
print("  Saved: results/figures/improved_simulation.png")
plt.close(fig1)

# Plot 2: Return distribution and Q-Q Plot
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
mu, sigma = returns.mean(), returns.std()
x = np.linspace(returns.min(), returns.max(), 100)
# Plot Normal Distribution curve
axes[0].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2),
            'r-', linewidth=2, label='Normal Distribution')
axes[0].set_xlabel('Returns', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Return Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# THIS LINE IS NOW CORRECT
stats.probplot(returns, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('results/figures/return_analysis.png', dpi=300, bbox_inches='tight')
print("  Saved: results/figures/return_analysis.png")
plt.close(fig2)

# Plot 3: Comparison with real data if available
if data_available:
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Ensure only the same number of steps are compared
    real_prices = merged_data['AUD_USD_Close'][:SIM_STEPS]
    real_returns = np.diff(np.log(real_prices.values))

    axes[0, 0].plot(real_prices.values, linewidth=1.5, alpha=0.7, label='Real', color='blue')
    axes[0, 0].set_title('Real AUD/USD Price', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Price', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Ensure simulated data length also matches
    axes[0, 1].plot(market.price_history[:SIM_STEPS], linewidth=1.5, alpha=0.7, label='Simulated', color='orange')
    axes[0, 1].set_title('Simulated AUD/USD Price', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(real_returns, bins=50, density=True, alpha=0.7, color='blue', label='Real')
    axes[1, 0].set_title('Real Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Returns', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(returns, bins=50, density=True, alpha=0.7, color='orange', label='Simulated')
    axes[1, 1].set_title('Simulated Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Returns', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig3.savefig('results/figures/real_vs_simulated.png', dpi=300, bbox_inches='tight')
    print("  Saved: results/figures/real_vs_simulated.png")
    plt.close(fig3)

    print(f"\n  Comparison with real data:")
    print(f"    Real volatility:      {real_returns.std()*100:.2f}%")
    print(f"    Simulated volatility: {returns.std()*100:.2f}%")
    print(f"    Real mean return:     {real_returns.mean()*100:.4f}%")
    print(f"    Simulated mean return: {returns.mean()*100:.4f}%")

# ============================================================
# Save results
# ============================================================
print("\n[6] Saving results...")

results_df = pd.DataFrame({
    'time_step': range(len(market.price_history)),
    'price': market.price_history
})
results_df.to_csv('results/improved_simulation.csv', index=False)
print("  Saved: results/improved_simulation.csv")

stats_df = pd.DataFrame({
    'metric': ['initial_price', 'final_price', 'total_return', 'volatility',
               'total_trades', 'mean_return', 'std_return', 'min_price', 'max_price'],
    'value': [
        prices[0], prices[-1], (prices[-1] - prices[0]) / prices[0],
        returns.std(), final_stats['total_trades'],
        returns.mean(), returns.std(), prices.min(), prices.max()
    ]
})
stats_df.to_csv('results/simulation_statistics.csv', index=False)
print("  Saved: results/simulation_statistics.csv")

# Save reproducibility metadata
metadata = {
    'seed': SEED,
    'sim_steps': int(SIM_STEPS),
    'data_available': bool(data_available),
    'data_file': 'data/merged_data.csv',
    'data_sha256': file_sha256('data/merged_data.csv'),
    'env': env_info(),
    'initial_price': float(prices[0]),
    'final_price': float(prices[-1]),
    'total_return': float((prices[-1] - prices[0]) / prices[0]),
    'volatility_logret': float(returns.std()),
    'total_trades': int(final_stats['total_trades'])
}
save_run_metadata('results/run_metadata.json', metadata)
print("  Saved: results/run_metadata.json (seed, env, data hash)")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("ADVANCED DYNAMICS SIMULATION COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nKey Findings (Advanced Model):")
print(f"  Price moved from ${prices[0]:.4f} to ${prices[-1]:.4f}")
print(f"  Total return: {(prices[-1]/prices[0]-1)*100:+.2f}%")
print(f"  Volatility: {returns.std()*100:.2f}% per day")
print(f"  Total trades: {final_stats['total_trades']}")
# Check for fat tails (kurtosis > 3 for leptokurtic distribution)
kurtosis_value = pd.Series(returns).kurtosis()
tail_status = 'fat tails' if kurtosis_value > 3 else 'normal'
print(f"  Market shows {tail_status} (kurtosis={kurtosis_value:.2f})")

print("\nGenerated files:")
print("  results/figures/improved_simulation.png (Price, Returns, Rolling Vol)")
print("  results/figures/return_analysis.png (Distribution, Q-Q Plot)")
if data_available:
    print("  results/figures/real_vs_simulated.png (Real Data Comparison)")
print("  results/improved_simulation.csv")
print("  results/simulation_statistics.csv")
print("  results/run_metadata.json")

print("\nThe model now incorporates dynamic fundamental value and volatility clustering!")
print("="*60)
