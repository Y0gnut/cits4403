"""
Data Collection Script for AUD Exchange Rate Modeling
Collects historical data for AUD/USD exchange rate and related economic indicators
"""

import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os
from functools import reduce
import requests
import io
import urllib3
import logging

# Disable urllib3's InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging, using utf-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ForexDataCollector:
    """Collects and processes forex and economic data"""

    def __init__(self, start_date='2019-01-01', end_date=None):
        """
        Initialize data collector

        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (default: today)
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = {}
        logger.info(f"Initialized ForexDataCollector for {self.start_date} to {self.end_date}")

    def collect_aud_usd_rate(self):
        """Collect AUD/USD exchange rate data from Yahoo Finance"""
        logger.info("Collecting AUD/USD exchange rate data...")
        try:
            aud_usd = yf.download('AUDUSD=X',
                                start=self.start_date,
                                end=self.end_date,
                                progress=False,
                                auto_adjust=False)
            if not aud_usd.empty:
                if isinstance(aud_usd.columns, pd.MultiIndex):
                    aud_usd.columns = aud_usd.columns.get_level_values(0)
                self.data['aud_usd'] = aud_usd
                logger.info(f"Success: Collected {len(aud_usd)} days of AUD/USD data")
                logger.info(f"Date range: {aud_usd.index[0].date()} to {aud_usd.index[-1].date()}")
                return True
            else:
                logger.error("Error: No data returned for AUD/USD")
                return False
        except Exception as e:
            logger.error(f"Error collecting AUD/USD data: {e}")
            return False

    def collect_iron_ore_price(self):
        """Collect iron ore price proxy (using commodity ETFs or indices)"""
        logger.info("Collecting iron ore price data...")
        try:
            # Using 'VALE' as a proxy for Iron Ore Price as in the original script
            iron_ore = yf.download('VALE',
                                 start=self.start_date,
                                 end=self.end_date,
                                 progress=False,
                                 auto_adjust=False)
            if not iron_ore.empty:
                if isinstance(iron_ore.columns, pd.MultiIndex):
                    iron_ore.columns = iron_ore.columns.get_level_values(0)
                self.data['iron_ore'] = iron_ore[['Close']].rename(columns={'Close': 'IronOre_Price'})
                logger.info(f"Success: Collected {len(iron_ore)} days of iron ore proxy data")
                return True
            else:
                logger.error("Error: No data returned for iron ore proxy")
                return False
        except Exception as e:
            logger.error(f"Error collecting iron ore proxy data: {e}")
            return False

    def collect_interest_rates(self):
        """
        Collect US and RBA interest rate data.
        US Rate from FRED (DFF). RBA Rate directly from RBA website.
        """
        logger.info("Collecting interest rate data...")
        overall_success = True

        # 1. Collect US Federal Funds Rate
        try:
            us_rate_df = pdr.get_data_fred('DFF',
                                          start=self.start_date,
                                          end=self.end_date)
            us_rate_df.columns = ['US_Rate']
            self.data['us_interest_rate'] = us_rate_df
            logger.info("Success: Collected US interest rate data (FRED: DFF)")
        except Exception as e:
            logger.error(f"Error collecting US interest rate data (DFF): {e}")
            overall_success = False

        # 2. Collect RBA Cash Rate Target
        try:
            RBA_URL = 'https://www.rba.gov.au/statistics/tables/csv/f1-data.csv'
            logger.info(f"Attempting to fetch RBA data from: {RBA_URL}")
            logger.warning("Using verify=False for RBA request (insecure). Consider updating certifi or configuring SSL.")
            response = requests.get(RBA_URL, verify=False)
            response.raise_for_status()
            rba_df = pd.read_csv(
                io.StringIO(response.text),
                skiprows=10,
                index_col=0,
                parse_dates=True
            )
            if len(rba_df.columns) > 0:
                target_col_name = rba_df.columns[0]
                rba_interest_rate = rba_df[[target_col_name]].rename(columns={target_col_name: 'AU_Rate'})
                logger.info(f"Note: Using first data column (Series ID: {target_col_name}) as 'AU_Rate'")
            else:
                raise ValueError("RBA CSV contains no data columns after skipping header rows.")
            rba_interest_rate = rba_interest_rate.ffill().dropna()
            start_dt = pd.to_datetime(self.start_date)
            end_dt = pd.to_datetime(self.end_date)
            rba_interest_rate = rba_interest_rate.loc[start_dt:end_dt]
            self.data['rba_interest_rate'] = rba_interest_rate
            logger.info("Success: Collected RBA interest rate data (RBA: F1 first series)")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error collecting RBA interest rate data: HTTP or connection error: {e}")
            overall_success = False
        except Exception as e:
            logger.error(f"Error collecting RBA interest rate data: Unable to parse data: {e}")
            overall_success = False

        return overall_success

    def collect_vix_index(self):
        """Collect VIX (market volatility index) as risk sentiment indicator"""
        logger.info("Collecting VIX index data...")
        try:
            vix = yf.download('^VIX',
                            start=self.start_date,
                            end=self.end_date,
                            progress=False,
                            auto_adjust=False)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                self.data['vix'] = vix[['Close']].rename(columns={'Close': 'VIX'})
                logger.info(f"Success: Collected {len(vix)} days of VIX data")
                return True
            else:
                logger.error("Error: No data returned for VIX")
                return False
        except Exception as e:
            logger.error(f"Error collecting VIX data: {e}")
            return False

    def collect_all_data(self):
        """Collect all required data"""
        logger.info("="*60)
        logger.info("Starting data collection for AUD Exchange Rate Model")
        logger.info("="*60)

        results = {
            'AUD/USD Rate': self.collect_aud_usd_rate(),
            'Iron Ore Proxy': self.collect_iron_ore_price(),
            'Interest Rates (AU/US)': self.collect_interest_rates(),
            'VIX Index': self.collect_vix_index()
        }

        # Keep Interest Rates in the summary despite collecting two series
        rate_success = results.pop('Interest Rates (AU/US)')
        results['Interest Rates (AU/US)'] = rate_success

        logger.info("\n" + "="*60)
        logger.info("Data Collection Summary:")
        logger.info("="*60)
        for name, success in results.items():
            status = "Success" if success else "Failed"
            logger.info(f"{name:25s}: {status}")

        return all(results.values())

    def merge_data(self):
        """Merge all collected data into a single DataFrame"""
        if not self.data:
            logger.error("No data to merge. Please collect data first.")
            return None

        logger.info("Merging all datasets...")

        data_frames_to_merge = []

        # 1. Base data: AUD/USD Close
        if 'aud_usd' in self.data:
            df = self.data['aud_usd'][['Close']].rename(columns={'Close': 'AUD_USD_Close'})
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("AUD/USD data index is not DatetimeIndex")
                return None
            data_frames_to_merge.append(df)
        else:
            logger.error("Error: Missing AUD/USD data - cannot merge")
            return None

        # 2. Add other single-column dataframes
        for key in ['iron_ore', 'vix', 'us_interest_rate', 'rba_interest_rate']:
            if key in self.data:
                df = self.data[key]
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.error(f"{key} data index is not DatetimeIndex")
                    return None
                data_frames_to_merge.append(df)

        # Merge on index (DatetimeIndex), anchoring to AUD dates only
        # Root cause: outer-joining then forward-filling propagated AUD_USD_Close across
        # dates where there was no new AUD quote, creating many zero returns.
        # Fix: start from AUD series and left-join others so the index stays on AUD dates.
        merged_df = data_frames_to_merge[0].copy()  # AUD base (has 'AUD_USD_Close')
        for right in data_frames_to_merge[1:]:
            merged_df = pd.merge(merged_df, right, left_index=True, right_index=True, how='left')

        merged_df = merged_df.sort_index()

        # Forward-fill only non-price columns; keep AUD_USD_Close as-is (no ffill)
        non_price_cols = [c for c in merged_df.columns if c != 'AUD_USD_Close']
        if non_price_cols:
            merged_df[non_price_cols] = merged_df[non_price_cols].ffill()

        # Ensure we have no missing prices (shouldn't happen with AUD base)
        merged_df = merged_df.dropna(subset=['AUD_USD_Close'])

        # Calculate Interest Rate Differential
        if 'AU_Rate' in merged_df.columns and 'US_Rate' in merged_df.columns:
            # Calculate the differential (AU Rate - US Rate)
            merged_df['interest_diff'] = merged_df['AU_Rate'] - merged_df['US_Rate']
            # Optionally keep US_Rate and AU_Rate for analysis (columns are kept by default due to merge strategy)
            # merged_df = merged_df.drop(columns=['AU_Rate', 'US_Rate'], errors='ignore')
        else:
            logger.warning("Missing AU_Rate or US_Rate, setting interest_diff to 0")
            merged_df['interest_diff'] = 0

        # Log missing values
        missing_values = merged_df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values in merged data:\n{missing_values[missing_values > 0]}")

        logger.info(f"Success: Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        logger.info(f"Columns: {list(merged_df.columns)}")
        logger.info(f"Data preview:\n{merged_df.head()}")

        return merged_df

    def save_data(self, output_dir='data'):
        """Save collected data to CSV files"""
        if not self.data:
            logger.error("No data to save. Please collect data first.")
            return False

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving data to '{output_dir}/' directory...")

        for name, df in self.data.items():
            filename = f"{output_dir}/{name}.csv"
            df.to_csv(filename, encoding='utf-8')
            logger.info(f"Success: Saved {name} to {filename}")

        merged = self.merge_data()
        if merged is not None:
            merged_filename = f"{output_dir}/merged_data.csv"
            merged.to_csv(merged_filename, encoding='utf-8')
            logger.info(f"Success: Saved merged data to {merged_filename}")

        logger.info("\n" + "="*60)
        logger.info("All data saved successfully!")
        logger.info("="*60)

        return True

    def get_data_statistics(self):
        """Print statistics about collected data"""
        if not self.data:
            logger.error("No data collected yet.")
            return

        logger.info("\n" + "="*60)
        logger.info("Data Statistics Summary")
        logger.info("="*60)

        for name, df in self.data.items():
            logger.info(f"\n{name.upper()}:")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            logger.info(f"Total days: {len(df)}")
            if len(df.columns) > 0:
                logger.info(f"Columns: {list(df.columns)}")

def main():
    """Main function to run data collection"""
    collector = ForexDataCollector(start_date='2019-01-01')
    success = collector.collect_all_data()

    if success:
        collector.get_data_statistics()
        collector.save_data()
    else:
        logger.warning("Some data collection failed. Saving available data.")
        collector.save_data()

if __name__ == "__main__":
    # Optional: Set console encoding to UTF-8 on Windows
    try:
        import sys
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.warning(f"Failed to set console encoding to UTF-8: {e}")
    main()
