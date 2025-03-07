import warnings
import mysql.connector
import pandas as pd
import numpy as np
import talib as ta
import time
from datetime import datetime, timedelta
import concurrent.futures
from multiprocessing import cpu_count
from threading import Lock

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None

class TechnicalIndicatorCalculator:
    def __init__(self):
        """Initialize the calculator with database configuration."""
        self.db_config = {
            'host': 'localhost',
            'user': 'dhan_hq',
            'password': 'Passw0rd@098',
            'database': 'dhanhq_db',
            'auth_plugin': 'mysql_native_password',
            'use_pure': True
        }
        # Add progress tracking attributes
        self.start_time = None
        self.processed_symbols = 0
        self.total_symbols = 0
        self.total_records_processed = 0
        self.errors_encountered = 0
        # Add threading lock for progress updates
        self.lock = Lock()
        # Determine optimal number of workers
        self.max_workers = min(12, cpu_count() * 2)  # Limit max workers to 32

    @property
    def expected_columns(self):
        """List of expected indicator columns."""
        return [
            # Original Technical Indicators
            'sma_20', 'sma_50', 'sma_200', 
            'ema_20', 'ema_50',
            'macd_line', 'macd_signal', 'macd_histogram',
            'adx_14', 'di_plus_14', 'di_minus_14',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'atr_14', 'natr',
            'rsi_14',
            'stochastic_k', 'stochastic_d',
            'cci_20', 'mfi_14', 'williams_r', 'roc',
            'obv', 'ad_line', 'adosc',
            'trix', 'ultosc', 'bop', 'stddev', 'var',
            
            # New Return Periods
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 
            'return_20d', 'return_40d', 'return_60d', 'return_120d',
            
            # Calendar Features
            'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekday_1', 'is_weekday_2', 'is_weekday_3', 'is_weekday_4', 'is_weekday_5',
            'is_month_start', 'is_month_end', 
            'is_quarter_start', 'is_quarter_end',
            'is_year_start', 'is_year_end',
            
            # Nifty Correlation
            'nifty_corr_20d', 'nifty_corr_60d', 'nifty_corr_120d', 'nifty_corr_full',
            
            # Relative Strength vs Nifty
            'rs_nifty_5d', 'rs_nifty_10d', 'rs_nifty_20d', 'rs_nifty_60d', 'rs_nifty_120d',
            
            # Volume Indicators
            'volume_sma_5d', 'volume_sma_10d', 'volume_sma_20d', 'volume_sma_50d',
            'volume_ratio_20d',
            'volume_roc_1d', 'volume_roc_5d', 'volume_roc_10d',
            'cmf_20d', 'pvt', 'volume_oscillator', 'eom_14d', 'vama_20d', 'vpci_20d'
        ]

    def process_symbol_batch(self, symbols):
        """Process a batch of symbols."""
        results = []
        for symbol in symbols:
            try:
                records_processed = self.process_symbol_data(symbol)
                with self.lock:
                    self.processed_symbols += 1
                    if records_processed:
                        self.total_records_processed += records_processed
                    
                    # Print progress every 10 symbols
                    if self.processed_symbols % 10 == 0:
                        self.print_progress_stats()
                
                results.append((symbol, records_processed))
            except Exception as e:
                with self.lock:
                    self.errors_encountered += 1
                results.append((symbol, 0))
        return results

    def log_progress(self, message, level="INFO", verbosity=1):
        """
        Log progress with verbosity control.
        verbosity levels:
        0 - Show only major progress and errors
        1 - Show basic processing info
        2 - Show detailed debug info
        """
        if level == "DEBUG" and verbosity < 2:
            return
        if level == "INFO" and verbosity < 1:
            return
        if level == "ERROR" or level == "WARNING":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
        
    def connect_to_db(self):
        """Create a new database connection for each thread."""
        return mysql.connector.connect(**self.db_config)

    def fetch_historical_data(self, symbol, last_processed_date=None):
        """Fetch historical data for a given symbol with explicit sorting."""
        conn = None
        try:
            conn = self.connect_to_db()
            
            if last_processed_date:
                query = """
                    SELECT date, trading_symbol, open, high, low, close, volume 
                    FROM historical_data 
                    WHERE trading_symbol = %s 
                        AND date >= DATE_SUB(%s, INTERVAL 365 DAY)
                    ORDER BY date ASC  -- Explicit ascending sort
                """
                df = pd.read_sql(query, conn, params=(symbol, last_processed_date))
            else:
                query = """
                    SELECT date, trading_symbol, open, high, low, close, volume 
                    FROM historical_data 
                    WHERE trading_symbol = %s 
                    ORDER BY date ASC  -- Explicit ascending sort
                """
                df = pd.read_sql(query, conn, params=(symbol,))
            
            # Double-check sorting in pandas
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            
            # Verify data integrity
            self.verify_data_integrity(df, symbol)
            
            return df
        except Exception as e:
            self.log_progress(f"Error fetching data for {symbol}: {str(e)}", "ERROR")
            raise
        finally:
            if conn:
                conn.close()

    def verify_data_integrity(self, df, symbol):
        """Verify data integrity and sorting."""
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Check for date sorting
        if not df['date'].is_monotonic_increasing:
            raise ValueError(f"Data for {symbol} is not properly sorted by date")

        # Check for duplicates
        duplicates = df[df.duplicated(['date'])]
        if not duplicates.empty:
            raise ValueError(f"Duplicate dates found for {symbol}")

        # Check for missing values
        if df.isnull().any().any():
            raise ValueError(f"Missing values found in data for {symbol}")

    def get_last_processed_date(self, symbol):
        """Get the last processed date for a given symbol."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) 
                FROM technical_indicators 
                WHERE trading_symbol = %s
            """, (symbol,))
            last_date = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return last_date
        except Exception as e:
            self.log_progress(f"Error getting last date for {symbol}: {str(e)}", "ERROR")
            if 'conn' in locals():
                conn.close()
            return None
    
    def calculate_correlation_with_nifty(self, symbol, df):
        """Calculate correlation between the stock and Nifty index."""
        try:
            data_length = len(df)
            
            # If the symbol is Nifty itself, return zeros (no correlation)
            if symbol == 'NIFTY':
                return {
                    'nifty_corr_20d': np.zeros(data_length),
                    'nifty_corr_60d': np.zeros(data_length),
                    'nifty_corr_120d': np.zeros(data_length),
                    'nifty_corr_full': np.zeros(data_length)
                }
                
            # Connect to DB
            conn = self.connect_to_db()
            
            # Get the date range from the stock data
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            # Fetch Nifty data for the same period
            query = """
                SELECT date, close 
                FROM historical_data 
                WHERE trading_symbol = 'NIFTY' 
                    AND date BETWEEN %s AND %s
                ORDER BY date ASC
            """
            nifty_df = pd.read_sql(query, conn, params=(min_date, max_date))
            conn.close()
            
            if nifty_df.empty:
                self.log_progress(f"No Nifty data available for correlation with {symbol}", "WARNING")
                # Return zeros (neutral) instead of NaN
                return {
                    'nifty_corr_20d': np.zeros(data_length),
                    'nifty_corr_60d': np.zeros(data_length),
                    'nifty_corr_120d': np.zeros(data_length),
                    'nifty_corr_full': np.zeros(data_length)
                }
            
            # Merge the data
            merged_df = pd.merge(
                df[['date', 'close']], 
                nifty_df, 
                on='date', 
                how='left',
                suffixes=('_stock', '_nifty')
            )
            
            # Initialize correlation arrays with zeros (neutral)
            corr_20d = np.zeros(data_length)
            corr_60d = np.zeros(data_length)
            corr_120d = np.zeros(data_length)
            corr_full = np.zeros(data_length)
            
            # Calculate rolling correlation with different windows
            if len(merged_df) >= 20:
                rolling_corr_20 = merged_df['close_stock'].rolling(window=20).corr(merged_df['close_nifty'])
                # Fill only where we have valid values
                valid_mask = ~rolling_corr_20.isna()
                if valid_mask.any():
                    corr_20d[valid_mask.values] = rolling_corr_20.values[valid_mask.values]
            
            if len(merged_df) >= 60:
                rolling_corr_60 = merged_df['close_stock'].rolling(window=60).corr(merged_df['close_nifty'])
                # Fill only where we have valid values
                valid_mask = ~rolling_corr_60.isna()
                if valid_mask.any():
                    corr_60d[valid_mask.values] = rolling_corr_60.values[valid_mask.values]
                
            if len(merged_df) >= 120:
                rolling_corr_120 = merged_df['close_stock'].rolling(window=120).corr(merged_df['close_nifty'])
                # Fill only where we have valid values
                valid_mask = ~rolling_corr_120.isna()
                if valid_mask.any():
                    corr_120d[valid_mask.values] = rolling_corr_120.values[valid_mask.values]
                
            # Calculate full period correlation
            if len(merged_df) >= 10:  # Minimum data points for meaningful correlation
                try:
                    full_corr = merged_df['close_stock'].corr(merged_df['close_nifty'])
                    if not pd.isna(full_corr):
                        # Fill all rows with the same value
                        corr_full = np.full(data_length, full_corr)
                except Exception as e:
                    self.log_progress(f"Error calculating full correlation for {symbol}: {str(e)}", "WARNING")
                
            return {
                'nifty_corr_20d': corr_20d,
                'nifty_corr_60d': corr_60d,
                'nifty_corr_120d': corr_120d,
                'nifty_corr_full': corr_full
            }
            
        except Exception as e:
            self.log_progress(f"Error calculating Nifty correlation for {symbol}: {str(e)}", "ERROR")
            # Return zeros (neutral) instead of NaN
            return {
                'nifty_corr_20d': np.zeros(data_length),
                'nifty_corr_60d': np.zeros(data_length),
                'nifty_corr_120d': np.zeros(data_length),
                'nifty_corr_full': np.zeros(data_length)
            }
    
    def calculate_relative_strength(self, symbol, df):
        """Calculate relative strength of stock vs Nifty."""
        try:
            data_length = len(df)
            
            # If the symbol is Nifty itself, return ones (neutral RS)
            if symbol == 'NIFTY':
                return {
                    f'rs_nifty_{period}d': np.ones(data_length) 
                    for period in [5, 10, 20, 60, 120]
                }
                
            # Connect to DB
            conn = self.connect_to_db()
            
            # Get the date range from the stock data
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            # Fetch Nifty data for the same period with buffer for calculations
            buffer_days = 120  # Add buffer for longest calculation period
            query = """
                SELECT date, close 
                FROM historical_data 
                WHERE trading_symbol = 'NIFTY' 
                    AND date BETWEEN DATE_SUB(%s, INTERVAL %s DAY) AND %s
                ORDER BY date ASC
            """
            nifty_df = pd.read_sql(query, conn, params=(min_date, buffer_days, max_date))
            conn.close()
            
            if nifty_df.empty:
                self.log_progress(f"No Nifty data available for relative strength with {symbol}", "WARNING")
                # Return ones (neutral RS) instead of NaN
                return {
                    f'rs_nifty_{period}d': np.ones(data_length) 
                    for period in [5, 10, 20, 60, 120]
                }
            
            # Merge the data
            merged_df = pd.merge(
                df[['date', 'close']], 
                nifty_df, 
                on='date', 
                how='left',
                suffixes=('_stock', '_nifty')
            )
            
            # Initialize result arrays with ones (neutral RS)
            results = {
                f'rs_nifty_{period}d': np.ones(data_length)
                for period in [5, 10, 20, 60, 120]
            }
            
            # Calculate relative strength for different periods
            periods = [5, 10, 20, 60, 120]  # 1-week, 2-week, 1-month, 3-month, 6-month
            
            for period in periods:
                if len(merged_df) >= period + 1:  # Need at least period+1
                    try:
                        # Calculate percentage change for both stock and Nifty
                        stock_pct = merged_df['close_stock'].pct_change(period)
                        nifty_pct = merged_df['close_nifty'].pct_change(period, fill_method=None)
                        
                        # Calculate relative strength (stock performance relative to Nifty)
                        # Add a small number to avoid division by zero
                        rs = (1 + stock_pct) / (1 + nifty_pct + 1e-10)
                        
                        # Replace infinity and NaN with 1.0 (neutral)
                        rs = rs.replace([np.inf, -np.inf, np.nan], 1.0)
                        
                        # Only store valid values (skip the first 'period' entries which will be NaN)
                        valid_mask = ~rs.isna()
                        if valid_mask.any():
                            # Only replace values where we have valid data
                            valid_indices = np.where(valid_mask.values)[0]
                            results[f'rs_nifty_{period}d'][valid_indices] = rs.values[valid_indices]
                        
                    except Exception as e:
                        self.log_progress(f"Error calculating {period}d RS for {symbol}: {str(e)}", "WARNING")
                    
            return results
            
        except Exception as e:
            self.log_progress(f"Error calculating relative strength for {symbol}: {str(e)}", "ERROR")
            # Return ones (neutral RS) instead of NaN
            return {
                f'rs_nifty_{period}d': np.ones(data_length)
                for period in [5, 10, 20, 60, 120]
            }
    
    def add_calendar_features(self, df):
        """Add calendar-based features to the dataframe."""
        try:
            # Make a copy of the original date column to preserve it
            df['original_date'] = df['date']
            
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date_temp'] = pd.to_datetime(df['date'])
            else:
                df['date_temp'] = df['date']
            
            # Extract basic date components using the temporary column
            df['day_of_week'] = df['date_temp'].dt.dayofweek  # Monday=0, Sunday=6
            df['day_of_month'] = df['date_temp'].dt.day
            df['month'] = df['date_temp'].dt.month
            df['quarter'] = df['date_temp'].dt.quarter
            
            # One-hot encode day of week (create 5 columns for trading days)
            for i in range(5):  # 0=Monday, 4=Friday
                df[f'is_weekday_{i+1}'] = (df['date_temp'].dt.dayofweek == i).astype(int)
            
            # Month/quarter/year boundaries
            df['is_month_start'] = df['date_temp'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date_temp'].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df['date_temp'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['date_temp'].dt.is_quarter_end.astype(int)
            df['is_year_start'] = df['date_temp'].dt.is_year_start.astype(int)
            df['is_year_end'] = df['date_temp'].dt.is_year_end.astype(int)
            
            # Restore the original date column
            df['date'] = df['original_date']
            
            # Drop temporary columns
            df = df.drop(['date_temp', 'original_date'], axis=1)
            
            return df
        except Exception as e:
            self.log_progress(f"Error adding calendar features: {str(e)}", "ERROR")
            raise

    def calculate_returns(self, prices):
        """
        Calculate returns for different time periods focused on short to mid-term trading.
        Returns are calculated as: (current_price / past_price - 1) * 100
        """
        try:
            returns = {}
            
            # Short-term returns (more relevant for trading signals)
            periods = [1, 3, 5, 10, 20, 40, 60, 120]  # days
            # 1=daily, 3=3-day, 5=weekly, 10=bi-weekly, 20=monthly, 40=bi-monthly, 60=quarterly, 120=6-month
            
            # Convert to numpy array if not already
            prices = np.array(prices)
            
            for period in periods:
                # Initialize returns array with NaN values
                return_values = np.full_like(prices, np.nan, dtype=float)
                
                # Calculate returns only where possible
                for i in range(period, len(prices)):
                    if prices[i-period] != 0:  # Avoid division by zero
                        return_values[i] = ((prices[i] / prices[i-period]) - 1) * 100
                
                returns[f'return_{period}d'] = return_values
            
            # Verify calculations
            self.verify_returns(returns, prices)
            
            return returns
        except Exception as e:
            self.log_progress(f"Error calculating returns: {str(e)}", "ERROR")
            raise

    def verify_returns(self, returns, prices):
        """Verify return calculations."""
        try:
            for period in [1, 3, 5, 10, 20, 40, 60, 120]:
                key = f'return_{period}d'
                if key in returns:
                    return_values = returns[key]
                    
                    if np.any(np.isinf(return_values)) or np.any(np.isnan(return_values[period:])):
                        problem_indices = np.where(np.isinf(return_values) | np.isnan(return_values[period:]))[0]
                        if len(problem_indices) > 0:
                            self.log_progress(f"Warning: Invalid values in {key} at indices {problem_indices[:5]}", "WARNING")
        except Exception as e:
            self.log_progress(f"Error in return verification: {str(e)}", "ERROR")
            raise
    
    def calculate_volume_indicators(self, df):
        """Calculate volume-based technical indicators."""
        try:
            # Extract required data
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            open_price = df['open'].values
            data_length = len(df)
            
            # Initialize results dictionary with zeros
            volume_indicators = {
                indicator: np.zeros(data_length) 
                for indicator in self.get_volume_indicator_columns()
            }
            
            # 1. Volume Moving Averages
            periods = [5, 10, 20, 50]
            for period in periods:
                if data_length >= period:
                    sma = ta.SMA(volume, timeperiod=period)
                    # Only update non-NaN values
                    valid_mask = ~np.isnan(sma)
                    volume_indicators[f'volume_sma_{period}d'][valid_mask] = sma[valid_mask]
            
            # 2. Volume Ratio (current volume / average volume)
            if data_length >= 20:
                avg_volume = ta.SMA(volume, timeperiod=20)
                volume_ratio = np.ones(data_length)  # Initialize with neutral value
                
                # Avoid division by zero
                mask = (avg_volume != 0) & (~np.isnan(avg_volume))
                volume_ratio[mask] = volume[mask] / avg_volume[mask]
                volume_indicators['volume_ratio_20d'] = volume_ratio
            
            # 3. Volume Rate of Change
            periods = [1, 5, 10]
            for period in periods:
                if data_length >= period + 1:
                    vol_roc = np.zeros(data_length)  # Initialize with zero
                    for i in range(period, data_length):
                        if volume[i-period] > 0:
                            vol_roc[i] = ((volume[i] / volume[i-period]) - 1) * 100
                    volume_indicators[f'volume_roc_{period}d'] = vol_roc
            
            # 4. Money Flow Index (already in main indicators as mfi_14)
            
            # 5. Chaikin Money Flow (CMF)
            if data_length >= 20:
                cmf = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=20)
                valid_mask = ~np.isnan(cmf)
                volume_indicators['cmf_20d'][valid_mask] = cmf[valid_mask]
            
            # 6. Price-Volume Trend (PVT)
            if data_length >= 2:
                pvt = np.zeros(data_length)
                for i in range(1, data_length):
                    if close[i-1] != 0:
                        price_change_pct = (close[i] - close[i-1]) / close[i-1]
                        pvt[i] = pvt[i-1] + (volume[i] * price_change_pct)
                volume_indicators['pvt'] = pvt
            
            # 7. Volume Oscillator (VO)
            if data_length >= 28:
                fast_vol_ma = ta.SMA(volume, timeperiod=14)
                slow_vol_ma = ta.SMA(volume, timeperiod=28)
                
                vo = np.zeros(data_length)  # Default to zero
                mask = (slow_vol_ma != 0) & (~np.isnan(slow_vol_ma)) & (~np.isnan(fast_vol_ma))
                vo[mask] = ((fast_vol_ma[mask] - slow_vol_ma[mask]) / slow_vol_ma[mask]) * 100
                volume_indicators['volume_oscillator'] = vo
            
            # 8. Ease of Movement (EOM) - Safe version
            if data_length >= 14:
                # Initialize eom_raw array
                eom_raw = np.zeros(data_length)
                
                for i in range(1, data_length):
                    # Calculate today's midpoint
                    today_mid = (high[i] + low[i]) / 2
                    
                    # Calculate yesterday's midpoint
                    yesterday_mid = (high[i-1] + low[i-1]) / 2
                    
                    # Calculate distance moved
                    distance_moved = today_mid - yesterday_mid
                    
                    # Calculate box ratio (avoid division by zero)
                    if volume[i] > 0 and (high[i] - low[i]) > 0:
                        box_ratio = (high[i] - low[i]) / volume[i] * 100000000
                        eom_raw[i] = distance_moved / box_ratio
                    else:
                        eom_raw[i] = 0
                
                # Apply smoothing
                if np.any(~np.isnan(eom_raw)):
                    eom_sma = ta.SMA(eom_raw, timeperiod=14)
                    valid_mask = ~np.isnan(eom_sma)
                    volume_indicators['eom_14d'][valid_mask] = eom_sma[valid_mask]
            
            # 9. Volume-Adjusted Moving Average (VAMA)
            if data_length >= 20:
                vama = np.zeros(data_length)
                
                for i in range(20, data_length):
                    vol_slice = volume[i-20:i]
                    price_slice = close[i-20:i]
                    vol_sum = np.sum(vol_slice)
                    
                    if vol_sum > 0:
                        # Weighted average based on volume
                        weights = vol_slice / vol_sum
                        vama[i] = np.sum(weights * price_slice)
                
                volume_indicators['vama_20d'] = vama
            
            # 10. Volume-Price Confirmation Indicator (VPCI)
            if data_length >= 21:
                vpci = np.zeros(data_length)
                
                price_ma = ta.SMA(close, timeperiod=20)
                vol_ma = ta.SMA(volume, timeperiod=20)
                
                for i in range(20, data_length):
                    if price_ma[i] > 0 and vol_ma[i] > 0:
                        price_change = (close[i] - close[i-20]) / price_ma[i]
                        vol_change = (volume[i] - volume[i-20]) / vol_ma[i]
                        vpci[i] = price_change * vol_change
                
                volume_indicators['vpci_20d'] = vpci
                
            return volume_indicators
            
        except Exception as e:
            self.log_progress(f"Error calculating volume indicators: {str(e)}", "ERROR")
            # Return zeros for all volume indicators instead of NaN
            return {
                indicator: np.zeros(data_length) 
                for indicator in self.get_volume_indicator_columns()
            }
    
    def get_volume_indicator_columns(self):
        """Get list of volume indicator column names."""
        return [
            'volume_sma_5d', 'volume_sma_10d', 'volume_sma_20d', 'volume_sma_50d',
            'volume_ratio_20d', 'volume_roc_1d', 'volume_roc_5d', 'volume_roc_10d',
            'cmf_20d', 'pvt', 'volume_oscillator', 'eom_14d', 'vama_20d', 'vpci_20d'
        ]

    def calculate_indicators(self, df):
        """Calculate technical indicators using Ta-Lib with data length awareness."""
        try:
            # Convert price data to numpy arrays
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            open_price = df['open'].values
            data_length = len(df)
            symbol = df['trading_symbol'].iloc[0]

            # Create a dictionary to store all results
            all_indicators = {}
            
            # Add basic columns
            all_indicators['date'] = df['date']
            all_indicators['trading_symbol'] = df['trading_symbol']

            # Calculate indicators based on available data length
            
            # Short-term indicators (20 days minimum)
            if data_length >= 20:
                all_indicators['sma_20'] = ta.SMA(close, timeperiod=20)
                all_indicators['ema_20'] = ta.EMA(close, timeperiod=20)
                bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
                all_indicators['bollinger_upper'] = bb_upper
                all_indicators['bollinger_middle'] = bb_middle
                all_indicators['bollinger_lower'] = bb_lower
                all_indicators['stddev'] = ta.STDDEV(close, timeperiod=20)
            else:
                # Only the first N-1 values should be NaN, not the entire array
                all_indicators['sma_20'] = np.zeros(data_length)
                all_indicators['sma_20'][:min(data_length, 19)] = np.nan
                all_indicators['ema_20'] = np.zeros(data_length)
                all_indicators['ema_20'][:min(data_length, 19)] = np.nan
                all_indicators['bollinger_upper'] = np.zeros(data_length)
                all_indicators['bollinger_upper'][:min(data_length, 19)] = np.nan
                all_indicators['bollinger_middle'] = np.zeros(data_length)
                all_indicators['bollinger_middle'][:min(data_length, 19)] = np.nan
                all_indicators['bollinger_lower'] = np.zeros(data_length)
                all_indicators['bollinger_lower'][:min(data_length, 19)] = np.nan
                all_indicators['stddev'] = np.zeros(data_length)
                all_indicators['stddev'][:min(data_length, 19)] = np.nan

            # Medium-term indicators (50 days minimum)
            if data_length >= 50:
                all_indicators['sma_50'] = ta.SMA(close, timeperiod=50)
                all_indicators['ema_50'] = ta.EMA(close, timeperiod=50)
            else:
                all_indicators['sma_50'] = np.zeros(data_length)
                all_indicators['sma_50'][:min(data_length, 49)] = np.nan
                all_indicators['ema_50'] = np.zeros(data_length)
                all_indicators['ema_50'][:min(data_length, 49)] = np.nan

            # Long-term indicators (200 days minimum)
            if data_length >= 200:
                all_indicators['sma_200'] = ta.SMA(close, timeperiod=200)
            else:
                all_indicators['sma_200'] = np.zeros(data_length)
                all_indicators['sma_200'][:min(data_length, 199)] = np.nan

            # Short-term technical indicators
            if data_length >= 14:
                all_indicators['rsi_14'] = ta.RSI(close, timeperiod=14)
                all_indicators['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
                all_indicators['adx_14'] = ta.ADX(high, low, close, timeperiod=14)
                all_indicators['di_plus_14'] = ta.PLUS_DI(high, low, close, timeperiod=14)
                all_indicators['di_minus_14'] = ta.MINUS_DI(high, low, close, timeperiod=14)
                all_indicators['mfi_14'] = ta.MFI(high, low, close, volume)
                k, d = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                all_indicators['stochastic_k'] = k
                all_indicators['stochastic_d'] = d
            else:
                # Only mark the first N-1 values as NaN
                for indicator in ['rsi_14', 'atr_14', 'adx_14', 'di_plus_14', 'di_minus_14', 'mfi_14', 'stochastic_k', 'stochastic_d']:
                    all_indicators[indicator] = np.zeros(data_length)
                    all_indicators[indicator][:min(data_length, 13)] = np.nan

            # MACD (26 days minimum)
            if data_length >= 26:
                macd, signal, hist = ta.MACD(close)
                all_indicators['macd_line'] = macd
                all_indicators['macd_signal'] = signal
                all_indicators['macd_histogram'] = hist
            else:
                # Only mark the first 25 values as NaN
                for indicator in ['macd_line', 'macd_signal', 'macd_histogram']:
                    all_indicators[indicator] = np.zeros(data_length)
                    all_indicators[indicator][:min(data_length, 25)] = np.nan

            # Other indicators - use a more intelligent approach
            indicators_with_periods = {
                'natr': 14, 
                'cci_20': 20, 
                'williams_r': 14, 
                'roc': 10, 
                'adosc': 10, 
                'trix': 30, 
                'ultosc': 28, 
                'var': 20
            }
            
            for indicator, period in indicators_with_periods.items():
                # Only apply minimum length check if there's enough data
                if data_length >= period:
                    if indicator == 'natr':
                        all_indicators[indicator] = ta.NATR(high, low, close)
                    elif indicator == 'cci_20':
                        all_indicators[indicator] = ta.CCI(high, low, close)
                    elif indicator == 'williams_r':
                        all_indicators[indicator] = ta.WILLR(high, low, close)
                    elif indicator == 'roc':
                        all_indicators[indicator] = ta.ROC(close)
                    elif indicator == 'adosc':
                        all_indicators[indicator] = ta.ADOSC(high, low, close, volume)
                    elif indicator == 'trix':
                        all_indicators[indicator] = ta.TRIX(close)
                    elif indicator == 'ultosc':
                        all_indicators[indicator] = ta.ULTOSC(high, low, close)
                    elif indicator == 'var':
                        all_indicators[indicator] = ta.VAR(close, timeperiod=20)
                else:
                    # Only mark the first (period-1) values as NaN
                    all_indicators[indicator] = np.zeros(data_length)
                    all_indicators[indicator][:min(data_length, period-1)] = np.nan
            
            # Indicators that don't need minimum data length
            all_indicators['obv'] = ta.OBV(close, volume)
            all_indicators['ad_line'] = ta.AD(high, low, close, volume)
            all_indicators['bop'] = ta.BOP(open_price, high, low, close)

            # Calculate Returns based on available data
            returns_dict = self.calculate_returns(close)
            
            # Add returns to all_indicators dict
            for key, values in returns_dict.items():
                all_indicators[key] = values
                
            # Calculate calendar features and add to dictionary
            calendar_features = self.add_calendar_features(df)
            for col in calendar_features.columns:
                if col not in ['date', 'trading_symbol', 'original_date', 'date_temp']:
                    all_indicators[col] = calendar_features[col].values
            
            # Calculate correlation with Nifty and relative strength only for non-Nifty symbols
            if symbol != 'NIFTY':
                try:
                    nifty_correlations = self.calculate_correlation_with_nifty(symbol, df)
                    for key, values in nifty_correlations.items():
                        all_indicators[key] = values
                except Exception as e:
                    self.log_progress(f"Error calculating Nifty correlation for {symbol}: {str(e)}", "WARNING")
                    # Set correlation values to zero instead of NaN when there's an error
                    for period in [20, 60, 120]:
                        all_indicators[f'nifty_corr_{period}d'] = np.zeros(data_length)
                    all_indicators['nifty_corr_full'] = np.zeros(data_length)
                
                try:
                    rs_results = self.calculate_relative_strength(symbol, df)
                    for key, values in rs_results.items():
                        all_indicators[key] = values
                except Exception as e:
                    self.log_progress(f"Error calculating relative strength for {symbol}: {str(e)}", "WARNING")
                    # Set relative strength values to one (neutral) instead of NaN when there's an error
                    for period in [5, 10, 20, 60, 120]:
                        all_indicators[f'rs_nifty_{period}d'] = np.ones(data_length)
            else:
                # Set correlation and RS values to zero for Nifty itself (zero correlation and neutral RS)
                for period in [20, 60, 120]:
                    all_indicators[f'nifty_corr_{period}d'] = np.zeros(data_length)
                all_indicators['nifty_corr_full'] = np.zeros(data_length)
                
                for period in [5, 10, 20, 60, 120]:
                    all_indicators[f'rs_nifty_{period}d'] = np.ones(data_length)
            
            # Calculate volume-based indicators
            try:
                volume_indicators = self.calculate_volume_indicators(df)
                for key, values in volume_indicators.items():
                    all_indicators[key] = values
            except Exception as e:
                self.log_progress(f"Error calculating volume indicators for {symbol}: {str(e)}", "WARNING")
                # Set volume indicator values to default values instead of NaN
                for indicator in self.get_volume_indicator_columns():
                    all_indicators[indicator] = np.zeros(data_length)

            # Create DataFrame from dictionary all at once
            results = pd.DataFrame(all_indicators)
            
            # Clean and validate the data
            results = self.clean_and_validate_data(results)

            return results
        except Exception as e:
            self.log_progress(f"Error in calculate_indicators: {str(e)}", "ERROR")
            raise

    def verify_calculations(self, results, symbol):
        """Verify indicator calculations."""
        # Check for any completely missing indicator columns
        missing_columns = [col for col in self.expected_columns if col not in results.columns]
        if missing_columns:
            raise ValueError(f"Missing indicators for {symbol}: {missing_columns}")

        # Check for excessive NaN values
        nan_threshold = 0.7  # 70% threshold - allow more nulls for new complex features
        nan_percentages = results.isnull().mean()
        problem_columns = nan_percentages[nan_percentages > nan_threshold].index.tolist()
        if problem_columns:
            self.log_progress(f"Warning: High NaN count in {symbol} for columns: {problem_columns}", "WARNING")

        # Verify basic relationships (original code)
        if not results.empty:
            # Verify Bollinger Bands relationship
            if all(col in results.columns for col in ['bollinger_upper', 'bollinger_middle', 'bollinger_lower']):
                # Filter out NaN values before comparison
                valid_mask = ~(results['bollinger_upper'].isna() | results['bollinger_middle'].isna() | results['bollinger_lower'].isna())
                if valid_mask.any():
                    if not (results.loc[valid_mask, 'bollinger_upper'] >= results.loc[valid_mask, 'bollinger_middle']).all():
                        problem_indices = results[~(results['bollinger_upper'] >= results['bollinger_middle']) & valid_mask].index
                        if len(problem_indices) > 0:
                            sample_idx = problem_indices[0]
                            self.log_progress(f"BB validation failed at index {sample_idx}: Upper={results.iloc[sample_idx]['bollinger_upper']}, Middle={results.iloc[sample_idx]['bollinger_middle']}", "WARNING")
                        raise ValueError(f"Invalid Bollinger Bands calculation for {symbol}")
                    
                    if not (results.loc[valid_mask, 'bollinger_middle'] >= results.loc[valid_mask, 'bollinger_lower']).all():
                        problem_indices = results[~(results['bollinger_middle'] >= results['bollinger_lower']) & valid_mask].index
                        if len(problem_indices) > 0:
                            sample_idx = problem_indices[0]
                            self.log_progress(f"BB validation failed at index {sample_idx}: Middle={results.iloc[sample_idx]['bollinger_middle']}, Lower={results.iloc[sample_idx]['bollinger_lower']}", "WARNING")
                        raise ValueError(f"Invalid Bollinger Bands calculation for {symbol}")
                        
            # Add logical checks for new calculations
            # 1. Check day_of_week is between 0-6
            if 'day_of_week' in results.columns:
                if not ((results['day_of_week'] >= 0) & (results['day_of_week'] <= 6)).all():
                    raise ValueError(f"Invalid day_of_week values for {symbol}")
                    
            # 2. Check month is between 1-12
            if 'month' in results.columns:
                if not ((results['month'] >= 1) & (results['month'] <= 12)).all():
                    raise ValueError(f"Invalid month values for {symbol}")
                    
            # 3. Check quarter is between 1-4
            if 'quarter' in results.columns:
                if not ((results['quarter'] >= 1) & (results['quarter'] <= 4)).all():
                    raise ValueError(f"Invalid quarter values for {symbol}")

    def clean_and_validate_data(self, df):
        """
        Clean and validate the calculated indicators.
        This version minimizes NaN values to prevent data loss.
        """
        try:
            # Make a copy to prevent fragmentation warnings
            df = df.copy()
            
            # Replace inf and -inf with neutral values (0 or 1 depending on indicator)
            # For relative strength indicators, use 1.0 for neutral
            rs_columns = [col for col in df.columns if col.startswith('rs_nifty_')]
            for col in rs_columns:
                df[col] = df[col].replace([np.inf, -np.inf], 1.0)
            
            # For correlation columns, use 0.0 for neutral
            corr_columns = [col for col in df.columns if col.startswith('nifty_corr_')]
            for col in corr_columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
            
            # For other indicators, use 0.0 for neutral
            other_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col not in rs_columns and col not in corr_columns]
            for col in other_columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # Clip extreme values (only for non-NaN values)
            for col in numeric_columns:
                if col in df.columns:  # Check if column exists
                    if 'return_' in col:
                        # Allow wider range for returns
                        df[col] = df[col].clip(-1000, 1000)
                    else:
                        # Standard range for other indicators
                        df[col] = df[col].clip(-1e6, 1e6)
            
            # Round to 6 decimal places (preserving NaN)
            df[numeric_columns] = df[numeric_columns].round(6)
            
            return df
        except Exception as e:
            self.log_progress(f"Error in data cleaning: {str(e)}", "ERROR")
            raise
    def remove_null_rows(self, df):
        """
        Modify the null removal strategy to be more selective.
        Only remove rows where critical indicators are NULL.
        """
        try:
            # Get original row count for logging
            original_row_count = len(df)
            
            if original_row_count == 0:
                self.log_progress("Empty DataFrame provided, nothing to clean", "WARNING")
                return df
                    
            # Define critical indicator columns that must have values
            # These are indicators that are essential for analysis
            critical_columns = [
                'sma_20', 'sma_50', 'ema_20',
                'macd_line', 'macd_signal', 'macd_histogram',
                'rsi_14', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'return_5d', 'return_20d'
            ]
            
            # Filter for columns that actually exist in the dataframe
            critical_columns = [col for col in critical_columns if col in df.columns]
            
            # Drop rows with NULL values in ALL critical columns
            # (only remove a row if it's missing all important indicators)
            if critical_columns:
                # Check if all critical columns are NULL for each row
                all_null_mask = df[critical_columns].isnull().all(axis=1)
                clean_df = df[~all_null_mask]
            else:
                clean_df = df
                
            # Log the results
            removed_rows = len(df) - len(clean_df)
            if removed_rows > 0:
                self.log_progress(
                    f"Removed {removed_rows} rows with NULL values in all critical columns "
                    f"({removed_rows/original_row_count:.2%} of original data)", 
                    "INFO"
                )
                    
            return clean_df
                
        except Exception as e:
            self.log_progress(f"Error in remove_null_rows: {str(e)}", "ERROR")
            raise


    def save_indicators(self, df):
        """Save calculated indicators to the database."""
        try:
            # Clean and validate data before saving
            df = self.clean_and_validate_data(df)
            
            # Ensure date is in the correct format for MySQL
            if pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = df['date'].dt.date
            
            # Filter out price/volume data columns that don't exist in technical_indicators table
            columns_to_exclude = ['open', 'high', 'low', 'close', 'volume']
            df_to_save = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])
            
            conn = self.connect_to_db()
            cursor = conn.cursor()

            # Get column names
            columns = df_to_save.columns.tolist()
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            # Prepare insert query
            query = f"""
                INSERT INTO technical_indicators (
                    {columns_str}
                ) VALUES (
                    {placeholders}
                )
            """

            # Convert DataFrame to list of tuples
            values = [tuple(x) for x in df_to_save.values]
            
            # Execute batch insert
            cursor.executemany(query, values)
            conn.commit()
            
            cursor.close()
            conn.close()

        except Exception as e:
            self.log_progress(f"Error in save_indicators: {str(e)}", "ERROR")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise

    def process_symbol_data(self, symbol):
        """Process data for a single symbol."""
        try:
            # Get last processed date
            last_processed_date = self.get_last_processed_date(symbol)
            
            # Fetch historical data
            df = self.fetch_historical_data(symbol, last_processed_date)

            if len(df) < 20:  # Minimum data required for basic indicators
                self.log_progress(f"Insufficient data for {symbol}, minimum required: 20 days", "WARNING")
                return 0

            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df)
            
            # Remove rows with null values
            df_with_indicators = self.remove_null_rows(df_with_indicators)
            
            # Verify calculations
            self.verify_calculations(df_with_indicators, symbol)
            
            if last_processed_date:
                # Only save records after the last processed date
                df_to_save = df_with_indicators[df_with_indicators['date'] > last_processed_date]
            else:
                # Save all records for new symbols
                df_to_save = df_with_indicators

            # Save indicators if we have new data
            if not df_to_save.empty:
                self.save_indicators(df_to_save)
                return len(df_to_save)
            
            return 0

        except Exception as e:
            self.log_progress(f"Error processing {symbol}: {str(e)}", "ERROR")
            return 0
        
    def fetch_lookback_data(self, symbol, start_date, lookback_days=365):
        """Fetch historical data including lookback period."""
        try:
            conn = self.connect_to_db()
            
            query = """
                SELECT date, trading_symbol, open, high, low, close, volume 
                FROM historical_data 
                WHERE trading_symbol = %s 
                    AND date BETWEEN DATE_SUB(%s, INTERVAL %s DAY) AND CURRENT_DATE()
                ORDER BY date
            """
            
            df = pd.read_sql(query, conn, params=(symbol, start_date, lookback_days))
            conn.close()
            return df

        except Exception as e:
            self.log_progress(f"Error fetching lookback data for {symbol}: {str(e)}", "ERROR")
            if 'conn' in locals():
                conn.close()
            raise

    def process_all_symbols(self):
        """Process all symbols with minimal output."""
        try:
            self.start_time = time.time()
            self.processed_symbols = 0
            self.total_records_processed = 0
            self.errors_encountered = 0
            
            # Get symbols to process
            conn = self.connect_to_db()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT h.trading_symbol
                FROM historical_data h
                LEFT JOIN (
                    SELECT trading_symbol, MAX(date) as last_date
                    FROM technical_indicators
                    GROUP BY trading_symbol
                ) t ON h.trading_symbol = t.trading_symbol
                WHERE t.last_date IS NULL 
                    OR h.date > t.last_date
            """)
            
            symbols_to_process = [row[0] for row in cursor.fetchall()]
            self.total_symbols = len(symbols_to_process)
            
            cursor.close()
            conn.close()

            self.log_progress(f"Processing {self.total_symbols} symbols...", verbosity=0)
            print("\n")  # Add extra line for readability

            # Process symbols
            for symbol in symbols_to_process:
                try:
                    records_processed = self.process_symbol_data(symbol)
                    if records_processed:
                        self.total_records_processed += records_processed
                    
                    self.processed_symbols += 1
                    
                    # Print progress every 10 symbols
                    if self.processed_symbols % 10 == 0:
                        self.print_progress_stats()

                except Exception as e:
                    self.errors_encountered += 1
                    continue

            # Final progress update
            self.print_progress_stats()
            self.log_progress("\nProcessing completed successfully!", verbosity=0)

        except Exception as e:
            self.log_progress(f"Fatal error: {str(e)}", "ERROR")
            raise

    def print_progress_stats(self):
        """Print current progress statistics."""
        elapsed_time = time.time() - self.start_time
        remaining_time = self.estimate_remaining_time()

        # Clear the current line
        print('\r', end='')

        # Print progress with a check for zero total_symbols
        if self.total_symbols > 0:
            progress_percentage = (self.processed_symbols/self.total_symbols)*100
        else:
            progress_percentage = 0.0  # Default when no symbols to process
            
        progress_msg = (
            f"Progress: {self.processed_symbols}/{self.total_symbols} symbols "
            f"({progress_percentage:.1f}%) | "
            f"Records: {self.total_records_processed:,} | "
            f"Errors: {self.errors_encountered} | "
            f"Time: {str(timedelta(seconds=int(elapsed_time)))} | "
            f"Remaining: {remaining_time}"
        )

        # Print without newline and flush immediately
        print(progress_msg, end='', flush=True)

    def estimate_remaining_time(self):
        """Estimate remaining processing time."""
        if self.processed_symbols == 0 or self.start_time is None:
            return "Calculating..."
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_symbol = elapsed_time / self.processed_symbols
        remaining_symbols = self.total_symbols - self.processed_symbols
        remaining_seconds = remaining_symbols * avg_time_per_symbol
        
        return str(timedelta(seconds=int(remaining_seconds)))

    def cleanup_duplicate_entries(self):
        """Clean up any duplicate entries in the database."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE t1 FROM technical_indicators t1
                INNER JOIN technical_indicators t2
                WHERE t1.id < t2.id
                    AND t1.date = t2.date
                    AND t1.trading_symbol = t2.trading_symbol
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.log_progress("Duplicate cleanup completed successfully")

        except Exception as e:
            self.log_progress(f"Error in cleanup_duplicate_entries: {str(e)}", "ERROR")
            if 'conn' in locals():
                conn.rollback()
                conn.close()

    def clean_data_for_ml(self, df):
        """
        Prepare data for machine learning by handling null values.
        This ensures all rows have complete data for all indicators.
        """
        try:
            # Count initial rows
            initial_row_count = len(df)
            
            # First, check if we're dealing with a completely empty dataframe
            if df.empty:
                return df
                
            # Get list of all indicator columns (excluding date and trading_symbol)
            indicator_columns = [col for col in df.columns 
                                if col not in ['date', 'trading_symbol', 'id']]
            
            # Calculate percentage of missing values by column
            missing_by_column = df[indicator_columns].isnull().mean() * 100
            
            # Log columns with high missing percentages (for debugging)
            high_missing_cols = missing_by_column[missing_by_column > 10].index.tolist()
            if high_missing_cols:
                self.log_progress(f"Warning: High missing values in columns: {high_missing_cols}", "WARNING")
            
            # Drop rows with any null values in indicator columns
            df_clean = df.dropna(subset=indicator_columns)
            
            # Log dropped rows count
            dropped_rows = initial_row_count - len(df_clean)
            if dropped_rows > 0:
                self.log_progress(f"Dropped {dropped_rows} rows with null values ({dropped_rows/initial_row_count:.1%})", "INFO")
                
            return df_clean
            
        except Exception as e:
            self.log_progress(f"Error in clean_data_for_ml: {str(e)}", "ERROR")
            raise
def main():
 """Main execution function."""
 try:
     print("\n" + "="*50)
     print("Technical Indicator Calculator")
     print("="*50 + "\n")
     
     calculator = TechnicalIndicatorCalculator()
     
     # Process all symbols (this will use your new remove_null_rows function)
     calculator.process_all_symbols()
     
     # Cleanup existing NULL rows in the database
     # calculator.cleanup_null_rows_in_database()
     
 except Exception as e:
     print(f"\nFatal error: {str(e)}")
     raise

if __name__ == "__main__":
    main()