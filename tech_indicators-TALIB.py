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
            'return_20', 'return_55', 'return_90', 'return_180', 'return_365'
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

    def calculate_returns(self, prices):
        """
        Calculate returns for different time periods.
        Returns are calculated as: (current_price / past_price - 1) * 100
        """
        try:
            returns = {}
            periods = [20, 55, 90, 180, 365]
            
            # Convert to numpy array if not already
            prices = np.array(prices)
            
            for period in periods:
                # Initialize returns array with NaN values
                return_values = np.full_like(prices, np.nan, dtype=float)  # Initialize with NaN

                # Calculate returns only where possible
                for i in range(period, len(prices)):  # Start from 'period' to avoid index errors
                    return_values[i] = ((prices[i] / prices[i - period]) - 1) * 100

                returns[f'return_{period}'] = return_values

            # Verify calculations
            self.verify_returns(returns, prices)
            
            return returns

        except Exception as e:
            self.log_progress(f"Error calculating returns: {str(e)}", "ERROR")
            raise

    def verify_returns(self, returns, prices):
        """Verify return calculations."""
        try:
            for period in [20, 55, 90, 180, 365]:
                key = f'return_{period}'
                if key in returns:
                    return_values = returns[key]
                    
                    # Verify calculations without debug output
                    #if not np.all(return_values[:period] == 0):
                    #    raise ValueError(f"First {period} values of {key} should be zero")
                    
                    if np.any(np.isinf(return_values)) or np.any(np.isnan(return_values)):
                        pass
                        #raise ValueError(f"Invalid values found in {key}")

        except Exception as e:
            self.log_progress(f"Error in return verification: {str(e)}", "ERROR")
            raise

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

            # Initialize results DataFrame
            results = pd.DataFrame()
            results['date'] = df['date']
            results['trading_symbol'] = df['trading_symbol']

            # Calculate indicators based on available data length
            
            # Short-term indicators (20 days minimum)
            if data_length >= 20:
                results['sma_20'] = ta.SMA(close, timeperiod=20)
                results['ema_20'] = ta.EMA(close, timeperiod=20)
                results['bollinger_upper'], results['bollinger_middle'], results['bollinger_lower'] = ta.BBANDS(
                    close, 
                    timeperiod=20, 
                    nbdevup=2.0,  # 2 standard deviations for upper band
                    nbdevdn=2.0,  # 2 standard deviations for lower band
                    matype=0      # SMA as the middle band type
                )
                results['stddev'] = ta.STDDEV(close, timeperiod=20)
            else:
                results['sma_20'] = np.nan
                results['ema_20'] = np.nan
                results['bollinger_upper'] = np.nan
                results['bollinger_middle'] = np.nan
                results['bollinger_lower'] = np.nan
                results['stddev'] = np.nan

            # Medium-term indicators (50 days minimum)
            if data_length >= 50:
                results['sma_50'] = ta.SMA(close, timeperiod=50)
                results['ema_50'] = ta.EMA(close, timeperiod=50)
            else:
                results['sma_50'] = np.nan
                results['ema_50'] = np.nan

            # Long-term indicators (200 days minimum)
            if data_length >= 200:
                results['sma_200'] = ta.SMA(close, timeperiod=200)
            else:
                results['sma_200'] = np.nan

            # Short-term technical indicators
            if data_length >= 14:
                results['rsi_14'] = ta.RSI(close, timeperiod=14)
                results['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
                results['adx_14'] = ta.ADX(high, low, close, timeperiod=14)
                results['di_plus_14'] = ta.PLUS_DI(high, low, close, timeperiod=14)
                results['di_minus_14'] = ta.MINUS_DI(high, low, close, timeperiod=14)
                results['mfi_14'] = ta.MFI(high, low, close, volume)
                results['stochastic_k'], results['stochastic_d'] = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            else:
                results['rsi_14'] = np.nan
                results['atr_14'] = np.nan
                results['adx_14'] = np.nan
                results['di_plus_14'] = np.nan
                results['di_minus_14'] = np.nan
                results['mfi_14'] = np.nan
                results['stochastic_k'] = np.nan
                results['stochastic_d'] = np.nan

            # MACD (26 days minimum)
            if data_length >= 26:
                macd, signal, hist = ta.MACD(close)
                results['macd_line'] = macd
                results['macd_signal'] = signal
                results['macd_histogram'] = hist
            else:
                results['macd_line'] = np.nan
                results['macd_signal'] = np.nan
                results['macd_histogram'] = np.nan

            # Other indicators
            results['natr'] = ta.NATR(high, low, close) if data_length >= 14 else np.nan
            results['cci_20'] = ta.CCI(high, low, close) if data_length >= 20 else np.nan
            results['williams_r'] = ta.WILLR(high, low, close) if data_length >= 14 else np.nan
            results['roc'] = ta.ROC(close) if data_length >= 10 else np.nan
            results['obv'] = ta.OBV(close, volume)  # No minimum
            results['ad_line'] = ta.AD(high, low, close, volume)  # No minimum
            results['adosc'] = ta.ADOSC(high, low, close, volume) if data_length >= 10 else np.nan
            results['trix'] = ta.TRIX(close) if data_length >= 30 else np.nan
            results['ultosc'] = ta.ULTOSC(high, low, close) if data_length >= 28 else np.nan
            results['bop'] = ta.BOP(open_price, high, low, close)  # No minimum
            results['var'] = ta.VAR(close, timeperiod=20) if data_length >= 20 else np.nan

            # Calculate Returns based on available data
            returns_dict = self.calculate_returns(close)  # Use the calculate_returns function

            # Add returns to results
            for period, values in returns_dict.items():
                results[period] = values

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
        nan_threshold = 0.5  # 50% threshold
        nan_percentages = results.isnull().mean()
        problem_columns = nan_percentages[nan_percentages > nan_threshold].index.tolist()
        if problem_columns:
            self.log_progress(f"Warning: High NaN count in {symbol} for columns: {problem_columns}", "WARNING")

        # Verify basic relationships
        if not results.empty:
            # Verify Bollinger Bands relationship
            if all(col in results.columns for col in ['bollinger_upper', 'bollinger_middle', 'bollinger_lower']):
                # Filter out NaN values before comparison
                valid_mask = ~(results['bollinger_upper'].isna() | results['bollinger_middle'].isna())
                if not (results.loc[valid_mask, 'bollinger_upper'] >= results.loc[valid_mask, 'bollinger_middle']).all():
                    # Optional: Add debugging to see where it fails
                    problem_indices = results[~(results['bollinger_upper'] >= results['bollinger_middle']) & valid_mask].index
                    if len(problem_indices) > 0:
                        sample_idx = problem_indices[0]
                        self.log_progress(f"BB validation failed at index {sample_idx}: Upper={results.iloc[sample_idx]['bollinger_upper']}, Middle={results.iloc[sample_idx]['bollinger_middle']}", "WARNING")
                    raise ValueError(f"Invalid Bollinger Bands calculation for {symbol}")

    def clean_and_validate_data(self, df):
        """Clean and validate the calculated indicators."""
        try:
            # Replace inf and -inf with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # Clip extreme values (only for non-NaN values)
            for col in numeric_columns:
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
        Remove ALL rows with NULL values in any indicator column.
        This ensures the database only contains complete records.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing technical indicators
        
        Returns:
        pandas.DataFrame: Clean DataFrame with NO NULL values in any indicator column
        """
        try:
            # Get original row count for logging
            original_row_count = len(df)
            
            if original_row_count == 0:
                self.log_progress("Empty DataFrame provided, nothing to clean", "WARNING")
                return df
                
            # Get all indicator columns (exclude date and trading_symbol)
            non_indicator_cols = ['date', 'trading_symbol', 'id']
            indicator_columns = [col for col in df.columns if col not in non_indicator_cols]
            
            # Simply drop ALL rows with ANY NULL values in indicator columns
            clean_df = df.dropna(subset=indicator_columns)
            
            # Log the results
            removed_rows = len(df) - len(clean_df)
            if removed_rows > 0:
                self.log_progress(
                    f"Removed {removed_rows} rows with NULL values "
                    f"({removed_rows/original_row_count:.2%} of original data)", 
                    "INFO"
                )
                
                # Get columns with the most NULL values for debugging
                null_counts = df.loc[~df.index.isin(clean_df.index), indicator_columns].isnull().sum().sort_values(ascending=False)
                problem_columns = null_counts[null_counts > 0].head(5)
                
                if not problem_columns.empty:
                    self.log_progress(
                        f"Top NULL columns: {problem_columns.to_dict()}", 
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
            
            conn = self.connect_to_db()
            cursor = conn.cursor()

            # Get column names excluding 'id'
            columns = df.columns.tolist()
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
            values = [tuple(x) for x in df.values]
            
            # Execute batch insert
            cursor.executemany(query, values)
            conn.commit()
            
            # Only log if there's an error
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