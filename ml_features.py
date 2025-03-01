import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import pytz
import tqdm
import decimal
import logging
import sys
from typing import List, Dict, Optional, Union, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FeatureEngineering')

class FeatureEngineering:
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the FeatureEngineering class with database configuration.
        
        Args:
            db_config: Dictionary containing MySQL connection parameters
        """
        self.db_config = db_config

    def connect_to_db(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        """Create a new database connection."""
        try:
            return mysql.connector.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None

    def fetch_data(self, trading_symbols: List[str], start_date: str, end_date: Union[str, date]) -> pd.DataFrame:
        """
        Fetches historical and technical indicator data for given symbols and date range.
        
        Args:
            trading_symbols: List of trading symbols to fetch data for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format or as a date object
            
        Returns:
            DataFrame containing historical data and technical indicators
        """
        if not trading_symbols:
            logger.error("No trading symbols provided")
            return pd.DataFrame()
            
        conn = None
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor(dictionary=True)

            # Process in batches to avoid memory issues
            batch_size = 50  # Process 50 symbols at a time
            all_data = []
            
            for i in range(0, len(trading_symbols), batch_size):
                batch_symbols = trading_symbols[i:i+batch_size]
                placeholders = ', '.join(['%s'] * len(batch_symbols))
                
                query = f"""
                    SELECT
                        hd.date,
                        hd.trading_symbol,
                        hd.open,
                        hd.high,
                        hd.low,
                        hd.close,
                        hd.volume,
                        ti.sma_20, ti.sma_50, ti.sma_200,
                        ti.ema_20, ti.ema_50,
                        ti.macd_line, ti.macd_signal, ti.macd_histogram,
                        ti.adx_14, ti.di_plus_14, ti.di_minus_14,
                        ti.bollinger_upper, ti.bollinger_middle, ti.bollinger_lower,
                        ti.atr_14, ti.natr,
                        ti.rsi_14,
                        ti.stochastic_k, ti.stochastic_d,
                        ti.cci_20, ti.mfi_14, ti.williams_r, ti.roc,
                        ti.trix, ti.ultosc, ti.bop, ti.stddev, ti.var,
                        ti.return_20, ti.return_55, ti.return_90, ti.return_180, ti.return_365
                    FROM
                        historical_data hd
                    JOIN
                        technical_indicators ti ON hd.date = ti.date AND hd.trading_symbol = ti.trading_symbol
                    WHERE
                        hd.trading_symbol IN ({placeholders})
                        AND hd.date BETWEEN %s AND %s
                    ORDER BY
                        hd.trading_symbol, hd.date;
                """

                params = batch_symbols + [start_date, end_date]
                logger.info(f"Fetching data for {len(batch_symbols)} symbols")
                cursor.execute(query, params)
                batch_data = cursor.fetchall()
                logger.info(f"Fetched {len(batch_data)} rows for current batch")
                
                # Convert all numeric values to float
                for row in batch_data:
                    for key, value in row.items():
                        if isinstance(value, decimal.Decimal):
                            row[key] = float(value)
                
                all_data.extend(batch_data)
                
            cursor.close()
            conn.close()
            
            logger.info(f"Total rows fetched: {len(all_data)}")
            
            if not all_data:
                logger.warning("No data was fetched from the database")
                return pd.DataFrame()
            
            # Create DataFrame and explicitly convert date column to datetime
            df = pd.DataFrame(all_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Check for NaN trading_symbol values
            if 'trading_symbol' in df.columns and df['trading_symbol'].isna().any():
                nan_count = df['trading_symbol'].isna().sum()
                logger.warning(f"Found {nan_count} rows with NaN trading_symbol. These will be dropped.")
                df = df.dropna(subset=['trading_symbol'])
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}", exc_info=True)
            if conn and conn.is_connected():
                conn.close()
            return pd.DataFrame()

    def create_features(self, df: pd.DataFrame, prediction_horizon: int = 5, 
                        lag_days: List[int] = [1, 2, 3, 5, 10], 
                        threshold: float = 0.01) -> pd.DataFrame:
        """
        Creates features and target variable for machine learning.
        
        Args:
            df: DataFrame containing historical data and technical indicators
            prediction_horizon: Number of days ahead to predict
            lag_days: List of lag periods to create lagged features
            threshold: Threshold for target variable classification
            
        Returns:
            DataFrame with engineered features
        """
        
        if df.empty:
            logger.warning("DataFrame is empty. Returning empty DataFrame.")
            return df

        logger.info(f"Initial DataFrame shape: {df.shape}")
        
        # Data quality check: Display sample data
        logger.info(f"Sample data (first row):\n{df.iloc[0]}")
        logger.info(f"Column data types:\n{df.dtypes}")
        
        # Check for missing columns
        required_columns = ['date', 'trading_symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Ensure all numeric columns are float
        numeric_cols = df.select_dtypes(include=[np.number, object]).columns
        for col in numeric_cols:
            try:
                if col != 'trading_symbol' and col != 'date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {e}")
        
        # Create a copy to avoid warnings
        result_df = df.copy()
        
        # Get unique trading symbols and count occurrences for each
        unique_symbols = result_df['trading_symbol'].unique()
        symbol_counts = result_df['trading_symbol'].value_counts()
        logger.info(f"Processing {len(unique_symbols)} unique trading symbols")
        logger.info(f"Top 5 symbols by count: \n{symbol_counts.head()}")
        
        # Create an empty list to store processed dataframes
        all_processed_dfs = []
        
        # Process each trading symbol separately to avoid mixing data
        for symbol in tqdm.tqdm(unique_symbols, desc="Processing symbols"):
            # Skip symbols with NaN values
            if pd.isna(symbol):
                logger.warning(f"Found NaN symbol. Skipping.")
                continue
                
            # Filter data for current symbol
            symbol_df = result_df[result_df['trading_symbol'] == symbol].copy()
            
            # Check if we have enough data for this symbol
            min_required_days = max(lag_days + [prediction_horizon])
            if len(symbol_df) <= min_required_days:
                logger.warning(f"Not enough data for symbol {symbol}. Need at least {min_required_days} days, got {len(symbol_df)}. Skipping.")
                continue
                
            # Sort by date to ensure proper lag calculation
            symbol_df = symbol_df.sort_values('date')
            
            try:
                # 1. Lagged Features - process each column individually
                for lag in lag_days:
                    for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi_14', 'macd_line', 'atr_14']:
                        if col in symbol_df.columns:
                            symbol_df[f'{col}_lag{lag}'] = symbol_df[col].shift(lag)
                
                # 2. Derived Features
                symbol_df['prev_close'] = symbol_df['close'].shift(1)
                symbol_df['prev_volume'] = symbol_df['volume'].shift(1)
                
                # Avoid division by zero or very small numbers
                symbol_df['price_change'] = (symbol_df['close'] - symbol_df['prev_close']) / (symbol_df['prev_close'].replace(0, np.nan) + 1e-6)
                symbol_df['volume_change'] = (symbol_df['volume'] - symbol_df['prev_volume']) / (symbol_df['prev_volume'].replace(0, np.nan) + 1e-6)
                symbol_df['high_low_range'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close'].replace(0, np.nan)
                symbol_df['close_open_range'] = (symbol_df['close'] - symbol_df['open']) / symbol_df['close'].replace(0, np.nan)
                
                # 3. Target Variable
                symbol_df['future_close'] = symbol_df['close'].shift(-prediction_horizon)
                symbol_df['future_return'] = (symbol_df['future_close'] - symbol_df['close']) / symbol_df['close'].replace(0, np.nan)
                symbol_df['target'] = np.where(symbol_df['future_return'] > threshold, 1, 0)
                
                # Remove inf values and NaNs
                symbol_df = symbol_df.replace([np.inf, -np.inf], np.nan)
                
                # Drop rows with NaN in future_close (can't train without target)
                symbol_df = symbol_df.dropna(subset=['future_close'])
                
                # Log the data quality for this symbol
                logger.debug(f"Symbol {symbol}: Before NaN removal: {len(symbol_df)} rows")
                nan_counts = symbol_df.isna().sum()
                columns_with_nans = nan_counts[nan_counts > 0]
                if not columns_with_nans.empty:
                    logger.debug(f"Symbol {symbol}: Columns with NaNs: {columns_with_nans}")
                
                # Drop all rows with NaN values
                symbol_df = symbol_df.dropna()
                logger.debug(f"Symbol {symbol}: After NaN removal: {len(symbol_df)} rows")
                
                if not symbol_df.empty:
                    all_processed_dfs.append(symbol_df)
                else:
                    logger.warning(f"After processing, no data remained for symbol {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
                continue
        
        if not all_processed_dfs:
            logger.warning("No data remained after processing. Check your filters and data quality.")
            return pd.DataFrame()
            
        # Combine all processed dataframes
        result_df = pd.concat(all_processed_dfs, ignore_index=True)
        
        logger.info(f"Final DataFrame shape: {result_df.shape}")
        
        return result_df

    def get_trading_symbols(self, start_date: str, end_date: Union[str, date]) -> List[str]:
        """
        Retrieves a list of distinct trading symbols available within the specified date range.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format or as a date object
            
        Returns:
            List of trading symbols
        """
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()

            query = """
                SELECT DISTINCT trading_symbol
                FROM historical_data
                WHERE date BETWEEN %s AND %s
            """
            cursor.execute(query, (start_date, end_date))
            symbols = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(symbols)} trading symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching trading symbols: {e}", exc_info=True)
            if 'conn' in locals() and conn.is_connected():
                conn.close()
            return []

    def store_features(self, df: pd.DataFrame, table_name: str = 'ml_features', batch_size: int = 1000) -> bool:
        """
        Stores the generated features in a MySQL table.
        
        Args:
            df: DataFrame containing the features to store
            table_name: Name of the table to store the features in
            batch_size: Number of rows to insert in each batch
            
        Returns:
            True if successful, False otherwise
        """
        if df.empty:
            logger.warning("DataFrame is empty. Nothing to store.")
            return False
            
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()

            # Get the actual columns in the table
            cursor.execute(f"SHOW COLUMNS FROM {table_name}")
            table_columns = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Table columns: {len(table_columns)}, DataFrame columns: {len(df.columns)}")
            
            # Filter dataframe to include only columns that exist in the table
            df_columns = set(df.columns)
            table_columns_set = set(table_columns)
            
            # Find columns in DataFrame but not in table
            missing_in_table = df_columns - table_columns_set
            if missing_in_table:
                logger.warning(f"Columns in DataFrame but not in table: {missing_in_table}")
            
            # Find columns in table but not in DataFrame
            missing_in_df = table_columns_set - df_columns
            if missing_in_df:
                logger.warning(f"Columns in table but not in DataFrame: {missing_in_df}")
            
            # Get common columns
            common_columns = list(df_columns.intersection(table_columns_set))
            
            # Ensure we have at least some common columns
            if not common_columns:
                logger.error("No common columns between DataFrame and table")
                cursor.close()
                conn.close()
                return False
                
            logger.info(f"Using {len(common_columns)} common columns for insertion")
            
            # Create filtered DataFrame with only common columns
            df_filtered = df[common_columns].copy()
            
            # Ensure all numeric columns are float
            numeric_cols = df_filtered.select_dtypes(include=np.number).columns
            df_filtered[numeric_cols] = df_filtered[numeric_cols].astype(float)
            
            # Replace any remaining NaN/inf values
            df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
            df_filtered = df_filtered.fillna(0)  # Fill NaN with zeros
            
            # Convert datetime columns to string format that MySQL can handle
            if 'date' in df_filtered.columns:
                # Format as 'YYYY-MM-DD' string for MySQL compatibility
                df_filtered['date'] = df_filtered['date'].dt.strftime('%Y-%m-%d')
            
            # First, let's delete existing data for these symbols and dates to avoid duplicates
            symbols = df_filtered['trading_symbol'].unique().tolist()
            
            # Extract min and max dates as strings in YYYY-MM-DD format to avoid timestamp conversion issues
            min_date_str = df_filtered['date'].min()
            max_date_str = df_filtered['date'].max()
            
            symbol_placeholders = ', '.join(['%s'] * len(symbols))
            delete_query = f"""
                DELETE FROM {table_name}
                WHERE trading_symbol IN ({symbol_placeholders})
                AND date BETWEEN %s AND %s
            """
            delete_params = symbols + [min_date_str, max_date_str]
            
            logger.info(f"Deleting existing data for {len(symbols)} symbols between {min_date_str} and {max_date_str}")
            cursor.execute(delete_query, delete_params)
            deleted_rows = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_rows} existing rows that would conflict with new data")

            # Prepare the INSERT query
            columns = ", ".join([f"`{col}`" for col in common_columns])
            placeholders = ", ".join(["%s"] * len(common_columns))
            insert_query = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
            """

            # Execute the INSERT query for the data in batches
            total_rows = len(df_filtered)
            rows_inserted = 0
            
            for i in tqdm.tqdm(range(0, total_rows, batch_size), desc=f"Storing features in {table_name}"):
                batch_df = df_filtered[i:i + batch_size]
                
                # Convert DataFrame to list of tuples for MySQL executemany
                # This is more efficient than iterating through rows
                data = list(batch_df.itertuples(index=False, name=None))
                
                try:
                    cursor.executemany(insert_query, data)
                    conn.commit()
                    rows_inserted += len(batch_df)
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}", exc_info=True)
                    conn.rollback()
                    
                    # Try row by row if batch insert fails
                    if len(batch_df) > 1:
                        logger.info("Trying row-by-row insertion for problematic batch")
                        for j, row in batch_df.iterrows():
                            try:
                                cursor.execute(insert_query, tuple(row))
                                conn.commit()
                                rows_inserted += 1
                            except Exception as row_error:
                                logger.error(f"Error inserting row {j}: {row_error}")
                                conn.rollback()

            logger.info(f"Successfully stored {rows_inserted}/{total_rows} rows in {table_name}")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Error storing features in MySQL: {e}", exc_info=True)
            if 'conn' in locals() and conn.is_connected():
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
                finally:
                    conn.close()
            return False

    def validate_data(self, trading_symbols: List[str], start_date: str, end_date: Union[str, date]) -> pd.DataFrame:
        """
        Validates data availability and quality for the specified symbols and date range.
        
        Args:
            trading_symbols: List of trading symbols to validate
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format or as a date object
            
        Returns:
            DataFrame with validation results
        """
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor(dictionary=True)
            
            validation_results = []
            
            for symbol in trading_symbols:
                # Check historical_data
                hd_query = """
                    SELECT 
                        COUNT(*) as count, 
                        MIN(date) as min_date, 
                        MAX(date) as max_date
                    FROM historical_data
                    WHERE trading_symbol = %s
                    AND date BETWEEN %s AND %s
                """
                cursor.execute(hd_query, (symbol, start_date, end_date))
                hd_result = cursor.fetchone()
                
                # Check technical_indicators
                ti_query = """
                    SELECT 
                        COUNT(*) as count, 
                        MIN(date) as min_date, 
                        MAX(date) as max_date
                    FROM technical_indicators
                    WHERE trading_symbol = %s
                    AND date BETWEEN %s AND %s
                """
                cursor.execute(ti_query, (symbol, start_date, end_date))
                ti_result = cursor.fetchone()
                
                # Check data completeness for technical_indicators
                if ti_result['count'] > 0:
                    missing_query = """
                        SELECT COUNT(*) as missing_count
                        FROM technical_indicators
                        WHERE trading_symbol = %s
                        AND date BETWEEN %s AND %s
                        AND (
                            sma_20 IS NULL OR
                            rsi_14 IS NULL OR
                            macd_line IS NULL OR
                            atr_14 IS NULL
                        )
                    """
                    cursor.execute(missing_query, (symbol, start_date, end_date))
                    missing_result = cursor.fetchone()
                    missing_count = missing_result['missing_count']
                else:
                    missing_count = 0
                
                validation_results.append({
                    'trading_symbol': symbol,
                    'hd_count': hd_result['count'],
                    'hd_min_date': hd_result['min_date'],
                    'hd_max_date': hd_result['max_date'],
                    'ti_count': ti_result['count'],
                    'ti_min_date': ti_result['min_date'],
                    'ti_max_date': ti_result['max_date'],
                    'missing_indicators': missing_count,
                    'has_sufficient_data': hd_result['count'] > 10 and ti_result['count'] > 10 and missing_count == 0
                })
            
            cursor.close()
            conn.close()
            
            return pd.DataFrame(validation_results)
            
        except Exception as e:
            logger.error(f"Error validating data: {e}", exc_info=True)
            if 'conn' in locals() and conn.is_connected():
                conn.close()
            return pd.DataFrame()

    def test_database_connection(self) -> bool:
        """Test the database connection and timestamp handling."""
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            
            # Test a simple insert with a date to verify timestamp handling
            test_query = """
                CREATE TEMPORARY TABLE IF NOT EXISTS timestamp_test (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    test_date DATE,
                    test_string VARCHAR(50)
                )
            """
            cursor.execute(test_query)
            
            # Try inserting a date
            today_str = datetime.now().date().strftime('%Y-%m-%d')
            insert_query = "INSERT INTO timestamp_test (test_date, test_string) VALUES (%s, %s)"
            cursor.execute(insert_query, (today_str, "Test string"))
            conn.commit()
            
            # Verify it worked
            cursor.execute("SELECT * FROM timestamp_test")
            results = cursor.fetchall()
            logger.info(f"Database connection test successful. Inserted and retrieved {len(results)} rows.")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}", exc_info=True)
            if 'conn' in locals() and conn.is_connected():
                conn.close()
            return False


def main():
    """Main function to run the feature engineering pipeline."""
    db_config = {
        'host': 'localhost',
        'user': 'dhan_hq',
        'password': 'Passw0rd@098',
        'database': 'dhanhq_db',
        'auth_plugin': 'mysql_native_password',
        'use_pure': True
    }

    feature_engineer = FeatureEngineering(db_config)

    # Test database connection first
    logger.info("Testing database connection...")
    if not feature_engineer.test_database_connection():
        logger.error("Database connection test failed. Exiting.")
        sys.exit(1)

    # Set date range
    ist = pytz.timezone('Asia/Kolkata')
    now_utc = datetime.utcnow()
    now_ist = now_utc.replace(tzinfo=pytz.utc).astimezone(ist)
    end_date = now_ist.date()
    
    # Use a more recent start date to reduce data volume and improve quality
    start_date = '2023-01-01'
    
    logger.info(f"Processing data from {start_date} to {end_date}")

    # Get trading symbols
    all_symbols = feature_engineer.get_trading_symbols(start_date, end_date)
    
    if not all_symbols:
        logger.error("No trading symbols found. Exiting.")
        sys.exit(1)

    # Test with limited symbols
    test_mode = False
    if test_mode:
        # Try to get symbols known to have good data quality
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        # Keep only those that exist in our database
        trading_symbols = [symbol for symbol in test_symbols if symbol in all_symbols]
        
        # If none of our preferred symbols exist, use the first few from all_symbols
        if not trading_symbols and all_symbols:
            trading_symbols = all_symbols[:5]
            
        logger.info(f"Test mode enabled. Using symbols: {trading_symbols}")
    else:
        trading_symbols = all_symbols
    
    # Validate data quality before proceeding
    logger.info("Validating data quality...")
    validation_df = feature_engineer.validate_data(trading_symbols, start_date, end_date)
    
    if not validation_df.empty:
        logger.info("\nData validation results:")
        for _, row in validation_df.iterrows():
            status = "GOOD" if row['has_sufficient_data'] else "INSUFFICIENT"
            logger.info(f"{row['trading_symbol']}: {status} - Historical: {row['hd_count']} rows, Technical: {row['ti_count']} rows, Missing: {row['missing_indicators']}")
        
        # Filter to only use symbols with sufficient data
        valid_symbols = validation_df[validation_df['has_sufficient_data']]['trading_symbol'].tolist()
        
        if not valid_symbols:
            logger.error("No symbols have sufficient data quality. Exiting.")
            sys.exit(1)
            
        logger.info(f"Proceeding with {len(valid_symbols)} validated symbols")
        trading_symbols = valid_symbols
    
    # Fetch data
    logger.info("Fetching data...")
    df = feature_engineer.fetch_data(trading_symbols, start_date, end_date)
    
    if df.empty:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)
    
    # Display data stats
    logger.info(f"Data fetched: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create features
    logger.info("Creating features...")
    df_featured = feature_engineer.create_features(df)
    
    if not df_featured.empty:
        # Display feature stats
        logger.info(f"Features created: {df_featured.shape[0]} rows, {df_featured.shape[1]} columns")
        
        # Store features
        logger.info("Storing features...")
        success = feature_engineer.store_features(df_featured)
        
        if success:
            logger.info("Process completed successfully")
        else:
            logger.error("Failed to store features")
    else:
        logger.error("Feature creation resulted in empty dataset. Check your data and parameters.")


if __name__ == '__main__':
    main()