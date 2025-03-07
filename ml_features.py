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
            batch_size = 100  # Process 50 symbols at a time
            all_data = []
            
            for i in range(0, len(trading_symbols), batch_size):
                batch_symbols = trading_symbols[i:i+batch_size]
                placeholders = ', '.join(['%s'] * len(batch_symbols))
                
                query = f"""
                    SELECT
                        hd.date,
                        hd.trading_symbol,
                        hd.open, hd.high, hd.low, hd.close, hd.volume,
                        ti.sma_20, ti.sma_50, ti.sma_200,
                        ti.ema_20, ti.ema_50,
                        ti.vama_20d, ti.vpci_20d, ti.volume_sma_50d, ti.nifty_corr_full,
                        ti.macd_line, ti.macd_signal, ti.macd_histogram,
                        ti.adx_14, ti.di_plus_14, ti.di_minus_14,
                        ti.bollinger_upper, ti.bollinger_middle, ti.bollinger_lower,
                        ti.atr_14, ti.natr,
                        ti.rsi_14,
                        ti.stochastic_k, ti.stochastic_d,
                        ti.cci_20, ti.mfi_14, ti.williams_r, ti.roc,
                        ti.trix, ti.ultosc, ti.bop, ti.stddev, ti.var,
                        ti.return_1d, ti.return_3d, ti.return_5d, ti.return_10d, 
                        ti.return_20d, ti.return_40d, ti.return_60d, ti.return_120d,
                        -- Additional calendar features
                        ti.day_of_week, ti.day_of_month, ti.month, ti.quarter,
                        ti.is_weekday_1, ti.is_weekday_2, ti.is_weekday_3, ti.is_weekday_4, ti.is_weekday_5,
                        ti.is_month_start, ti.is_month_end, ti.is_quarter_start, ti.is_quarter_end,
                        -- Correlation features
                        ti.nifty_corr_20d, ti.nifty_corr_60d, ti.nifty_corr_120d, 
                        -- Relative strength features
                        ti.rs_nifty_5d, ti.rs_nifty_10d, ti.rs_nifty_20d, ti.rs_nifty_60d, ti.rs_nifty_120d,
                        -- Volume metrics
                        ti.volume_sma_5d, ti.volume_sma_10d, ti.volume_sma_20d, ti.volume_ratio_20d,
                        ti.volume_roc_1d, ti.volume_roc_5d, ti.volume_roc_10d,
                        -- Money flow indicators
                        ti.cmf_20d, ti.pvt, ti.volume_oscillator, ti.eom_14d
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
                        lag_days: List[int] = [1, 2, 3, 5, 10, 15, 20],
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
                # Apply optimized NaN handling for technical indicators
                # This is more effective than just dropping rows with NaNs
                symbol_df = self.clean_data_for_ml(symbol_df)
                
                # Instead of adding features one by one, create a dictionary of features
                new_features = {}
                
                # 1. Lagged Features - process each column individually
                for lag in lag_days:
                    for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'rsi_14', 'macd_line', 'atr_14']:
                        if col in symbol_df.columns:
                            new_features[f'{col}_lag{lag}'] = symbol_df[col].shift(lag)
                
                # 2. Derived Features
                new_features['prev_close'] = symbol_df['close'].shift(1)
                new_features['prev_volume'] = symbol_df['volume'].shift(1)
                
                # Avoid division by zero or very small numbers - reference from new_features
                prev_close = new_features['prev_close']
                prev_volume = new_features['prev_volume']
                
                new_features['price_change'] = (symbol_df['close'] - prev_close) / (prev_close.replace(0, np.nan) + 1e-6)
                new_features['volume_change'] = (symbol_df['volume'] - prev_volume) / (prev_volume.replace(0, np.nan) + 1e-6)
                new_features['high_low_range'] = (symbol_df['high'] - symbol_df['low']) / symbol_df['close'].replace(0, np.nan)
                new_features['close_open_range'] = (symbol_df['close'] - symbol_df['open']) / symbol_df['close'].replace(0, np.nan)

                # Market regime features
                if 'nifty_corr_20d' in symbol_df.columns:
                    # Correlation divergence (when stock moves differently from market)
                    new_features['corr_change'] = symbol_df['nifty_corr_20d'] - symbol_df['nifty_corr_60d']
                    # Market regime feature (high correlation is different regime than low correlation)
                    new_features['market_regime'] = pd.cut(symbol_df['nifty_corr_20d'], 
                                                    bins=[-1.1, -0.5, 0, 0.5, 1.1],
                                                    labels=[0, 1, 2, 3])

                # Seasonal effects
                if 'day_of_week' in symbol_df.columns:
                    # Convert categorical day of week to dummy variables if not already in is_weekday format
                    if 'is_weekday_1' not in symbol_df.columns:
                        for day in range(1, 6):  # 1 through 5 for weekdays
                            new_features[f'is_day_{day}'] = (symbol_df['day_of_week'] == day).astype(int)
                    
                    # Month effect
                    if 'month' in symbol_df.columns:
                        # Create quarter dummy variables
                        for q in range(1, 5):
                            new_features[f'is_quarter_{q}'] = (symbol_df['quarter'] == q).astype(int)

                # Volume pattern features
                if 'volume_ratio_20d' in symbol_df.columns:
                    # High volume breakout signal - use price_change from new_features
                    new_features['high_volume_signal'] = ((symbol_df['volume_ratio_20d'] > 1.5) & 
                                                    (new_features['price_change'] > 0)).astype(int)
                    
                    # Volume trend features
                    if 'volume_roc_5d' in symbol_df.columns and 'volume_roc_10d' in symbol_df.columns:
                        new_features['volume_trend'] = np.where(
                            (symbol_df['volume_roc_5d'] > 0) & (symbol_df['volume_roc_10d'] > 0), 1,
                            np.where((symbol_df['volume_roc_5d'] < 0) & (symbol_df['volume_roc_10d'] < 0), -1, 0)
                        )

                # Money flow confirmation
                if 'cmf_20d' in symbol_df.columns:
                    # Money flow confirmation of price trend - use price_change from new_features
                    new_features['price_cmf_confirm'] = ((new_features['price_change'] > 0) & 
                                                    (symbol_df['cmf_20d'] > 0)).astype(int)
                
                # 3. Target Variable
                new_features['future_close'] = symbol_df['close'].shift(-prediction_horizon)
                new_features['future_return'] = (new_features['future_close'] - symbol_df['close']) / symbol_df['close'].replace(0, np.nan)
                new_features['target'] = np.where(new_features['future_return'] > threshold, 1, 0)
                # Additional targets for more specific predictions
                # Exit signal (when to sell)
                new_features['exit_signal'] = np.where(new_features['future_return'] < -threshold, 1, 0)  

                # Time to target - number of days to reach peak return within prediction horizon
                def days_to_peak(row_idx, horizon, close_series):
                    if row_idx + horizon >= len(close_series):
                        return np.nan
                    prices = close_series.iloc[row_idx:row_idx+horizon+1].values
                    base_price = prices[0]
                    returns = (prices - base_price) / base_price
                    max_return_idx = np.argmax(returns)
                    return max_return_idx if max_return_idx > 0 else np.nan

                # Apply days to peak calculation
                new_features['days_to_target'] = [days_to_peak(i, prediction_horizon, symbol_df['close']) 
                                            for i in range(len(symbol_df))]
                
                # Price pattern features
                price_change_series = pd.Series(new_features['price_change'])
                new_features['price_acceleration'] = price_change_series - price_change_series.shift(1)
                
                new_features['breakout'] = ((symbol_df['high'] > symbol_df['high'].rolling(20).max().shift(1)) & 
                                        (symbol_df['volume'] > symbol_df['volume'].rolling(20).mean() * 1.5)).astype(int)

                # Volatility features
                new_features['atr_ratio'] = symbol_df['atr_14'] / symbol_df['close']
                new_features['bollinger_width'] = (symbol_df['bollinger_upper'] - symbol_df['bollinger_lower']) / symbol_df['bollinger_middle']

                # Support/resistance features
                new_features['dist_to_upper_band'] = (symbol_df['bollinger_upper'] - symbol_df['close']) / symbol_df['close']
                new_features['dist_to_lower_band'] = (symbol_df['close'] - symbol_df['bollinger_lower']) / symbol_df['close']
                new_features['dist_to_sma200'] = (symbol_df['close'] - symbol_df['sma_200']) / symbol_df['close']

                # Momentum divergence
                new_features['price_rsi_divergence'] = ((symbol_df['close'] > symbol_df['close'].shift(5)) & 
                                                    (symbol_df['rsi_14'] < symbol_df['rsi_14'].shift(5))).astype(int)
                
                # These are columns that exist in the database table but not in our new features
                if 'vama_20d' in symbol_df.columns and 'vama_20d' not in new_features:
                    new_features['vama_20d'] = symbol_df['vama_20d']

                if 'vpci_20d' in symbol_df.columns and 'vpci_20d' not in new_features:
                    new_features['vpci_20d'] = symbol_df['vpci_20d']

                if 'volume_sma_50d' in symbol_df.columns and 'volume_sma_50d' not in new_features:
                    new_features['volume_sma_50d'] = symbol_df['volume_sma_50d']

                if 'nifty_corr_full' in symbol_df.columns and 'nifty_corr_full' not in new_features:
                    new_features['nifty_corr_full'] = symbol_df['nifty_corr_full']

                # Calendar year start/end indicators
                # These might not be in the fetched data, so we'll calculate them
                if 'is_year_start' not in new_features:
                    if 'is_year_start' in symbol_df.columns:
                        new_features['is_year_start'] = symbol_df['is_year_start']
                    elif 'date' in symbol_df.columns:
                        dates = pd.to_datetime(symbol_df['date'])
                        new_features['is_year_start'] = ((dates.dt.month == 1) & (dates.dt.day == 1)).astype(int)

                if 'is_year_end' not in new_features:
                    if 'is_year_end' in symbol_df.columns:
                        new_features['is_year_end'] = symbol_df['is_year_end']
                    elif 'date' in symbol_df.columns:
                        dates = pd.to_datetime(symbol_df['date'])
                        new_features['is_year_end'] = ((dates.dt.month == 12) & (dates.dt.day == 31)).astype(int)

                # Mean reversion potential
                new_features['zscore_20d'] = (symbol_df['close'] - symbol_df['close'].rolling(20).mean()) / symbol_df['close'].rolling(20).std()

                # Trend strength features
                new_features['adx_trend_strength'] = np.where(symbol_df['adx_14'] > 25, 1, 0)
                new_features['di_crossover'] = np.where(symbol_df['di_plus_14'] > symbol_df['di_minus_14'], 1, -1)

                # Remove columns that aren't in the database table
                if 'is_quarter_3' in result_df.columns:
                    logger.info("Removing 'is_quarter_3' column as it's not in the database table")
                    result_df = result_df.drop(columns=['is_quarter_3'])

                # Historical performance during similar market conditions
                new_features['return_ratio_20_5'] = symbol_df['return_20d'] / (symbol_df['return_5d'] + 1e-6)  # Momentum consistency
                new_features['return_consistency'] = (symbol_df['return_5d'] > 0).rolling(10).mean()  # % of positive 5-day returns recently

                # Relative performance
                sector_columns = [col for col in symbol_df.columns if 'rs_nifty' in col]
                if sector_columns:
                    new_features['sector_strength'] = symbol_df[sector_columns].mean(axis=1)
                
                # Create a DataFrame from the new features
                features_df = pd.DataFrame(new_features, index=symbol_df.index)
                
                # Combine with original data
                symbol_df = pd.concat([symbol_df, features_df], axis=1)
                
                # Remove inf values
                symbol_df = symbol_df.replace([np.inf, -np.inf], np.nan)
                
                # Clean newly created features with NaNs using similar approach
                # These will mostly be in the lagged and derived features
                new_feature_cols = list(new_features.keys())

                if new_feature_cols:
                    for col in new_feature_cols:
                        if col in symbol_df.columns:
                            # Check if the column has any non-NA values - convert to a boolean scalar first
                            has_valid_values = symbol_df[col].notna().any()
                            if isinstance(has_valid_values, bool) and has_valid_values:
                                # For lag features, use next value in series (bfill)
                                if 'lag' in col:
                                    symbol_df[col] = symbol_df[col].bfill(limit=1)
                                    symbol_df[col] = symbol_df[col].ffill(limit=1)
                                    symbol_df[col] = symbol_df[col].fillna(0.0)
                                # For price change features, use median or zero
                                elif any(x in col for x in ['change', 'range']):
                                    median_val = symbol_df[col].median()
                                    symbol_df[col] = symbol_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                                else:
                                    symbol_df[col] = symbol_df[col].fillna(0.0)
                            else:
                                # All values are NA, just fill with zeros
                                symbol_df[col] = symbol_df[col].fillna(0.0)
                
                # Drop rows with NaN in target variables - we can't train without valid targets
                # This is the only place where we still need to drop rows
                target_columns = ['future_close', 'future_return', 'target']
                rows_before = len(symbol_df)
                symbol_df = symbol_df.dropna(subset=target_columns)
                rows_after = len(symbol_df)
                
                if rows_before > rows_after:
                    logger.debug(f"Removed {rows_before - rows_after} rows with NaN in target variables for {symbol}")
                
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
    
    def clean_data_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for machine learning using an optimized approach for handling NaN values
        in technical indicators based on their financial meaning.
        
        Args:
            df: DataFrame containing technical indicators with potential NaN values
            
        Returns:
            DataFrame with NaN values appropriately handled for ML
        """
        try:
            # Count initial rows
            initial_row_count = len(df)
            if df.empty:
                logger.warning("Empty DataFrame provided, nothing to clean")
                return df
            
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Get list of all indicator columns (excluding date and trading_symbol)
            indicator_columns = [col for col in df.columns 
                                if col not in ['date', 'trading_symbol', 'id', 'open', 'high', 'low', 'close', 'volume']]
            
            # Calculate percentage of missing values by column before cleaning
            missing_by_column = df[indicator_columns].isnull().mean() * 100
            high_missing_cols = missing_by_column[missing_by_column > 25].index.tolist()
            if high_missing_cols:
                logger.warning(f"Columns with >25% missing values: {high_missing_cols}")
            
            # Group indicators by type for appropriate handling
            trend_indicators = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'bollinger_'))]

            # 1. Calendar features - these should be kept as is, no cleaning needed as they're discrete values
            calendar_indicators = ['day_of_week', 'day_of_month', 'month', 'quarter',
                                'is_weekday_1', 'is_weekday_2', 'is_weekday_3', 'is_weekday_4', 'is_weekday_5',
                                'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
                                'is_year_start', 'is_year_end']
            
            for col in calendar_indicators:
                if col in df.columns and df[col].isnull().any():
                    # For day_of_week, month, etc., use the most common value
                    if col in ['day_of_week', 'day_of_month', 'month', 'quarter']:
                        most_common = df[col].mode().iloc[0]
                        df[col] = df[col].fillna(most_common)
                    # For boolean flags, fill with 0 (False)
                    else:
                        df[col] = df[col].fillna(0)
            
            # 2. Correlation features
            correlation_indicators = [col for col in df.columns if 'corr' in col]
            for col in correlation_indicators:
                if col in df.columns and df[col].isnull().any():
                    # For correlation, 0 means no correlation
                    df[col] = df[col].fillna(0.0)
            
            # 3. Relative strength indicators
            rs_indicators = [col for col in df.columns if col.startswith('rs_')]
            for col in rs_indicators:
                if col in df.columns and df[col].isnull().any():
                    # 1.0 means equal performance to the reference
                    df[col] = df[col].fillna(1.0)

            # 5. Money flow indicators
            money_flow_indicators = ['cmf_20d', 'pvt', 'volume_oscillator', 'eom_14d', 'vama_20d', 'vpci_20d']
            for col in money_flow_indicators:
                if col in df.columns and df[col].isnull().any():
                    # For money flow indicators, 0 is neutral
                    df[col] = df[col].fillna(0.0)

            momentum_oscillators = [
                'rsi_14', 'stochastic_k', 'stochastic_d', 'macd_line', 'macd_signal', 
                'macd_histogram', 'cci_20', 'mfi_14', 'williams_r', 'roc'
            ]
            
            volatility_indicators = ['atr_14', 'natr', 'stddev', 'var']
            
            volume_indicators = [col for col in df.columns if 'volume' in col] + ['obv', 'ad_line', 'adosc']

            # Handle the new volume metrics specifically
            volume_metric_indicators = [col for col in df.columns if any(x in col for x in ['volume_sma_', 'volume_ratio_', 'volume_roc_'])]
            for col in volume_metric_indicators:
                if col in df.columns and df[col].isnull().any():
                    if 'ratio' in col:
                        # For volume ratios, 1.0 means normal volume
                        df[col] = df[col].fillna(1.0)
                    elif 'roc' in col:
                        # For rate of change, 0 means no change
                        df[col] = df[col].fillna(0.0)
                    else:
                        # For other volume metrics, use median
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
            directional_indicators = ['adx_14', 'di_plus_14', 'di_minus_14']
            
            correlation_indicators = [col for col in df.columns if 'corr' in col]
            
            relative_strength_indicators = [col for col in df.columns if col.startswith('rs_')]
            
            return_indicators = [col for col in df.columns if col.startswith('return_')]
            
            # Filter for indicators that actually exist in the dataframe
            all_indicator_groups = {
                'trend': [col for col in trend_indicators if col in df.columns],
                'momentum': [col for col in momentum_oscillators if col in df.columns],
                'volatility': [col for col in volatility_indicators if col in df.columns],
                'volume': [col for col in volume_indicators if col in df.columns],
                'directional': [col for col in directional_indicators if col in df.columns],
                'correlation': [col for col in correlation_indicators if col in df.columns],
                'relative_strength': [col for col in relative_strength_indicators if col in df.columns],
                'returns': [col for col in return_indicators if col in df.columns]
            }
            
            # Apply appropriate NaN handling strategies by indicator type
            
            # 1. Trend indicators: Forward fill (continues the trend)
            for col in all_indicator_groups['trend']:
                if col in df.columns and df[col].isnull().any():
                    # Forward fill with limit to avoid extending too far
                    df[col] = df[col].ffill(limit=5)
                    # If any NaNs remain at the beginning of the series, backward fill
                    df[col] = df[col].bfill(limit=5)
                    # Any remaining NaNs get the column median
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
            
            # 2. Momentum oscillators: Fill with neutral values
            for col in all_indicator_groups['momentum']:
                if col in df.columns and df[col].isnull().any():
                    if col == 'rsi_14':
                        df[col] = df[col].fillna(50.0)  # Neutral RSI
                    elif col in ['stochastic_k', 'stochastic_d']:
                        df[col] = df[col].fillna(50.0)  # Neutral stochastic
                    elif col in ['macd_line', 'macd_signal', 'macd_histogram']:
                        df[col] = df[col].fillna(0.0)  # Neutral MACD
                    elif col == 'cci_20':
                        df[col] = df[col].fillna(0.0)  # Neutral CCI
                    elif col == 'williams_r':
                        df[col] = df[col].fillna(-50.0)  # Neutral Williams %R
                    else:
                        df[col] = df[col].fillna(0.0)  # Default neutral value
            
            # 3. Volatility indicators: Fill with median for the series
            for col in all_indicator_groups['volatility']:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)
            
            # 4. Volume indicators: Fill with median or zero depending on indicator
            for col in all_indicator_groups['volume']:
                if col in df.columns and df[col].isnull().any():
                    if 'ratio' in col or col.endswith('_sma'):
                        df[col] = df[col].fillna(1.0)  # Neutral volume ratio
                    else:
                        df[col] = df[col].fillna(0.0)  # Neutral for other volume indicators
            
            # 5. Directional indicators: Fill with neutral values
            for col in all_indicator_groups['directional']:
                if col in df.columns and df[col].isnull().any():
                    if col == 'adx_14':
                        df[col] = df[col].fillna(25.0)  # Weak trend strength
                    else:
                        df[col] = df[col].fillna(25.0)  # Neutral directional indicator
            
            # 6. Correlation indicators: Fill with zero (no correlation)
            for col in all_indicator_groups['correlation']:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(0.0)  # No correlation
            
            # 7. Relative strength indicators: Fill with one (equal performance)
            for col in all_indicator_groups['relative_strength']:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(1.0)  # Equal relative performance
            
            # 8. Return indicators: Special handling - more complex
            for col in all_indicator_groups['returns']:
                if col in df.columns and df[col].isnull().any():
                    # For returns, we'll use forward and backward fill with smaller limits 
                    # then fall back to zero (no change)
                    df[col] = df[col].ffill(limit=3)
                    df[col] = df[col].bfill(limit=3)
                    df[col] = df[col].fillna(0.0)  # No change
            # Add to the clean_data_for_ml method
            pattern_indicators = [col for col in df.columns if any(x in col for x in ['breakout', 'divergence', 'crossover'])]

            if pattern_indicators:
                for col in pattern_indicators:
                    if col in df.columns and df[col].isnull().any():
                        df[col] = df[col].fillna(0)  # For pattern indicators, no pattern is the default
            
            # Handle any remaining columns with NaN values
            remaining_cols = [col for col in indicator_columns 
                            if col in df.columns and df[col].isnull().any()]
            
            if remaining_cols:
                logger.warning(f"Handling {len(remaining_cols)} remaining columns with NaNs")
                for col in remaining_cols:
                    # Use column median if available, otherwise zero
                    median_val = df[col].median()
                    fill_value = median_val if not pd.isna(median_val) else 0.0
                    df[col] = df[col].fillna(fill_value)
            
            # Check for any remaining NaN values
            remaining_nulls = df[indicator_columns].isnull().sum().sum()
            if remaining_nulls > 0:
                # Find columns with remaining NaN values
                cols_with_nulls = df[indicator_columns].columns[df[indicator_columns].isnull().any()].tolist()
                logger.warning(f"Warning: {remaining_nulls} NaN values remain in columns: {cols_with_nulls}")
                
                # Final fallback: fill any remaining NaNs with zeros
                for col in cols_with_nulls:
                    df[col] = df[col].fillna(0.0)
            
            # Log the results
            removed_nans = initial_row_count * len(indicator_columns) - df[indicator_columns].isnull().sum().sum()
            logger.info(f"Cleaned {removed_nans} NaN values while preserving {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in clean_data_for_ml: {e}", exc_info=True)
            # In case of error, return the original DataFrame
            return df

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
            # First ensure all columns exist in the DataFrame
            valid_numeric_cols = [col for col in numeric_cols if col in df_filtered.columns]
            # Then perform the conversion
            if valid_numeric_cols:
                df_filtered[valid_numeric_cols] = df_filtered[valid_numeric_cols].astype(float)
            
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
    logger.info("Creating features with optimized NaN handling...")
    df_featured = feature_engineer.create_features(df, lag_days=[1, 2, 3, 5, 10, 15, 20])
    
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