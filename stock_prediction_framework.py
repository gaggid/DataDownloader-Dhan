import pandas as pd
import numpy as np
import mysql.connector
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mysql.connector import Error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost, Pool
import talib as ta
import optuna
import joblib
import warnings
import os
import logging
import gc
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockPredictionFramework:
    def __init__(self, db_password, prediction_horizon_days=[1, 3, 5, 7, 10], 
                upside_threshold_min=0.01, upside_threshold_max=0.03,
                downside_threshold_min=0.01, downside_threshold_max=0.03,
                use_pytorch=True, primary_horizon=7):
        """
        Initialize the stock prediction framework
        
        Args:
            db_password: Password for database connection
            prediction_horizon_days: List of days for prediction horizons
            upside_threshold_min: Minimum price movement threshold for long (1%)
            upside_threshold_max: Maximum price movement threshold for long (3%)
            downside_threshold_min: Minimum price movement threshold for short (1%)
            downside_threshold_max: Maximum price movement threshold for short (3%)
            use_pytorch: Whether to use PyTorch models
            primary_horizon: The primary prediction horizon to focus on
        """
        self.db_password = db_password
        self.prediction_horizon_days = prediction_horizon_days
        self.upside_threshold_min = upside_threshold_min
        self.upside_threshold_max = upside_threshold_max
        self.downside_threshold_min = downside_threshold_min
        self.downside_threshold_max = downside_threshold_max
        self.use_pytorch = use_pytorch
        self.primary_horizon = primary_horizon
        self.optimal_thresholds = {}
        
        # Configure PyTorch device - optimized for RTX 4060Ti
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Set this to True if you encounter memory issues
            torch.cuda.empty_cache()
            # Optional: Set precision to reduce memory usage
            torch.set_float32_matmul_precision('high')
            logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("PyTorch using CPU")
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'user': 'dhan_hq',
            'password': self.db_password,
            'database': 'dhanhq_db',
            'auth_plugin': 'mysql_native_password',
            'use_pure': True
        }
        
        # Model paths
        self.models_dir = 'trained_models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Results paths
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Feature selection results
        self.feature_importance = {}
        self.selected_features = {}
        
        # Model dictionaries
        self.models = {}
        self.scalers = {}
        
    def connect_to_database(self):
        """Establish connection to MySQL database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            logger.info("Database connection successful")
            return conn
        except Error as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            return None
        
    def fetch_historical_data(self, start_date='2020-01-01', limit=None):
        """
        Fetch historical stock data from the database
        
        Args:
            start_date: Starting date for historical data retrieval
            limit: Optional limit for number of rows to fetch
        
        Returns:
            DataFrame containing historical stock data
        """
        conn = self.connect_to_database()
        if conn is None:
            return None
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            query = f"""
                SELECT id, date, trading_symbol, company_name, exchange, 
                       security_id, open, high, low, close, volume, timestamp
                FROM historical_data
                WHERE date >= '{start_date}'
                ORDER BY trading_symbol, date
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query)
            data = cursor.fetchall()
            
            df = pd.DataFrame(data)
            
            # Convert data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Fetched {len(df)} records from database")
            
            return df
        
        except Error as e:
            logger.error(f"Error fetching data: {e}")
            return None
        
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def add_time_features(self, stock_df):
        """Add time-based features to the dataframe"""
        # Extract time components
        stock_df['day_of_week'] = stock_df['date'].dt.dayofweek
        stock_df['day_of_month'] = stock_df['date'].dt.day
        stock_df['week_of_year'] = stock_df['date'].dt.isocalendar().week
        stock_df['month'] = stock_df['date'].dt.month
        stock_df['quarter'] = stock_df['date'].dt.quarter
        
        # End-of-month effect (5 days before month end)
        stock_df['month_end'] = (stock_df['date'].dt.days_in_month - stock_df['date'].dt.day) <= 5
        
        # Day of week effect (Monday=0, Friday=4)
        for i in range(5):
            stock_df[f'weekday_{i}'] = (stock_df['day_of_week'] == i).astype(float)
        
        # Earnings season (Q1: Jan-Mar, Q2: Apr-Jun, etc.)
        for i in range(1, 5):
            stock_df[f'earnings_q{i}'] = (stock_df['quarter'] == i).astype(float)
            
        return stock_df
    
    def add_candle_features(self, stock_df):
        """Add candlestick pattern features to the dataframe"""
        # Total candle range
        stock_df['candle_range'] = stock_df['high'] - stock_df['low']
        
        # Bullish/bearish identification based on close position
        stock_df['close_position'] = (stock_df['close'] - stock_df['low']) / stock_df['candle_range']
        stock_df['bullish_candle'] = (stock_df['close_position'] > 0.55).astype(float)
        stock_df['bearish_candle'] = (stock_df['close_position'] < 0.45).astype(float)
        stock_df['doji_candle'] = ((stock_df['close_position'] >= 0.45) & 
                                (stock_df['close_position'] <= 0.55)).astype(float)
        
        # Body/shadow ratios
        stock_df['upper_shadow_ratio'] = (stock_df['high'] - stock_df[['open', 'close']].max(axis=1)) / stock_df['candle_range']
        stock_df['lower_shadow_ratio'] = (stock_df[['open', 'close']].min(axis=1) - stock_df['low']) / stock_df['candle_range']
        stock_df['body_ratio'] = abs(stock_df['close'] - stock_df['open']) / stock_df['candle_range']
        
        # Consecutive candle patterns (2-day patterns)
        # Convert to boolean before using logical operators
        stock_df['consecutive_bullish'] = ((stock_df['bullish_candle'] > 0.5) & 
                                        (stock_df['bullish_candle'].shift(1) > 0.5)).astype(float)
        stock_df['consecutive_bearish'] = ((stock_df['bearish_candle'] > 0.5) & 
                                        (stock_df['bearish_candle'].shift(1) > 0.5)).astype(float)
        
        # Engulfing patterns
        stock_df['bullish_engulfing'] = (
            (stock_df['open'] < stock_df['close']) &  # Current candle is bullish
            (stock_df['open'].shift(1) > stock_df['close'].shift(1)) &  # Previous candle is bearish
            (stock_df['open'] < stock_df['close'].shift(1)) &  # Current open below previous close
            (stock_df['close'] > stock_df['open'].shift(1))  # Current close above previous open
        ).astype(float)
        
        stock_df['bearish_engulfing'] = (
            (stock_df['open'] > stock_df['close']) &  # Current candle is bearish
            (stock_df['open'].shift(1) < stock_df['close'].shift(1)) &  # Previous candle is bullish
            (stock_df['open'] > stock_df['close'].shift(1)) &  # Current open above previous close
            (stock_df['close'] < stock_df['open'].shift(1))  # Current close below previous open
        ).astype(float)
        
        return stock_df
    
    def calculate_enhanced_volatility(self, stock_df):
        """
        Calculate enhanced volatility metrics including Parkinson and Garman-Klass volatility
        
        Args:
            stock_df: DataFrame for a specific stock
            
        Returns:
            DataFrame with added volatility features
        """
        # Calculate basic returns if they don't exist yet
        if 'return_1d' not in stock_df.columns:
            stock_df['return_1d'] = stock_df['close'].pct_change(1)
        
        # Calculate basic volatility features
        for window in [3, 5, 10, 15, 20, 30]:
            stock_df[f'volatility_{window}d'] = stock_df['return_1d'].rolling(window=window).std()
        
        # Parkinson volatility (uses high-low range)
        # Formula: √[1/(4*ln(2)) * Σ(ln(High/Low)²)]
        stock_df['parkinson_vol_5d'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * 
            (np.log(stock_df['high'] / stock_df['low']) ** 2).rolling(window=5).mean()
        )
        
        stock_df['parkinson_vol_10d'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * 
            (np.log(stock_df['high'] / stock_df['low']) ** 2).rolling(window=10).mean()
        )
        
        stock_df['parkinson_vol_20d'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * 
            (np.log(stock_df['high'] / stock_df['low']) ** 2).rolling(window=20).mean()
        )
        
        # Garman-Klass volatility (uses open, high, low, close)
        # Formula: √[0.5 * ln(High/Low)² - (2*ln(2)-1) * ln(Close/Open)²]
        stock_df['gk_vol_5d'] = np.sqrt(
            (0.5 * np.log(stock_df['high'] / stock_df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(stock_df['close'] / stock_df['open']) ** 2
            ).rolling(window=5).mean()
        )
        
        stock_df['gk_vol_10d'] = np.sqrt(
            (0.5 * np.log(stock_df['high'] / stock_df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(stock_df['close'] / stock_df['open']) ** 2
            ).rolling(window=10).mean()
        )
        
        stock_df['gk_vol_20d'] = np.sqrt(
            (0.5 * np.log(stock_df['high'] / stock_df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(stock_df['close'] / stock_df['open']) ** 2
            ).rolling(window=20).mean()
        )
        
        # Volatility ratio features (comparing different calculation methods)
        stock_df['vol_ratio_std_park_10d'] = stock_df['volatility_10d'] / stock_df['parkinson_vol_10d']
        stock_df['vol_ratio_park_gk_10d'] = stock_df['parkinson_vol_10d'] / stock_df['gk_vol_10d']
        
        # Volatility regime features
        stock_df['vol_regime_10d'] = pd.qcut(
            stock_df['volatility_10d'].fillna(stock_df['volatility_10d'].median()), 
            5, 
            labels=False, 
            duplicates='drop'
        ).astype(float)
        
        return stock_df
    
    def calculate_anchored_vwap(self, stock_df, lookback=30, volume_threshold=1.5):
        """
        Calculate anchored VWAP from significant high/low points
        
        Args:
            stock_df: Data for a specific symbol
            lookback: Window to look back for significant points
            volume_threshold: Volume ratio threshold to identify significant points
        
        Returns:
            DataFrame with added anchored VWAP features
        """
        # Skip if not enough data
        if len(stock_df) < lookback:
            logger.warning(f"Not enough data to calculate anchored VWAP for {stock_df['trading_symbol'].iloc[0]}")
            return stock_df
        
        # Calculate relative volume
        stock_df['rel_volume'] = stock_df['volume'] / stock_df['volume'].rolling(20).mean()
        
        # Find significant high and low points with high volume
        high_anchors = []
        low_anchors = []
        
        # Fill NaN values in rel_volume for first 20 rows
        stock_df['rel_volume'] = stock_df['rel_volume'].fillna(1.0)
        
        # Need to have bullish_candle and bearish_candle as boolean masks
        if 'bullish_candle' not in stock_df.columns:
            # Add them if they don't exist yet
            stock_df['candle_range'] = stock_df['high'] - stock_df['low']
            stock_df['close_position'] = (stock_df['close'] - stock_df['low']) / stock_df['candle_range']
            stock_df['bullish_candle'] = (stock_df['close_position'] > 0.55).astype(float)
            stock_df['bearish_candle'] = (stock_df['close_position'] < 0.45).astype(float)
        
        # Find anchor points
        for i in range(lookback, len(stock_df)):
            current_row = stock_df.iloc[i]
            
            # High volume day
            if current_row['rel_volume'] > volume_threshold:
                # Bullish candle - anchor from low
                if current_row['bullish_candle'] > 0.5:  # Treating as binary 0/1
                    low_anchors.append(i)
                # Bearish candle - anchor from high
                elif current_row['bearish_candle'] > 0.5:  # Treating as binary 0/1
                    high_anchors.append(i)
        
        # Calculate VWAP for each anchor point
        vwap_columns = []
        
        # From low points (support)
        for idx, anchor_idx in enumerate(low_anchors[-3:]):  # Keep only the 3 most recent
            try:
                anchor_date = stock_df.iloc[anchor_idx]['date']
                
                # Calculate VWAP from this anchor point
                vwap_col = f'vwap_support_{idx+1}'
                data_from_anchor = stock_df[stock_df['date'] >= anchor_date].copy()
                
                if len(data_from_anchor) > 0:
                    # Cumulative calculations
                    data_from_anchor['cum_pv'] = data_from_anchor['close'] * data_from_anchor['volume']
                    data_from_anchor['cum_vol'] = data_from_anchor['volume']
                    
                    # Calculate cumulative sum
                    data_from_anchor['cum_pv'] = data_from_anchor['cum_pv'].cumsum()
                    data_from_anchor['cum_vol'] = data_from_anchor['cum_vol'].cumsum()
                    
                    # Calculate VWAP
                    data_from_anchor[vwap_col] = data_from_anchor['cum_pv'] / data_from_anchor['cum_vol']
                    
                    # Merge back to the original dataframe
                    date_col_name = 'date'
                    stock_df = pd.merge(
                        stock_df, 
                        data_from_anchor[['date', vwap_col]], 
                        on=date_col_name, 
                        how='left'
                    )
                    vwap_columns.append(vwap_col)
            except Exception as e:
                logger.warning(f"Error calculating support VWAP: {e}")
        
        # From high points (resistance)
        for idx, anchor_idx in enumerate(high_anchors[-3:]):  # Keep only the 3 most recent
            try:
                anchor_date = stock_df.iloc[anchor_idx]['date']
                
                # Calculate VWAP from this anchor point
                vwap_col = f'vwap_resistance_{idx+1}'
                data_from_anchor = stock_df[stock_df['date'] >= anchor_date].copy()
                
                if len(data_from_anchor) > 0:
                    # Cumulative calculations
                    data_from_anchor['cum_pv'] = data_from_anchor['close'] * data_from_anchor['volume']
                    data_from_anchor['cum_vol'] = data_from_anchor['volume']
                    
                    # Calculate cumulative sum
                    data_from_anchor['cum_pv'] = data_from_anchor['cum_pv'].cumsum()
                    data_from_anchor['cum_vol'] = data_from_anchor['cum_vol'].cumsum()
                    
                    # Calculate VWAP
                    data_from_anchor[vwap_col] = data_from_anchor['cum_pv'] / data_from_anchor['cum_vol']
                    
                    # Merge back to the original dataframe
                    date_col_name = 'date'
                    stock_df = pd.merge(
                        stock_df, 
                        data_from_anchor[['date', vwap_col]], 
                        on=date_col_name, 
                        how='left'
                    )
                    vwap_columns.append(vwap_col)
            except Exception as e:
                logger.warning(f"Error calculating resistance VWAP: {e}")
        
        # Add relative features
        for col in vwap_columns:
            try:
                stock_df[f'dist_from_{col}'] = (stock_df['close'] / stock_df[col]) - 1
            except Exception as e:
                logger.warning(f"Error calculating distance from {col}: {e}")
        
        # Fill NaN values with 0 for VWAP features
        for col in vwap_columns:
            if col in stock_df.columns:
                stock_df[col] = stock_df[col].fillna(0)
                if f'dist_from_{col}' in stock_df.columns:
                    stock_df[f'dist_from_{col}'] = stock_df[f'dist_from_{col}'].fillna(0)
        
        return stock_df

    def engineer_features(self, df):
        """
        Create a rich set of features from the historical price data with optimized memory usage
            
        Args:
            df: DataFrame with historical stock data
                
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering process")
        
        # Add more debugging information
        total_stocks = len(df[df['exchange'] == 'NSE_EQ']['trading_symbol'].unique())
        total_indices = len(df[df['exchange'] == 'NSE_Indx']['trading_symbol'].unique()) if 'NSE_Indx' in df['exchange'].unique() else 0
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        avg_days_per_stock = df[df['exchange'] == 'NSE_EQ'].groupby('trading_symbol').size().mean()
        
        logger.info(f"Total stocks: {total_stocks}, Total indices: {total_indices}")
        logger.info(f"Date range: {date_range}")
        logger.info(f"Average days of data per stock: {avg_days_per_stock}")
        
        # Process indices first to create benchmark data
        index_returns_dict = {}
        
        if 'NSE_Indx' in df['exchange'].unique():
            indices_df = df[df['exchange'] == 'NSE_Indx'].copy()
            
            # Use NIFTY as the main index, or the first available index
            main_indices = ['NIFTY', 'NIFTY50', 'SENSEX']
            main_index = None
            
            for idx in main_indices:
                if idx in indices_df['trading_symbol'].unique():
                    main_index = idx
                    break
                    
            if main_index is None and not indices_df.empty:
                main_index = indices_df['trading_symbol'].iloc[0]
                
            logger.info(f"Using {main_index} as the benchmark index")
            
            if main_index:
                main_index_df = indices_df[indices_df['trading_symbol'] == main_index].sort_values('date')
                
                # Create daily returns for the index
                main_index_df['index_return_1d'] = main_index_df['close'].pct_change(1)
                
                # Create multi-day returns for the index
                for window in [3, 5, 10, 15, 20, 30]:
                    main_index_df[f'index_return_{window}d'] = main_index_df['close'].pct_change(window)
                    
                # Create a dictionary of index returns by date for easy lookup
                for col in [col for col in main_index_df.columns if 'index_return' in col]:
                    index_returns_dict[col] = dict(zip(main_index_df['date'], main_index_df[col]))
            
            # Clean up to free memory
            del indices_df
            if main_index:
                del main_index_df
            gc.collect()
        
        # Group by stock
        symbols = df[df['exchange'] == 'NSE_EQ']['trading_symbol'].unique()
        all_stocks_with_features = []
        
        # Track stocks with insufficient data
        insufficient_data_stocks = []
        processed_stocks = 0
        
        # List of features to exclude (low importance)
        exclude_features = [
            'min_low_20d', 'min_low_30d', 'bb_lower', 'ema_50d', 'sma_30d', 'obv', 
            'macd_signal', 'macd_hist'
        ]
        
        # Function to process each stock efficiently
        def process_stock(symbol):
            nonlocal processed_stocks
            
            logger.info(f"Engineering features for {symbol}")
            
            # Get data for this symbol
            stock_df = df[df['trading_symbol'] == symbol].sort_values('date').copy()
            
            # Log stock data count
            logger.info(f"Stock {symbol} has {len(stock_df)} days of data")
            
            if len(stock_df) < 20:  # Minimum requirement: 20 days
                return None, (symbol, len(stock_df))
            
            processed_stocks += 1
                
            # Create an empty dictionary to store all new columns
            new_columns = {}
            
            # Basic price features
            new_columns['return_1d'] = stock_df['close'].pct_change(1)
            new_columns['log_return_1d'] = np.log(stock_df['close'] / stock_df['close'].shift(1))
            
            # Price movement features
            new_columns['price_range'] = stock_df['high'] - stock_df['low']
            new_columns['price_range_pct'] = new_columns['price_range'] / stock_df['open']
            new_columns['body_size'] = abs(stock_df['close'] - stock_df['open'])
            new_columns['body_size_pct'] = new_columns['body_size'] / stock_df['open']
            new_columns['upper_shadow'] = stock_df['high'] - stock_df[['open', 'close']].max(axis=1)
            new_columns['lower_shadow'] = stock_df[['open', 'close']].min(axis=1) - stock_df['low']
            
            # Rolling price features - different windows
            for window in [3, 5, 10, 15, 20, 30]:
                # Returns and volatility
                new_columns[f'return_{window}d'] = stock_df['close'].pct_change(window)
                new_columns[f'volatility_{window}d'] = new_columns['return_1d'].rolling(window=window).std()
                
                # Skip low importance features for longer timeframes
                if f'min_low_{window}d' not in exclude_features:
                    new_columns[f'min_low_{window}d'] = stock_df['low'].rolling(window=window).min()
                    
                new_columns[f'max_high_{window}d'] = stock_df['high'].rolling(window=window).max()
                
                # Price location relative to range
                if f'min_low_{window}d' not in exclude_features:
                    new_columns[f'price_position_{window}d'] = (stock_df['close'] - new_columns[f'min_low_{window}d']) / \
                                                        (new_columns[f'max_high_{window}d'] - new_columns[f'min_low_{window}d'])
                
                # Rolling means - skip sma_30d
                if f'sma_{window}d' not in exclude_features:
                    new_columns[f'sma_{window}d'] = stock_df['close'].rolling(window=window).mean()
                    
                    # Distance from moving averages
                    new_columns[f'dist_from_sma_{window}d'] = (stock_df['close'] / new_columns[f'sma_{window}d']) - 1
            
            # Exponential Moving Averages - skip ema_50d
            for span in [5, 10, 20]:  # Removed 50 from the list
                new_columns[f'ema_{span}d'] = stock_df['close'].ewm(span=span, adjust=False).mean()
                new_columns[f'dist_from_ema_{span}d'] = (stock_df['close'] / new_columns[f'ema_{span}d']) - 1
            
            # Create EMA crossover features after all EMAs are calculated
            new_columns['ema_5_10_cross'] = np.where(new_columns['ema_5d'] > new_columns['ema_10d'], 1, -1)
            new_columns['ema_10_20_cross'] = np.where(new_columns['ema_10d'] > new_columns['ema_20d'], 1, -1)
            # Remove ema_20_50_cross since we're not calculating ema_50d
            
            # Volume features
            new_columns['volume_pct_change'] = stock_df['volume'].pct_change()
            for window in [3, 5, 10, 20]:
                new_columns[f'volume_sma_{window}d'] = stock_df['volume'].rolling(window=window).mean()
                new_columns[f'volume_ratio_{window}d'] = stock_df['volume'] / new_columns[f'volume_sma_{window}d']
            
            # Add relative performance vs index if available
            if index_returns_dict:
                # Map index returns to corresponding dates in stock data
                for col, values in index_returns_dict.items():
                    new_columns[col] = stock_df['date'].map(values)
                
                # Calculate relative performance
                new_columns['rel_return_1d'] = new_columns['return_1d'] - new_columns['index_return_1d']
                
                # Calculate relative returns for multiple periods
                for window in [3, 5, 10, 15, 20, 30]:
                    idx_col = f'index_return_{window}d'
                    stock_col = f'return_{window}d'
                    if idx_col in new_columns and stock_col in new_columns:
                        new_columns[f'rel_return_{window}d'] = new_columns[stock_col] - new_columns[idx_col]
                
                # Calculate relative strength indicators
                for window in [5, 10, 20]:
                    # Relative strength (ratio of cumulative returns)
                    new_columns[f'rel_strength_{window}d'] = (
                        (1 + new_columns['return_1d']).rolling(window=window).apply(lambda x: np.prod(x), raw=True) / 
                        (1 + new_columns['index_return_1d']).rolling(window=window).apply(lambda x: np.prod(x), raw=True)
                    ) - 1
            
            # Add TA-Lib indicators - exclude low importance ones
            try:
                # Momentum indicators
                new_columns['rsi_14'] = ta.RSI(stock_df['close'].values, timeperiod=14)
                
                # MACD - only keep the main MACD line, exclude signal and histogram
                macd, _, _ = ta.MACD(
                    stock_df['close'].values, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                new_columns['macd'] = macd
                
                # Bollinger Bands - exclude bb_lower
                bb_upper, bb_middle, _ = ta.BBANDS(
                    stock_df['close'].values,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2
                )
                new_columns['bb_upper'] = bb_upper
                new_columns['bb_middle'] = bb_middle
                # Skip bb_lower since it's in exclude_features
                new_columns['bb_width'] = (bb_upper - stock_df['low']) / bb_middle
                
                # Stochastic
                slowk, slowd = ta.STOCH(
                    stock_df['high'].values,
                    stock_df['low'].values,
                    stock_df['close'].values,
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0
                )
                new_columns['stoch_k'] = slowk
                new_columns['stoch_d'] = slowd
                
                # ADX
                new_columns['adx'] = ta.ADX(
                    stock_df['high'].values,
                    stock_df['low'].values,
                    stock_df['close'].values,
                    timeperiod=14
                )
                
                # Skip OBV since it's in exclude_features
                
                # ATR (Average True Range)
                new_columns['atr'] = ta.ATR(
                    stock_df['high'].values,
                    stock_df['low'].values,
                    stock_df['close'].values,
                    timeperiod=14
                )
                
            except Exception as e:
                logger.warning(f"Error calculating TA-Lib indicators for {symbol}: {e}")
            
            # Calculate enhanced volatility metrics
            stock_df = self.calculate_enhanced_volatility(stock_df)
            
            # Add new features
            
            # 1. Add time-based features
            stock_df_with_time = self.add_time_features(stock_df.copy())
            for col in stock_df_with_time.columns:
                if col not in stock_df.columns and col not in new_columns:
                    new_columns[col] = stock_df_with_time[col]
            
            # 2. Add candle pattern features
            stock_df_with_candles = self.add_candle_features(stock_df.copy())
            for col in stock_df_with_candles.columns:
                if col not in stock_df.columns and col not in new_columns:
                    new_columns[col] = stock_df_with_candles[col]
            
            # Prepare target variables dictionary
            target_columns = {}

            # Generate target variables for each prediction horizon
            for days in self.prediction_horizon_days:
                # Future return
                target_columns[f'future_return_{days}d'] = stock_df['close'].pct_change(days).shift(-days)
                
                # Future price
                target_columns[f'future_price_{days}d'] = stock_df['close'].shift(-days)
                
                # Target Long: 1 if future return is above upside threshold, 0 otherwise
                target_col_long = f'target_long_{days}d'
                target_long = pd.Series(0, index=stock_df.index)
                mask_long = (target_columns[f'future_return_{days}d'] >= self.upside_threshold_min)
                target_long.loc[mask_long] = 1
                target_columns[target_col_long] = target_long

                # Target Short: 1 if future return is below downside threshold, 0 otherwise
                target_col_short = f'target_short_{days}d'
                target_short = pd.Series(0, index=stock_df.index)
                mask_short = (target_columns[f'future_return_{days}d'] <= -self.downside_threshold_min)
                target_short.loc[mask_short] = 1
                target_columns[target_col_short] = target_short

                # Log class distribution for monitoring
                positive_long = target_long.sum()
                positive_short = target_short.sum()
                total_count = len(target_long)
                logger.info(f"Target {days}d - Long cases: {positive_long}/{total_count} ({positive_long/total_count*100:.2f}%)")
                logger.info(f"Target {days}d - Short cases: {positive_short}/{total_count} ({positive_short/total_count*100:.2f}%)")
                
                # Days to target (filled when backtesting)
                target_columns[f'days_to_long_target_{days}d'] = pd.Series(np.nan, index=stock_df.index)
                target_columns[f'days_to_short_target_{days}d'] = pd.Series(np.nan, index=stock_df.index)
                
                # Target prices
                target_columns[f'target_long_price_{days}d'] = stock_df['close'] * (1 + self.upside_threshold_min)
                target_columns[f'target_short_price_{days}d'] = stock_df['close'] * (1 - self.downside_threshold_min)
            
            # Add all feature columns to stock_df at once
            stock_df = pd.concat([stock_df, pd.DataFrame(new_columns, index=stock_df.index)], axis=1)
            
            # Add all target columns to stock_df
            stock_df = pd.concat([stock_df, pd.DataFrame(target_columns, index=stock_df.index)], axis=1)
            
            # 3. Add anchored VWAP features
            stock_df = self.calculate_anchored_vwap(stock_df)
            
            return stock_df, None

        # Process stocks in batches to reduce memory pressure
        batch_size = 50  # Adjust based on memory constraints
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            
            for symbol in batch_symbols:
                processed_df, insufficient_data = process_stock(symbol)
                if processed_df is not None:
                    all_stocks_with_features.append(processed_df)
                elif insufficient_data:
                    insufficient_data_stocks.append(insufficient_data)
                    
            # Force garbage collection after each batch
            gc.collect()
            logger.info(f"Processed batch {i//batch_size + 1} of {(len(symbols)-1)//batch_size + 1}")
        
        # Log information about stocks with insufficient data
        if insufficient_data_stocks:
            logger.warning(f"Skipped {len(insufficient_data_stocks)} stocks due to insufficient data:")
            for symbol, count in insufficient_data_stocks[:10]:  # Only log the first 10
                logger.warning(f"  {symbol}: {count} days of data")
            if len(insufficient_data_stocks) > 10:
                logger.warning(f"  ... and {len(insufficient_data_stocks)-10} more")
        
        # Combine all stocks
        if all_stocks_with_features:
            combined_df = pd.concat(all_stocks_with_features, ignore_index=True)
            
            # Get all feature columns (excluding original data columns and target columns)
            original_columns = {'id', 'date', 'trading_symbol', 'company_name', 'exchange', 
                            'security_id', 'open', 'high', 'low', 'close', 'volume', 'timestamp'}
            target_patterns = {'target_', 'future_', 'days_to_', 'target_price'}
            
            all_features = []
            for col in combined_df.columns:
                if col not in original_columns and not any(pattern in col for pattern in target_patterns):
                    all_features.append(col)
            
            # Save feature list
            feature_list_path = f'{self.results_dir}/feature_list.csv'
            feature_df = pd.DataFrame({'feature_name': all_features})
            feature_df.to_csv(feature_list_path, index=False)
            logger.info(f"Saved list of {len(all_features)} generated features to {feature_list_path}")
            
            logger.info(f"Feature engineering completed. DataFrame shape: {combined_df.shape}")
            logger.info(f"Processed {processed_stocks} stocks successfully")
            
            # Clear memory
            all_stocks_with_features = None
            gc.collect()
            
            return combined_df
        else:
            logger.error("No stocks with sufficient data to create features")
            return None
    
    def get_model_weights(self, horizon):
        """
        Get the optimal weights for each model based on horizon
        
        Args:
            horizon: Prediction horizon in days
            
        Returns:
            Dictionary of model weights
        """
        # Based on the evaluation results in model_evaluation_results.csv
        # Assign weights differently for each horizon
        
        if horizon == 1:
            # For 1-day, LightGBM has better AUC but PyTorch has better precision
            return {'lgbm': 0.6, 'pytorch': 0.4}
        elif horizon == 3:
            # For 3-day, both models are comparable
            return {'lgbm': 0.5, 'pytorch': 0.5}
        elif horizon == 5:
            # For 5-day, PyTorch has better F1 score
            return {'lgbm': 0.45, 'pytorch': 0.55}
        elif horizon == 7:
            # For 7-day, LightGBM has better AUC and Recall
            return {'lgbm': 0.65, 'pytorch': 0.35}
        elif horizon == 10:
            # For 10-day, LightGBM has significantly better performance
            return {'lgbm': 0.7, 'pytorch': 0.3}
        else:
            # Default equal weights
            return {'lgbm': 0.5, 'pytorch': 0.5}
    
    def find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        """
        Find the optimal probability threshold for a specific metric
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold value
        """
        # Define the metrics
        def precision_at_threshold(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return precision_score(y_true, y_pred, zero_division=0)
        
        def recall_at_threshold(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return recall_score(y_true, y_pred, zero_division=0)
        
        def f1_at_threshold(threshold):
            y_pred = (y_proba >= threshold).astype(int)
            return f1_score(y_true, y_pred, zero_division=0)
        
        # Select the appropriate optimization function
        if metric == 'precision':
            optimize_func = precision_at_threshold
        elif metric == 'recall':
            optimize_func = recall_at_threshold
        else:  # default to f1
            optimize_func = f1_at_threshold
        
        # Test a range of thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = [optimize_func(t) for t in thresholds]
        
        # Find the threshold with the highest score
        best_score_idx = np.argmax(scores)
        best_threshold = thresholds[best_score_idx]
        best_score = scores[best_score_idx]
        
        logger.info(f"Best {metric} score: {best_score:.4f} at threshold: {best_threshold:.2f}")
        return best_threshold
        
        
    def analyze_features(self, df, target_days=5, target_col=None, correlation_threshold=0.05, horizon_specific=True):
        """
        Analyze and select the most important features for prediction - optimized for different horizons
        
        Args:
            df: DataFrame with engineered features
            target_days: Target prediction horizon
            target_col: Target column name (for long/short differentiation)
            correlation_threshold: Minimum correlation to consider a feature important
            horizon_specific: Whether to use horizon-specific feature selection strategies
                
        Returns:
            List of selected feature names
        """
        # Determine direction from target column
        direction = 'long'
        if target_col is not None and 'short' in target_col:
            direction = 'short'
        elif target_col is None:
            target_col = f'target_{direction}_{target_days}d'
        
        logger.info(f"Analyzing features for {direction} {target_days}-day prediction horizon")
        
        # Sample data to reduce memory usage if dataset is large
        if len(df) > 100000:
            sample_size = min(100000, int(len(df) * 0.3))
            logger.info(f"Sampling {sample_size} rows for feature analysis")
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
        
        # Drop rows with NaN in target column
        df_clean = df_sample.dropna(subset=[target_col]).copy()
        
        # Get all numeric feature columns
        feature_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclude target columns and non-feature columns
        exclude_patterns = [
            'target_', 'future_', 'days_to_', 'target_price', 'id', 'security_id', 'timestamp'
        ]
        for pattern in exclude_patterns:
            feature_cols = [col for col in feature_cols if pattern not in col]
        
        # Early filtering: Drop features with >50% NaN values
        nan_ratio = df_clean[feature_cols].isna().mean()
        valid_features = nan_ratio[nan_ratio <= 0.5].index.tolist()
        logger.info(f"Removed {len(feature_cols) - len(valid_features)} features with >50% NaN values")
        feature_cols = valid_features
        
        # Calculate correlation with target (in batches if many features)
        batch_size = 100  # Adjust based on memory constraints
        correlations = {}
        
        for i in range(0, len(feature_cols), batch_size):
            batch_features = feature_cols[i:i+batch_size]
            batch_corr = df_clean[batch_features + [target_col]].corr()[target_col]
            for feat, corr in batch_corr.items():
                if feat != target_col:
                    correlations[feat] = corr
        
        # Convert to Series and sort
        correlations = pd.Series(correlations).sort_values(ascending=False)
        
        # Save all feature correlations to CSV
        correlation_df = pd.DataFrame({'feature': correlations.index, 'correlation': correlations.values})
        correlation_csv_path = f'{self.results_dir}/feature_correlations_{direction}_{target_days}d.csv'
        correlation_df.to_csv(correlation_csv_path, index=False)
        logger.info(f"Saved feature correlations to {correlation_csv_path}")
        
        # Horizon-specific feature selection strategies
        if horizon_specific:
            if target_days <= 3:  # Short-term predictions (1-3 days)
                # For short-term, prioritize momentum and recent volatility features
                priority_patterns = [
                    'return_1d', 'return_3d', 'volatility_3d', 'volatility_5d', 
                    'rsi', 'stoch', 'parkinson_vol_5d', 'gk_vol_5d',
                    'price_range_pct', 'body_size_pct', 'ema_5', 'close_position',
                    'bullish_candle', 'bearish_candle', 'volume_ratio'
                ]
                
                # Find features matching priority patterns
                priority_features = []
                for pattern in priority_patterns:
                    matched_features = [f for f in feature_cols if pattern in f]
                    priority_features.extend(matched_features)
                
                # Filter correlations to keep only priority features
                prioritized_corr = correlations[correlations.index.isin(priority_features)]
                
                # If we have enough priority features with good correlation, use them
                strong_priority_corr = prioritized_corr[abs(prioritized_corr) > correlation_threshold]
                if len(strong_priority_corr) >= 15:
                    filtered_corr = strong_priority_corr
                    logger.info(f"Using {len(filtered_corr)} short-term priority features")
                else:
                    # Otherwise fall back to standard approach but with lower threshold
                    filtered_corr = correlations[abs(correlations) > correlation_threshold/2]
                    logger.info(f"Not enough strong short-term features, using {len(filtered_corr)} general features")
                    
            elif target_days <= 5:  # Medium-term predictions (5 days)
                # For medium-term, balance between momentum and trend features
                priority_patterns = [
                    'return_5d', 'volatility_5d', 'volatility_10d', 
                    'parkinson_vol_10d', 'gk_vol_10d', 'price_range_pct',
                    'ema_10', 'max_high_10d', 'price_position', 'bullish_engulfing', 
                    'bearish_engulfing', 'adx', 'macd'
                ]
                
                # Find features matching priority patterns
                priority_features = []
                for pattern in priority_patterns:
                    matched_features = [f for f in feature_cols if pattern in f]
                    priority_features.extend(matched_features)
                
                # Filter correlations to keep only priority features
                prioritized_corr = correlations[correlations.index.isin(priority_features)]
                
                # If we have enough priority features with good correlation, use them
                strong_priority_corr = prioritized_corr[abs(prioritized_corr) > correlation_threshold]
                if len(strong_priority_corr) >= 15:
                    filtered_corr = strong_priority_corr
                    logger.info(f"Using {len(filtered_corr)} medium-term priority features")
                else:
                    # Otherwise fall back to standard approach
                    filtered_corr = correlations[abs(correlations) > correlation_threshold]
                    logger.info(f"Not enough strong medium-term features, using {len(filtered_corr)} general features")
                    
            else:  # Long-term predictions (7-10 days)
                # For long-term, prioritize trend, earnings and relative performance features
                priority_patterns = [
                    'return_10d', 'return_15d', 'volatility_15d', 'volatility_20d',
                    'parkinson_vol_20d', 'gk_vol_20d', 'max_high_15d', 'ema_20',
                    'earnings_q', 'rel_return', 'rel_strength', 'price_position',
                    'vol_regime'
                ]
                
                # Find features matching priority patterns
                priority_features = []
                for pattern in priority_patterns:
                    matched_features = [f for f in feature_cols if pattern in f]
                    priority_features.extend(matched_features)
                
                # Filter correlations to keep only priority features
                prioritized_corr = correlations[correlations.index.isin(priority_features)]
                
                # If we have enough priority features with good correlation, use them
                strong_priority_corr = prioritized_corr[abs(prioritized_corr) > correlation_threshold]
                if len(strong_priority_corr) >= 15:
                    filtered_corr = strong_priority_corr
                    logger.info(f"Using {len(filtered_corr)} long-term priority features")
                else:
                    # Otherwise fall back to standard approach
                    filtered_corr = correlations[abs(correlations) > correlation_threshold]
                    logger.info(f"Not enough strong long-term features, using {len(filtered_corr)} general features")
        else:
            # Standard approach without horizon-specific optimization
            initial_threshold = correlation_threshold
            filtered_corr = correlations[abs(correlations) > initial_threshold]
            
            # If too few features meet the threshold, gradually reduce it
            while len(filtered_corr) < 10 and initial_threshold > 0.01:
                initial_threshold -= 0.01
                filtered_corr = correlations[abs(correlations) > initial_threshold]
                logger.info(f"Reducing correlation threshold to {initial_threshold:.2f}")
        
        # If too many features meet the threshold, take top ones
        max_features = 50  # Adjust based on your needs
        if len(filtered_corr) > max_features:
            logger.info(f"Too many features ({len(filtered_corr)}), taking top {max_features}")
            filtered_corr = correlations.abs().nlargest(max_features)
            filtered_corr = correlations[filtered_corr.index]
        
        strong_corr_features = filtered_corr.index.tolist()
        
        if not strong_corr_features:
            logger.warning(f"No features with correlation > {correlation_threshold} found. Using top 10 features instead.")
            # If still no features, take top 10 by absolute correlation
            strong_corr_features = correlations.abs().nlargest(10).index.tolist()
        else:
            logger.info(f"Found {len(strong_corr_features)} features with correlation > {correlation_threshold}")
        
        # Check for multicollinearity among strong features
        if len(strong_corr_features) > 1:
            # Calculate correlation matrix for selected features only
            corr_matrix = df_clean[strong_corr_features].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Identify pairs with high correlation (>0.95)
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            
            # Remove one from each highly correlated pair
            final_features = [f for f in strong_corr_features if f not in to_drop]
            logger.info(f"Removed {len(to_drop)} features due to high multicollinearity")
        else:
            final_features = strong_corr_features
        
        # Apply SelectKBest for univariate feature selection
        if len(final_features) > 10:
            X = df_clean[final_features].fillna(df_clean[final_features].mean())
            y = df_clean[target_col]
            
            k = min(30, len(final_features))
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            
            # Get selected feature names and their scores
            mask = selector.get_support()
            scores = selector.scores_
            selected_features_scores = [(final_features[i], scores[i]) for i in range(len(final_features)) if mask[i]]
            selected_features_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Save SelectKBest results to CSV
            selectk_df = pd.DataFrame(selected_features_scores, columns=['feature', 'selectk_score'])
            selectk_csv_path = f'{self.results_dir}/feature_selectk_{target_days}d.csv'
            selectk_df.to_csv(selectk_csv_path, index=False)
            logger.info(f"Saved SelectKBest feature scores to {selectk_csv_path}")
            
            final_features = [f[0] for f in selected_features_scores]
            
            logger.info(f"Selected top {k} features using SelectKBest")
        
        # Store selected features with direction info
        feature_key = f'{direction}_{target_days}d'
        self.selected_features[feature_key] = final_features
        
        # Save final selected features to CSV
        selected_df = pd.DataFrame({'feature': final_features})
        selected_csv_path = f'{self.results_dir}/selected_features_{direction}_{target_days}d.csv'
        selected_df.to_csv(selected_csv_path, index=False)
        logger.info(f"Saved final selected features to {selected_csv_path}")
        
        # Generate and save feature importance plot
        if len(filtered_corr) > 0:
            plt.figure(figsize=(12, 8))
            filtered_corr.sort_values().plot(kind='barh')
            plt.title(f'Feature Correlations with {direction} {target_days}-day Target')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/feature_importance_{direction}_{target_days}d.png')
            plt.close()
        
        return final_features
            
    def prepare_data_for_training(self, df, target_days=5, target_col=None, test_size=0.2, val_size=0.2):
        """
        Prepare data for model training including normalization
        
        Args:
            df: DataFrame with features
            target_days: Target prediction horizon
            target_col: Target column name (for long/short differentiation)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary with prepared data
        """
        logger.info(f"Preparing data for {target_days}-day prediction")
        
        # If target_col not specified, use default naming
        if target_col is None:
            target_col = f'target_{target_days}d'
        
        # Determine direction from target column
        direction = 'long'
        if 'short' in target_col:
            direction = 'short'
        
        # Get features for this direction and horizon
        feature_key = f'{direction}_{target_days}d'
        features = self.selected_features.get(feature_key, [])
        
        if not features:
            logger.error(f"No features selected for {direction} {target_days}-day horizon")
            return None
            
        # Drop rows with NaN
        df_clean = df.dropna(subset=features + [target_col]).copy()
        
        # Create time-based split to avoid data leakage
        # Sort by date for proper time-based splitting
        df_clean = df_clean.sort_values('date')
        
        # Define cutoff dates for train/val/test
        dates = df_clean['date'].unique()
        dates = np.sort(dates)
        
        test_cutoff = int(len(dates) * (1 - test_size))
        val_cutoff = int(test_cutoff * (1 - val_size))
        
        train_dates = dates[:val_cutoff]
        val_dates = dates[val_cutoff:test_cutoff]
        test_dates = dates[test_cutoff:]
        
        # Split data
        train_df = df_clean[df_clean['date'].isin(train_dates)]
        val_df = df_clean[df_clean['date'].isin(val_dates)]
        test_df = df_clean[df_clean['date'].isin(test_dates)]
        
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        
        # Extract features and targets
        X_train = train_df[features]
        y_train = train_df[target_col]
        
        X_val = val_df[features]
        y_val = val_df[target_col]
        
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler with direction info
        scaler_path = f'{self.models_dir}/scaler_{direction}_{target_days}d.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler for {direction} {target_days}-day horizon to {scaler_path}")
        
        # Convert to numpy arrays
        data = {
            'X_train': X_train_scaled,
            'y_train': y_train.values,
            'X_val': X_val_scaled,
            'y_val': y_val.values,
            'X_test': X_test_scaled,
            'y_test': y_test.values,
            'feature_names': features,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'scaler': scaler,
            'direction': direction
        }
    
        return data
    
    def get_default_params(self, total_samples, positive_samples, n_features):
        """Generate conservative default parameters based on data characteristics"""
        class_ratio = positive_samples / total_samples
        
        # Safe min_child_samples based on minority class
        min_samples = max(5, int(positive_samples * 0.05))
        
        # Safe max_depth and num_leaves based on data size
        if total_samples < 1000:
            max_depth = 3
            num_leaves = 7  # 2^3 - 1
        elif total_samples < 10000:
            max_depth = 5
            num_leaves = 15  # Fewer than 2^5 - 1 to be conservative
        else:
            max_depth = 6
            num_leaves = 31  # 2^6 - 1
        
        # Safe regularization based on number of features
        if n_features < 10:
            reg_alpha = 0.01
            reg_lambda = 0.01
        elif n_features < 50:
            reg_alpha = 0.05
            reg_lambda = 0.05
        else:
            reg_alpha = 0.1
            reg_lambda = 0.1
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'device': 'cpu',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 100,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_samples,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
        }
        
        # Handle class imbalance
        if class_ratio < 0.05 or class_ratio > 0.95:
            params['is_unbalance'] = True
        elif class_ratio < 0.2 or class_ratio > 0.8:
            if class_ratio < 0.5:
                params['scale_pos_weight'] = (1 - class_ratio) / class_ratio
            else:
                params['scale_pos_weight'] = class_ratio / (1 - class_ratio)
        
        return params
    
    def optimize_lightgbm(self, data, target_days, direction="long", n_trials=30):
        """
        Optimize LightGBM hyperparameters using Optuna with data-aware constraints
        """
        logger.info(f"Optimizing LightGBM for {direction} {target_days}-day prediction with {n_trials} trials")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Analyze class balance
        positive_samples = np.sum(y_train == 1)
        total_samples = len(y_train)
        class_ratio = positive_samples / total_samples
        
        logger.info(f"Class distribution - Positive: {positive_samples}/{total_samples} ({class_ratio:.2%})")
        
        # Analyze feature characteristics
        n_features = X_train.shape[1]
        
        # Calculate parameter ranges based on data characteristics
        # Ensure min_child_samples has a reasonable range - never larger than 5% of data
        min_samples_min = max(5, min(20, int(positive_samples * 0.01)))
        min_samples_max = max(min_samples_min + 5, min(50, int(total_samples * 0.05)))
        
        # Ensure num_leaves has a reasonable range
        max_num_leaves = min(127, total_samples // 20)  # At least 20 samples per leaf on average
        
        def objective(trial):
            # Calculate safe max_depth based on data size
            # This ensures we don't create too many nodes for small datasets
            safe_max_depth = min(8, max(3, int(np.log2(total_samples / min_samples_min))))
            
            # Calculate safe num_leaves range based on max_depth
            current_max_depth = trial.suggest_int('max_depth', 3, safe_max_depth)
            min_leaves = min(10, 2**current_max_depth - 1)
            safe_max_leaves = min(2**current_max_depth - 1, max_num_leaves)
            
            # Ensure safe_max_leaves is greater than min_leaves
            if safe_max_leaves <= min_leaves:
                safe_max_leaves = min_leaves + 1
            
            # Define parameters with constraints that respect data characteristics
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'device': 'cpu',  # Explicitly set to CPU to avoid potential GPU issues
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': current_max_depth,
                'num_leaves': trial.suggest_int('num_leaves', min_leaves, safe_max_leaves),
                'min_child_samples': trial.suggest_int('min_child_samples', min_samples_min, min_samples_max),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            }
            
            # Handle class imbalance differently based on severity
            if class_ratio < 0.05 or class_ratio > 0.95:
                # Severe imbalance: use is_unbalance
                param['is_unbalance'] = True
            elif class_ratio < 0.2 or class_ratio > 0.8:
                # Moderate imbalance: use scale_pos_weight
                if class_ratio < 0.5:
                    param['scale_pos_weight'] = (1 - class_ratio) / class_ratio
                else:
                    param['scale_pos_weight'] = class_ratio / (1 - class_ratio)
            
            try:
                gbm = lgb.LGBMClassifier(**param)
                
                gbm.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(20, verbose=False)]
                )
                
                y_pred_proba = gbm.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                return auc_score
                
            except Exception as e:
                logger.warning(f"Trial failed with parameters: {param}")
                logger.warning(f"Error message: {str(e)}")
                
                # Instead of just returning a low value, we can help Optuna learn from failures
                # Check which parameters might be causing issues
                if 'num_leaves' in str(e) or 'min_child_samples' in str(e) or 'left_count' in str(e):
                    # Error related to tree structure - likely from aggressive parameter values
                    return 0.1
                
                # Generic failure - give a very low score
                return 0.0
        
        # Use TPESampler with multivariate=True for better parameter exploration
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        try:
            study.optimize(objective, n_trials=n_trials)
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            # Return conservative default parameters based on data characteristics
            return self.get_default_params(total_samples, positive_samples, n_features)
        
        # Check if we got any successful trials
        if len(study.trials) == 0 or study.best_value <= 0.1:
            logger.warning("No successful trials. Using data-aware default parameters.")
            return self.get_default_params(total_samples, positive_samples, n_features)
        
        logger.info(f"Best trial: score {study.best_value}, params {study.best_params}")
        
        # Build complete parameter dict with additional fixed parameters
        best_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'device': 'cpu',
        }
        
        # Handle class imbalance in the best parameters
        if class_ratio < 0.05 or class_ratio > 0.95:
            best_params['is_unbalance'] = True
        elif class_ratio < 0.2 or class_ratio > 0.8:
            if class_ratio < 0.5:
                best_params['scale_pos_weight'] = (1 - class_ratio) / class_ratio
            else:
                best_params['scale_pos_weight'] = class_ratio / (1 - class_ratio)
        
        best_params.update(study.best_params)
        
        return best_params
    

        
    def train_lightgbm_model(self, data, target_days=5, params=None):
        """
        Train LightGBM model for stock prediction
        
        Args:
            data: Dictionary with prepared data
            target_days: Target prediction horizon
            params: Optional hyperparameters
            
        Returns:
            Trained model
        """
        direction = data.get('direction', 'long')
        logger.info(f"Training LightGBM model for {direction} {target_days}-day prediction")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'device': 'gpu',  # Add this line to use GPU
                'learning_rate': 0.05,
                'n_estimators': 200,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'class_weight': 'balanced'
            }
        
        # Option 2: Calculate specific weights
        # Count samples in each class
        n_samples = len(y_train)
        n_pos = np.sum(y_train == 1)
        n_neg = n_samples - n_pos
        
        # Calculate weight for each class (inverse frequency)
        weight_dict = {
            0: n_samples / (2 * n_neg),
            1: n_samples / (2 * n_pos)
        }
        
        # Create sample weights array
        sample_weight = np.array([weight_dict[y] for y in y_train])
        
        model = lgb.LGBMClassifier(**params)
        
        # Add sample_weight to the fit method
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,  # Add this line for manual weighting
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50, verbose=100)]
        )
        
        # Save feature importance
        feature_importance = model.feature_importances_
        self.feature_importance[target_days] = dict(zip(data['feature_names'], feature_importance))
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_csv_path = f'{self.results_dir}/lgbm_feature_importance_{target_days}d.csv'
        importance_df.to_csv(importance_csv_path, index=False)
        logger.info(f"Saved LightGBM feature importance to {importance_csv_path}")
        
        # Generate feature importance plot
        lgb.plot_importance(model, max_num_features=20, figsize=(10, 8))
        plt.title(f'LightGBM Feature Importance for {target_days}-day Prediction')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/lgbm_feature_importance_{target_days}d.png')
        plt.close()
        
        # Example of updated path:
        model_path = f'{self.models_dir}/lgbm_model_{direction}_{target_days}d.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
  
    def apply_smote(self, data, target_days):
        """
        Apply SMOTE to handle class imbalance with safeguards for sparse data
        
        Args:
            data: Dictionary with prepared data
            target_days: Target prediction horizon
            
        Returns:
            Updated data dictionary with resampled training data
        """
        from imblearn.over_sampling import SMOTE, SMOTENC
        
        logger.info(f"Applying SMOTE for {target_days}-day prediction")
        
        X_train, y_train = data['X_train'], data['y_train']
        
        # Check if minority class has enough samples
        positive_samples = np.sum(y_train == 1)
        
        if positive_samples >= 5:  # SMOTE needs at least 5 samples of minority class
            try:
                # Analyze data sparsity
                sparse_features = []
                for i in range(X_train.shape[1]):
                    # Check if feature has very few unique values
                    unique_values = np.unique(X_train[:, i])
                    if len(unique_values) < 5:
                        sparse_features.append(i)
                
                # If we have sparse features, use SMOTENC which is better for categorical/sparse features
                if len(sparse_features) > 0:
                    logger.info(f"Using SMOTENC for {len(sparse_features)} sparse features")
                    smote = SMOTENC(categorical_features=sparse_features, random_state=42, k_neighbors=min(positive_samples-1, 5))
                else:
                    # Regular SMOTE with reduced k_neighbors for small datasets
                    k_neighbors = min(positive_samples-1, 5)  # k must be <= n_minority_samples - 1
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                
                # Apply SMOTE with sampling_strategy < 1.0 to avoid excessive oversampling
                # This creates a more realistic but still improved class balance
                sampling_strategy = min(0.5, (len(y_train) - positive_samples) / positive_samples)
                smote.sampling_strategy = sampling_strategy
                
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                # Log resampling results
                original_shape = X_train.shape
                new_shape = X_train_resampled.shape
                original_pos = np.sum(y_train == 1)
                new_pos = np.sum(y_train_resampled == 1)
                
                logger.info(f"SMOTE resampling: {original_shape[0]} -> {new_shape[0]} samples")
                logger.info(f"Positive class: {original_pos} -> {new_pos} samples")
                
                # Update data dictionary
                data['X_train'] = X_train_resampled
                data['y_train'] = y_train_resampled
                
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Using original imbalanced data.")
        else:
            logger.warning(f"Not enough minority samples for SMOTE. Using original data.")
        
        return data
    
    def train_pytorch_model(self, data, target_days=5, enhanced=False):
        """
        Train PyTorch neural network for stock prediction utilizing GPU
        
        Args:
            data: Dictionary with prepared data
            target_days: Target prediction horizon
            enhanced: Whether to use enhanced architecture and training
            
        Returns:
            Trained PyTorch model
        """
        direction = data.get('direction', 'long')
        logger.info(f"Training PyTorch model for {direction} {target_days}-day prediction")
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        # Convert numpy arrays to PyTorch tensors - keep on CPU initially
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Optimize batch size for RTX 4060Ti (16GB)
        batch_size = 512  # Increased for RTX 4060Ti
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        
        # Calculate class weights for imbalanced data
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        pos_weight_tensor = torch.tensor([float(pos_weight)]).to(self.device)
        
        # Define model architecture optimized for your hardware
        input_size = X_train.shape[1]
        
        if enhanced:
            # Enhanced model for primary horizon
            hidden_size1 = 256
            hidden_size2 = 128
            hidden_size3 = 64
            dropout_rate1 = 0.4
            dropout_rate2 = 0.3
            dropout_rate3 = 0.2
            learning_rate = 0.0005
            num_epochs = 150
            patience = 15
        else:
            # Standard model for other horizons
            hidden_size1 = 128
            hidden_size2 = 64
            hidden_size3 = 32
            dropout_rate1 = 0.3
            dropout_rate2 = 0.2
            dropout_rate3 = 0.2
            learning_rate = 0.001
            num_epochs = 100
            patience = 10
        
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate3),
            nn.Linear(hidden_size3, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss(weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler for enhanced training
        if enhanced:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        
        # Training loop
        best_val_auc = 0.0
        best_model = None
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
        
        # For computing AUC
        from sklearn.metrics import roc_auc_score
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            y_true_train = []
            y_pred_train = []
            
            for inputs, labels in train_loader:
                # Move data to the same device as model
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                y_true_train.extend(labels.cpu().detach().numpy())
                y_pred_train.extend(outputs.cpu().detach().numpy())
            
            train_loss /= len(train_loader)
            train_auc = roc_auc_score(y_true_train, y_pred_train)
            
            # Validation
            model.eval()
            val_loss = 0.0
            y_true_val = []
            y_pred_val = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move data to the same device as model
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    y_true_val.extend(labels.cpu().numpy())
                    y_pred_val.extend(outputs.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(y_true_val, y_pred_val)
            
            # Update learning rate scheduler if using enhanced training
            if enhanced:
                scheduler.step(val_auc)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                        f'Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        model.load_state_dict(best_model)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_auc'], label='Train AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/pytorch_training_history_{target_days}d.png')
        plt.close()
        
        # Save model
        model_path = f'{self.models_dir}/pytorch_model_{target_days}d.pt'
        torch.save(model.state_dict(), model_path)
        logger.info(f"PyTorch model saved to {model_path}")
        
        # Example of updated path:
        model_path = f'{self.models_dir}/pytorch_model_{direction}_{target_days}d.pt'
        torch.save(model.state_dict(), model_path)
        logger.info(f"PyTorch model saved to {model_path}")
        
        return model
    def train_all_models(self):
        """
        Train models for all prediction horizons and both long/short directions
        """
        # Fetch and preprocess data
        df = self.fetch_historical_data()
        if df is None:
            logger.error("Unable to fetch data. Exiting.")
            return
        
        # Engineer features
        df_with_features = self.engineer_features(df)
        if df_with_features is None:
            logger.error("Feature engineering failed. Exiting.")
            return
        
        # Train models for each horizon and direction
        for days in self.prediction_horizon_days:
            is_primary = days == self.primary_horizon
            
            # Train long models
            logger.info(f"Starting training pipeline for LONG {days}-day prediction horizon" + 
                    (" (PRIMARY)" if is_primary else ""))
            self.train_direction_models(df_with_features, days, direction="long", is_primary=is_primary)
            
            # Train short models
            logger.info(f"Starting training pipeline for SHORT {days}-day prediction horizon" + 
                    (" (PRIMARY)" if is_primary else ""))
            self.train_direction_models(df_with_features, days, direction="short", is_primary=is_primary)
            
        logger.info("All models trained successfully")

    def train_direction_models(self, df_with_features, days, direction="long", is_primary=False):
        """Helper method to train models for a specific direction (long or short)"""
        target_col = f'target_{direction}_{days}d'
        
        # Analyze and select features
        correlation_threshold = 0.04 if is_primary else 0.05
        selected_features = self.analyze_features(df_with_features, target_days=days, 
                                            target_col=target_col,
                                            correlation_threshold=correlation_threshold)
        
        # Store features with direction
        self.selected_features[f'{direction}_{days}d'] = selected_features
        
        # Prepare data
        data = self.prepare_data_for_training(df_with_features, target_days=days, target_col=target_col)
        
        if data is None:
            logger.error(f"Failed to prepare data for {direction} {days}d model. Skipping.")
            return
        
        # Check class balance and apply appropriate sampling strategy
        positive_samples = np.sum(data['y_train'] == 1)
        total_samples = len(data['y_train'])
        class_ratio = positive_samples / total_samples
        
        logger.info(f"{direction} {days}d - Class balance: {positive_samples}/{total_samples} ({class_ratio:.2%})")
        
        # Only proceed if we have enough minority class samples
        # For binary classification, we need at least some positive examples
        if positive_samples < 5:
            logger.warning(f"Too few positive samples ({positive_samples}) for {direction} {days}d model. Skipping.")
            return
        
        # For severe imbalance, apply SMOTE if we have enough positive samples
        if class_ratio < 0.1 and positive_samples >= 5:
            data = self.apply_smote(data, target_days=days)
            # Recalculate class distribution after SMOTE
            new_positive = np.sum(data['y_train'] == 1)
            new_total = len(data['y_train'])
            logger.info(f"After SMOTE: {new_positive}/{new_total} ({new_positive/new_total:.2%})")
        
        try:
            # Optimize hyperparameters with data-aware parameter constraints
            n_trials = 50 if is_primary else 30
            best_params = self.optimize_lightgbm(data, target_days=days, direction=direction, n_trials=n_trials)
            
            # Train LightGBM model with optimized parameters
            lgbm_model = self.train_lightgbm_model(data, target_days=days, params=best_params)
            if lgbm_model is not None:
                self.models[f'lgbm_{direction}_{days}d'] = lgbm_model
            
            # Train PyTorch model
            pytorch_model = self.train_pytorch_model(data, target_days=days, enhanced=is_primary)
            if pytorch_model is not None:
                self.models[f'pytorch_{direction}_{days}d'] = pytorch_model
            
            # Save scaler
            self.scalers[f'{direction}_{days}d'] = data['scaler']
            
        except Exception as e:
            logger.error(f"Error training models for {direction} {days}d: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def create_feature_importance_report(self):
        """
        Create a consolidated report of feature importance across all horizons and directions
        """
        logger.info("Creating feature importance report")
        
        # Check if feature importance data is available
        if not self.feature_importance:
            logger.warning("No feature importance data available")
            return
        
        # Consolidate feature importance across all horizons and directions
        all_features = set()
        for key, importances in self.feature_importance.items():
            all_features.update(importances.keys())
        
        # Create a DataFrame with all features and their importance across horizons
        consolidated_data = []
        
        for feature in all_features:
            feature_data = {'feature': feature}
            
            # Group by direction
            long_importances = []
            short_importances = []
            
            # Add importance for each horizon and direction
            for key, importances in self.feature_importance.items():
                if 'long' in key:
                    if feature in importances:
                        horizon = int(key.split('_')[-1].replace('d', ''))
                        feature_data[f'long_{horizon}d'] = importances[feature]
                        long_importances.append(importances[feature])
                elif 'short' in key:
                    if feature in importances:
                        horizon = int(key.split('_')[-1].replace('d', ''))
                        feature_data[f'short_{horizon}d'] = importances[feature]
                        short_importances.append(importances[feature])
            
            # Calculate average importance by direction
            if long_importances:
                feature_data['avg_long_importance'] = sum(long_importances) / len(long_importances)
            else:
                feature_data['avg_long_importance'] = 0
                
            if short_importances:
                feature_data['avg_short_importance'] = sum(short_importances) / len(short_importances)
            else:
                feature_data['avg_short_importance'] = 0
                
            # Overall average
            all_importances = long_importances + short_importances
            if all_importances:
                feature_data['avg_overall_importance'] = sum(all_importances) / len(all_importances)
            else:
                feature_data['avg_overall_importance'] = 0
                
            consolidated_data.append(feature_data)
        
        # Create DataFrame and sort by average importance
        consolidated_df = pd.DataFrame(consolidated_data)
        consolidated_df = consolidated_df.sort_values('avg_overall_importance', ascending=False)
        
        # Save to CSV
        report_path = f'{self.results_dir}/feature_importance_report.csv'
        consolidated_df.to_csv(report_path, index=False)
        logger.info(f"Feature importance report saved to {report_path}")
        
        # Create visualizations for long vs short feature importance
        plt.figure(figsize=(14, 10))
        
        # Top 20 features
        top_features = consolidated_df.head(20)
        
        # Create bar chart comparing long vs short importance
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(top_features))
        width = 0.35
        
        plt.bar(x - width/2, top_features['avg_long_importance'], width, label='Long')
        plt.bar(x + width/2, top_features['avg_short_importance'], width, label='Short')
        
        plt.xlabel('Features')
        plt.ylabel('Average Importance')
        plt.title('Feature Importance: Long vs Short')
        plt.xticks(x, top_features['feature'], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{self.results_dir}/long_vs_short_feature_importance.png')
        plt.close()
        
        return consolidated_df
    
    def evaluate_models(self):
        """
        Evaluate all trained models for both long and short predictions
        
        Returns:
            DataFrame with evaluation metrics
        """
        results = []
        
        for days in self.prediction_horizon_days:
            for direction in ['long', 'short']:
                logger.info(f"Evaluating {direction} models for {days}-day prediction horizon")
                
                # Load data
                df = self.fetch_historical_data()
                df_with_features = self.engineer_features(df)
                
                # Use direction-specific target
                target_col = f'target_{direction}_{days}d'
                data = self.prepare_data_for_training(df_with_features, target_days=days, target_col=target_col)
                
                if data is None:
                    logger.error(f"Data preparation failed for {direction} {days}-day prediction")
                    continue
                    
            X_test, y_test = data['X_test'], data['y_test']
            test_df = data['test_df']
            
            # Evaluate LightGBM model
            lgbm_model = self.models.get(f'lgbm_{direction}_{days}d')
            if lgbm_model is None:
                # Try to load from file
                try:
                    model_path = f'{self.models_dir}/lgbm_model_{direction}_{days}d.pkl'
                    lgbm_model = joblib.load(model_path)
                    self.models[f'lgbm_{direction}_{days}d'] = lgbm_model
                except:
                    logger.error(f"Could not load LightGBM model for {direction} {days}-day prediction")
                    continue
            
            # Make predictions
            y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
            y_pred_lgbm = (y_pred_proba_lgbm >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
            precision_lgbm = precision_score(y_test, y_pred_lgbm, zero_division=0)
            recall_lgbm = recall_score(y_test, y_pred_lgbm, zero_division=0)
            f1_lgbm = f1_score(y_test, y_pred_lgbm, zero_division=0)
            auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)

            # Find optimal threshold for F1 score
            optimal_threshold_lgbm = self.find_optimal_threshold(y_test, y_pred_proba_lgbm, metric='f1')
            # Store the optimal threshold
            if not hasattr(self, 'optimal_thresholds'):
                self.optimal_thresholds = {}
            self.optimal_thresholds[f'lgbm_{days}d'] = optimal_threshold_lgbm

            # Recalculate metrics with optimal threshold
            y_pred_lgbm_optimal = (y_pred_proba_lgbm >= optimal_threshold_lgbm).astype(int)
            accuracy_lgbm_optimal = accuracy_score(y_test, y_pred_lgbm_optimal)
            precision_lgbm_optimal = precision_score(y_test, y_pred_lgbm_optimal, zero_division=0)
            recall_lgbm_optimal = recall_score(y_test, y_pred_lgbm_optimal, zero_division=0)
            f1_lgbm_optimal = f1_score(y_test, y_pred_lgbm_optimal, zero_division=0)

            # Add results with optimal threshold
            results.append({
                'Model': 'LightGBM-F1Optimized',
                'Horizon (days)': days,
                'Threshold': optimal_threshold_lgbm,
                'Accuracy': accuracy_lgbm_optimal,
                'Precision': precision_lgbm_optimal,
                'Recall': recall_lgbm_optimal,
                'F1 Score': f1_lgbm_optimal,
                'AUC': auc_lgbm  # AUC remains the same
            })
            
            # Store results
            results.append({
                'Model': 'LightGBM',
                'Horizon (days)': days,
                'Accuracy': accuracy_lgbm,
                'Precision': precision_lgbm,
                'Recall': recall_lgbm,
                'F1 Score': f1_lgbm,
                'AUC': auc_lgbm
            })
            
            # Initialize lists for storing ROC curve data for plotting
            roc_curves = []
            
            # Add LightGBM ROC curve data
            fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_pred_proba_lgbm)
            roc_curves.append((fpr_lgbm, tpr_lgbm, f'LightGBM (AUC = {auc_lgbm:.3f})'))
            
            # Evaluate PyTorch model
            pytorch_model = self.models.get(f'pytorch_{direction}_{days}d')
            if pytorch_model is None:
                # Try to load from file
                try:
                    model_path = f'{self.models_dir}/pytorch_model_{days}d.pt'
                    input_size = X_test.shape[1]
                    model = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model = model.to(self.device)
                    pytorch_model = model
                    self.models[f'pytorch_{days}d'] = pytorch_model
                except Exception as e:
                    logger.warning(f"Could not load PyTorch model for {days}-day prediction: {e}")
                    continue
            
            # Make predictions
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            pytorch_model.eval()
            with torch.no_grad():
                y_pred_proba_pt = pytorch_model(X_test_tensor).squeeze().cpu().numpy()
            y_pred_pt = (y_pred_proba_pt >= 0.5).astype(int)
            
            # Calculate metrics
            accuracy_pt = accuracy_score(y_test, y_pred_pt)
            precision_pt = precision_score(y_test, y_pred_pt, zero_division=0)
            recall_pt = recall_score(y_test, y_pred_pt, zero_division=0)
            f1_pt = f1_score(y_test, y_pred_pt, zero_division=0)
            auc_pt = roc_auc_score(y_test, y_pred_proba_pt)
            
            # Store results
            results.append({
                'Model': 'PyTorch',
                'Horizon (days)': days,
                'Accuracy': accuracy_pt,
                'Precision': precision_pt,
                'Recall': recall_pt,
                'F1 Score': f1_pt,
                'AUC': auc_pt
            })

            # Find optimal threshold for F1 score for PyTorch model
            optimal_threshold_pt = self.find_optimal_threshold(y_test, y_pred_proba_pt, metric='f1')
            # Store the optimal threshold
            if not hasattr(self, 'optimal_thresholds'):
                self.optimal_thresholds = {}
            self.optimal_thresholds[f'pytorch_{days}d'] = optimal_threshold_pt

            # Recalculate metrics with optimal threshold
            y_pred_pt_optimal = (y_pred_proba_pt >= optimal_threshold_pt).astype(int)
            accuracy_pt_optimal = accuracy_score(y_test, y_pred_pt_optimal)
            precision_pt_optimal = precision_score(y_test, y_pred_pt_optimal, zero_division=0)
            recall_pt_optimal = recall_score(y_test, y_pred_pt_optimal, zero_division=0)
            f1_pt_optimal = f1_score(y_test, y_pred_pt_optimal, zero_division=0)

            # Add results with optimal threshold
            results.append({
                'Model': 'PyTorch-F1Optimized',
                'Horizon (days)': days,
                'Threshold': optimal_threshold_pt,
                'Accuracy': accuracy_pt_optimal,
                'Precision': precision_pt_optimal,
                'Recall': recall_pt_optimal,
                'F1 Score': f1_pt_optimal,
                'AUC': auc_pt  # AUC remains the same
            })
            
            # Add PyTorch ROC curve data
            fpr_pt, tpr_pt, _ = roc_curve(y_test, y_pred_proba_pt)
            roc_curves.append((fpr_pt, tpr_pt, f'PyTorch (AUC = {auc_pt:.3f})'))
            
            # Generate ROC curve plot
            plt.figure(figsize=(10, 8))
            
            # Plot all ROC curves
            for fpr, tpr, label in roc_curves:
                plt.plot(fpr, tpr, label=label)
            
            # Reference line
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {days}-day Prediction')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.results_dir}/roc_curve_{days}d.png')
            plt.close()
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
        
        # Create results DataFrame
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(f'{self.results_dir}/model_evaluation_results.csv', index=False)
        logger.info("Model evaluation completed")
        
        return results_df
    
    def ensure_features_selected(self, days, df):
        """Ensure features are selected for the given horizon"""
        if days not in self.selected_features or not self.selected_features[days]:
            logger.info(f"No features selected for {days}-day horizon. Running feature selection...")
            self.selected_features[days] = self.analyze_features(df, target_days=days)
        
        return self.selected_features[days]
    
    def ensure_model_loaded(self, model_type, days):
        """Ensure model is loaded, attempt to load from file if not in memory"""
        model_key = f"{model_type}_{days}d"
        
        if model_key in self.models and self.models[model_key] is not None:
            return self.models[model_key]
        
        # Try to load from file
        try:
            if model_type == 'lgbm':
                model_path = f'{self.models_dir}/lgbm_model_{days}d.pkl'
                model = joblib.load(model_path)
            elif model_type == 'pytorch':
                model_path = f'{self.models_dir}/pytorch_model_{days}d.pt'
                input_size = len(self.selected_features.get(days, []))
                if input_size == 0:
                    logger.warning(f"No features selected for {days}-day horizon")
                    return None
                    
                model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model = model.to(self.device)
            
            self.models[model_key] = model
            logger.info(f"Successfully loaded {model_type} model for {days}-day horizon")
            return model
        except Exception as e:
            logger.warning(f"Could not load {model_type} model for {days}-day horizon: {e}")
            return None
    
    def save_scalers(self):
        """Save all scalers to disk"""
        logger.info("Saving scalers...")
        for days, scaler in self.scalers.items():
            scaler_path = f'{self.models_dir}/scaler_{days}d.pkl'
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler for {days}-day horizon to {scaler_path}")
    
    def load_scalers(self):
        """Load all scalers from disk for both long and short predictions"""
        logger.info("Loading scalers...")
        loaded_scalers = 0
        
        for days in self.prediction_horizon_days:
            for direction in ['long', 'short']:
                scaler_path = f'{self.models_dir}/scaler_{direction}_{days}d.pkl'
                try:
                    scaler = joblib.load(scaler_path)
                    self.scalers[f'{direction}_{days}d'] = scaler
                    loaded_scalers += 1
                    logger.info(f"Loaded scaler for {direction} {days}-day horizon")
                except Exception as e:
                    logger.warning(f"Could not load scaler for {direction} {days}-day horizon: {e}")
        
        return loaded_scalers > 0
    
    def backtest(self, start_date=None, end_date=None, confidence_threshold=0.7, batch_size=64):
        """
        GPU-accelerated backtesting process for both long and short predictions
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            confidence_threshold: Confidence threshold for taking trades
            batch_size: Batch size for GPU processing
            
        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Starting GPU-accelerated backtesting from {start_date} to {end_date}")
        
        # Ensure scalers are loaded
        if not self.load_scalers():
            logger.error("Failed to load scalers. Cannot proceed with backtesting.")
            return None
        
        # Fetch data
        df = self.fetch_historical_data()
        if df is None:
            logger.error("Failed to fetch data for backtesting")
            return None
            
        # Engineer features
        df_with_features = self.engineer_features(df)
        if df_with_features is None:
            logger.error("Feature engineering failed for backtesting")
            return None
            
        # Filter data by date
        if start_date:
            start_date = pd.to_datetime(start_date)
            df_with_features = df_with_features[df_with_features['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df_with_features = df_with_features[df_with_features['date'] <= end_date]
        
        # Get all unique dates and sort
        dates = sorted(df_with_features['date'].unique())
        logger.info(f"Backtesting over {len(dates)} trading days")
        
        # Initialize results storage
        trades = []
        portfolio_values = []
        initial_capital = 100000  # Initial capital of ₹1 lakh
        available_capital = initial_capital
        open_positions = {}  # Dictionary to track open positions
        
        # Batch processing for faster prediction
        def process_prediction_batch(batch_data, direction, days):
            # Get model path keys
            lgbm_key = f'lgbm_{direction}_{days}d'
            pytorch_key = f'pytorch_{direction}_{days}d'
            scaler_key = f'{direction}_{days}d'
            
            # Ensure models and scaler exist
            if lgbm_key not in self.models or pytorch_key not in self.models or scaler_key not in self.scalers:
                return None
            
            # Get feature names
            feature_names = self.selected_features.get(f'{direction}_{days}d', [])
            if not feature_names:
                return None
            
            # Extract features
            features = batch_data[feature_names].values
            
            # Scale features
            scaler = self.scalers[scaler_key]
            features_scaled = scaler.transform(features)
            
            # LightGBM prediction
            lgbm_model = self.models[lgbm_key]
            lgbm_proba = lgbm_model.predict_proba(features_scaled)[:, 1]
            
            # PyTorch prediction (using GPU for batch processing)
            pytorch_model = self.models[pytorch_key]
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            pytorch_model.eval()
            with torch.no_grad():
                batch_nn_proba = pytorch_model(features_tensor).squeeze().cpu().numpy()
            
            # Get model weights for this horizon
            weights = self.get_model_weights(days)
            
            # Calculate ensemble probabilities
            ensemble_proba = lgbm_proba * weights['lgbm'] + batch_nn_proba * weights['pytorch']
            
            return ensemble_proba
        
        # Process each date
        for date_idx, current_date in enumerate(dates[:-max(self.prediction_horizon_days)]):
            logger.info(f"Backtesting day {date_idx+1}/{len(dates)}: {current_date}")
            
            # Get data for the current date
            current_data = df_with_features[df_with_features['date'] == current_date]
            
            # Skip if no data for current date
            if len(current_data) == 0:
                continue
                
            # Process batches of stocks for each horizon and direction
            for days in self.prediction_horizon_days:
                for direction in ['long', 'short']:
                    # Process in batches to utilize GPU effectively
                    for batch_start in range(0, len(current_data), batch_size):
                        batch_end = min(batch_start + batch_size, len(current_data))
                        batch = current_data.iloc[batch_start:batch_end]
                        
                        # Get predictions for this batch
                        confidence_scores = process_prediction_batch(batch, direction, days)
                        
                        if confidence_scores is None:
                            continue
                        
                        # Process each stock in the batch
                        for i, (idx, row) in enumerate(batch.iterrows()):
                            if confidence_scores[i] >= confidence_threshold:
                                symbol = row['trading_symbol']
                                current_price = row['close']
                                
                                # Determine target price based on direction
                                if direction == 'long':
                                    target_price = current_price * (1 + self.upside_threshold_min)
                                    stop_loss = current_price * (1 - 0.005)  # 0.5% stop loss
                                else:  # short
                                    target_price = current_price * (1 - self.downside_threshold_min)
                                    stop_loss = current_price * (1 + 0.005)  # 0.5% stop loss
                                
                                # Simulate trade
                                position_size = min(available_capital * 0.1, 10000)  # 10% of capital or max ₹10k
                                
                                # Add to open positions
                                position_id = f"{symbol}_{direction}_{current_date}_{days}d"
                                open_positions[position_id] = {
                                    'symbol': symbol,
                                    'entry_date': current_date,
                                    'direction': direction,
                                    'horizon': days,
                                    'entry_price': current_price,
                                    'target_price': target_price,
                                    'stop_loss': stop_loss,
                                    'position_size': position_size,
                                    'shares': position_size / current_price,
                                    'confidence': confidence_scores[i]
                                }
                                
                                # Reduce available capital
                                available_capital -= position_size
                                
                                # Log trade
                                logger.info(f"Opening {direction} position: {symbol} at {current_price} (Target: {target_price})")
            
            # Check for position exits (both target and stop-loss)
            positions_to_remove = []
            
            for position_id, position in open_positions.items():
                symbol = position['symbol']
                entry_date = position['entry_date']
                direction = position['direction']
                entry_price = position['entry_price']
                target_price = position['target_price']
                stop_loss = position['stop_loss']
                
                # Get data for this symbol since entry
                future_data = df_with_features[
                    (df_with_features['trading_symbol'] == symbol) & 
                    (df_with_features['date'] > entry_date) & 
                    (df_with_features['date'] <= current_date)
                ]
                
                if len(future_data) == 0:
                    continue
                    
                # Check if target or stop-loss was hit
                exit_triggered = False
                exit_date = None
                exit_price = None
                exit_type = None
                
                for _, row in future_data.iterrows():
                    if direction == 'long':
                        # For long positions
                        if row['high'] >= target_price:
                            exit_triggered = True
                            exit_date = row['date']
                            exit_price = target_price
                            exit_type = 'target'
                            break
                        elif row['low'] <= stop_loss:
                            exit_triggered = True
                            exit_date = row['date']
                            exit_price = stop_loss
                            exit_type = 'stop_loss'
                            break
                    else:
                        # For short positions
                        if row['low'] <= target_price:
                            exit_triggered = True
                            exit_date = row['date']
                            exit_price = target_price
                            exit_type = 'target'
                            break
                        elif row['high'] >= stop_loss:
                            exit_triggered = True
                            exit_date = row['date']
                            exit_price = stop_loss
                            exit_type = 'stop_loss'
                            break
                
                # If position is still open and max holding period is reached
                if not exit_triggered and (current_date - entry_date).days >= position['horizon']:
                    exit_triggered = True
                    exit_date = current_date
                    exit_price = future_data.iloc[-1]['close']
                    exit_type = 'horizon'
                
                # Process exit if triggered
                if exit_triggered:
                    # Calculate profit/loss
                    if direction == 'long':
                        pnl = (exit_price - entry_price) * position['shares']
                    else:  # short
                        pnl = (entry_price - exit_price) * position['shares']
                    
                    # Update available capital
                    available_capital += position['position_size'] + pnl
                    
                    # Record the trade
                    trades.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'holding_days': (exit_date - entry_date).days,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_type': exit_type,
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'return_pct': (pnl / position['position_size']) * 100,
                        'confidence': position['confidence']
                    })
                    
                    # Log exit
                    logger.info(f"Closing {direction} position: {symbol} at {exit_price} ({exit_type}, PnL: {pnl:.2f})")
                    
                    # Mark for removal
                    positions_to_remove.append(position_id)
            
            # Remove closed positions
            for position_id in positions_to_remove:
                del open_positions[position_id]
            
            # Calculate portfolio value at end of day
            current_portfolio_value = available_capital
            for position in open_positions.values():
                # Get the latest price
                latest_data = df_with_features[
                    (df_with_features['trading_symbol'] == position['symbol']) & 
                    (df_with_features['date'] == current_date)
                ]
                
                if len(latest_data) > 0:
                    current_price = latest_data.iloc[0]['close']
                    
                    # Calculate position value
                    if position['direction'] == 'long':
                        position_value = position['shares'] * current_price
                    else:  # short
                        position_value = position['position_size'] + ((position['entry_price'] - current_price) * position['shares'])
                    
                    current_portfolio_value += position_value
            
            # Record portfolio value
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': current_portfolio_value,
                'available_capital': available_capital,
                'open_positions': len(open_positions)
            })
        
        # Close any remaining open positions at the last date
        last_date = dates[-1]
        positions_to_remove = []
        
        for position_id, position in open_positions.items():
            symbol = position['symbol']
            entry_date = position['entry_date']
            direction = position['direction']
            entry_price = position['entry_price']
            
            # Get latest data for this symbol
            latest_data = df_with_features[
                (df_with_features['trading_symbol'] == symbol) & 
                (df_with_features['date'] == last_date)
            ]
            
            if len(latest_data) > 0:
                exit_price = latest_data.iloc[0]['close']
            else:
                # If no data on last date, find the most recent price
                symbol_data = df_with_features[
                    (df_with_features['trading_symbol'] == symbol) & 
                    (df_with_features['date'] > entry_date)
                ].sort_values('date', ascending=False)
                
                if len(symbol_data) > 0:
                    exit_price = symbol_data.iloc[0]['close']
                else:
                    exit_price = entry_price  # Fall back to entry price
            
            # Calculate profit/loss
            if direction == 'long':
                pnl = (exit_price - entry_price) * position['shares']
            else:  # short
                pnl = (entry_price - exit_price) * position['shares']
            
            # Record the trade
            trades.append({
                'symbol': symbol,
                'direction': direction,
                'entry_date': entry_date,
                'exit_date': last_date,
                'holding_days': (last_date - entry_date).days,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_type': 'end_of_period',
                'position_size': position['position_size'],
                'pnl': pnl,
                'return_pct': (pnl / position['position_size']) * 100,
                'confidence': position['confidence']
            })
            
            positions_to_remove.append(position_id)
        
        # Remove closed positions
        for position_id in positions_to_remove:
            del open_positions[position_id]
        
        # Create results DataFrames
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_values)
        
        # Calculate backtest metrics
        if len(trades_df) > 0:
            # Calculate basic metrics
            total_trades = len(trades_df)
            profitable_trades = sum(trades_df['pnl'] > 0)
            win_rate = profitable_trades / total_trades
            avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
            
            # Calculate returns
            initial_value = portfolio_df.iloc[0]['portfolio_value']
            final_value = portfolio_df.iloc[-1]['portfolio_value']
            total_return = (final_value / initial_value) - 1
            trading_days = len(portfolio_df)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
            
            # Calculate drawdown
            portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cummax']) - 1
            max_drawdown = portfolio_df['drawdown'].min()
            
            # Calculate CAGR and Sharpe ratio
            daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            cagr = (final_value / initial_value) ** (252 / trading_days) - 1
            volatility = daily_returns.std() * (252 ** 0.5)
            sharpe_ratio = cagr / volatility if volatility > 0 else 0
            
            # Collect metrics
            metrics = {
                'Start Date': portfolio_df.iloc[0]['date'],
                'End Date': portfolio_df.iloc[-1]['date'],
                'Trading Days': trading_days,
                'Initial Capital': initial_value,
                'Final Portfolio Value': final_value,
                'Total Return': total_return * 100,
                'Annualized Return': annualized_return * 100,
                'CAGR': cagr * 100,
                'Volatility (Annual)': volatility * 100,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown * 100,
                'Total Trades': total_trades,
                'Winning Trades': profitable_trades,
                'Losing Trades': total_trades - profitable_trades,
                'Win Rate': win_rate * 100,
                'Average Profit': avg_profit,
                'Average Loss': avg_loss,
                'Profit Factor': profit_factor,
                'Average Holding Period': trades_df['holding_days'].mean()
            }
            
            # Calculate additional Indian market metrics
            # Kurtosis - measures "tailedness" of returns distribution
            metrics['Kurtosis'] = daily_returns.kurtosis()
            
            # Sortino ratio - variation of Sharpe that only considers downside volatility
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * (252 ** 0.5)
            metrics['Sortino Ratio'] = cagr / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum consecutive wins and losses
            trades_df['win'] = trades_df['pnl'] > 0
            win_streak = 0
            max_win_streak = 0
            loss_streak = 0
            max_loss_streak = 0
            
            for win in trades_df['win']:
                if win:
                    win_streak += 1
                    loss_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                else:
                    loss_streak += 1
                    win_streak = 0
                    max_loss_streak = max(max_loss_streak, loss_streak)
            
            metrics['Max Consecutive Wins'] = max_win_streak
            metrics['Max Consecutive Losses'] = max_loss_streak
            
            # Analyze by trade direction
            metrics['Long Trades'] = len(trades_df[trades_df['direction'] == 'long'])
            metrics['Short Trades'] = len(trades_df[trades_df['direction'] == 'short'])
            
            if metrics['Long Trades'] > 0:
                long_win_rate = sum(trades_df[trades_df['direction'] == 'long']['pnl'] > 0) / metrics['Long Trades']
                metrics['Long Win Rate'] = long_win_rate * 100
            else:
                metrics['Long Win Rate'] = 0
                
            if metrics['Short Trades'] > 0:
                short_win_rate = sum(trades_df[trades_df['direction'] == 'short']['pnl'] > 0) / metrics['Short Trades']
                metrics['Short Win Rate'] = short_win_rate * 100
            else:
                metrics['Short Win Rate'] = 0
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame([metrics])
            
            # Save results
            base_path = f'{self.results_dir}/backtest_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}'
            trades_df.to_csv(f'{base_path}_trades.csv', index=False)
            portfolio_df.to_csv(f'{base_path}_portfolio.csv', index=False)
            metrics_df.to_csv(f'{base_path}_metrics.csv', index=False)
            
            # Generate performance charts
            self.generate_backtest_charts(trades_df, portfolio_df, metrics, base_path)
            
            logger.info(f"Backtest completed with {total_trades} trades")
            logger.info(f"Win Rate: {win_rate:.2%}, Profit Factor: {profit_factor:.2f}, Total Return: {total_return:.2%}")
            
            return {
                'trades': trades_df,
                'portfolio': portfolio_df,
                'metrics': metrics_df
            }
        else:
            logger.warning("No trades executed during backtest period")
            return None
    
    def generate_backtest_charts(self, trades_df, portfolio_df, metrics, base_path):
        """
        Generate comprehensive charts for backtest analysis
        """
        # Set style
        plt.style.use('ggplot')
        
        # 1. Equity Curve
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'])
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_equity_curve.png')
        plt.close()
        
        # 2. Drawdown Chart
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_df['date'], portfolio_df['drawdown'] * 100)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_drawdown.png')
        plt.close()
        
        # 3. Trade PnL Distribution
        plt.figure(figsize=(14, 7))
        sns.histplot(trades_df['pnl'], kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Trade PnL Distribution')
        plt.xlabel('PnL (₹)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_pnl_distribution.png')
        plt.close()
        
        # 4. Trade PnL by Direction
        plt.figure(figsize=(14, 7))
        sns.boxplot(x='direction', y='pnl', data=trades_df)
        plt.title('Trade PnL by Direction')
        plt.xlabel('Direction')
        plt.ylabel('PnL (₹)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_pnl_by_direction.png')
        plt.close()
        
        # 5. Monthly Returns Heatmap
        if len(portfolio_df) > 20:
            # Calculate daily returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            # Resample to monthly returns
            portfolio_df['month'] = portfolio_df['date'].dt.to_period('M')
            portfolio_df['year'] = portfolio_df['date'].dt.year
            
            monthly_returns = portfolio_df.groupby(['year', 'month'])['daily_return'].apply(
                lambda x: (1 + x).prod() - 1
            ).reset_index()
            
            # Pivot for heatmap format
            monthly_pivot = pd.pivot_table(
                monthly_returns,
                values='daily_return',
                index='year',
                columns=monthly_returns['month'].dt.month,
                aggfunc='sum'
            )
            
            # Create month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pivot.columns = [month_labels[i-1] for i in monthly_pivot.columns]
            
            # Plot heatmap
            plt.figure(figsize=(14, 7))
            sns.heatmap(monthly_pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn",
                        linewidths=1, center=0, cbar_kws={'label': 'Monthly Return (%)'})
            plt.title('Monthly Returns Heatmap (%)')
            plt.tight_layout()
            plt.savefig(f'{base_path}_monthly_returns.png')
            plt.close()
        
        # 6. Performance metrics summary
        plt.figure(figsize=(12, 10))
        metrics_to_plot = [
            'Total Return', 'Annualized Return', 'CAGR', 'Sharpe Ratio', 
            'Sortino Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor'
        ]
        
        # Extract values
        values = [metrics[m] for m in metrics_to_plot]
        
        # Create horizontal bar chart
        plt.barh(metrics_to_plot, values, color='steelblue')
        plt.xlabel('Value')
        plt.title('Performance Metrics Summary')
        plt.grid(True, axis='x')
        
        # Add value labels
        for i, v in enumerate(values):
            if 'Ratio' in metrics_to_plot[i]:
                plt.text(v + 0.1, i, f"{v:.2f}", va='center')
            elif 'Factor' in metrics_to_plot[i]:
                plt.text(v + 0.1, i, f"{v:.2f}x", va='center')
            else:
                plt.text(v + 0.1, i, f"{v:.2f}%", va='center')
                
        plt.tight_layout()
        plt.savefig(f'{base_path}_metrics_summary.png')
        plt.close()
        
        # 7. Confidence Score vs Return
        plt.figure(figsize=(10, 6))
        plt.scatter(trades_df['confidence'], trades_df['return_pct'], alpha=0.6)
        plt.title('Confidence Score vs Trade Return')
        plt.xlabel('Model Confidence Score')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_confidence_vs_return.png')
        plt.close()
        
        # 8. Rolling Win Rate
        if len(trades_df) > 30:
            plt.figure(figsize=(14, 7))
            trades_df = trades_df.sort_values('entry_date')
            trades_df['win'] = trades_df['pnl'] > 0
            trades_df['rolling_win_rate'] = trades_df['win'].rolling(window=20).mean() * 100
            
            plt.plot(trades_df['entry_date'], trades_df['rolling_win_rate'])
            plt.axhline(y=50, color='r', linestyle='--')
            plt.title('20-Trade Rolling Win Rate')
            plt.xlabel('Entry Date')
            plt.ylabel('Win Rate (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{base_path}_rolling_win_rate.png')
            plt.close()
            
        # 9. Compare Long vs Short Performance
        if 'direction' in trades_df.columns and len(trades_df) > 10:
            # Aggregate performance by direction
            direction_perf = trades_df.groupby('direction').agg({
                'pnl': ['sum', 'mean'],
                'return_pct': 'mean',
                'win': 'mean'
            })
            
            direction_perf.columns = ['Total_PnL', 'Avg_PnL', 'Avg_Return_Pct', 'Win_Rate']
            direction_perf['Win_Rate'] *= 100  # Convert to percentage
            
            # Plot comparison
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Total PnL
            axs[0, 0].bar(direction_perf.index, direction_perf['Total_PnL'])
            axs[0, 0].set_title('Total PnL by Direction')
            axs[0, 0].set_ylabel('Total PnL (₹)')
            axs[0, 0].grid(True)
            
            # Avg PnL
            axs[0, 1].bar(direction_perf.index, direction_perf['Avg_PnL'])
            axs[0, 1].set_title('Average PnL per Trade')
            axs[0, 1].set_ylabel('Avg PnL (₹)')
            axs[0, 1].grid(True)
            
            # Avg Return %
            axs[1, 0].bar(direction_perf.index, direction_perf['Avg_Return_Pct'])
            axs[1, 0].set_title('Average Return % per Trade')
            axs[1, 0].set_ylabel('Avg Return (%)')
            axs[1, 0].grid(True)
            
            # Win Rate
            axs[1, 1].bar(direction_perf.index, direction_perf['Win_Rate'])
            axs[1, 1].set_title('Win Rate by Direction')
            axs[1, 1].set_ylabel('Win Rate (%)')
            axs[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{base_path}_direction_comparison.png')
            plt.close()
        
        # 10. Holding Period Analysis
        plt.figure(figsize=(12, 6))
        sns.histplot(trades_df['holding_days'], kde=True, bins=15)
        plt.axvline(trades_df['holding_days'].mean(), color='r', linestyle='--', 
                    label=f'Mean: {trades_df["holding_days"].mean():.1f} days')
        plt.title('Distribution of Holding Periods')
        plt.xlabel('Holding Period (days)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{base_path}_holding_period.png')
        plt.close()
    
    def align_features(self, X, features, expected_features):
        """
        Align input features with the expected features for a model
        
        Args:
            X: Input data (single row or batch)
            features: Current feature names
            expected_features: Feature names the model expects
            
        Returns:
            Aligned feature array with correct dimensions
        """
        if len(features) == len(expected_features) and all(f == e for f, e in zip(features, expected_features)):
            # Features already match
            return X
            
        logger.info(f"Aligning features: have {len(features)}, need {len(expected_features)}")
        
        # For a single sample (prediction case)
        if X.ndim == 2 and X.shape[0] == 1:
            aligned_X = np.zeros((1, len(expected_features)))
            
            # Map available features to their correct positions
            for i, feature in enumerate(expected_features):
                if feature in features:
                    # Find position in current features
                    idx = features.index(feature)
                    aligned_X[0, i] = X[0, idx]
                else:
                    # Feature is missing, fill with 0 or median value
                    logger.warning(f"Missing feature: {feature}")
                    aligned_X[0, i] = 0  # Use appropriate default value
                    
            return aligned_X
        
        # For multiple samples (batch processing case)
        # Implement similar logic for batches if needed
        
        return X

    def predict_stocks_with_potential(self, confidence_threshold=0.7):
        """
        Identify stocks with potential for upside and downside movement
        """
        logger.info(f"Identifying stocks with potential (threshold: {confidence_threshold})")

        # Ensure scalers are loaded
        if sum(1 for _ in self.scalers.values()) < len(self.prediction_horizon_days):
            logger.info("Loading scalers from disk...")
            self.load_scalers()
            
        # Log current scalers
        logger.info(f"Available scalers: {list(self.scalers.keys())}")
        
        # Fetch latest data
        df = self.fetch_historical_data()
        if df is None:
            logger.error("Failed to fetch data for prediction")
            return None
            
        # Engineer features
        df_with_features = self.engineer_features(df)
        if df_with_features is None:
            logger.error("Feature engineering failed for prediction")
            return None
            
        # Get latest date
        latest_date = df_with_features['date'].max()
        logger.info(f"Making predictions for date: {latest_date}")

        # Get all stocks on latest date
        latest_data = df_with_features[df_with_features['date'] == latest_date].copy()
        logger.info(f"Found {len(latest_data)} stocks on latest date")

        # Initialize prediction results
        predictions = []
        
        # Process each prediction horizon separately
        for days in self.prediction_horizon_days:
            for direction in ['long', 'short']:
                logger.info(f"Processing {direction} {days}-day horizon predictions")
                
                # Use direction-specific feature sets and models
                feature_key = f'{direction}_{days}d'
                lgbm_model_key = f'lgbm_{direction}_{days}d'
                pytorch_model_key = f'pytorch_{direction}_{days}d'
            
            # First, load the expected features for this horizon
            feature_path = f'{self.results_dir}/selected_features_{days}d.csv'
            if not os.path.exists(feature_path):
                logger.warning(f"No feature selection file found for {days}-day horizon")
                continue
                
            try:
                feature_df = pd.read_csv(feature_path)
                expected_features = feature_df['feature'].tolist()
                logger.info(f"Using {len(expected_features)} features for {days}-day horizon")
            except Exception as e:
                logger.error(f"Error loading features for {days}-day horizon: {e}")
                continue
                
            # Check if we have a scaler for this horizon
            if days not in self.scalers:
                logger.warning(f"No scaler available for {days}-day horizon")
                continue
            
            # Load the models
            lgbm_model_key = f'lgbm_{days}d'
            if lgbm_model_key not in self.models:
                try:
                    model_path = f'{self.models_dir}/lgbm_model_{days}d.pkl'
                    if not os.path.exists(model_path):
                        logger.warning(f"LightGBM model not found: {model_path}")
                        continue
                        
                    self.models[lgbm_model_key] = joblib.load(model_path)
                    logger.info(f"Loaded LightGBM model for {days}-day horizon")
                except Exception as e:
                    logger.error(f"Error loading LightGBM model: {e}")
                    continue
                    
            pytorch_model_key = f'pytorch_{days}d'
            if pytorch_model_key not in self.models:
                try:
                    model_path = f'{self.models_dir}/pytorch_model_{days}d.pt'
                    if not os.path.exists(model_path):
                        logger.warning(f"PyTorch model not found: {model_path}")
                        continue
                        
                    # Create model with correct input size
                    input_size = len(expected_features)
                    model = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.models[pytorch_model_key] = model.to(self.device)
                    logger.info(f"Loaded PyTorch model for {days}-day horizon")
                except Exception as e:
                    logger.error(f"Error loading PyTorch model: {e}")
                    continue
            
            # Get the models
            lgbm_model = self.models[lgbm_model_key]
            pytorch_model = self.models[pytorch_model_key]
            
            # Process each stock
            stocks_processed = 0
            stocks_with_predictions = 0
            
            for idx, row in latest_data.iterrows():
                symbol = row['trading_symbol']
                
                # Check if all required features are present and not NaN
                if not all(f in row.index and not pd.isna(row[f]) for f in expected_features):
                    continue
                    
                # Extract features in the correct order
                X = np.array([row[f] for f in expected_features]).reshape(1, -1)
                
                try:
                    # Scale data
                    X_scaled = self.scalers[days].transform(X)
                    
                    # Make LightGBM prediction
                    lgbm_proba = lgbm_model.predict_proba(X_scaled)[0, 1]
                    
                    # Make PyTorch prediction
                    X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                    pytorch_model.eval()
                    with torch.no_grad():
                        nn_proba = pytorch_model(X_tensor).item()
                    
                    # Get model weights for this horizon
                    weights = self.get_model_weights(days)
                    
                    # Weighted ensemble prediction
                    ensemble_proba = (lgbm_proba * weights['lgbm'] + nn_proba * weights['pytorch'])
                    
                    # Only consider high confidence predictions
                    if ensemble_proba >= confidence_threshold:
                        stocks_with_predictions += 1
                        current_price = row['close']
                        target_price = current_price * (1 + self.upside_threshold_min)
                        max_target_price = current_price * (1 + self.upside_threshold_max)
                        
                        predictions.append({
                            'trading_symbol': symbol,
                            'company_name': row['company_name'],
                            'prediction_date': latest_date,
                            'direction': direction,
                            'prediction_horizon': days,
                            'confidence_score': ensemble_proba,
                            'current_price': current_price,
                            'target_price': target_price,
                            'max_target_price': max_target_price,
                            'expected_return_min': self.upside_threshold_min,
                            'expected_return_max': self.upside_threshold_max,
                            'lgbm_confidence': lgbm_proba,
                            'nn_confidence': nn_proba
                        })
                        logger.info(f"Found signal: {symbol}, {days}d horizon, confidence: {ensemble_proba:.4f}")
                    
                    stocks_processed += 1
                    if stocks_processed % 100 == 0:
                        logger.info(f"Processed {stocks_processed} stocks for {days}d horizon")
                        
                except Exception as e:
                    logger.error(f"Error predicting for {symbol} on {days}d horizon: {e}")
                    continue
                    
            logger.info(f"Completed {days}d horizon: processed {stocks_processed} stocks, found {stocks_with_predictions} signals")

        # Create DataFrame from results
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            
            # Sort by confidence score
            predictions_df = predictions_df.sort_values('confidence_score', ascending=False)
            
            # Save predictions
            predictions_df.to_csv(f'{self.results_dir}/stock_predictions_{latest_date.strftime("%Y%m%d")}.csv', index=False)
            
            logger.info(f"Generated {len(predictions_df)} stock predictions")
            return predictions_df
        else:
            logger.warning("No predictions generated")
            return None
    
    def run_complete_pipeline(self):
        """
        Run the complete pipeline: train models, evaluate, backtest, and make predictions
        """
        logger.info("Starting complete pipeline")
        
        # Train all models (both long and short)
        self.train_all_models()
        
        # Create feature importance report
        self.create_feature_importance_report()
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Backtest with GPU acceleration
        backtest_results = self.backtest(
            start_date='2024-10-01',
            end_date='2024-12-31',
            confidence_threshold=0.7,
            batch_size=64  # GPU batch size
        )
        
        # Generate predictions (both long and short)
        predictions = self.predict_stocks_with_potential(confidence_threshold=0.7)
        
        logger.info("Pipeline completed successfully")
        return {
            'evaluation': evaluation_results,
            'backtest': backtest_results,
            'predictions': predictions
        }
    
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Stock Prediction Framework')
    parser.add_argument('--db_password', required=True, help='Database password')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'backtest', 'predict', 'full'],
                      default='full', help='Mode to run')
    parser.add_argument('--confidence', type=float, default=0.7, 
                      help='Confidence threshold for predictions')
    parser.add_argument('--batch_size', type=int, default=512,
                      help='Batch size for training')
    parser.add_argument('--use_pytorch', action='store_true', default=True,
                      help='Use PyTorch for neural network models')
    parser.add_argument('--upside_threshold_min', type=float, default=0.025,
                      help='Minimum upside target threshold for long positions')
    parser.add_argument('--upside_threshold_max', type=float, default=0.05,
                      help='Maximum upside target threshold for long positions')
    parser.add_argument('--downside_threshold_min', type=float, default=0.025,
                      help='Minimum downside target threshold for short positions')
    parser.add_argument('--downside_threshold_max', type=float, default=0.05,
                      help='Maximum downside target threshold for short positions')
    parser.add_argument('--backtest_start_date', type=str, default=None,
                      help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--backtest_end_date', type=str, default=None,
                      help='End date for backtesting (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = StockPredictionFramework(
        db_password=args.db_password,
        prediction_horizon_days=[1, 3, 5, 7, 10],
        upside_threshold_min=args.upside_threshold_min,
        upside_threshold_max=args.upside_threshold_max,
        downside_threshold_min=args.downside_threshold_min,
        downside_threshold_max=args.downside_threshold_max,
        use_pytorch=args.use_pytorch
    )
    
    # Run based on mode
    if args.mode == 'train':
        framework.train_all_models()
    elif args.mode == 'evaluate':
        framework.evaluate_models()
    elif args.mode == 'backtest':
        framework.backtest(
            start_date=args.backtest_start_date,
            end_date=args.backtest_end_date,
            confidence_threshold=args.confidence
        )
    elif args.mode == 'predict':
        framework.predict_stocks_with_potential(confidence_threshold=args.confidence)
    elif args.mode == 'full':
        framework.run_complete_pipeline()