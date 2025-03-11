# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import joblib
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Machine learning libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('TradingModel')

# Suppress warnings
warnings.filterwarnings('ignore')

class TradingModelTrainer:
    def __init__(self, db_config: Dict[str, str], output_dir: str = None):
        """
        Initialize the trading model trainer
        
        Args:
            db_config: MySQL database configuration
            output_dir: Directory to save models and results
        """
        self.db_config = db_config
        
        # Create output directory if not provided
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(script_dir, f"trading_models_{timestamp}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Model outputs will be saved to: {self.output_dir}")
        
        # Define the optimal features based on your analysis
        self.optimal_features = [
            'nifty_corr_120d', 'macd_line_lag20', 'corr_change', 'nifty_corr_20d',
            'month', 'bollinger_width', 'nifty_corr_60d', 'rsi_14_lag10',
            'volume_oscillator', 'macd_histogram', 'rs_nifty_120d', 'day_of_month',
            'di_minus_14', 'trix', 'di_plus_14', 'sma_200', 'dist_to_lower_band',
            'ultosc', 'sma_20_lag15', 'macd_line', 'rs_nifty_60d', 'return_40d',
            'stochastic_d', 'roc', 'rs_nifty_20d', 'natr', 'cci_20', 'return_120d',
            'adx_14', 'dist_to_sma200'
        ]
        
        # Track execution time
        self.start_time = time.time()
    def connect_to_db(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        """Create a database connection."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def load_data(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data from the database
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of trading symbols to include (None for all)
            
        Returns:
            DataFrame with ml_features data
        """
        try:
            conn = self.connect_to_db()
            if not conn:
                return pd.DataFrame()
                
            query = "SELECT * FROM ml_features WHERE date BETWEEN %s AND %s"
            params = [start_date, end_date]
            
            # Add symbol filter if provided
            if symbols:
                placeholder = ', '.join(['%s'] * len(symbols))
                query += f" AND trading_symbol IN ({placeholder})"
                params.extend(symbols)
            
            logger.info(f"Loading data from {start_date} to {end_date}")
            df = pd.read_sql(query, conn, params=params)
            
            conn.close()
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date and then by symbol for time-series integrity
            df = df.sort_values(['date', 'trading_symbol'])
            
            logger.info(f"Loaded {len(df)} rows with {df['trading_symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            if 'conn' in locals() and conn:
                conn.close()
            return pd.DataFrame()
    
    def engineer_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features based on optimal feature set
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_new = df.copy()
        
        # Initialize the engineered columns with NaN values
        engineered_cols = [
            'nifty_corr_ratio', 'macd_line_change', 'bollinger_position', 
            'rsi_macd_confirm', 'trend_strength', 'nifty_rs_ratio',
            'di_crossover', 'trix_direction', 'return_ratio', 'volatility_regime'
        ]
        
        # Initialize columns first
        for col in engineered_cols:
            df_new[col] = np.nan
        
        # Group by symbol to ensure calculations are within each symbol
        for symbol in df_new['trading_symbol'].unique():
            symbol_mask = df_new['trading_symbol'] == symbol
            symbol_data = df_new.loc[symbol_mask].copy()
            
            # 1. Correlation ratios and changes
            symbol_data['nifty_corr_ratio'] = symbol_data['nifty_corr_20d'] / (symbol_data['nifty_corr_120d'] + 1e-6)
            
            # 2. MACD related features
            symbol_data['macd_line_change'] = symbol_data['macd_line'] - symbol_data['macd_line_lag20']
            
            # 3. Bollinger band relative position
            # Where is price relative to bollinger bands (normalized between 0 and 1)
            if 'bollinger_upper' in symbol_data.columns and 'bollinger_lower' in symbol_data.columns:
                band_width = symbol_data['bollinger_upper'] - symbol_data['bollinger_lower']
                symbol_data['bollinger_position'] = (symbol_data['close'] - symbol_data['bollinger_lower']) / (band_width + 1e-6)
            
            # 4. RSI and MACD confirmation
            symbol_data['rsi_macd_confirm'] = np.sign(symbol_data['rsi_14_lag10'] - 50) * np.sign(symbol_data['macd_line'])
            
            # 5. Combined trend strength
            symbol_data['trend_strength'] = (
                np.sign(symbol_data['macd_line']) +
                np.sign(symbol_data['trix']) +
                np.sign(symbol_data['return_40d'])
            )
            
            # 6. Relative strength ratio
            symbol_data['nifty_rs_ratio'] = symbol_data['rs_nifty_20d'] / (symbol_data['rs_nifty_120d'] + 1e-6)
            
            # 7. DI crossover indicator
            symbol_data['di_crossover'] = symbol_data['di_plus_14'] - symbol_data['di_minus_14']
            
            # 8. Trix direction
            symbol_data['trix_direction'] = np.sign(symbol_data['trix'])
            
            # 9. Return ratio (shorter term vs longer term)
            symbol_data['return_ratio'] = symbol_data['return_40d'] / (symbol_data['return_120d'] + 1e-6)
            
            # 10. Volatility regime based on natr
            symbol_data['volatility_regime'] = symbol_data['natr'] / symbol_data['natr'].rolling(20).mean()
            
            # Update each column individually in the main dataframe
            for col in engineered_cols:
                df_new.loc[symbol_mask, col] = symbol_data[col]
        
        # Replace infinities and NaNs
        df_new = df_new.replace([np.inf, -np.inf], np.nan)
        
        # For engineered features, fill NaNs with 0
        df_new[engineered_cols] = df_new[engineered_cols].fillna(0)
        
        logger.info(f"Added {len(engineered_cols)} engineered features")
        return df_new
    
    def prepare_features_and_target(
        self, 
        df: pd.DataFrame, 
        include_engineered: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.Series]]]:
        """
        Prepare features and target variables for each symbol
        
        Args:
            df: Input DataFrame
            include_engineered: Whether to include engineered features
            
        Returns:
            Two dictionaries: features for each symbol and multiple target variables for each symbol
        """
        if df.empty:
            logger.error("DataFrame is empty")
            return {}, {}
        
        # Prepare feature list - start with optimal features
        feature_cols = self.optimal_features.copy()
        
        # Add engineered features if include_engineered is True and they exist in the dataframe
        if include_engineered:
            engineered_cols = [
                'nifty_corr_ratio', 'macd_line_change', 'bollinger_position', 
                'rsi_macd_confirm', 'trend_strength', 'nifty_rs_ratio',
                'di_crossover', 'trix_direction', 'return_ratio', 'volatility_regime'
            ]
            
            # Only include engineered columns that actually exist in the dataframe
            existing_engineered_cols = [col for col in engineered_cols if col in df.columns]
            feature_cols.extend(existing_engineered_cols)
        
        # Target variables - the 5 we need to predict
        target_cols = ['future_close', 'future_return', 'target', 'exit_signal', 'days_to_target']
        for target_col in target_cols:
            if target_col in df.columns:
                non_null_count = df[target_col].notna().sum()
                logger.info(f"Target column '{target_col}' found with {non_null_count} non-null values")
            else:
                logger.warning(f"Target column '{target_col}' NOT FOUND in loaded data")
        
        # Dictionaries to store X and y for each symbol
        X_dict = {}
        y_dict = {}
        
        # Process each symbol separately
        for symbol in df['trading_symbol'].unique():
            symbol_df = df[df['trading_symbol'] == symbol].copy()
            
            # Ensure chronological order
            symbol_df = symbol_df.sort_values('date')
            
            # Extract features - only include columns that actually exist
            available_features = [col for col in feature_cols if col in symbol_df.columns]
            if not available_features:
                logger.warning(f"No valid features found for symbol {symbol}. Skipping.")
                continue
            
            X = symbol_df[available_features]
            
            # Extract target variables
            y = {}
            missing_targets = []
            
            for target_col in target_cols:
                if target_col in symbol_df.columns:
                    y[target_col] = symbol_df[target_col]
                else:
                    missing_targets.append(target_col)
            
            if missing_targets:
                logger.warning(f"Symbol {symbol} missing target variables: {missing_targets}")
                
            # Only add the symbol if we have at least some target variables
            if y:
                X_dict[symbol] = X
                y_dict[symbol] = y
                logger.info(f"Symbol {symbol} prepared with {len(available_features)} features and {len(y)} target variables")
            else:
                logger.warning(f"Symbol {symbol} has no target variables. Skipping.")
        
        logger.info(f"Prepared data for {len(X_dict)} symbols using {len(feature_cols)} features")
        return X_dict, y_dict
    
    def train_test_split_time_series(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y_dict: Dict[str, Dict[str, pd.Series]],
        test_size: float = 0.2
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.Series]], Dict[str, Dict[str, pd.Series]]]:
        """
        Split data into train and test sets respecting time order for multiple targets
        
        Args:
            X_dict: Dictionary of feature DataFrames for each symbol
            y_dict: Dictionary of target dictionaries for each symbol
            test_size: Fraction of data to use for testing
            
        Returns:
            X_train_dict, X_test_dict, y_train_dict, y_test_dict
        """
        X_train_dict = {}
        X_test_dict = {}
        y_train_dict = {}
        y_test_dict = {}
        
        for symbol in X_dict.keys():
            X = X_dict[symbol]
            y_targets = y_dict[symbol]
            
            # Calculate split point (respecting time order)
            split_idx = int(len(X) * (1 - test_size))
            
            # Split features
            X_train_dict[symbol] = X.iloc[:split_idx].copy()
            X_test_dict[symbol] = X.iloc[split_idx:].copy()
            
            # Split each target variable
            y_train_dict[symbol] = {}
            y_test_dict[symbol] = {}
            
            for target_name, y in y_targets.items():
                y_train_dict[symbol][target_name] = y.iloc[:split_idx].copy()
                y_test_dict[symbol][target_name] = y.iloc[split_idx:].copy()
        
        logger.info(f"Split data into train/test sets with test_size={test_size}")
        return X_train_dict, X_test_dict, y_train_dict, y_test_dict
    
    def handle_class_imbalance(
        self,
        X_train_dict: Dict[str, pd.DataFrame],
        y_train_dict: Dict[str, Dict[str, pd.Series]]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.Series]]]:
        """
        Apply SMOTE to handle class imbalance in training data for classification targets
        
        Args:
            X_train_dict: Dictionary of training feature DataFrames
            y_train_dict: Dictionary of training target dictionaries
            
        Returns:
            X_resampled_dict, y_resampled_dict
        """
        X_resampled_dict = {}
        y_resampled_dict = {}
        
        # Classification targets
        classification_targets = ['target', 'exit_signal']
        
        for symbol in X_train_dict.keys():
            X_train = X_train_dict[symbol]
            y_train_targets = y_train_dict[symbol]
            
            # For classification targets, may need to handle class imbalance
            # For regression targets, keep as is
            y_resampled_targets = {}
            
            # Check if we need to apply SMOTE for any classification target
            need_smote = False
            
            for target_name in classification_targets:
                if target_name in y_train_targets:
                    y_train = y_train_targets[target_name]
                    class_counts = y_train.value_counts()
                    
                    # If both classes have at least 5 samples and there is imbalance
                    if len(class_counts) > 1 and all(class_counts > 5):
                        minority_pct = class_counts.min() / class_counts.sum()
                        # Only apply SMOTE if there's significant imbalance
                        if minority_pct < 0.3:
                            need_smote = True
                            break
            
            # If we need to apply SMOTE, do it for all classification targets
            if need_smote:
                logger.info(f"Applying SMOTE for {symbol} to handle class imbalance")
                
                # Get original index to align with regression targets later
                orig_index = X_train.index
                
                # Choose one classification target to use for resampling (target is preferred)
                resample_target = 'target' if 'target' in y_train_targets else 'exit_signal'
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train_targets[resample_target])
                
                # Convert back to DataFrame with original column names
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                
                # Store resampled X
                X_resampled_dict[symbol] = X_resampled
                
                # Handle all target variables
                y_resampled_targets = {}
                
                # For the used classification target, use the SMOTE result
                y_resampled_targets[resample_target] = pd.Series(y_resampled, name=resample_target)
                
                # For other classification targets, need to align or resample separately
                for target_name in classification_targets:
                    if target_name != resample_target and target_name in y_train_targets:
                        # For simplicity, we'll apply SMOTE separately for each classification target
                        # This isn't ideal but ensures each target has balanced classes
                        _, y_target_resampled = smote.fit_resample(X_train, y_train_targets[target_name])
                        y_resampled_targets[target_name] = pd.Series(y_target_resampled, name=target_name)
                
                # For regression targets, need to handle differently since SMOTE changes data points
                # Option 1: Skip them in resampled data
                # Option 2: Train separate models for regression targets on original data
                # Option 3: Use a weighted sampling approach instead of SMOTE
                # Here we'll use Option 1 for simplicity
                for target_name, y_train in y_train_targets.items():
                    if target_name not in classification_targets:
                        # Skip regression targets for now
                        logger.warning(f"Skipping regression target {target_name} during resampling for {symbol}")
                        # y_resampled_targets[target_name] = y_train
            else:
                # No SMOTE needed, use original data
                X_resampled_dict[symbol] = X_train
                y_resampled_targets = y_train_targets
            
            y_resampled_dict[symbol] = y_resampled_targets
            
        return X_resampled_dict, y_resampled_dict
    
    def train_lightgbm_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str
    ) -> lgb.Booster:
        """
        Train a LightGBM regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of the target variable
            
        Returns:
            Trained LightGBM model
        """
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model parameters for regression
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
        
        # Adjust parameters based on target variable
        if target_name == 'days_to_target':
            # For count data like days, use Poisson regression
            params['objective'] = 'poisson'
            params['metric'] = 'poisson'
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )
        
        return model

    def train_lightgbm_classification_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str
    ) -> lgb.Booster:
        """
        Train a LightGBM classification model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of the target variable
            
        Returns:
            Trained LightGBM model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        weight_for_0 = (total_samples / (2 * class_counts.get(0, 1)))
        weight_for_1 = (total_samples / (2 * class_counts.get(1, 1)))
        
        # Create sample weights array for each observation
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = weight_for_0
        sample_weights[y_train == 1] = weight_for_1
        
        # Prepare datasets with weights
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model parameters for classification
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'scale_pos_weight': weight_for_1 / weight_for_0
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )
        
        return model
    
    def train_lightgbm_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> lgb.Booster:
        """
        Train a LightGBM model with GPU acceleration
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained LightGBM model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        weight_for_0 = (total_samples / (2 * class_counts.get(0, 1)))
        weight_for_1 = (total_samples / (2 * class_counts.get(1, 1)))
        
        # Create sample weights array for each observation
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 0] = weight_for_0
        sample_weights[y_train == 1] = weight_for_1
        
        # Prepare datasets with weights
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model parameters (with GPU acceleration)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            # Removed class_weight dictionary
            'scale_pos_weight': weight_for_1 / weight_for_0  # This is sufficient for binary classification
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )
        
        return model
    
    def train_xgboost_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str
    ) -> xgb.Booster:
        """
        Train an XGBoost regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of the target variable
            
        Returns:
            Trained XGBoost model
        """
        # Model parameters for regression
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        
        # Adjust parameters based on target variable
        if target_name == 'days_to_target':
            # For count data like days, count:poisson might be better
            params['objective'] = 'count:poisson'
            params['eval_metric'] = 'poisson-nloglik'
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model

    def train_xgboost_classification_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> xgb.Booster:
        """
        Train an XGBoost classification model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
        
        # Model parameters for classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model

    def train_catboost_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str
    ) -> CatBoostClassifier:
        """
        Train a CatBoost regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target_name: Name of the target variable
            
        Returns:
            Trained CatBoost model
        """
        # Model parameters for regression
        loss_function = 'RMSE'
        
        # Adjust parameters based on target variable
        if target_name == 'days_to_target':
            loss_function = 'Poisson'
        
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function=loss_function,
            eval_metric='RMSE',
            random_seed=42,
            early_stopping_rounds=50,
            task_type='GPU',
            devices='0',
            verbose=100
        )
        
        # Prepare eval set
        eval_set = [(X_val, y_val)]
        
        # Train model
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        return model

    def train_catboost_classification_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> CatBoostClassifier:
        """
        Train a CatBoost classification model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained CatBoost model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
        
        # Model parameters for classification
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            early_stopping_rounds=50,
            task_type='GPU',
            devices='0',
            scale_pos_weight=scale_pos_weight,
            verbose=100
        )
        
        # Prepare eval set
        eval_set = [(X_val, y_val)]
        
        # Train model
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        return model

    def train_models_for_symbol_multi_target(
        self,
        symbol: str,
        X_train: pd.DataFrame,
        y_train_dict: Dict[str, pd.Series],
        X_val: pd.DataFrame,
        y_val_dict: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models for a specific symbol for each target variable
        
        Args:
            symbol: Trading symbol
            X_train: Training features
            y_train_dict: Dictionary of training targets
            X_val: Validation features
            y_val_dict: Dictionary of validation targets
            
        Returns:
            Dictionary of trained models and their evaluation metrics for each target
        """
        logger.info(f"Training models for {symbol} with multiple targets")
        
        # At the beginning of the method, define your target types
        classification_targets = ['target', 'exit_signal']
        regression_targets = ['future_close', 'future_return', 'days_to_target']
        
        try:
            models_by_target = {}
            
            # Train models for each target variable
            # And check that each target is being properly identified
            for target_name, y_train in y_train_dict.items():
                y_val = y_val_dict[target_name]
                
                # Determine if this is a classification or regression problem
                is_classification = target_name in classification_targets
                
                if is_classification:
                    logger.info(f"Training classification model for {symbol} - {target_name}")
                else:
                    logger.info(f"Training regression model for {symbol} - {target_name}")
                
                models = {}
                
                # Train LightGBM model
                start_time = time.time()
                if is_classification:
                    lgb_model = self.train_lightgbm_classification_model(X_train, y_train, X_val, y_val, target_name)
                else:
                    lgb_model = self.train_lightgbm_regression_model(X_train, y_train, X_val, y_val, target_name)
                lgb_time = time.time() - start_time
                models['lightgbm'] = {
                    'model': lgb_model,
                    'training_time': lgb_time
                }
                
                # Train XGBoost model
                start_time = time.time()
                if is_classification:
                    xgb_model = self.train_xgboost_classification_model(X_train, y_train, X_val, y_val)
                else:
                    xgb_model = self.train_xgboost_regression_model(X_train, y_train, X_val, y_val, target_name)
                xgb_time = time.time() - start_time
                models['xgboost'] = {
                    'model': xgb_model,
                    'training_time': xgb_time
                }
                
                # Train CatBoost model
                start_time = time.time()
                if is_classification:
                    catboost_model = self.train_catboost_classification_model(X_train, y_train, X_val, y_val)
                else:
                    catboost_model = self.train_catboost_regression_model(X_train, y_train, X_val, y_val, target_name)
                catboost_time = time.time() - start_time
                models['catboost'] = {
                    'model': catboost_model,
                    'training_time': catboost_time
                }
                
                # Evaluate models
                for model_name, model_info in models.items():
                    if is_classification:
                        if model_name == 'lightgbm':
                            y_pred_proba = model_info['model'].predict(X_val)
                        elif model_name == 'xgboost':
                            dval = xgb.DMatrix(X_val)
                            y_pred_proba = model_info['model'].predict(dval)
                        elif model_name == 'catboost':
                            y_pred_proba = model_info['model'].predict_proba(X_val)[:, 1]
                        
                        # Evaluate classification metrics
                        metrics = self.evaluate_classification_predictions(y_val, y_pred_proba)
                        models[model_name]['metrics'] = metrics
                        
                        logger.info(f"{symbol} - {target_name} - {model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
                    else:
                        if model_name == 'lightgbm':
                            y_pred = model_info['model'].predict(X_val)
                        elif model_name == 'xgboost':
                            dval = xgb.DMatrix(X_val)
                            y_pred = model_info['model'].predict(dval)
                        elif model_name == 'catboost':
                            y_pred = model_info['model'].predict(X_val)
                        
                        # Evaluate regression metrics
                        metrics = self.evaluate_regression_predictions(y_val, y_pred)
                        models[model_name]['metrics'] = metrics
                        
                        logger.info(f"{symbol} - {target_name} - {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                
                # Store models for this target
                models_by_target[target_name] = models
                
            return models_by_target
                
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
            return {}
    
    def train_xgboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> xgb.Booster:
        """
        Train an XGBoost model with GPU acceleration
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
        
        # Model parameters (with GPU acceleration)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def evaluate_regression_predictions(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def evaluate_classification_predictions(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification predictions
        
        Args:
            y_true: True values
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train_catboost_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> CatBoostClassifier:
        """
        Train a CatBoost model with GPU acceleration
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained CatBoost model
        """
        # Calculate class weights
        total_samples = len(y_train)
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
        
        # Model parameters (with GPU acceleration)
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            early_stopping_rounds=50,
            task_type='GPU',
            devices='0',
            scale_pos_weight=scale_pos_weight,
            verbose=100
        )
        
        # Prepare eval set
        eval_set = [(X_val, y_val)]
        
        # Train model
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        return model
    
    def train_models_for_symbol(
        self,
        symbol: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train multiple models for a specific symbol
        
        Args:
            symbol: Trading symbol
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary of trained models and their evaluation metrics
        """
        logger.info(f"Training models for {symbol}")
        
        try:
            models = {}
            
            # Train LightGBM model
            start_time = time.time()
            lgb_model = self.train_lightgbm_model(X_train, y_train, X_val, y_val)
            lgb_time = time.time() - start_time
            models['lightgbm'] = {
                'model': lgb_model,
                'training_time': lgb_time
            }
            
            # Train XGBoost model
            start_time = time.time()
            xgb_model = self.train_xgboost_model(X_train, y_train, X_val, y_val)
            xgb_time = time.time() - start_time
            models['xgboost'] = {
                'model': xgb_model,
                'training_time': xgb_time
            }
            
            # Train CatBoost model
            start_time = time.time()
            catboost_model = self.train_catboost_model(X_train, y_train, X_val, y_val)
            catboost_time = time.time() - start_time
            models['catboost'] = {
                'model': catboost_model,
                'training_time': catboost_time
            }
            
            # Evaluate models
            for model_name, model_info in models.items():
                if model_name == 'lightgbm':
                    y_pred_proba = model_info['model'].predict(X_val)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                elif model_name == 'xgboost':
                    dval = xgb.DMatrix(X_val)
                    y_pred_proba = model_info['model'].predict(dval)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                elif model_name == 'catboost':
                    y_pred_proba = model_info['model'].predict_proba(X_val)[:, 1]
                    y_pred = model_info['model'].predict(X_val)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_pred_proba)
                
                # Store metrics
                models[model_name]['metrics'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
                
                logger.info(f"{symbol} - {model_name}: AUC={auc:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")
            
            return models
            
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
            return {}
    
    def find_best_models(
        self,
        models_by_symbol_target: Dict[str, Dict[str, Dict]]
    ) -> Dict[str, Dict[str, Tuple[str, Any]]]:
        """
        Find the best model for each symbol and target variable
        
        Args:
            models_by_symbol_target: Dictionary of models for each symbol and target
            
        Returns:
            Dictionary with the best model for each symbol and target
        """
        best_models = {}
        
        # Classification targets
        classification_targets = ['target', 'exit_signal']
        
        # Regression targets
        regression_targets = ['future_close', 'future_return', 'days_to_target']
        
        for symbol, targets_dict in models_by_symbol_target.items():
            best_models[symbol] = {}
            
            for target_name, models in targets_dict.items():
                best_model_name = None
                best_model = None
                best_score = -float('inf')  # Initialize with negative infinity
                
                for model_name, model_info in models.items():
                    if 'metrics' not in model_info:
                        continue
                        
                    # Use appropriate scoring metric based on the target type
                    if target_name in classification_targets:
                        score = model_info['metrics'].get('auc', -float('inf'))
                    elif target_name in regression_targets:
                        # For regression, higher R² is better
                        score = model_info['metrics'].get('r2', -float('inf'))
                    else:
                        # Default to a generic approach
                        if 'auc' in model_info['metrics']:
                            score = model_info['metrics']['auc']
                        elif 'r2' in model_info['metrics']:
                            score = model_info['metrics']['r2']
                        else:
                            score = -float('inf')
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_model = model_info['model']
                
                if best_model_name and best_model:
                    best_models[symbol][target_name] = (best_model_name, best_model)
                    
                    if target_name in classification_targets:
                        metric_name = "AUC"
                    else:
                        metric_name = "R²"
                        
                    logger.info(f"{symbol} - {target_name}: Best model is {best_model_name} with {metric_name}={best_score:.4f}")
        
        return best_models
    
    def save_models(
        self,
        best_models: Dict[str, Dict[str, Tuple[str, Any]]]
    ) -> bool:
        """
        Save the best models to disk
        
        Args:
            best_models: Dictionary of best models for each symbol and target
            
        Returns:
            True if successful, False otherwise
        """
        try:
            models_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model info
            model_info = {}
            
            for symbol, targets_dict in best_models.items():
                model_info[symbol] = {}
                
                for target_name, (model_name, model) in targets_dict.items():
                    model_path = os.path.join(models_dir, f"{symbol}_{target_name}_{model_name}.model")
                    
                    if model_name == 'lightgbm':
                        model.save_model(model_path)
                    elif model_name == 'xgboost':
                        model.save_model(model_path)
                    elif model_name == 'catboost':
                        model.save_model(model_path)
                    
                    model_info[symbol][target_name] = {
                        'model_type': model_name,
                        'model_path': model_path
                    }
            
            # Save model info
            info_path = os.path.join(self.output_dir, 'model_info.joblib')
            joblib.dump(model_info, info_path)
            
            model_count = sum(len(targets_dict) for targets_dict in best_models.values())
            logger.info(f"Saved {model_count} models for {len(best_models)} symbols to {models_dir}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def save_performance_metrics(
        self,
        models_by_symbol: Dict[str, Dict[str, Dict]]
    ) -> None:
        """
        Save detailed performance metrics to CSV
        
        Args:
            models_by_symbol: Dictionary of models and metrics for each symbol
        """
        try:
            metrics_records = []
            
            for symbol, models in models_by_symbol.items():
                for model_name, model_info in models.items():
                    if 'metrics' in model_info:
                        record = {
                            'symbol': symbol,
                            'model': model_name,
                            'accuracy': model_info['metrics'].get('accuracy', float('nan')),
                            'precision': model_info['metrics'].get('precision', float('nan')),
                            'recall': model_info['metrics'].get('recall', float('nan')),
                            'f1': model_info['metrics'].get('f1', float('nan')),
                            'auc': model_info['metrics'].get('auc', float('nan')),
                            'training_time': model_info.get('training_time', float('nan'))
                        }
                        metrics_records.append(record)
            
            metrics_df = pd.DataFrame(metrics_records)
            metrics_path = os.path.join(self.output_dir, 'model_performance_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved performance metrics to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def plot_confusion_matrices(
        self,
        X_test_dict: Dict[str, pd.DataFrame],
        y_test_dict: Dict[str, pd.Series],
        best_models: Dict[str, Tuple[str, Any]]
    ) -> None:
        """
        Plot confusion matrices for each model
        
        Args:
            X_test_dict: Dictionary of test features for each symbol
            y_test_dict: Dictionary of test targets for each symbol
            best_models: Dictionary of best models for each symbol
        """
        try:
            cm_dir = os.path.join(self.output_dir, 'confusion_matrices')
            os.makedirs(cm_dir, exist_ok=True)
            
            for symbol, (model_name, model) in best_models.items():
                if symbol not in X_test_dict or symbol not in y_test_dict:
                    continue
                    
                X_test = X_test_dict[symbol]
                y_test = y_test_dict[symbol]
                
                # Get predictions
                if model_name == 'lightgbm':
                    y_pred = (model.predict(X_test) > 0.5).astype(int)
                elif model_name == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    y_pred = (model.predict(dtest) > 0.5).astype(int)
                elif model_name == 'catboost':
                    y_pred = model.predict(X_test)
                    
                # Plot confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Down/No Change', 'Up'],
                            yticklabels=['Down/No Change', 'Up'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'{symbol} - {model_name} Confusion Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(cm_dir, f'{symbol}_{model_name}_cm.png'))
                plt.close()
                
            logger.info(f"Generated confusion matrices in {cm_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrices: {e}")
    
    def plot_feature_importance(
        self,
        best_models: Dict[str, Tuple[str, Any]],
        X_train_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Plot feature importance for each model
        
        Args:
            best_models: Dictionary of best models for each symbol
            X_train_dict: Dictionary of training features for each symbol
        """
        try:
            fi_dir = os.path.join(self.output_dir, 'feature_importance')
            os.makedirs(fi_dir, exist_ok=True)
            
            for symbol, (model_name, model) in best_models.items():
                if symbol not in X_train_dict:
                    continue
                    
                X_train = X_train_dict[symbol]
                feature_names = X_train.columns
                
                # Get feature importance
                if model_name == 'lightgbm':
                    importance = model.feature_importance(importance_type='gain')
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    })
                elif model_name == 'xgboost':
                    importance = model.get_score(importance_type='gain')
                    # XGBoost uses feature indices, convert to names
                    feature_importance = pd.DataFrame({
                        'feature': [feature_names[int(f.replace('f', ''))] if f.startswith('f') else f for f in importance.keys()],
                        'importance': list(importance.values())
                    })
                elif model_name == 'catboost':
                    importance = model.get_feature_importance()
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    })
                else:
                    continue
                    
                # Sort by importance
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance)
                plt.title(f'{symbol} - {model_name} Feature Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(fi_dir, f'{symbol}_{model_name}_feature_importance.png'))
                plt.close()
                
                # Save to CSV
                feature_importance.to_csv(os.path.join(fi_dir, f'{symbol}_{model_name}_feature_importance.csv'), index=False)
                
            logger.info(f"Generated feature importance plots and CSVs in {fi_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    def generate_training_summary(
        self,
        models_by_symbol: Dict[str, Dict[str, Dict]],
        best_models: Dict[str, Tuple[str, Any]],
        execution_time: float
    ) -> None:
        """
        Generate a summary of the training process
        
        Args:
            models_by_symbol: Dictionary of models for each symbol
            best_models: Dictionary of best models for each symbol
            execution_time: Total execution time in seconds
        """
        try:
            summary_path = os.path.join(self.output_dir, 'training_summary.txt')
            
            with open(summary_path, 'w') as f:
                f.write(f"Training Summary Report\n")
                f.write(f"=====================\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Execution Time: {execution_time:.2f} seconds\n\n")
                
                f.write(f"Models Trained: {len(models_by_symbol)} symbols\n")
                f.write(f"Best Models Selected: {len(best_models)} symbols\n\n")
                
                f.write(f"Best Model Distribution:\n")
                model_counts = {}
                for _, (model_name, _) in best_models.items():
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1
                
                for model_name, count in model_counts.items():
                    f.write(f"  - {model_name}: {count} symbols ({count/len(best_models)*100:.1f}%)\n")
                
                f.write(f"\nSymbol-wise Best Models:\n")
                f.write(f"------------------------\n")
                for symbol, (model_name, _) in sorted(best_models.items()):
                    # Find AUC for this model
                    auc = "N/A"
                    if symbol in models_by_symbol and model_name in models_by_symbol[symbol]:
                        if 'metrics' in models_by_symbol[symbol][model_name]:
                            auc = f"{models_by_symbol[symbol][model_name]['metrics'].get('auc', 'N/A'):.4f}"
                    
                    f.write(f"{symbol}: {model_name} (AUC={auc})\n")
            
            logger.info(f"Generated training summary at {summary_path}")
            
        except Exception as e:
            logger.error(f"Error generating training summary: {e}")
    
    def generate_trading_signals(
        self,
        X_test_dict: Dict[str, pd.DataFrame],
        y_test_dict: Dict[str, Dict[str, pd.Series]],
        best_models: Dict[str, Dict[str, Tuple[str, Any]]],
        dates_dict: Dict[str, pd.Series],
        threshold_buy: float = 0.6,
        threshold_sell: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions for multiple targets
        """
        try:
            all_signals = []
            
            for symbol, target_models in best_models.items():
                if symbol not in X_test_dict or symbol not in dates_dict:
                    logger.warning(f"Skipping signal generation for {symbol} due to missing data")
                    continue
                    
                X_test = X_test_dict[symbol]
                
                # Get dates that align with X_test indices
                try:
                    # Find dates that correspond to the test set
                    symbol_df = dates_dict[symbol].reset_index(drop=True)
                    
                    # Use only the last part corresponding to the test set
                    test_dates = symbol_df.iloc[-len(X_test):].reset_index(drop=True)
                    
                    if len(test_dates) != len(X_test):
                        logger.warning(f"Date length mismatch for {symbol}: dates={len(test_dates)}, X_test={len(X_test)}")
                        # Adjust to the minimum length
                        min_len = min(len(test_dates), len(X_test))
                        test_dates = test_dates.iloc[:min_len]
                        X_test = X_test.iloc[:min_len]
                except Exception as e:
                    logger.error(f"Error aligning dates for {symbol}: {e}")
                    continue
                
                # Make predictions for each target
                predictions = {}
                model_names = {}
                
                try:
                    # Check which targets we have models for
                    target_list = list(target_models.keys())
                    logger.info(f"Symbol {symbol} has models for targets: {target_list}")
                    
                    # Get predictions for primary targets (target and exit_signal)
                    for target_name, (model_name, model) in target_models.items():
                        model_names[target_name] = model_name
                        
                        if model_name == 'lightgbm':
                            predictions[target_name] = model.predict(X_test)
                        elif model_name == 'xgboost':
                            dtest = xgb.DMatrix(X_test)
                            predictions[target_name] = model.predict(dtest)
                        elif model_name == 'catboost':
                            if target_name in ['target', 'exit_signal']:
                                predictions[target_name] = model.predict_proba(X_test)[:, 1]
                            else:
                                predictions[target_name] = model.predict(X_test)
                except Exception as e:
                    logger.error(f"Error making predictions for {symbol}: {e}")
                    continue
                
                # Check which predictions we actually have
                logger.info(f"Symbol {symbol} has predictions for: {list(predictions.keys())}")
                
                # Ensure required targets are available
                required_targets = ['target', 'exit_signal']
                if not all(target in predictions for target in required_targets):
                    logger.warning(f"Symbol {symbol} missing required prediction targets. Available: {list(predictions.keys())}")
                    continue
                
                # Prepare signal data
                signal_data = {
                    'date': test_dates.values,
                    'trading_symbol': [symbol] * len(test_dates),
                    'target_prob': predictions['target'],
                    'exit_signal_prob': predictions['exit_signal'],
                }
                
                # Add additional predictions if available
                for target_name in ['future_close', 'future_return', 'days_to_target']:
                    if target_name in predictions:
                        signal_data[f'predicted_{target_name}'] = predictions[target_name]
                
                # Add model names used
                for target_name, model_name in model_names.items():
                    signal_data[f'{target_name}_model'] = [model_name] * len(test_dates)
                
                # Create DataFrame
                symbol_signals = pd.DataFrame(signal_data)
                
                # Determine signals
                # Default signal is HOLD
                symbol_signals['signal'] = 'HOLD'
                
                # BUY signal: target probability > threshold_buy
                buy_condition = symbol_signals['target_prob'] > threshold_buy
                symbol_signals.loc[buy_condition, 'signal'] = 'BUY'
                
                # SELL signal: exit_signal probability > threshold_sell
                sell_condition = symbol_signals['exit_signal_prob'] > threshold_sell
                symbol_signals.loc[sell_condition, 'signal'] = 'SELL'
                
                # Add expected return if available
                if 'predicted_future_return' in symbol_signals.columns:
                    symbol_signals['expected_return'] = np.nan
                    symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'expected_return'] = \
                        symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'predicted_future_return']
                
                # Add expected days to target if available
                if 'predicted_days_to_target' in symbol_signals.columns:
                    symbol_signals['expected_days'] = np.nan
                    symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'expected_days'] = \
                        symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'predicted_days_to_target']
                
                # Add signal strength (confidence)
                symbol_signals['signal_strength'] = np.nan
                symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'signal_strength'] = \
                    symbol_signals.loc[symbol_signals['signal'] == 'BUY', 'target_prob']
                symbol_signals.loc[symbol_signals['signal'] == 'SELL', 'signal_strength'] = \
                    symbol_signals.loc[symbol_signals['signal'] == 'SELL', 'exit_signal_prob']
                
                # Add to all signals
                all_signals.append(symbol_signals)
            
            if not all_signals:
                logger.warning("No signals generated for any symbol")
                return pd.DataFrame()
                
            # Combine all signals
            signals_df = pd.concat(all_signals, ignore_index=True)
            
            # Sort by date and symbol
            signals_df = signals_df.sort_values(['date', 'trading_symbol'])
            
            logger.info(f"Generated {len(signals_df)} trading signals for {len(all_signals)} symbols")
            
            return signals_df
                
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return pd.DataFrame()
    def backtest_signals(
        self,
        signals_df: pd.DataFrame,
        price_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.05,
        stop_loss_atr_multiple: float = 2.0
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Backtest trading signals
        """
        # Ensure price_data has the necessary columns
        required_cols = ['date', 'trading_symbol', 'open', 'high', 'low', 'close']
        
        if not all(col in price_data.columns for col in required_cols):
            logger.error(f"Price data missing required columns. Available: {price_data.columns.tolist()}")
            # Try to construct any missing columns if possible
            if 'date' in price_data.columns and 'trading_symbol' in price_data.columns:
                if 'open' not in price_data.columns and 'close' in price_data.columns:
                    # Use close as a proxy for open if missing
                    price_data['open'] = price_data['close']
                    logger.warning("Using 'close' as a proxy for missing 'open' prices")
                    
                if 'high' not in price_data.columns and 'close' in price_data.columns:
                    price_data['high'] = price_data['close']
                    logger.warning("Using 'close' as a proxy for missing 'high' prices")
                    
                if 'low' not in price_data.columns and 'close' in price_data.columns:
                    price_data['low'] = price_data['close']
                    logger.warning("Using 'close' as a proxy for missing 'low' prices")
                
                # If still missing required columns, return empty results
                if not all(col in price_data.columns for col in required_cols):
                    logger.error("Cannot proceed with backtest due to missing price data columns")
                    return pd.DataFrame(), {}
        
        # Add natr column if missing (for volatility-based position sizing)
        if 'natr' not in price_data.columns and 'atr_14' in price_data.columns:
            price_data['natr'] = price_data['atr_14'] / price_data['close']
        elif 'natr' not in price_data.columns:
            # Calculate a simple approximation of ATR if missing
            logger.warning("Calculating simple volatility estimate for position sizing")
            price_data['natr'] = (price_data['high'] - price_data['low']) / price_data['close']
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'position_history': []
        }
        
        # Convert dates to datetime if they aren't already
        if not pd.api.types.is_datetime64_any_dtype(signals_df['date']):
            signals_df['date'] = pd.to_datetime(signals_df['date'])
        if not pd.api.types.is_datetime64_any_dtype(price_data['date']):
            price_data['date'] = pd.to_datetime(price_data['date'])
        
        # Merge signals with price data
        backtest_data = signals_df.merge(
            price_data,
            on=['date', 'trading_symbol'],
            how='inner'
        )
        
        # Check if we have data after the merge
        if backtest_data.empty:
            logger.error("No data available for backtest after merging signals with price data")
            return pd.DataFrame(), {}
        
        # Sort by date
        backtest_data = backtest_data.sort_values('date')
        logger.info(f"Backtest data contains {len(backtest_data)} rows from {backtest_data['date'].min()} to {backtest_data['date'].max()}")
        
        # Group by date to process day by day
        for date, day_data in backtest_data.groupby('date'):
            # Process SELL signals first (close positions)
            for _, row in day_data[day_data['signal'] == 'SELL'].iterrows():
                symbol = row['trading_symbol']
                close_price = row['close']
                
                # Check if we have an open position for this symbol
                if symbol in portfolio['positions']:
                    position = portfolio['positions'][symbol]
                    
                    # Calculate profit/loss
                    entry_price = position['entry_price']
                    shares = position['shares']
                    cost_basis = entry_price * shares
                    position_value = close_price * shares
                    profit_loss = position_value - cost_basis
                    
                    # Close position
                    portfolio['cash'] += position_value
                    
                    # Record the closed position
                    closed_position = position.copy()
                    closed_position.update({
                        'exit_date': date,
                        'exit_price': close_price,
                        'exit_reason': 'SELL_SIGNAL',
                        'profit_loss': profit_loss,
                        'return_pct': (profit_loss / cost_basis) * 100
                    })
                    portfolio['position_history'].append(closed_position)
                    
                    # Remove from active positions
                    del portfolio['positions'][symbol]
                    logger.info(f"{date}: Sold {shares} shares of {symbol} at {close_price:.2f} for P/L: {profit_loss:.2f}")
            
            # Update stop-loss for existing positions
            positions_to_close = []
            for symbol, position in portfolio['positions'].items():
                # Find the symbol's data for this day
                symbol_data = day_data[day_data['trading_symbol'] == symbol]
                if not symbol_data.empty:
                    # Update stop-loss if using trailing stop
                    current_price = symbol_data.iloc[0]['close']
                    symbol_natr = symbol_data.iloc[0].get('natr', 0.02)  # Default to 2% if missing
                    
                    # Check if price hit stop-loss
                    if current_price <= position['stop_loss']:
                        positions_to_close.append((symbol, current_price, 'STOP_LOSS'))
            
            # Close positions hit by stop-loss
            for symbol, close_price, reason in positions_to_close:
                position = portfolio['positions'][symbol]
                
                # Calculate profit/loss
                entry_price = position['entry_price']
                shares = position['shares']
                cost_basis = entry_price * shares
                position_value = close_price * shares
                profit_loss = position_value - cost_basis
                
                # Close position
                portfolio['cash'] += position_value
                
                # Record the closed position
                closed_position = position.copy()
                closed_position.update({
                    'exit_date': date,
                    'exit_price': close_price,
                    'exit_reason': reason,
                    'profit_loss': profit_loss,
                    'return_pct': (profit_loss / cost_basis) * 100
                })
                portfolio['position_history'].append(closed_position)
                
                # Remove from active positions
                del portfolio['positions'][symbol]
                logger.info(f"{date}: {reason} triggered for {symbol} at {close_price:.2f} for P/L: {profit_loss:.2f}")
            
            # Process BUY signals
            for _, row in day_data[day_data['signal'] == 'BUY'].iterrows():
                symbol = row['trading_symbol']
                close_price = row['close']
                symbol_natr = row.get('natr', 0.02)  # Default to 2% if missing
                
                # Skip if we already have a position for this symbol
                if symbol in portfolio['positions']:
                    continue
                
                # Calculate position size based on volatility (ATR)
                # Higher ATR (volatility) means smaller position size
                volatility_factor = 1.0 / (1.0 + symbol_natr)
                max_position_value = portfolio['cash'] * position_size_pct * volatility_factor
                
                # Calculate number of shares to buy (minimum 1)
                shares = max(1, int(max_position_value / close_price))
                
                # Calculate cost and update cash
                cost = shares * close_price
                if cost > portfolio['cash']:
                    # Not enough cash, adjust shares
                    shares = int(portfolio['cash'] / close_price)
                    cost = shares * close_price
                
                # Skip if can't buy at least 1 share
                if shares < 1:
                    continue
                
                # Calculate stop loss based on ATR
                stop_loss = close_price - (symbol_natr * close_price * stop_loss_atr_multiple)
                
                # Open new position
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = {
                    'symbol': symbol,
                    'entry_date': date,
                    'entry_price': close_price,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'initial_stop_loss': stop_loss,
                    'cost_basis': cost,
                    'risk_per_share': close_price - stop_loss
                }
                
                logger.info(f"{date}: Bought {shares} shares of {symbol} at {close_price:.2f}, stop at {stop_loss:.2f}")
        
        # Close any remaining positions using the last known price
        for symbol, position in list(portfolio['positions'].items()):
            # Find last day data for this symbol
            symbol_data = backtest_data[backtest_data['trading_symbol'] == symbol].iloc[-1]
            close_price = symbol_data['close']
            
            # Calculate profit/loss
            entry_price = position['entry_price']
            shares = position['shares']
            cost_basis = entry_price * shares
            position_value = close_price * shares
            profit_loss = position_value - cost_basis
            
            # Close position
            portfolio['cash'] += position_value
            
            # Record the closed position
            closed_position = position.copy()
            closed_position.update({
                'exit_date': symbol_data['date'],
                'exit_price': close_price,
                'exit_reason': 'END_OF_BACKTEST',
                'profit_loss': profit_loss,
                'return_pct': (profit_loss / cost_basis) * 100
            })
            portfolio['position_history'].append(closed_position)
            
            # Remove from active positions
            del portfolio['positions'][symbol]
        
        # Create DataFrame from position history
        if portfolio['position_history']:
            position_df = pd.DataFrame(portfolio['position_history'])
        else:
            position_df = pd.DataFrame(columns=[
                'symbol', 'entry_date', 'entry_price', 'shares', 'stop_loss',
                'exit_date', 'exit_price', 'exit_reason', 'profit_loss', 'return_pct'
            ])
        
        # Calculate backtest metrics
        metrics = {}
        if not position_df.empty:
            # Basic metrics
            metrics['total_trades'] = len(position_df)
            metrics['winning_trades'] = len(position_df[position_df['profit_loss'] > 0])
            metrics['losing_trades'] = len(position_df[position_df['profit_loss'] <= 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            # Profit metrics
            metrics['total_profit'] = position_df[position_df['profit_loss'] > 0]['profit_loss'].sum()
            metrics['total_loss'] = position_df[position_df['profit_loss'] <= 0]['profit_loss'].sum()
            metrics['net_profit'] = metrics['total_profit'] + metrics['total_loss']
            metrics['profit_factor'] = abs(metrics['total_profit'] / metrics['total_loss']) if metrics['total_loss'] != 0 else float('inf')
            
            # Return metrics
            metrics['avg_return_pct'] = position_df['return_pct'].mean()
            metrics['return_std'] = position_df['return_pct'].std()
            
            # Final equity
            metrics['final_equity'] = initial_capital + metrics['net_profit']
            metrics['total_return_pct'] = (metrics['final_equity'] / initial_capital - 1) * 100
        
        return position_df, metrics

    def visualize_backtest(
        self,
        position_df: pd.DataFrame,
        metrics: Dict[str, float],
        price_data: pd.DataFrame
    ) -> None:
        """
        Visualize backtest results
        
        Args:
            position_df: DataFrame with position history
            metrics: Dictionary with backtest metrics
            price_data: DataFrame with price data
        """
        if position_df.empty:
            logger.warning("No positions to visualize")
            return
        
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Profit/Loss Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(position_df['profit_loss'], bins=20, kde=True)
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss ($)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(vis_dir, 'pnl_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Cumulative Returns
        position_df = position_df.sort_values('exit_date')
        position_df['cumulative_pnl'] = position_df['profit_loss'].cumsum()
        
        plt.figure(figsize=(12, 6))
        plt.plot(position_df['exit_date'], position_df['cumulative_pnl'])
        plt.title('Cumulative Profit/Loss')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P/L ($)')
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, 'cumulative_pnl.png'), dpi=300)
        plt.close()
        
        # 3. Win/Loss by Symbol
        symbol_performance = position_df.groupby('symbol').agg({
            'profit_loss': 'sum',
            'return_pct': 'mean',
            'symbol': 'count'
        }).rename(columns={'symbol': 'trade_count'}).sort_values('profit_loss', ascending=False)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(symbol_performance.index, symbol_performance['profit_loss'])
        
        # Color bars based on profit/loss
        for i, bar in enumerate(bars):
            bar.set_color('green' if symbol_performance['profit_loss'].iloc[i] > 0 else 'red')
        
        plt.title('Profit/Loss by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Profit/Loss ($)')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'pnl_by_symbol.png'), dpi=300)
        plt.close()
        
        # 4. Summary Metrics Table
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        metrics_text = (
            f"Backtest Summary Metrics\n"
            f"------------------------\n\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']*100:.2f}%)\n"
            f"Losing Trades: {metrics['losing_trades']}\n\n"
            f"Total Profit: ${metrics['total_profit']:.2f}\n"
            f"Total Loss: ${metrics['total_loss']:.2f}\n"
            f"Net Profit: ${metrics['net_profit']:.2f}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n\n"
            f"Average Return: {metrics['avg_return_pct']:.2f}%\n"
            f"Return Std Dev: {metrics['return_std']:.2f}%\n\n"
            f"Final Equity: ${metrics['final_equity']:.2f}\n"
            f"Total Return: {metrics['total_return_pct']:.2f}%"
        )
        plt.text(0.1, 0.1, metrics_text, fontsize=12, family='monospace')
        plt.savefig(os.path.join(vis_dir, 'metrics_summary.png'), dpi=300)
        plt.close()
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_csv(os.path.join(vis_dir, 'metrics_summary.csv'), index=False)
        
        logger.info(f"Backtest visualizations saved to {vis_dir}")
    
    def save_performance_metrics_multi_target(
    self,
    models_by_symbol_target: Dict[str, Dict[str, Dict[str, Dict]]]
    ) -> None:
        """
        Save detailed performance metrics to CSV for multiple targets
        
        Args:
            models_by_symbol_target: Dictionary of models and metrics for each symbol and target
        """
        try:
            metrics_records = []
            
            for symbol, targets_dict in models_by_symbol_target.items():
                for target_name, models in targets_dict.items():
                    for model_name, model_info in models.items():
                        if 'metrics' in model_info:
                            record = {
                                'symbol': symbol,
                                'target': target_name,
                                'model': model_name,
                                'training_time': model_info.get('training_time', float('nan'))
                            }
                            
                            # Add all metrics (different for classification vs regression)
                            for metric_name, metric_value in model_info['metrics'].items():
                                record[metric_name] = metric_value
                            
                            metrics_records.append(record)
            
            metrics_df = pd.DataFrame(metrics_records)
            metrics_path = os.path.join(self.output_dir, 'model_performance_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved performance metrics to {metrics_path}")
                
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def plot_feature_importance_multi_target(
        self,
        best_models: Dict[str, Dict[str, Tuple[str, Any]]],
        X_train_dict: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Plot feature importance for each model and target
        
        Args:
            best_models: Dictionary of best models for each symbol and target
            X_train_dict: Dictionary of training features for each symbol
        """
        try:
            fi_dir = os.path.join(self.output_dir, 'feature_importance')
            os.makedirs(fi_dir, exist_ok=True)
            
            # Classification and regression targets
            classification_targets = ['target', 'exit_signal']
            regression_targets = ['future_close', 'future_return', 'days_to_target']
            
            # Track feature importance across targets
            global_importance = {
                'classification': defaultdict(float),
                'regression': defaultdict(float)
            }
            
            feature_count = {
                'classification': defaultdict(int),
                'regression': defaultdict(int)
            }
            
            for symbol, targets_dict in best_models.items():
                if symbol not in X_train_dict:
                    continue
                    
                X_train = X_train_dict[symbol]
                feature_names = X_train.columns
                
                for target_name, (model_name, model) in targets_dict.items():
                    # Get feature importance
                    if model_name == 'lightgbm':
                        importance = model.feature_importance(importance_type='gain')
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance
                        })
                    elif model_name == 'xgboost':
                        importance = model.get_score(importance_type='gain')
                        # XGBoost uses feature indices, convert to names
                        feature_importance = pd.DataFrame({
                            'feature': [feature_names[int(f.replace('f', ''))] if f.startswith('f') else f for f in importance.keys()],
                            'importance': list(importance.values())
                        })
                    elif model_name == 'catboost':
                        importance = model.get_feature_importance()
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importance
                        })
                    else:
                        continue
                        
                    # Sort by importance
                    feature_importance = feature_importance.sort_values('importance', ascending=False)
                    
                    # Determine target type
                    target_type = 'classification' if target_name in classification_targets else 'regression'
                    
                    # Add to global importance
                    for _, row in feature_importance.iterrows():
                        global_importance[target_type][row['feature']] += row['importance']
                        feature_count[target_type][row['feature']] += 1
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
                    plt.title(f'{symbol} - {target_name} - {model_name} Feature Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(fi_dir, f'{symbol}_{target_name}_{model_name}_feature_importance.png'))
                    plt.close()
                    
                    # Save to CSV
                    feature_importance.to_csv(os.path.join(fi_dir, f'{symbol}_{target_name}_{model_name}_feature_importance.csv'), index=False)
            
            # Plot global feature importance for classification and regression targets
            for target_type in ['classification', 'regression']:
                if not global_importance[target_type]:
                    continue
                    
                # Calculate average importance
                avg_importance = {feature: importance / feature_count[target_type][feature] 
                                for feature, importance in global_importance[target_type].items()}
                
                # Convert to DataFrame and sort
                global_fi_df = pd.DataFrame({
                    'feature': list(avg_importance.keys()),
                    'importance': list(avg_importance.values())
                }).sort_values('importance', ascending=False)
                
                # Plot top 30 features
                plt.figure(figsize=(12, 10))
                sns.barplot(x='importance', y='feature', data=global_fi_df.head(30))
                plt.title(f'Global Feature Importance for {target_type.capitalize()} Targets')
                plt.tight_layout()
                plt.savefig(os.path.join(fi_dir, f'global_{target_type}_feature_importance.png'))
                plt.close()
                
                # Save to CSV
                global_fi_df.to_csv(os.path.join(fi_dir, f'global_{target_type}_feature_importance.csv'), index=False)
            
            logger.info(f"Generated feature importance plots and CSVs in {fi_dir}")
                
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    def generate_training_summary_multi_target(
        self,
        models_by_symbol_target: Dict[str, Dict[str, Dict[str, Dict]]],
        best_models: Dict[str, Dict[str, Tuple[str, Any]]],
        execution_time: float
    ) -> None:
        """
        Generate a summary of the training process for multiple targets
        
        Args:
            models_by_symbol_target: Dictionary of models for each symbol and target
            best_models: Dictionary of best models for each symbol and target
            execution_time: Total execution time in seconds
        """
        try:
            summary_path = os.path.join(self.output_dir, 'training_summary.txt')
            
            # Classification and regression targets
            classification_targets = ['target', 'exit_signal']
            regression_targets = ['future_close', 'future_return', 'days_to_target']
            
            with open(summary_path, 'w') as f:
                f.write(f"Multi-Target Training Summary Report\n")
                f.write(f"===============================\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Execution Time: {execution_time:.2f} seconds\n\n")
                
                f.write(f"Models Trained: {len(models_by_symbol_target)} symbols\n")
                f.write(f"Best Models Selected: {len(best_models)} symbols\n\n")
                
                # Get count of models by target and algorithm
                model_counts = defaultdict(lambda: defaultdict(int))
                
                for symbol, targets_dict in best_models.items():
                    for target_name, (model_name, _) in targets_dict.items():
                        model_counts[target_name][model_name] += 1
                
                f.write(f"Best Model Distribution by Target:\n")
                f.write(f"-------------------------------\n")
                
                for target_name, model_dict in model_counts.items():
                    f.write(f"\n{target_name}:\n")
                    total = sum(model_dict.values())
                    
                    for model_name, count in model_dict.items():
                        f.write(f"  - {model_name}: {count} symbols ({count/total*100:.1f}%)\n")
                
                # Performance summary by target type
                f.write(f"\n\nPerformance Summary by Target Type:\n")
                f.write(f"-------------------------------\n")
                
                # Process classification targets
                f.write(f"\nClassification Targets:\n")
                for target_name in classification_targets:
                    f.write(f"\n{target_name}:\n")
                    
                    # Collect metrics across symbols
                    auc_scores = []
                    f1_scores = []
                    
                    for symbol, targets_dict in models_by_symbol_target.items():
                        if target_name in targets_dict:
                            best_model_name = best_models[symbol][target_name][0] if symbol in best_models and target_name in best_models[symbol] else None
                            
                            if best_model_name and best_model_name in targets_dict[target_name]:
                                metrics = targets_dict[target_name][best_model_name].get('metrics', {})
                                auc = metrics.get('auc')
                                f1 = metrics.get('f1')
                                
                                if auc is not None:
                                    auc_scores.append(auc)
                                if f1 is not None:
                                    f1_scores.append(f1)
                    
                    if auc_scores:
                        f.write(f"  - Average AUC: {np.mean(auc_scores):.4f} (min: {min(auc_scores):.4f}, max: {max(auc_scores):.4f})\n")
                    if f1_scores:
                        f.write(f"  - Average F1: {np.mean(f1_scores):.4f} (min: {min(f1_scores):.4f}, max: {max(f1_scores):.4f})\n")
                
                # Process regression targets
                f.write(f"\nRegression Targets:\n")
                for target_name in regression_targets:
                    f.write(f"\n{target_name}:\n")
                    
                    # Collect metrics across symbols
                    r2_scores = []
                    rmse_scores = []
                    
                    for symbol, targets_dict in models_by_symbol_target.items():
                        if target_name in targets_dict:
                            best_model_name = best_models[symbol][target_name][0] if symbol in best_models and target_name in best_models[symbol] else None
                            
                            if best_model_name and best_model_name in targets_dict[target_name]:
                                metrics = targets_dict[target_name][best_model_name].get('metrics', {})
                                r2 = metrics.get('r2')
                                rmse = metrics.get('rmse')
                                
                                if r2 is not None:
                                    r2_scores.append(r2)
                                if rmse is not None:
                                    rmse_scores.append(rmse)
                    
                    if r2_scores:
                        f.write(f"  - Average R²: {np.mean(r2_scores):.4f} (min: {min(r2_scores):.4f}, max: {max(r2_scores):.4f})\n")
                    if rmse_scores:
                        f.write(f"  - Average RMSE: {np.mean(rmse_scores):.4f} (min: {min(rmse_scores):.4f}, max: {max(rmse_scores):.4f})\n")
                
                # Symbol-wise Best Models
                f.write(f"\n\nSymbol-wise Best Models:\n")
                f.write(f"------------------------\n")
                
                for symbol, targets_dict in sorted(best_models.items()):
                    f.write(f"\n{symbol}:\n")
                    
                    for target_name, (model_name, _) in sorted(targets_dict.items()):
                        # Find metric for this model
                        metric_value = "N/A"
                        
                        if (symbol in models_by_symbol_target and
                            target_name in models_by_symbol_target[symbol] and
                            model_name in models_by_symbol_target[symbol][target_name] and
                            'metrics' in models_by_symbol_target[symbol][target_name][model_name]):
                            
                            metrics = models_by_symbol_target[symbol][target_name][model_name]['metrics']
                            
                            if target_name in classification_targets:
                                metric_name = "AUC"
                                metric_value = f"{metrics.get('auc', 'N/A'):.4f}"
                            else:
                                metric_name = "R²"
                                metric_value = f"{metrics.get('r2', 'N/A'):.4f}"
                        else:
                            metric_name = "Score"
                        
                        f.write(f"  - {target_name}: {model_name} ({metric_name}={metric_value})\n")
            
            logger.info(f"Generated training summary at {summary_path}")
                
        except Exception as e:
            logger.error(f"Error generating training summary: {e}")

    def run_training_pipeline(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        test_size: float = 0.2,
        include_engineered: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data loading to model evaluation for multiple targets
        """
        try:
            results = {}
            
            # 1. Load data
            df = self.load_data(start_date, end_date, symbols)
            if df.empty:
                logger.error("Failed to load data. Exiting pipeline.")
                return results
            
            # Save symbol dates for later use
            symbols_dates = {}
            for symbol in df['trading_symbol'].unique():
                symbol_df = df[df['trading_symbol'] == symbol]
                symbols_dates[symbol] = symbol_df['date']
            
            # 2. Engineer additional features
            if include_engineered:
                df = self.engineer_additional_features(df)
            
            # 3. Prepare features and multiple targets
            X_dict, y_dict = self.prepare_features_and_target(df, include_engineered)
            
            # 4. Split data into train and test sets
            X_train_dict, X_test_dict, y_train_dict, y_test_dict = self.train_test_split_time_series(
                X_dict, y_dict, test_size
            )
            
            # 5. Handle class imbalance for classification targets
            X_train_resampled_dict, y_train_resampled_dict = self.handle_class_imbalance(
                X_train_dict, y_train_dict
            )
            
            # 6. Train models for each symbol and target
            models_by_symbol_target = {}
            
            for symbol in X_train_resampled_dict:
                # Check if we have enough data
                if len(X_train_resampled_dict[symbol]) < 100 or len(X_test_dict[symbol]) < 30:
                    logger.warning(f"Symbol {symbol} has insufficient data for training. Skipping.")
                    continue
                
                # Set aside a validation set (last 20% of train data)
                val_size = int(len(X_train_resampled_dict[symbol]) * 0.2)
                X_val = X_train_resampled_dict[symbol].iloc[-val_size:].copy()
                X_train = X_train_resampled_dict[symbol].iloc[:-val_size].copy()
                
                # Prepare validation targets
                # When preparing validation datasets, respect all the targets that exist for this symbol
                y_val_dict_symbol = {}
                y_train_dict_symbol = {}

                for target_name, y_train in y_train_resampled_dict[symbol].items():
                    if len(y_train) > len(X_train_resampled_dict[symbol]):
                        y_train = y_train.iloc[:len(X_train_resampled_dict[symbol])]
                    
                    y_val_dict_symbol[target_name] = y_train.iloc[-val_size:].copy()
                    y_train_dict_symbol[target_name] = y_train.iloc[:-val_size].copy()
                    
                    logger.info(f"Prepared {len(y_train_dict_symbol[target_name])} training samples for {symbol} - {target_name}")
                
                # Train models for all targets
                models_by_target = self.train_models_for_symbol_multi_target(
                    symbol, X_train, y_train_dict_symbol, X_val, y_val_dict_symbol
                )
                
                if models_by_target:
                    models_by_symbol_target[symbol] = models_by_target
            
            # 7. Find the best model for each symbol and target
            best_models = self.find_best_models(models_by_symbol_target)
            
            # 8. Save models
            self.save_models(best_models)
            
            # 9. Save performance metrics
            self.save_performance_metrics_multi_target(models_by_symbol_target)
            
            # 10. Plot feature importance for each target type
            self.plot_feature_importance_multi_target(best_models, X_train_resampled_dict)
            
            # 11. Generate training summary
            execution_time = time.time() - self.start_time
            self.generate_training_summary_multi_target(models_by_symbol_target, best_models, execution_time)
            
            # 12. Generate trading signals for test set
            signals_df = self.generate_trading_signals(
                X_test_dict, y_test_dict, best_models, symbols_dates
            )
            
            # 13. Save signals
            if not signals_df.empty:
                signals_path = os.path.join(self.output_dir, 'trading_signals.csv')
                signals_df.to_csv(signals_path, index=False)
                logger.info(f"Saved trading signals to {signals_path}")
                
                # 14. Run backtest on the generated signals
                # Load price data for backtest period (using the same test data period)
                test_start_date = signals_df['date'].min().strftime('%Y-%m-%d')
                test_end_date = signals_df['date'].max().strftime('%Y-%m-%d')
                
                logger.info(f"Running backtest from {test_start_date} to {test_end_date}")
                
                # Get price data for backtest
                price_data = self.load_data(test_start_date, test_end_date, symbols)
                
                # Run backtest
                backtest_results = self.run_backtest(signals_df, price_data, initial_capital=100000.0)
                
                # Store backtest results
                results['backtest_results'] = backtest_results
            
            # Store results
            results['models_by_symbol_target'] = models_by_symbol_target
            results['best_models'] = best_models
            results['signals_df'] = signals_df
            
            # Log execution time
            execution_time = time.time() - self.start_time
            logger.info(f"Training pipeline completed in {execution_time:.2f} seconds")
            
            return results
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}", exc_info=True)
            return {}

    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Run a backtest using generated signals
        
        Args:
            signals_df: DataFrame with trading signals
            price_data: DataFrame with price data (if None, will be loaded from DB)
            start_date: Start date for backtest (if price_data is None)
            end_date: End date for backtest (if price_data is None)
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Load price data if not provided
            if price_data is None:
                if start_date is None or end_date is None:
                    # Use dates from signals
                    start_date = signals_df['date'].min().strftime('%Y-%m-%d')
                    end_date = signals_df['date'].max().strftime('%Y-%m-%d')
                
                # Get unique symbols from signals
                symbols = signals_df['trading_symbol'].unique().tolist()
                
                # Load price data
                price_data = self.load_data(start_date, end_date, symbols)
            
            # Run backtest
            position_df, metrics = self.backtest_signals(
                signals_df, price_data, initial_capital
            )
            
            # Visualize backtest results
            self.visualize_backtest(position_df, metrics, price_data)
            
            # Save backtest results
            position_df.to_csv(os.path.join(self.output_dir, 'backtest_positions.csv'), index=False)
            
            # Return results
            return {
                'position_df': position_df,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
            return {}

def main():
    """Main function to run the trading model pipeline"""
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'dhan_hq',
        'password': 'Passw0rd@098',
        'database': 'dhanhq_db',
        'auth_plugin': 'mysql_native_password',
        'use_pure': True
    }
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trading_models_{timestamp}")
    
    # Create trainer
    trainer = TradingModelTrainer(db_config, output_dir)
    
    # Define date range for training (use the last 2 years of data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Define symbols to train on (use liquid stocks from NSE)
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
        'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'KOTAKBANK',
        'ITC', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
        'SUNPHARMA', 'TATAMOTORS', 'ULTRACEMCO', 'ADANIENT', 'WIPRO'
    ]
    
    # Run training pipeline
    logger.info(f"Starting training pipeline for {len(symbols)} symbols from {start_date} to {end_date}")
    results = trainer.run_training_pipeline(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        test_size=0.2,
        include_engineered=True
    )
    
    # Print summary
    if 'best_models' in results:
        logger.info("\nTraining Results Summary:")
        logger.info(f"Total symbols trained: {len(results.get('models_by_symbol_target', {}))}")
        logger.info(f"Best models saved: {len(results.get('best_models', {}))}")
        
        if 'signals_df' in results and not results['signals_df'].empty:
            signal_counts = results['signals_df']['signal'].value_counts()
            logger.info("\nSignal Distribution:")
            for signal, count in signal_counts.items():
                logger.info(f"  {signal}: {count}")
        
        # Print backtest results if available
        if 'backtest_results' in results and results['backtest_results']:
            backtest_metrics = results['backtest_results'].get('metrics', {})
            logger.info("\nBacktest Results:")
            logger.info(f"Total Trades: {backtest_metrics.get('total_trades', 0)}")
            logger.info(f"Win Rate: {backtest_metrics.get('win_rate', 0) * 100:.2f}%")
            logger.info(f"Net Profit: ${backtest_metrics.get('net_profit', 0):.2f}")
            logger.info(f"Profit Factor: {backtest_metrics.get('profit_factor', 0):.2f}")
            logger.info(f"Final Equity: ${backtest_metrics.get('final_equity', 0):.2f}")
            logger.info(f"Total Return: {backtest_metrics.get('total_return_pct', 0):.2f}%")
    
    logger.info(f"Trading model pipeline completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()