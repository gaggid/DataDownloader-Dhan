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
        
        # Define the optimal features based on our analysis
        self.optimal_features = [
            'return_90', 'trix', 'di_minus_14', 'return_180', 'sma_200',
            'return_365', 'return_55', 'macd_histogram', 'natr', 'cci_20'
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
            'return_ratio_90_180', 'return_ratio_90_365', 'sma_return_ratio',
            'trix_direction', 'return_90_direction', 'macd_hist_direction',
            'return_90_risk_adj', 'trend_strength', 'natr_trend', 'di_minus_change'
        ]
        
        # Initialize columns first
        for col in engineered_cols:
            df_new[col] = np.nan
        
        # Group by symbol to ensure calculations are within each symbol
        for symbol in df_new['trading_symbol'].unique():
            symbol_mask = df_new['trading_symbol'] == symbol
            symbol_data = df_new.loc[symbol_mask].copy()
            
            # 1. Ratio features
            symbol_data['return_ratio_90_180'] = symbol_data['return_90'] / (symbol_data['return_180'] + 1e-6)
            symbol_data['return_ratio_90_365'] = symbol_data['return_90'] / (symbol_data['return_365'] + 1e-6)
            symbol_data['sma_return_ratio'] = symbol_data['return_90'] / (symbol_data['sma_200'] + 1e-6)
            
            # 2. Trend direction features
            symbol_data['trix_direction'] = np.sign(symbol_data['trix'])
            symbol_data['return_90_direction'] = np.sign(symbol_data['return_90'])
            symbol_data['macd_hist_direction'] = np.sign(symbol_data['macd_histogram'])
            
            # 3. Volatility-adjusted returns
            symbol_data['return_90_risk_adj'] = symbol_data['return_90'] / (symbol_data['natr'] + 1e-6)
            
            # 4. Combined trend strength indicator
            symbol_data['trend_strength'] = (
                np.sign(symbol_data['return_90']) +
                np.sign(symbol_data['trix']) +
                np.sign(symbol_data['return_180'])
            )
            
            # 5. Volatility trend
            symbol_data['natr_trend'] = symbol_data['natr'] / symbol_data['natr'].rolling(10).mean() - 1
            
            # 6. Momentum change rate
            symbol_data['di_minus_change'] = symbol_data['di_minus_14'].pct_change(5)
            
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
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Prepare features and target variables for each symbol
        
        Args:
            df: Input DataFrame
            include_engineered: Whether to include engineered features
            
        Returns:
            Two dictionaries of DataFrames/Series for each symbol (X and y)
        """
        if df.empty:
            logger.error("DataFrame is empty")
            return {}, {}
        
        # Prepare feature list
        feature_cols = self.optimal_features.copy()
        
        if include_engineered:
            engineered_cols = [
                'return_ratio_90_180', 'return_ratio_90_365', 'sma_return_ratio',
                'trix_direction', 'return_90_direction', 'macd_hist_direction',
                'return_90_risk_adj', 'trend_strength', 'natr_trend', 'di_minus_change'
            ]
            feature_cols.extend(engineered_cols)
        
        # Dictionaries to store X and y for each symbol
        X_dict = {}
        y_dict = {}
        
        # Process each symbol separately
        for symbol in df['trading_symbol'].unique():
            symbol_df = df[df['trading_symbol'] == symbol].copy()
            
            # Ensure chronological order
            symbol_df = symbol_df.sort_values('date')
            
            # Extract features and target
            X = symbol_df[feature_cols]
            y = symbol_df['target']
            
            # Store in dictionaries
            X_dict[symbol] = X
            y_dict[symbol] = y
        
        logger.info(f"Prepared data for {len(X_dict)} symbols using {len(feature_cols)} features")
        return X_dict, y_dict
    
    def train_test_split_time_series(
        self,
        X_dict: Dict[str, pd.DataFrame],
        y_dict: Dict[str, pd.Series],
        test_size: float = 0.2
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Split data into train and test sets respecting time order
        
        Args:
            X_dict: Dictionary of feature DataFrames for each symbol
            y_dict: Dictionary of target Series for each symbol
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
            y = y_dict[symbol]
            
            # Calculate split point (respecting time order)
            split_idx = int(len(X) * (1 - test_size))
            
            # Split features and target
            X_train_dict[symbol] = X.iloc[:split_idx].copy()
            X_test_dict[symbol] = X.iloc[split_idx:].copy()
            y_train_dict[symbol] = y.iloc[:split_idx].copy()
            y_test_dict[symbol] = y.iloc[split_idx:].copy()
        
        logger.info(f"Split data into train/test sets with test_size={test_size}")
        return X_train_dict, X_test_dict, y_train_dict, y_test_dict
    
    def handle_class_imbalance(
        self,
        X_train_dict: Dict[str, pd.DataFrame],
        y_train_dict: Dict[str, pd.Series]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Apply SMOTE to handle class imbalance in training data
        
        Args:
            X_train_dict: Dictionary of training feature DataFrames
            y_train_dict: Dictionary of training target Series
            
        Returns:
            X_resampled_dict, y_resampled_dict
        """
        X_resampled_dict = {}
        y_resampled_dict = {}
        
        for symbol in X_train_dict.keys():
            X_train = X_train_dict[symbol]
            y_train = y_train_dict[symbol]
            
            # Check class distribution
            class_counts = y_train.value_counts()
            logger.info(f"Symbol {symbol} class distribution before resampling: {dict(class_counts)}")
            
            # Only apply SMOTE if both classes have at least 5 samples
            if len(class_counts) > 1 and all(class_counts > 5):
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                # Convert back to DataFrame/Series
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_resampled = pd.Series(y_resampled, name='target')
                
                logger.info(f"Symbol {symbol} applied SMOTE: {len(X_train)} → {len(X_resampled)} samples")
            else:
                logger.warning(f"Symbol {symbol} skipped SMOTE (insufficient samples)")
                X_resampled = X_train
                y_resampled = y_train
            
            X_resampled_dict[symbol] = X_resampled
            y_resampled_dict[symbol] = y_resampled
        
        return X_resampled_dict, y_resampled_dict
    
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
    
    def find_best_model(
        self,
        models_by_symbol: Dict[str, Dict[str, Dict]]
    ) -> Dict[str, Tuple[str, Any]]:
        """
        Find the best model for each symbol based on AUC score
        
        Args:
            models_by_symbol: Dictionary of models for each symbol
            
        Returns:
            Dictionary with the best model for each symbol
        """
        best_models = {}
        
        for symbol, models in models_by_symbol.items():
            best_model_name = None
            best_model = None
            best_score = -1
            
            for model_name, model_info in models.items():
                if 'metrics' in model_info and model_info['metrics']['auc'] > best_score:
                    best_score = model_info['metrics']['auc']
                    best_model_name = model_name
                    best_model = model_info['model']
            
            if best_model_name and best_model:
                best_models[symbol] = (best_model_name, best_model)
                logger.info(f"{symbol}: Best model is {best_model_name} with AUC={best_score:.4f}")
        
        return best_models
    
    def save_models(
        self,
        best_models: Dict[str, Tuple[str, Any]]
    ) -> bool:
        """
        Save the best models to disk
        
        Args:
            best_models: Dictionary of best models for each symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            models_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model info
            model_info = {}
            
            for symbol, (model_name, model) in best_models.items():
                model_path = os.path.join(models_dir, f"{symbol}_{model_name}.model")
                
                if model_name == 'lightgbm':
                    model.save_model(model_path)
                elif model_name == 'xgboost':
                    model.save_model(model_path)
                elif model_name == 'catboost':
                    model.save_model(model_path)
                
                model_info[symbol] = {
                    'model_type': model_name,
                    'model_path': model_path
                }
            
            # Save model info
            info_path = os.path.join(self.output_dir, 'model_info.joblib')
            joblib.dump(model_info, info_path)
            
            logger.info(f"Saved {len(best_models)} models to {models_dir}")
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
        y_test_dict: Dict[str, pd.Series],
        best_models: Dict[str, Tuple[str, Any]],
        dates_dict: Dict[str, pd.Series],
        threshold_buy: float = 0.6,
        threshold_sell: float = 0.7
    ) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions
        
        Args:
            X_test_dict: Dictionary of test features for each symbol
            y_test_dict: Dictionary of test targets for each symbol
            best_models: Dictionary of best models for each symbol
            dates_dict: Dictionary of dates for each symbol
            threshold_buy: Probability threshold for buy signals
            threshold_sell: Probability threshold for sell signals
            
        Returns:
            DataFrame with trading signals
        """
        try:
            all_signals = []
            
            for symbol, (model_name, model) in best_models.items():
                if symbol not in X_test_dict or symbol not in dates_dict or symbol not in y_test_dict:
                    logger.warning(f"Skipping signal generation for {symbol} due to missing data")
                    continue
                    
                X_test = X_test_dict[symbol]
                y_test = y_test_dict[symbol]
                
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
                        y_test = y_test.iloc[:min_len]
                except Exception as e:
                    logger.error(f"Error aligning dates for {symbol}: {e}")
                    continue
                
                # Make predictions
                try:
                    if model_name == 'lightgbm':
                        y_pred_proba = model.predict(X_test)
                    elif model_name == 'xgboost':
                        dtest = xgb.DMatrix(X_test)
                        y_pred_proba = model.predict(dtest)
                    elif model_name == 'catboost':
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        logger.warning(f"Unknown model type for {symbol}: {model_name}")
                        continue
                except Exception as e:
                    logger.error(f"Error making predictions for {symbol}: {e}")
                    continue
                
                # Check trend confirmation signals
                trend_signals = np.zeros(len(X_test))
                if 'return_90' in X_test.columns:
                    trend_signals += (X_test['return_90'] > 0).astype(int).values
                if 'trix' in X_test.columns:
                    trend_signals += (X_test['trix'] > 0).astype(int).values
                if 'sma_200' in X_test.columns:
                    sma_trend = (X_test['sma_200'] > X_test['sma_200'].shift(1)).fillna(0).astype(int).values
                    trend_signals += sma_trend
                
                # Ensure all arrays have the same length
                min_length = min(len(test_dates), len(y_test), len(y_pred_proba), len(trend_signals))
                
                # Create signal data
                signal_data = {
                    'date': test_dates.iloc[:min_length].values,
                    'trading_symbol': [symbol] * min_length,
                    'actual': y_test.iloc[:min_length].values,
                    'predicted_proba': y_pred_proba[:min_length],
                    'trend_confirmations': trend_signals[:min_length],
                    'model_name': [model_name] * min_length
                }
                
                # Create DataFrame
                symbol_signals = pd.DataFrame(signal_data)
                
                # Determine signals
                symbol_signals['signal'] = 'HOLD'
                
                # BUY signal: probability > threshold_buy AND at least 2 trend confirmations
                buy_condition = (symbol_signals['predicted_proba'] > threshold_buy) & (symbol_signals['trend_confirmations'] >= 2)
                symbol_signals.loc[buy_condition, 'signal'] = 'BUY'
                
                # SELL signal: probability < (1 - threshold_sell)
                sell_condition = symbol_signals['predicted_proba'] < (1 - threshold_sell)
                symbol_signals.loc[sell_condition, 'signal'] = 'SELL'
                
                # Add to all signals
                all_signals.append(symbol_signals)
            
            if not all_signals:
                logger.warning("No signals generated for any symbol")
                return pd.DataFrame()
                
            # Combine all signals
            signals_df = pd.concat(all_signals, ignore_index=True)
            
            # Sort by date and symbol
            signals_df = signals_df.sort_values(['date', 'trading_symbol'])
            
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
        
        Args:
            signals_df: DataFrame with trading signals
            price_data: DataFrame with price data (must have date, trading_symbol, open, high, low, close, natr)
            initial_capital: Initial capital for the backtest
            position_size_pct: Maximum percentage of capital per position
            stop_loss_atr_multiple: Multiple of ATR for stop loss
            
        Returns:
            DataFrame with backtest results and dictionary with backtest metrics
        """
        # Ensure price_data has the necessary columns
        required_cols = ['date', 'trading_symbol', 'open', 'high', 'low', 'close', 'natr']
        if not all(col in price_data.columns for col in required_cols):
            logger.error(f"Price data missing required columns: {required_cols}")
            return pd.DataFrame(), {}
        
        # Initialize portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'position_history': []
        }
        
        # Merge signals with price data
        backtest_data = signals_df.merge(
            price_data[required_cols],
            on=['date', 'trading_symbol'],
            how='inner'
        )
        
        # Sort by date
        backtest_data = backtest_data.sort_values('date')
        
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
                    symbol_natr = symbol_data.iloc[0]['natr']
                    
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
                symbol_natr = row['natr']
                
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

    def run_training_pipeline(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        test_size: float = 0.2,
        include_engineered: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data loading to model evaluation
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of trading symbols to include (None for all)
            test_size: Fraction of data to use for testing
            include_engineered: Whether to include engineered features
            
        Returns:
            Dictionary with results including models and evaluation metrics
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
            
            # 3. Prepare features and target
            X_dict, y_dict = self.prepare_features_and_target(df, include_engineered)
            
            # 4. Split data into train and test sets
            X_train_dict, X_test_dict, y_train_dict, y_test_dict = self.train_test_split_time_series(
                X_dict, y_dict, test_size
            )
            
            # 5. Handle class imbalance
            X_train_resampled_dict, y_train_resampled_dict = self.handle_class_imbalance(
                X_train_dict, y_train_dict
            )
            
            # 6. Train models for each symbol
            models_by_symbol = {}
            
            for symbol in X_train_resampled_dict:
                # Check if we have enough data
                if len(X_train_resampled_dict[symbol]) < 100 or len(X_test_dict[symbol]) < 30:
                    logger.warning(f"Symbol {symbol} has insufficient data for training. Skipping.")
                    continue
                
                # Set aside a validation set (last 20% of train data)
                val_size = int(len(X_train_resampled_dict[symbol]) * 0.2)
                X_val = X_train_resampled_dict[symbol].iloc[-val_size:].copy()
                y_val = y_train_resampled_dict[symbol].iloc[-val_size:].copy()
                X_train = X_train_resampled_dict[symbol].iloc[:-val_size].copy()
                y_train = y_train_resampled_dict[symbol].iloc[:-val_size].copy()
                
                # Train models
                models = self.train_models_for_symbol(
                    symbol, X_train, y_train, X_val, y_val
                )
                
                if models:
                    models_by_symbol[symbol] = models
            
            # 7. Find the best model for each symbol
            best_models = self.find_best_model(models_by_symbol)
            
            # 8. Save models
            self.save_models(best_models)
            
            # 9. Save performance metrics
            self.save_performance_metrics(models_by_symbol)
            
            # 10. Plot confusion matrices
            self.plot_confusion_matrices(X_test_dict, y_test_dict, best_models)
            
            # 11. Plot feature importance
            self.plot_feature_importance(best_models, X_train_resampled_dict)
            
            # 12. Generate training summary
            execution_time = time.time() - self.start_time
            self.generate_training_summary(models_by_symbol, best_models, execution_time)
            
            # 13. Generate trading signals for test set
            signals_df = self.generate_trading_signals(
                X_test_dict, y_test_dict, best_models, symbols_dates
            )
            
            # 14. Save signals
            if not signals_df.empty:
                signals_path = os.path.join(self.output_dir, 'trading_signals.csv')
                signals_df.to_csv(signals_path, index=False)
                logger.info(f"Saved trading signals to {signals_path}")
            
            # Store results
            results['models_by_symbol'] = models_by_symbol
            results['best_models'] = best_models
            results['signals_df'] = signals_df
            
            # Log execution time
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
        logger.info(f"Total symbols trained: {len(results.get('models_by_symbol', {}))}")
        logger.info(f"Best models saved: {len(results.get('best_models', {}))}")
        
        if 'signals_df' in results and not results['signals_df'].empty:
            signal_counts = results['signals_df']['signal'].value_counts()
            logger.info("\nSignal Distribution:")
            for signal, count in signal_counts.items():
                logger.info(f"  {signal}: {count}")
    
    logger.info(f"Trading model pipeline completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()