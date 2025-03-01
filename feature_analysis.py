import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues

import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import shap
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import time
import logging
import sys
import os
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FeatureAnalysis')

class FeatureAnalyzer:
    def __init__(self, db_config: Dict[str, str], output_dir: str = None):
        """
        Initialize the feature analyzer
        
        Args:
            db_config: MySQL database configuration
            output_dir: Directory to save analysis results (None for script directory)
        """
        self.db_config = db_config
        
        # If output_dir is None, use script directory
        if output_dir is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(script_dir, f"feature_analysis_{timestamp}")
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Analysis results will be saved to: {self.output_dir}")
        
        # Track execution time
        self.start_time = time.time()
    
    def connect_to_db(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        """Create a new database connection."""
        try:
            return mysql.connector.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return None
    
    def fetch_ml_features(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch ML features from the database
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of trading symbols to fetch (None for all)
            
        Returns:
            DataFrame containing ML features
        """
        try:
            conn = self.connect_to_db()
            
            # Construct query
            query = "SELECT * FROM ml_features WHERE date BETWEEN %s AND %s"
            params = [start_date, end_date]
            
            # Add symbol filter if provided
            if symbols:
                placeholder = ', '.join(['%s'] * len(symbols))
                query += f" AND trading_symbol IN ({placeholder})"
                params.extend(symbols)
            
            # Execute query
            logger.info(f"Fetching data from {start_date} to {end_date}")
            df = pd.read_sql(query, conn, params=params)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            conn.close()
            
            logger.info(f"Fetched {len(df)} rows, {df['trading_symbol'].nunique()} symbols")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ML features: {e}")
            if 'conn' in locals() and conn:
                conn.close()
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Preprocess data for analysis
        
        Args:
            df: DataFrame containing ML features
            
        Returns:
            Tuple containing (X, y, feature_names)
        """
        if df.empty:
            logger.error("Empty DataFrame, cannot preprocess")
            return pd.DataFrame(), pd.Series(), []
        
        # Drop rows with missing values
        df_clean = df.dropna()
        
        if len(df_clean) < len(df):
            logger.info(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        
        # Identify target and features
        y = df_clean['target']
        
        # List of columns to exclude from features
        exclude_cols = ['id', 'date', 'trading_symbol', 'future_close', 'future_return', 'target']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        
        logger.info(f"Prepared dataset with {X.shape[1]} features and {len(X)} samples")
        
        return X, y, feature_cols
    
    def check_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.9) -> Dict[str, Any]:
        """
        Check for multicollinearity among features
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold to identify multicollinearity
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        logger.info("Checking for multicollinearity...")
        
        # 1. Correlation Matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle of correlation matrix excluding diagonal
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        highly_correlated_features = [(upper.index[i], upper.columns[j], upper.iloc[i, j]) 
                                     for i in range(len(upper.index)) 
                                     for j in range(len(upper.columns)) 
                                     if upper.iloc[i, j] > threshold]
        
        # Sort by correlation value
        highly_correlated_features.sort(key=lambda x: x[2], reverse=True)
        
        # 2. Variance Inflation Factor (VIF)
        # Prepare a sample to calculate VIF (can be computationally expensive)
        sample_size = min(5000, len(X))
        X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        
        # Calculate VIF for each feature
        try:
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X_sample.columns
            vif_data["VIF"] = [variance_inflation_factor(X_sample.values, i) 
                               for i in range(X_sample.shape[1])]
            
            # Sort by VIF value
            vif_data = vif_data.sort_values("VIF", ascending=False)
            
            high_vif_features = vif_data[vif_data["VIF"] > 10]
        except Exception as e:
            logger.warning(f"Error calculating VIF: {e}")
            high_vif_features = pd.DataFrame()
        
        # Prepare results
        results = {
            "correlation_matrix": corr_matrix,
            "highly_correlated_pairs": highly_correlated_features,
            "high_vif_features": high_vif_features if not high_vif_features.empty else None
        }
        
        # Log findings
        logger.info(f"Found {len(highly_correlated_features)} highly correlated feature pairs (>{threshold})")
        if not high_vif_features.empty:
            logger.info(f"Found {len(high_vif_features)} features with high VIF (>10)")
        
        # Save correlation matrix visualization
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
        
        return results
    
    def feature_importance_rf(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance using Random Forest...")
        
        # Time series split for more robust importance estimation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize importance dataframe
        feature_importances = pd.DataFrame()
        feature_importances['Feature'] = X.columns
        
        # Random forest classifier with class_weight to handle imbalance
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        importance_values = []
        
        # Calculate importance across folds for stability
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf.fit(X_train, y_train)
            fold_importance = rf.feature_importances_
            importance_values.append(fold_importance)
        
        # Average importance across folds
        avg_importance = np.mean(np.array(importance_values), axis=0)
        feature_importances['Importance'] = avg_importance
        
        # Sort by importance
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        
        # Save feature importance plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(30))
        plt.title('Top 30 Features by Random Forest Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()
        
        return feature_importances
    
    def feature_importance_lgbm(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using LightGBM
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance using LightGBM...")
        
        # Convert data to LightGBM format
        lgb_train = lgb.Dataset(X, y)
        
        # Parameters
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train model
        gbm = lgb.train(params, lgb_train, num_boost_round=100)
        
        # Get feature importance
        importance = gbm.feature_importance(importance_type='gain')
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        })
        
        # Sort by importance
        feature_importances = feature_importances.sort_values('Importance', ascending=False)
        
        # Save feature importance plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(30))
        plt.title('Top 30 Features by LightGBM Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lgbm_feature_importance.png'), dpi=300)
        plt.close()
        
        return feature_importances
    
    def shap_analysis(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Perform SHAP analysis to understand feature contributions
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        logger.info("Performing SHAP analysis...")
        
        try:
            # Use a sample for SHAP analysis if data is large
            sample_size = min(2000, len(X))
            X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
            y_sample = y.loc[X_sample.index]
            
            # Train a model for SHAP analysis
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_sample, y_sample)
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Get shap values - handle different possible formats
            shap_values = explainer.shap_values(X_sample)
            
            # Create appropriate summary plot based on shap_values type
            plt.figure(figsize=(12, 10))
            
            # For binary classification, shap_values will be a list of two arrays
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Use the positive class (index 1)
                shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
                
                # Calculate mean absolute SHAP values for feature importance
                mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
            else:
                # For other cases (regression or single output)
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'shap_importance.png'), dpi=300)
            plt.close()
            
            # Get top features based on mean absolute SHAP values
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'SHAP_Value': mean_abs_shap
            })
            feature_importance = feature_importance.sort_values('SHAP_Value', ascending=False)
            
            # Save to CSV
            feature_importance.to_csv(os.path.join(self.output_dir, 'shap_importance.csv'), index=False)
            
            # Create dependence plots only for top features
            top_features = feature_importance.head(10)['Feature'].tolist()
            
            for feature in top_features:
                try:
                    plt.figure(figsize=(10, 7))
                    feature_idx = list(X_sample.columns).index(feature)
                    
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        # For binary classification
                        shap.dependence_plot(
                            feature_idx, 
                            shap_values[1], 
                            X_sample,
                            show=False
                        )
                    else:
                        # For other cases
                        shap.dependence_plot(
                            feature_idx, 
                            shap_values, 
                            X_sample,
                            show=False
                        )
                    
                    plt.title(f'SHAP Dependence Plot for {feature}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'shap_dependence_{feature}.png'), dpi=300)
                    plt.close()
                except Exception as e:
                    logger.error(f"Error creating dependence plot for feature {feature}: {e}")
            
            logger.info(f"SHAP analysis completed for {len(top_features)} top features")
        
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {e}", exc_info=True)
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Perform Recursive Feature Elimination with Cross-Validation
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected features
        """
        logger.info("Performing Recursive Feature Elimination...")
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Base estimator
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # RFECV
        selector = RFECV(
            estimator=estimator,
            step=5,  # Remove 5 features at each iteration
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_features = X.columns[selector.support_].tolist()
        
        # Plot number of features vs. CV score - handle different attribute names
        plt.figure(figsize=(10, 6))
        
        # Check which attribute exists in the selector object
        if hasattr(selector, 'cv_results_'):
            # For newer scikit-learn versions
            cv_scores = selector.cv_results_['mean_test_score']
            plt.plot(
                range(1, len(cv_scores) + 1),
                cv_scores,
                marker='o'
            )
        elif hasattr(selector, 'grid_scores_'):
            # For older scikit-learn versions
            plt.plot(
                range(1, len(selector.grid_scores_) + 1),
                selector.grid_scores_,
                marker='o'
            )
        else:
            # If neither attribute exists, just show the number of selected features
            plt.text(0.5, 0.5, f"Selected {len(selected_features)} features", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
        
        plt.title('Feature Selection Cross-Validation Score')
        plt.xlabel('Number of Features')
        plt.ylabel('Cross-Validation Score (F1)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rfecv_score.png'), dpi=300)
        plt.close()
        
        logger.info(f"RFE selected {len(selected_features)} features")
        
        return selected_features
    
    def pca_analysis(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Perform Principal Component Analysis
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple containing (component_df, explained_variance)
        """
        logger.info("Performing PCA analysis...")
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Calculate explained variance
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Number of components for 95% variance
        n_components_95 = np.argmax(explained_variance >= 0.95) + 1
        
        # Get feature loadings for top components
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=X.columns
        )
        
        # Get top features for first 5 components
        n_top_features = 10
        top_features = {}
        
        for i in range(min(5, pca.n_components_)):
            pc = f'PC{i+1}'
            top_pos = loadings[pc].nlargest(n_top_features).index.tolist()
            top_neg = loadings[pc].nsmallest(n_top_features).index.tolist()
            top_features[pc] = {'positive': top_pos, 'negative': top_neg}
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.axvline(x=n_components_95, color='g', linestyle='--', 
                   label=f'{n_components_95} Components')
        plt.title('PCA Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'), dpi=300)
        plt.close()
        
        # Create heatmap of top component loadings
        plt.figure(figsize=(15, 10))
        top_features_all = set()
        for pc_features in top_features.values():
            top_features_all.update(pc_features['positive'])
            top_features_all.update(pc_features['negative'])
        
        loadings_subset = loadings.loc[list(top_features_all), ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'][:min(5, pca.n_components_)]]
        sns.heatmap(loadings_subset, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('PCA Loadings for Top Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_loadings.png'), dpi=300)
        plt.close()
        
        logger.info(f"PCA analysis completed. {n_components_95} components explain 95% of variance")
        
        return top_features, explained_variance
    
    def evaluate_feature_subset(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict[str, float]:
        """
        Evaluate a subset of features using a classifier
        
        Args:
            X: Feature DataFrame
            y: Target Series
            features: List of features to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Select features
        X_subset = X[features]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize metrics
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # Classifier
        clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Evaluate across folds
        for train_idx, test_idx in tscv.split(X_subset):
            X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            metrics['auc'].append(roc_auc_score(y_test, y_proba))
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        logger.info(f"Feature subset evaluation completed with {len(features)} features")
        logger.info(f"Metrics: {avg_metrics}")
        
        return avg_metrics
    
    def find_optimal_feature_subset(self, X: pd.DataFrame, y: pd.Series, 
                                  importance_df: pd.DataFrame) -> List[str]:
        """
        Find the optimal subset of features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            importance_df: DataFrame with feature importance scores
            
        Returns:
            List of optimal features
        """
        logger.info("Finding optimal feature subset...")
        
        # Get features sorted by importance
        sorted_features = importance_df['Feature'].tolist()
        
        # Incremental evaluation
        feature_counts = list(range(5, min(50, len(sorted_features)), 5))
        feature_counts.extend([min(60, len(sorted_features)), min(75, len(sorted_features)), min(len(sorted_features), 100)])
        
        results = []
        
        for n_features in feature_counts:
            subset = sorted_features[:n_features]
            metrics = self.evaluate_feature_subset(X, y, subset)
            results.append({
                'n_features': n_features,
                **metrics
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot metrics by feature count
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in metrics_to_plot:
            plt.plot(results_df['n_features'], results_df[metric], marker='o', label=metric)
        
        plt.title('Performance Metrics by Feature Count')
        plt.xlabel('Number of Features')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_count_performance.png'), dpi=300)
        plt.close()
        
        # Find optimal feature count (highest F1 score)
        optimal_idx = results_df['f1'].idxmax()
        optimal_count = results_df.iloc[optimal_idx]['n_features']
        optimal_features = sorted_features[:int(optimal_count)]
        
        logger.info(f"Optimal feature count: {optimal_count} with F1 score: {results_df.iloc[optimal_idx]['f1']:.4f}")
        
        return optimal_features
    
    def run_analysis(self, start_date: str, end_date: str, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete feature analysis pipeline
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of trading symbols to analyze (None for all)
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        try:
            # 1. Fetch data
            df = self.fetch_ml_features(start_date, end_date, symbols)
            if df.empty:
                logger.error("No data fetched. Exiting analysis.")
                return {}
            
            # 2. Preprocess data
            X, y, feature_cols = self.preprocess_data(df)
            if X.empty:
                logger.error("Preprocessing failed. Exiting analysis.")
                return {}
            
            # 3. Check class distribution
            class_counts = y.value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            results['data_summary'] = {
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'class_distribution': class_counts.to_dict(),
                'date_range': (df['date'].min(), df['date'].max()),
                'symbols': df['trading_symbol'].unique().tolist()
            }
            
            # 4. Check multicollinearity
            try:
                multicollinearity_results = self.check_multicollinearity(X)
                results['multicollinearity'] = {
                    'high_correlation_count': len(multicollinearity_results['highly_correlated_pairs']),
                    'top_correlations': multicollinearity_results['highly_correlated_pairs'][:20],
                    'high_vif_count': len(multicollinearity_results['high_vif_features']) if multicollinearity_results['high_vif_features'] is not None else 0
                }
            except Exception as e:
                logger.error(f"Error in multicollinearity analysis: {e}", exc_info=True)
                results['multicollinearity'] = {'error': str(e)}
            
            # 5. Calculate feature importance with Random Forest
            try:
                rf_importance = self.feature_importance_rf(X, y)
                results['rf_importance'] = rf_importance
            except Exception as e:
                logger.error(f"Error in RF importance analysis: {e}", exc_info=True)
                rf_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.ones(len(X.columns))})
                results['rf_importance'] = {'error': str(e)}
            
            # 6. Calculate feature importance with LightGBM
            try:
                lgbm_importance = self.feature_importance_lgbm(X, y)
                results['lgbm_importance'] = lgbm_importance
            except Exception as e:
                logger.error(f"Error in LightGBM importance analysis: {e}", exc_info=True)
                lgbm_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.ones(len(X.columns))})
                results['lgbm_importance'] = {'error': str(e)}
            
            # 7. Combined feature importance (average rank from both methods)
            try:
                rf_ranks = rf_importance.reset_index().drop('index', axis=1, errors='ignore')
                rf_ranks['RF_Rank'] = rf_ranks.index + 1
                
                lgbm_ranks = lgbm_importance.reset_index().drop('index', axis=1, errors='ignore')
                lgbm_ranks['LGBM_Rank'] = lgbm_ranks.index + 1
                
                # Merge both rankings
                combined_ranks = pd.merge(
                    rf_ranks[['Feature', 'RF_Rank']], 
                    lgbm_ranks[['Feature', 'LGBM_Rank']], 
                    on='Feature'
                )
                
                combined_ranks['Avg_Rank'] = (combined_ranks['RF_Rank'] + combined_ranks['LGBM_Rank']) / 2
                combined_ranks = combined_ranks.sort_values('Avg_Rank')
                
                results['combined_ranks'] = combined_ranks
            except Exception as e:
                logger.error(f"Error in combined ranking: {e}", exc_info=True)
                results['combined_ranks'] = {'error': str(e)}
                # Create a dummy combined ranks if we need it later
                combined_ranks = pd.DataFrame({'Feature': X.columns, 'Avg_Rank': range(1, len(X.columns)+1)})
            
            # 8. PCA analysis
            try:
                pca_results, explained_variance = self.pca_analysis(X)
                results['pca'] = {
                    'components_for_95_var': np.argmax(explained_variance >= 0.95) + 1,
                    'top_component_features': pca_results
                }
            except Exception as e:
                logger.error(f"Error in PCA analysis: {e}", exc_info=True)
                results['pca'] = {'error': str(e), 'components_for_95_var': X.shape[1]}
            
            # 9. SHAP analysis for deeper understanding
            try:
                self.shap_analysis(X, y)
                results['shap'] = {'completed': True}
            except Exception as e:
                logger.error(f"Error in SHAP analysis: {e}", exc_info=True)
                results['shap'] = {'error': str(e)}
            
            # 10. RFE for feature selection
            try:
                rfe_features = self.recursive_feature_elimination(X, y)
                results['rfe'] = {
                    'selected_features': rfe_features,
                    'feature_count': len(rfe_features)
                }
            except Exception as e:
                logger.error(f"Error in RFE analysis: {e}", exc_info=True)
                results['rfe'] = {'error': str(e), 'selected_features': X.columns.tolist()[:20], 'feature_count': 20}
                rfe_features = X.columns.tolist()[:20]  # Use top 20 as fallback
            
            # 11. Find optimal feature subset
            try:
                optimal_features = self.find_optimal_feature_subset(X, y, combined_ranks[['Feature', 'Avg_Rank']])
                results['optimal_features'] = {
                    'features': optimal_features,
                    'feature_count': len(optimal_features)
                }
            except Exception as e:
                logger.error(f"Error finding optimal feature subset: {e}", exc_info=True)
                # Use top 20 features from combined ranking as fallback
                optimal_features = combined_ranks['Feature'].head(20).tolist()
                results['optimal_features'] = {
                    'error': str(e),
                    'features': optimal_features,
                    'feature_count': len(optimal_features)
                }
            
            # 12. Save final report
            try:
                self.save_report(results, rf_importance, lgbm_importance, combined_ranks)
            except Exception as e:
                logger.error(f"Error saving report: {e}", exc_info=True)
            
            # 13. Save optimal feature set
            try:
                joblib.dump(optimal_features, os.path.join(self.output_dir, 'optimal_features.pkl'))
            except Exception as e:
                logger.error(f"Error saving optimal features: {e}", exc_info=True)
            
            # Calculate execution time
            execution_time = time.time() - self.start_time
            logger.info(f"Analysis completed in {execution_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            return results  # Return whatever results we gathered before the error
    
    def save_report(self, results: Dict[str, Any], rf_importance: pd.DataFrame, 
                    lgbm_importance: pd.DataFrame, combined_ranks: pd.DataFrame) -> None:
        """
        Save analysis report to file
        
        Args:
            results: Analysis results dictionary
            rf_importance: Random Forest importance DataFrame
            lgbm_importance: LightGBM importance DataFrame
            combined_ranks: Combined feature rankings DataFrame
        """
        try:
            # Save CSV files
            rf_importance.to_csv(os.path.join(self.output_dir, 'rf_importance.csv'), index=False)
            lgbm_importance.to_csv(os.path.join(self.output_dir, 'lgbm_importance.csv'), index=False)
            combined_ranks.to_csv(os.path.join(self.output_dir, 'combined_rankings.csv'), index=False)
            
            # Save optimal features to text file
            with open(os.path.join(self.output_dir, 'optimal_features.txt'), 'w') as f:
                for feature in results['optimal_features']['features']:
                    f.write(f"{feature}\n")
            
            # Create HTML report
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Feature Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    h2 { color: #3498db; margin-top: 30px; }
                    h3 { color: #2980b9; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .figure { margin: 20px 0; text-align: center; }
                    .figure img { max-width: 100%; border: 1px solid #ddd; }
                    .summary { background-color: #eaf2f8; padding: 15px; border-radius: 5px; }
                    .high { color: #e74c3c; }
                    .medium { color: #f39c12; }
                    .low { color: #27ae60; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Feature Analysis Report</h1>
                    <div class="summary">
                        <h2>Executive Summary</h2>
                        <p>This report presents the results of a comprehensive feature analysis for a trading prediction model.</p>
                        <ul>
                            <li><strong>Data Coverage:</strong> {start_date} to {end_date}</li>
                            <li><strong>Symbols Analyzed:</strong> {symbol_count}</li>
                            <li><strong>Total Samples:</strong> {sample_count}</li>
                            <li><strong>Original Features:</strong> {feature_count}</li>
                            <li><strong>Optimal Feature Count:</strong> {optimal_count}</li>
                        </ul>
                    </div>
                    
                    <h2>Data Overview</h2>
                    <h3>Class Distribution</h3>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                        {class_distribution_rows}
                    </table>
                    
                    <h2>Multicollinearity Analysis</h2>
                    <p>{high_corr_count} feature pairs showed high correlation (>0.9).</p>
                    <h3>Top Correlated Feature Pairs</h3>
                    <table>
                        <tr>
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Correlation</th>
                        </tr>
                        {correlation_rows}
                    </table>
                    
                    <div class="figure">
                        <img src="correlation_matrix.png" alt="Correlation Matrix">
                        <p>Figure 1: Feature Correlation Matrix</p>
                    </div>
                    
                    <h2>Feature Importance</h2>
                    <h3>Random Forest Importance</h3>
                    <div class="figure">
                        <img src="rf_feature_importance.png" alt="Random Forest Feature Importance">
                        <p>Figure 2: Top Features by Random Forest Importance</p>
                    </div>
                    
                    <h3>LightGBM Importance</h3>
                    <div class="figure">
                        <img src="lgbm_feature_importance.png" alt="LightGBM Feature Importance">
                        <p>Figure 3: Top Features by LightGBM Importance</p>
                    </div>
                    
                    <h3>Top Features (Combined Ranking)</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Feature</th>
                            <th>RF Rank</th>
                            <th>LGBM Rank</th>
                            <th>Avg Rank</th>
                        </tr>
                        {combined_rank_rows}
                    </table>
                    
                    <h2>SHAP Analysis</h2>
                    <div class="figure">
                        <img src="shap_importance.png" alt="SHAP Feature Importance">
                        <p>Figure 4: SHAP Feature Importance</p>
                    </div>
                    
                    <h3>Feature Dependence Plots</h3>
                    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                        {shap_dependence_images}
                    </div>
                    
                    <h2>PCA Analysis</h2>
                    <p>{pca_components} principal components explain 95% of the variance.</p>
                    <div class="figure">
                        <img src="pca_variance.png" alt="PCA Explained Variance">
                        <p>Figure 5: PCA Explained Variance</p>
                    </div>
                    
                    <div class="figure">
                        <img src="pca_loadings.png" alt="PCA Feature Loadings">
                        <p>Figure 6: PCA Feature Loadings</p>
                    </div>
                    
                    <h2>Feature Selection</h2>
                    <h3>Recursive Feature Elimination</h3>
                    <p>RFE selected {rfe_count} features.</p>
                    <div class="figure">
                        <img src="rfecv_score.png" alt="RFE Cross-Validation Score">
                        <p>Figure 7: RFE Cross-Validation Score</p>
                    </div>
                    
                    <h3>Optimal Feature Subset</h3>
                    <p>The optimal feature subset contains {optimal_count} features.</p>
                    <div class="figure">
                        <img src="feature_count_performance.png" alt="Performance by Feature Count">
                        <p>Figure 8: Model Performance by Feature Count</p>
                    </div>
                    
                    <h3>Selected Features</h3>
                    <table>
                        <tr>
                            <th>#</th>
                            <th>Feature</th>
                        </tr>
                        {optimal_feature_rows}
                    </table>
                    
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Use the identified optimal feature set of {optimal_count} features for model training.</li>
                        <li>Consider removing highly correlated features to reduce multicollinearity.</li>
                        <li>Pay special attention to the top 10 features as they provide the most predictive power.</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Fill in dynamic content
            data_summary = results['data_summary']
            
            # Calculate total samples and percentage for each class
            class_dist = data_summary['class_distribution']
            total_samples = sum(class_dist.values())
            class_dist_rows = ""
            for cls, count in class_dist.items():
                percentage = (count / total_samples) * 100
                class_dist_rows += f"<tr><td>{cls}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
            
            # Top correlated feature pairs
            multicol = results['multicollinearity']
            corr_rows = ""
            for i, (feat1, feat2, corr) in enumerate(multicol['top_correlations'][:10]):
                corr_rows += f"<tr><td>{feat1}</td><td>{feat2}</td><td>{corr:.4f}</td></tr>"
            
            # Combined rank features
            rank_rows = ""
            for i, (_, row) in enumerate(combined_ranks.head(20).iterrows()):
                rank_rows += f"<tr><td>{i+1}</td><td>{row['Feature']}</td><td>{row['RF_Rank']}</td><td>{row['LGBM_Rank']}</td><td>{row['Avg_Rank']:.2f}</td></tr>"
            
            # SHAP dependence images
            shap_images = ""
            for feature in results['feature_importance']['combined_top_features'][:10]:
                img_path = f"shap_dependence_{feature}.png"
                if os.path.exists(os.path.join(self.output_dir, img_path)):
                    shap_images += f'<div style="margin: 10px;"><img src="{img_path}" alt="SHAP Dependence {feature}" style="width: 350px;"><p>SHAP Dependence: {feature}</p></div>'
            
            # Optimal features
            opt_features = results['optimal_features']['features']
            opt_rows = ""
            for i, feat in enumerate(opt_features):
                opt_rows += f"<tr><td>{i+1}</td><td>{feat}</td></tr>"
            
            # Fill template
            html_content = html_content.format(
                start_date=data_summary['date_range'][0].strftime('%Y-%m-%d'),
                end_date=data_summary['date_range'][1].strftime('%Y-%m-%d'),
                symbol_count=len(data_summary['symbols']),
                sample_count=data_summary['n_samples'],
                feature_count=data_summary['n_features'],
                optimal_count=results['optimal_features']['feature_count'],
                class_distribution_rows=class_dist_rows,
                high_corr_count=multicol['high_correlation_count'],
                correlation_rows=corr_rows,
                combined_rank_rows=rank_rows,
                shap_dependence_images=shap_images,
                pca_components=results['pca']['components_for_95_var'],
                rfe_count=results['rfe']['feature_count'],
                optimal_feature_rows=opt_rows
            )
            
            # Save HTML report
            with open(os.path.join(self.output_dir, 'feature_analysis_report.html'), 'w') as f:
                f.write(html_content)
            
            logger.info(f"Analysis report saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}", exc_info=True)


def main():
    """Main entry point for the feature analyzer"""
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'dhan_hq',
        'password': 'Passw0rd@098',
        'database': 'dhanhq_db',
        'auth_plugin': 'mysql_native_password',
        'use_pure': True
    }
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, f"feature_analysis_{timestamp}")
    
    # Create analyzer with output directory in script location
    analyzer = FeatureAnalyzer(db_config, output_dir=output_dir)
    
    # Define date range - use most recent 1 year of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Define symbols to analyze - focus on liquid stocks
    # These are some of the most liquid NSE stocks
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 
        'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'BAJFINANCE', 'KOTAKBANK',
        'ITC', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'TITAN',
        'SUNPHARMA', 'TATAMOTORS', 'ULTRACEMCO', 'ADANIENT', 'WIPRO'
    ]
    
    # Run analysis with GPU-accelerated libraries where possible
    logger.info(f"Starting feature analysis for {len(symbols)} symbols from {start_date} to {end_date}")
    results = analyzer.run_analysis(start_date, end_date, symbols)
    
    if not results:
        logger.error("Analysis failed. Please check logs for details.")
        return
    
    logger.info("Feature analysis completed successfully")
    
    # Extract and show optimal features
    optimal_features = results['optimal_features']['features']
    logger.info(f"Optimal feature set ({len(optimal_features)} features):")
    for i, feature in enumerate(optimal_features[:10]):
        logger.info(f"{i+1}. {feature}")
    
    if len(optimal_features) > 10:
        logger.info(f"... plus {len(optimal_features) - 10} more features")
    
    logger.info(f"Full results available in: {output_dir}")


if __name__ == "__main__":
    main()