import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues

import re
import os
import time
import logging
import warnings
import sys
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.feature_selection import mutual_info_classif

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
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
from scipy.stats import spearmanr
from lightgbm import LGBMClassifier  # For time_aware_feature_importance
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    def parallel_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                              n_jobs: int = -1) -> Dict[str, pd.DataFrame]:
        """
        Calculate feature importance using multiple methods in parallel
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary with different importance DataFrames
        """
        logger.info("Calculating feature importance using multiple methods in parallel...")
        
        # Define number of processes
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        
        # Define tasks to run in parallel
        tasks = [
            ('rf', self._rf_importance_task, (X, y)),
            ('lgbm', self._lgbm_importance_task, (X, y)),
            ('mutual_info', self._mutual_info_task, (X, y)),
            ('correlation', self._correlation_task, (X, y))
        ]
        
        # Run tasks in parallel
        results = {}
        
        with ProcessPoolExecutor(max_workers=min(len(tasks), n_jobs)) as executor:
            future_to_task = {executor.submit(task_func, *task_args): task_name 
                            for task_name, task_func, task_args in tasks}
            
            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    results[task_name] = result
                    logger.info(f"Completed {task_name} importance calculation")
                except Exception as e:
                    logger.error(f"Error in {task_name} importance calculation: {e}", exc_info=True)
        
        return results

    def _rf_importance_task(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Random Forest importance calculation task for parallel processing"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X, y)
        return pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)

    def _lgbm_importance_task(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """LightGBM importance calculation task for parallel processing"""
        lgb_train = lgb.Dataset(X, y)
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
        gbm = lgb.train(params, lgb_train, num_boost_round=100)
        importance = gbm.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

    def _mutual_info_task(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Mutual information calculation task for parallel processing"""
        mi_scores = mutual_info_classif(X, y, random_state=42)
        return pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)

    def _correlation_task(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Correlation calculation task for parallel processing"""
        # Calculate Pearson correlation with target
        pearson_corr = []
        for col in X.columns:
            corr = abs(np.corrcoef(X[col], y)[0, 1])
            pearson_corr.append((col, corr))
        
        return pd.DataFrame(pearson_corr, columns=['Feature', 'Target_Correlation']
                        ).sort_values('Target_Correlation', ascending=False)
        
    def cached_evaluate_feature_subset(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict[str, float]:
        """
        Cached version of evaluate_feature_subset to avoid repeated calculations

        Args:
            X: Feature DataFrame
            y: Target Series
            features: List of features to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create cache key from sorted feature list
        sorted_features = sorted(features)
        cache_key = '-'.join(sorted_features)

        # Check if we already have results for this feature set
        if hasattr(self, '_evaluation_cache') and cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        # Initialize cache if not exists
        if not hasattr(self, '_evaluation_cache'):
            self._evaluation_cache = {}

        # Calculate metrics
        metrics = self.evaluate_feature_subset(X, y, features)

        # Store in cache
        self._evaluation_cache[cache_key] = metrics

        return metrics
    
    # Update the preprocess_data method
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
        exclude_cols = [
            'id', 'date', 'trading_symbol', 
            'future_close', 'future_return', 'target',
            'exit_signal', 'days_to_target'
        ]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        
        logger.info(f"Prepared dataset with {X.shape[1]} features and {len(X)} samples")
        
        return X, y, feature_cols
    def enhanced_correlation_analysis(self, X, y, threshold=0.9):
        """
        Perform enhanced correlation analysis including Spearman rank correlation
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Correlation threshold
            
        Returns:
            Dictionary with analysis results
        """
        # Pearson correlation (already implemented)
        pearson_corr = X.corr().abs()
        
        # Add Spearman rank correlation for non-linear relationships
        spearman_corr = X.corr(method='spearman').abs()
        
        # Get feature-to-target correlations
        target_pearson = pd.DataFrame({
            'Feature': X.columns,
            'Pearson_Corr_Target': [abs(np.corrcoef(X[col], y)[0,1]) for col in X.columns]
        }).sort_values('Pearson_Corr_Target', ascending=False)
        
        target_spearman = pd.DataFrame({
            'Feature': X.columns,
            'Spearman_Corr_Target': [abs(spearmanr(X[col], y)[0]) for col in X.columns]
        }).sort_values('Spearman_Corr_Target', ascending=False)
        
        # Combined target correlation
        target_combined = pd.merge(target_pearson, target_spearman, on='Feature')
        target_combined['Combined_Rank'] = target_combined['Pearson_Corr_Target'].rank(ascending=False) + \
                                        target_combined['Spearman_Corr_Target'].rank(ascending=False)
        target_combined = target_combined.sort_values('Combined_Rank')
        
        # Find features with correlation divergence (high Spearman, low Pearson or vice versa)
        target_combined['Corr_Difference'] = abs(target_combined['Spearman_Corr_Target'] - target_combined['Pearson_Corr_Target'])
        divergent_features = target_combined.nlargest(10, 'Corr_Difference')
        
        return {
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'target_correlation': target_combined,
            'divergent_features': divergent_features
        }
    def enhanced_feature_selection_with_pruning(self, X: pd.DataFrame, y: pd.Series, 
                                           importance_df: pd.DataFrame, 
                                           corr_threshold: float = 0.8,
                                           max_features: int = 30) -> Dict[str, Any]:
        """
        Enhanced feature selection with correlation pruning
        
        Args:
            X: Feature DataFrame
            y: Target Series
            importance_df: DataFrame with feature importance scores
            corr_threshold: Correlation threshold for feature pruning
            max_features: Maximum number of features to select
            
        Returns:
            Dictionary with selection results and metrics
        """
        logger.info(f"Performing enhanced feature selection with correlation pruning...")
        
        # Get features sorted by importance
        sorted_features = importance_df['Feature'].tolist()
        
        # Calculate feature correlations
        corr_matrix = X[sorted_features].corr().abs()
        
        # Initialize selected features with the most important feature
        selected_features = [sorted_features[0]]
        excluded_features = []
        
        # Iteratively add features that are not highly correlated with already selected ones
        for feature in sorted_features[1:]:
            # Check if feature is highly correlated with any selected feature
            is_correlated = False
            for selected in selected_features:
                if corr_matrix.loc[feature, selected] > corr_threshold:
                    is_correlated = True
                    excluded_features.append((feature, selected, corr_matrix.loc[feature, selected]))
                    break
            
            # If not highly correlated, add to selected features
            if not is_correlated:
                selected_features.append(feature)
                
            # Stop when we have enough features
            if len(selected_features) >= max_features:
                break
        
        # Evaluate original (unpruned) feature set
        original_metrics = self.evaluate_feature_subset(X, y, sorted_features[:len(selected_features)])
        
        # Evaluate pruned feature set
        pruned_metrics = self.evaluate_feature_subset(X, y, selected_features)
        
        # Create comparison visualizations
        self.compare_feature_sets(X, y, 
                                set1=sorted_features[:len(selected_features)], 
                                set2=selected_features,
                                label1="Original Features", 
                                label2="Pruned Features")
        
        logger.info(f"Selected {len(selected_features)} features with correlation pruning")
        logger.info(f"Excluded {len(excluded_features)} correlated features")
        logger.info(f"Original metrics: {original_metrics}")
        logger.info(f"Pruned metrics: {pruned_metrics}")
        
        return {
            'selected_features': selected_features,
            'excluded_features': excluded_features,
            'original_metrics': original_metrics,
            'pruned_metrics': pruned_metrics,
            'feature_count': len(selected_features)
        }

    def analyze_feature_stability_across_regimes(self, X: pd.DataFrame, y: pd.Series, 
                                            regime_column: str = 'market_regime') -> Dict[str, Any]:
        """
        Analyze feature importance across different market regimes
        
        Args:
            X: Feature DataFrame
            y: Target Series
            regime_column: Column that defines the market regime
            
        Returns:
            Dictionary with regime-specific importance and stability metrics
        """
        logger.info("Analyzing feature stability across different market regimes...")
        
        if regime_column not in X.columns:
            logger.warning(f"Column {regime_column} not found. Creating simple market regime based on returns.")
            # Create simple regime based on return patterns if regime column doesn't exist
            if 'return_20d' in X.columns:
                X['tmp_regime'] = np.where(X['return_20d'] > 0, 'bull', 'bear')
                regime_column = 'tmp_regime'
            else:
                logger.error("Cannot create market regime. No suitable columns found.")
                return {}
        
        # Get unique regimes
        regimes = X[regime_column].unique()
        
        # Store importance by regime
        regime_importance = {}
        all_features = set()
        
        # Calculate importance for each regime
        for regime in regimes:
            regime_mask = X[regime_column] == regime
            if regime_mask.sum() < 100:  # Skip regimes with too few samples
                logger.warning(f"Skipping regime '{regime}' with only {regime_mask.sum()} samples")
                continue
                
            X_regime = X[regime_mask].drop(columns=[regime_column])
            y_regime = y[regime_mask]
            
            # Calculate feature importance for this regime
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_regime, y_regime)
            
            # Store importance
            importance = pd.DataFrame({
                'Feature': X_regime.columns,
                f'Importance_{regime}': model.feature_importances_
            }).sort_values(f'Importance_{regime}', ascending=False)
            
            regime_importance[regime] = importance
            all_features.update(importance['Feature'])
        
        # Combine all importance DataFrames
        combined_df = pd.DataFrame({'Feature': list(all_features)})
        for regime, imp_df in regime_importance.items():
            combined_df = combined_df.merge(imp_df, on='Feature', how='left')
        
        # Fill NaN values with zeros
        combined_df = combined_df.fillna(0)
        
        # Calculate stability metrics
        importance_columns = [col for col in combined_df.columns if col.startswith('Importance_')]
        combined_df['Mean_Importance'] = combined_df[importance_columns].mean(axis=1)
        combined_df['Std_Importance'] = combined_df[importance_columns].std(axis=1)
        combined_df['CV'] = combined_df['Std_Importance'] / (combined_df['Mean_Importance'] + 1e-10)  # Coefficient of variation
        combined_df['Stability'] = 1 - combined_df['CV']  # High stability = low CV
        
        # Sort by mean importance
        combined_df = combined_df.sort_values('Mean_Importance', ascending=False)
        
        # Identify features with high importance and high stability
        high_importance = combined_df['Mean_Importance'] > combined_df['Mean_Importance'].quantile(0.75)
        high_stability = combined_df['Stability'] > combined_df['Stability'].quantile(0.75)
        combined_df['Robust'] = high_importance & high_stability
        
        robust_features = combined_df[combined_df['Robust']]['Feature'].tolist()
        
        # Create visualizations
        self._plot_regime_importance(combined_df, importance_columns, regimes)
        self._plot_stability_vs_importance(combined_df)
        
        # Save results
        combined_df.to_csv(os.path.join(self.output_dir, 'regime_importance.csv'), index=False)
        
        logger.info(f"Identified {len(robust_features)} robust features across market regimes")
        
        return {
            'regime_importance': combined_df,
            'robust_features': robust_features
        }

    def _plot_regime_importance(self, df: pd.DataFrame, importance_columns: List[str], 
                            regimes: List[str], top_n: int = 20) -> None:
        """Plot feature importance across different market regimes"""
        plt.figure(figsize=(15, 10))
        
        # Get top features by mean importance
        top_features = df.nlargest(top_n, 'Mean_Importance')['Feature'].tolist()
        
        # Prepare data for heatmap
        heatmap_data = df[df['Feature'].isin(top_features)].set_index('Feature')[importance_columns]
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.3f')
        plt.title(f'Top {top_n} Features: Importance Across Market Regimes')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regime_importance_heatmap.png'), dpi=300)
        plt.close()
        
        # Create barplot for mean importance
        plt.figure(figsize=(12, 8))
        top_df = df[df['Feature'].isin(top_features)].sort_values('Mean_Importance')
        sns.barplot(x='Mean_Importance', y='Feature', data=top_df, palette='viridis')
        plt.title(f'Top {top_n} Features by Mean Importance Across Regimes')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mean_importance_regimes.png'), dpi=300)
        plt.close()

    def _plot_stability_vs_importance(self, df: pd.DataFrame) -> None:
        """Create scatter plot of feature stability vs importance"""
        plt.figure(figsize=(12, 8))
        
        # Color points by robustness
        colors = df['Robust'].map({True: 'green', False: 'gray'})
        
        plt.scatter(df['Mean_Importance'], df['Stability'], c=colors, alpha=0.7)
        
        # Add feature labels for robust features
        for _, row in df[df['Robust']].iterrows():
            plt.annotate(row['Feature'], 
                        (row['Mean_Importance'], row['Stability']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.xlabel('Mean Importance')
        plt.ylabel('Stability (1 - Coefficient of Variation)')
        plt.title('Feature Stability vs. Importance Across Market Regimes')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add quadrant lines
        plt.axhline(y=df['Stability'].quantile(0.75), color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=df['Mean_Importance'].quantile(0.75), color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stability_vs_importance.png'), dpi=300)
        plt.close()
    
    def analyze_individual_feature_contributions(self, X: pd.DataFrame, y: pd.Series, 
                                           top_features: List[str] = None, 
                                           n_features: int = 10) -> None:
        """
        Analyze predictive power of individual features
        
        Args:
            X: Feature DataFrame
            y: Target Series
            top_features: List of features to analyze (None to use importance ranking)
            n_features: Number of top features to analyze if top_features is None
        """
        logger.info("Analyzing individual feature contributions...")
        
        # If no features provided, select top n features by importance
        if top_features is None:
            # Get feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = importance_df.head(n_features)['Feature'].tolist()
        
        # Prepare for metrics
        feature_metrics = []
        
        # Train model on each feature individually
        for feature in top_features:
            try:
                X_single = X[[feature]]
                
                # Use TimeSeriesSplit for validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Track metrics across folds
                metrics = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc': []
                }
                
                # Train and evaluate across folds
                for train_idx, test_idx in tscv.split(X_single):
                    X_train, X_test = X_single.iloc[train_idx], X_single.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                    rf.fit(X_train, y_train)
                    
                    y_pred = rf.predict(X_test)
                    y_proba = rf.predict_proba(X_test)[:, 1]
                    
                    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                    metrics['precision'].append(precision_score(y_test, y_pred))
                    metrics['recall'].append(recall_score(y_test, y_pred))
                    metrics['f1'].append(f1_score(y_test, y_pred))
                    metrics['auc'].append(roc_auc_score(y_test, y_proba))
                
                # Average metrics
                avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
                
                # Add to results
                feature_metrics.append({
                    'Feature': feature,
                    **avg_metrics
                })
                
                # Create partial dependence plot
                self._create_partial_dependence_plot(X, y, feature)
                
            except Exception as e:
                logger.error(f"Error analyzing feature {feature}: {e}")
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(feature_metrics)
        metrics_df = metrics_df.sort_values('f1', ascending=False)
        
        # Create summary visualization
        plt.figure(figsize=(14, 8))
        
        # Plot F1 score by feature
        sns.barplot(x='f1', y='Feature', data=metrics_df)
        plt.title('Individual Feature F1 Scores')
        plt.xlabel('F1 Score')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'individual_feature_f1.png'), dpi=300)
        plt.close()
        
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(self.output_dir, 'individual_feature_metrics.csv'), index=False)
        
        logger.info(f"Completed individual feature analysis for {len(top_features)} features")

    def _create_partial_dependence_plot(self, X: pd.DataFrame, y: pd.Series, feature: str) -> None:
        """Create partial dependence plot for a single feature"""
        try:
            # Train model on all features
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # Get feature values for x-axis
            x_values = np.linspace(X[feature].min(), X[feature].max(), 100)
            
            # Create copy of dataset
            X_temp = X.copy()
            
            # Initialize predicted probabilities
            y_pred_probs = []
            
            # For each value, set the feature to that value and predict
            for value in x_values:
                X_temp[feature] = value
                y_pred = model.predict_proba(X_temp)[:, 1]
                y_pred_probs.append(np.mean(y_pred))
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_pred_probs)
            plt.title(f'Partial Dependence Plot: {feature}')
            plt.xlabel(feature)
            plt.ylabel('Predicted Probability')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add histogram of feature distribution on secondary axis
            ax2 = plt.twinx()
            ax2.hist(X[feature], bins=30, alpha=0.3, color='gray')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'pdp_{feature}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating partial dependence plot for {feature}: {e}")
    def compare_feature_sets(self, X: pd.DataFrame, y: pd.Series, 
                            set1: List[str], set2: List[str],
                            label1: str = "Set 1", label2: str = "Set 2") -> None:
        """
        Compare performance metrics between two feature sets
        
        Args:
            X: Feature DataFrame
            y: Target Series
            set1: First feature set
            set2: Second feature set
            label1: Label for first set
            label2: Label for second set
        """
        # Evaluate metrics for each set
        metrics1 = self.evaluate_feature_subset(X, y, set1)
        metrics2 = self.evaluate_feature_subset(X, y, set2)
        
        # Create comparison bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, [metrics1[m] for m in metrics], width, label=f"{label1} ({len(set1)} features)")
        plt.bar(x + width/2, [metrics2[m] for m in metrics], width, label=f"{label2} ({len(set2)} features)")
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison Between Feature Sets')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_sets_comparison.png'), dpi=300)
        plt.close()
    
    def mutual_information_analysis(self, X: pd.DataFrame, y: pd.Series, n_top: int = 30) -> pd.DataFrame:
        """
        Calculate mutual information between features and target
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_top: Number of top features to return
            
        Returns:
            DataFrame with mutual information scores
        """
        logger.info("Calculating mutual information between features and target...")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Create DataFrame with results
        mi_df = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        # Save MI plot
        plt.figure(figsize=(14, 10))
        sns.barplot(x='MI_Score', y='Feature', data=mi_df.head(n_top))
        plt.title(f'Top {n_top} Features by Mutual Information')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mutual_information.png'), dpi=300)
        plt.close()
        
        # Log top features
        logger.info(f"Top {min(n_top, len(mi_df))} features by mutual information:")
        for i, (feature, score) in enumerate(zip(mi_df['Feature'].head(n_top), mi_df['MI_Score'].head(n_top))):
            logger.info(f"{i+1}. {feature}: {score:.6f}")
        
        return mi_df
    
    def enhanced_feature_selection(self, X, y, importance_df, corr_threshold=0.8):
        """
        Enhanced feature selection with correlation pruning
        
        Args:
            X: Feature DataFrame
            y: Target Series
            importance_df: DataFrame with feature importance scores
            corr_threshold: Correlation threshold for feature pruning
            
        Returns:
            List of selected features
        """
        # Get features sorted by importance
        sorted_features = importance_df['Feature'].tolist()
        
        # Calculate feature correlations
        corr_matrix = X[sorted_features].corr().abs()
        
        # Initialize selected features with the most important feature
        selected_features = [sorted_features[0]]
        remaining_features = sorted_features[1:]
        
        # Iteratively add features that are not highly correlated with already selected ones
        for feature in remaining_features:
            # Check if feature is highly correlated with any selected feature
            is_correlated = False
            for selected in selected_features:
                if corr_matrix.loc[feature, selected] > corr_threshold:
                    is_correlated = True
                    break
            
            # If not highly correlated, add to selected features
            if not is_correlated:
                selected_features.append(feature)
                
            # Stop when we have a reasonable number of features (e.g., 30)
            if len(selected_features) >= 30:
                break
        
        # Evaluate selected feature subset performance
        metrics = self.evaluate_feature_subset(X, y, selected_features)
        
        logger.info(f"Selected {len(selected_features)} features with correlation pruning")
        logger.info(f"Performance metrics: {metrics}")
        
        return selected_features
    
    def time_aware_feature_importance(self, X, y, n_splits=5):
        """
        Calculate feature importance with time-series awareness
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of time splits
            
        Returns:
            DataFrame with time-aware importance scores
        """
        # Create time-series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Importance scores across splits
        importances = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = LGBMClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Get feature importance for this split
            split_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            })
            
            importances.append(split_importance)
        
        # Average importance across splits
        avg_importance = pd.concat(importances).groupby('Feature').mean().reset_index()
        avg_importance = avg_importance.sort_values('Importance', ascending=False)
        
        # Calculate importance stability (standard deviation across splits)
        std_importance = pd.concat(importances).groupby('Feature').std().reset_index()
        std_importance.columns = ['Feature', 'Std_Importance']
        
        # Combine average and stability
        combined = pd.merge(avg_importance, std_importance, on='Feature')
        combined['Stability'] = 1 - (combined['Std_Importance'] / (combined['Importance'] + 1e-6))
        combined = combined.sort_values('Importance', ascending=False)
        
        return combined
    
    def evaluate_feature_groups(self, X, y):
        """
        Evaluate performance of different feature groups
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with group performance metrics
        """
        # Identify feature groups
        feature_groups = self.identify_feature_groups(X)
        
        # Evaluate each group
        results = []
        
        for group_name, features in feature_groups.items():
            if len(features) < 2:
                continue
                
            try:
                metrics = self.evaluate_feature_subset(X, y, features)
                results.append({
                    'Group': group_name,
                    'Feature_Count': len(features),
                    **metrics
                })
            except Exception as e:
                logger.warning(f"Error evaluating group {group_name}: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot group performance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Group', y='f1', data=results_df)
        plt.title('Feature Group Performance (F1 Score)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'group_performance.png'), dpi=300)
        plt.close()
        
        return results_df


    # Add a new method for analyzing feature characteristics
    def check_feature_characteristics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze characteristics of features
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with feature characteristics
        """
        logger.info("Analyzing feature characteristics...")
        
        # Identify different types of features
        results = {
            'categorical_features': [],
            'binary_features': [],
            'temporal_features': [],
            'technical_indicators': [],
            'volume_features': [],
            'price_features': [],
            'derived_features': [],
            'lag_features': []
        }
        
        # Identify binary features (tinyint or values are mostly 0 and 1)
        for col in X.columns:
            unique_vals = X[col].nunique()
            
            # Check for binary features
            if unique_vals <= 2 or ('is_' in col or 'tinyint' in str(X[col].dtype)):
                results['binary_features'].append(col)
            
            # Identify lag features
            elif '_lag' in col:
                results['lag_features'].append(col)
                
            # Identify temporal features
            elif any(x in col for x in ['day', 'month', 'quarter', 'year', 'weekday']):
                results['temporal_features'].append(col)
            
            # Identify volume features
            elif 'volume' in col:
                results['volume_features'].append(col)
            
            # Identify price features (open, high, low, close)
            elif any(x in col for x in ['open', 'high', 'low', 'close', 'price']):
                results['price_features'].append(col)
            
            # Technical indicators
            elif any(x in col for x in ['sma', 'ema', 'macd', 'rsi', 'atr', 'adx', 'bollinger', 'stochastic']):
                results['technical_indicators'].append(col)
            
            # Derived features
            else:
                results['derived_features'].append(col)
        
        # Log summary
        for category, features in results.items():
            logger.info(f"Found {len(features)} {category}")
        
        return results
    
    def check_multicollinearity(self, X: pd.DataFrame, threshold: float = 0.9) -> Dict[str, Any]:
        """
        Check for multicollinearity among features with optimizations for larger feature sets
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold to identify multicollinearity
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        logger.info("Checking for multicollinearity...")
        
        # For very large feature sets, analyze by feature groups
        n_features = X.shape[1]
        
        if n_features > 150:
            logger.info(f"Large feature set detected ({n_features} features). Using optimized correlation analysis.")
            # Get feature characteristics
            feature_chars = self.check_feature_characteristics(X)
            
            # Create groups of related features
            feature_groups = []
            for category in ['price_features', 'technical_indicators', 'volume_features', 'temporal_features', 'lag_features']:
                if feature_chars[category]:
                    feature_groups.append(feature_chars[category])
            
            # Limit each group to a reasonable size
            max_group_size = 50
            limited_groups = []
            for group in feature_groups:
                if len(group) > max_group_size:
                    limited_groups.append(group[:max_group_size])
                else:
                    limited_groups.append(group)
            
            # Analyze correlations within each group
            highly_correlated_features = []
            for group in limited_groups:
                if len(group) < 2:
                    continue
                    
                group_df = X[group]
                corr_matrix = group_df.corr().abs()
                
                # Get upper triangle
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Find features with correlation greater than threshold
                for i in range(len(upper.index)):
                    for j in range(len(upper.columns)):
                        val = upper.iloc[i, j]
                        if val > threshold:
                            highly_correlated_features.append((upper.index[i], upper.columns[j], val))
            
            # Set corr_matrix to None - we'll create visualizations by group
            corr_matrix = None
        else:
            # Standard approach for smaller feature sets
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
        
        # Calculate VIF - for large feature sets, use a subset
        vif_feature_limit = 100
        
        try:
            if n_features > vif_feature_limit:
                # Use top features from correlation analysis
                features_for_vif = list(set([x[0] for x in highly_correlated_features[:50]] + 
                                        [x[1] for x in highly_correlated_features[:50]]))
                # Limit to a reasonable number
                features_for_vif = features_for_vif[:vif_feature_limit]
                X_vif = X[features_for_vif]
            else:
                X_vif = X
                
            # Prepare a sample to reduce computation time
            sample_size = min(5000, len(X))
            X_sample = X_vif.sample(sample_size, random_state=42) if len(X_vif) > sample_size else X_vif
            
            # Calculate VIF for each feature
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
        
        # Create correlation visualizations
        if n_features <= 150 and corr_matrix is not None:
            plt.figure(figsize=(20, 16))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300)
            plt.close()
        else:
            # Create multiple correlation heatmaps for groups of features
            feature_chars = self.check_feature_characteristics(X)
            for category, features in feature_chars.items():
                if len(features) >= 2 and len(features) <= 100:
                    plt.figure(figsize=(20, 16))
                    group_corr = X[features].corr().abs()
                    sns.heatmap(group_corr, annot=False, cmap='coolwarm', linewidths=0.5)
                    plt.title(f'Correlation Matrix: {category}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'correlation_matrix_{category}.png'), dpi=300)
                    plt.close()
        
        return results

    # Add a method to identify logical feature groups
    def identify_feature_groups(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify logical groupings of features to aid in analysis
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with feature groups
        """
        # Define regex patterns for different feature types
        patterns = {
            'price_basic': [r'^open$', r'^high$', r'^low$', r'^close$', r'^volume$'],
            'price_derived': [r'price_', r'_range$', r'_return_', r'breakout'],
            'moving_averages': [r'sma_', r'ema_', r'vama_'],
            'oscillators': [r'rsi_', r'stochastic_', r'williams_r', r'mfi_', r'cci_'],
            'trend_indicators': [r'macd_', r'adx_', r'di_', r'cmf_', r'vpci_'],
            'volatility_indicators': [r'bollinger_', r'atr_', r'stddev', r'var', r'zscore_'],
            'volume_indicators': [r'volume_', r'pvt', r'eom_', r'volume_oscillator'],
            'lag_variables': [r'_lag\d+$'],
            'relative_strength': [r'rs_', r'nifty_corr_', r'sector_strength'],
            'categorical': [r'is_', r'day_of_', r'month$', r'quarter$', r'market_regime']
        }
        
        # Initialize groups
        groups = {group: [] for group in patterns.keys()}
        
        # Categorize each feature
        for col in X.columns:
            found_match = False
            for group, regexes in patterns.items():
                for regex in regexes:
                    if re.search(regex, col):
                        groups[group].append(col)
                        found_match = True
                        break
                if found_match:
                    break
            
            # If no match found, add to misc
            if not found_match:
                if 'misc' not in groups:
                    groups['misc'] = []
                groups['misc'].append(col)
        
        # Log summary
        for group, features in groups.items():
            if features:  # Only log non-empty groups
                logger.info(f"Feature group '{group}': {len(features)} features")
        
        return groups
    
    def feature_importance_rf(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest with better handling for larger feature sets
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info("Calculating feature importance using Random Forest...")
        
        # Get feature characteristics
        feature_chars = self.check_feature_characteristics(X)
        
        # Time series split for more robust importance estimation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize importance dataframe
        feature_importances = pd.DataFrame()
        feature_importances['Feature'] = X.columns
        
        # Adjust parameters based on feature count
        n_estimators = min(100, max(50, int(X.shape[1] / 5)))
        max_depth = min(15, max(5, int(np.log2(X.shape[1]) * 2)))
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
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
        
        # Save feature importance plot - adjust for larger feature sets
        plt.figure(figsize=(14, 12))
        top_n = min(50, len(feature_importances))
        sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n))
        plt.title(f'Top {top_n} Features by Random Forest Importance')
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
            # For large feature sets, use a sample and top features
            n_features = X.shape[1]
            sample_size = min(2000, len(X))
            
            # If feature set is very large, select top features from RF importance first
            if n_features > 100:
                logger.info(f"Large feature set detected ({n_features} features). Selecting top features for SHAP analysis.")
                # Get feature importance
                rf_importance = self.feature_importance_rf(X, y)
                # Select top 100 features
                top_features = rf_importance['Feature'].head(100).tolist()
                X_top = X[top_features]
                logger.info(f"Selected top {len(top_features)} features for SHAP analysis")
            else:
                X_top = X
            
            X_sample = X_top.sample(sample_size, random_state=42) if len(X_top) > sample_size else X_top
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
                'Feature': X_sample.columns,
                'SHAP_Value': mean_abs_shap
            })
            feature_importance = feature_importance.sort_values('SHAP_Value', ascending=False)
            
            # Save to CSV
            feature_importance.to_csv(os.path.join(self.output_dir, 'shap_importance.csv'), index=False)
            
            # Create dependence plots only for top features (limit to avoid too many plots)
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
        
        # For very large feature sets, first filter with a simple importance method
        if X.shape[1] > 100:
            logger.info(f"Large feature set detected ({X.shape[1]} features). Pre-filtering features.")
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importance = rf.feature_importances_
            
            # Keep top 100 features or features with importance > 0.001, whichever is more
            top_features_idx = np.argsort(importance)[::-1][:100]
            top_features = X.columns[top_features_idx].tolist()
            
            important_features = X.columns[importance > 0.001].tolist()
            
            # Combine both selection methods
            selected_features = list(set(top_features).union(set(important_features)))
            logger.info(f"Pre-filtered to {len(selected_features)} features for RFE")
            
            # Use the filtered feature set
            X_filtered = X[selected_features]
        else:
            X_filtered = X
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Base estimator
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Adjust step size based on number of features
        step_size = max(1, min(5, X_filtered.shape[1] // 20))
        
        # RFECV
        selector = RFECV(
            estimator=estimator,
            step=step_size,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            min_features_to_select=10
        )
        
        selector.fit(X_scaled, y)
        
        # Get selected features
        selected_features = X_filtered.columns[selector.support_].tolist()
        
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
    
    def comparative_importance_analysis(self, rf_importance, lgbm_importance, mi_df, time_aware_importance):
        """
        Create visualizations comparing different feature importance methods
        """
        # Get top 20 features from each method
        top_rf = rf_importance.head(20)['Feature'].tolist()
        top_lgbm = lgbm_importance.head(20)['Feature'].tolist()
        top_mi = mi_df.head(20)['Feature'].tolist()
        top_time = time_aware_importance.head(20)['Feature'].tolist()
        
        # Create a set of all unique top features
        all_top = list(set(top_rf + top_lgbm + top_mi + top_time))
        
        # Create a DataFrame showing which features appear in which methods
        comparison = pd.DataFrame(index=all_top)
        comparison['RF'] = [1 if f in top_rf else 0 for f in all_top]
        comparison['LGBM'] = [1 if f in top_lgbm else 0 for f in all_top]
        comparison['MI'] = [1 if f in top_mi else 0 for f in all_top]
        comparison['Time_Aware'] = [1 if f in top_time else 0 for f in all_top]
        comparison['Total'] = comparison.sum(axis=1)
        comparison = comparison.sort_values('Total', ascending=False)
        
        # Plot heatmap
        plt.figure(figsize=(12, 14))
        sns.heatmap(comparison.drop('Total', axis=1), cmap='YlGnBu', cbar=False, linewidths=.5)
        plt.title('Features Identified as Important by Different Methods')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'importance_comparison.png'), dpi=300)
        plt.close()
        
        # Return features that appear in multiple methods
        robust_features = comparison[comparison['Total'] >= 3].index.tolist()
        return robust_features
    
    def enhanced_correlation_analysis(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.9) -> Dict[str, Any]:
        """
        Perform enhanced correlation analysis including Spearman rank correlation
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Correlation threshold
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Performing enhanced correlation analysis...")
        
        # Pearson correlation (linear relationships)
        pearson_corr = X.corr().abs()
        
        # Spearman rank correlation (monotonic relationships)
        spearman_corr = X.corr(method='spearman').abs()
        
        # Get feature-to-target correlations
        target_pearson = pd.DataFrame({
            'Feature': X.columns,
            'Pearson_Corr_Target': [abs(np.corrcoef(X[col], y)[0,1]) for col in X.columns]
        }).sort_values('Pearson_Corr_Target', ascending=False)
        
        target_spearman = pd.DataFrame({
            'Feature': X.columns,
            'Spearman_Corr_Target': [abs(spearmanr(X[col], y)[0]) for col in X.columns]
        }).sort_values('Spearman_Corr_Target', ascending=False)
        
        # Combined target correlation
        target_combined = pd.merge(target_pearson, target_spearman, on='Feature')
        target_combined['Combined_Rank'] = target_combined['Pearson_Corr_Target'].rank(ascending=False) + \
                                        target_combined['Spearman_Corr_Target'].rank(ascending=False)
        target_combined = target_combined.sort_values('Combined_Rank')
        
        # Find features with correlation divergence (high Spearman, low Pearson or vice versa)
        target_combined['Corr_Difference'] = abs(target_combined['Spearman_Corr_Target'] - target_combined['Pearson_Corr_Target'])
        divergent_features = target_combined.nlargest(10, 'Corr_Difference')
        
        # Create visualizations
        self._plot_correlation_comparison(target_combined)
        self._plot_divergent_features(divergent_features)
        
        # Save results
        target_combined.to_csv(os.path.join(self.output_dir, 'correlation_analysis.csv'), index=False)
        
        logger.info(f"Enhanced correlation analysis completed")
        logger.info(f"Found {len(divergent_features)} features with notable correlation divergence")
        
        return {
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'target_correlation': target_combined,
            'divergent_features': divergent_features
        }

    def _plot_correlation_comparison(self, df: pd.DataFrame, top_n: int = 20) -> None:
        """Plot comparison of Pearson vs Spearman correlations"""
        plt.figure(figsize=(12, 8))
        
        # Get top features by combined rank
        top_features = df.head(top_n)
        
        # Create grouped bar chart
        x = np.arange(len(top_features))
        width = 0.35
        
        plt.bar(x - width/2, top_features['Pearson_Corr_Target'], width, label='Pearson')
        plt.bar(x + width/2, top_features['Spearman_Corr_Target'], width, label='Spearman')
        
        plt.xlabel('Features')
        plt.ylabel('Correlation with Target')
        plt.title('Pearson vs Spearman Correlation with Target')
        plt.xticks(x, top_features['Feature'], rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pearson_vs_spearman.png'), dpi=300)
        plt.close()
    
    def create_feature_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Create and evaluate multiple feature subsets based on different selection methods
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with ensemble results
        """
        logger.info("Creating feature ensemble...")
        
        # Generate different feature subsets
        feature_sets = {}
        
        try:
            # Subset 1: Random Forest importance-based
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            rf_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_sets['RF'] = rf_importance.head(30)['Feature'].tolist()
        except Exception as e:
            logger.error(f"Error creating RF feature set: {e}")
        
        try:
            # Subset 2: Correlation with target (Pearson)
            pearson_corr = []
            for col in X.columns:
                corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                pearson_corr.append((col, corr))
            pearson_df = pd.DataFrame(pearson_corr, columns=['Feature', 'Corr'])
            feature_sets['Pearson'] = pearson_df.sort_values('Corr', ascending=False).head(30)['Feature'].tolist()
        except Exception as e:
            logger.error(f"Error creating Pearson feature set: {e}")
        
        try:
            # Subset 3: Correlation with target (Spearman)
            spearman_corr = []
            for col in X.columns:
                corr = np.abs(spearmanr(X[col], y)[0])
                spearman_corr.append((col, corr))
            spearman_df = pd.DataFrame(spearman_corr, columns=['Feature', 'Corr'])
            feature_sets['Spearman'] = spearman_df.sort_values('Corr', ascending=False).head(30)['Feature'].tolist()
        except Exception as e:
            logger.error(f"Error creating Spearman feature set: {e}")
        
        try:
            # Subset 4: Mutual Information
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_df = pd.DataFrame({
                'Feature': X.columns,
                'MI': mi_scores
            }).sort_values('MI', ascending=False)
            feature_sets['MI'] = mi_df.head(30)['Feature'].tolist()
        except Exception as e:
            logger.error(f"Error creating MI feature set: {e}")
        
        # Evaluate each feature set
        results = []
        
        for name, features in feature_sets.items():
            try:
                metrics = self.evaluate_feature_subset(X, y, features)
                results.append({
                    'Method': name,
                    'Features': features,
                    'Feature_Count': len(features),
                    **metrics
                })
            except Exception as e:
                logger.error(f"Error evaluating {name} feature set: {e}")
        
        # Create combined feature set
        try:
            # Find features that appear in multiple sets
            all_features = []
            for features in feature_sets.values():
                all_features.extend(features)
            
            # Count occurrences
            feature_counts = pd.Series(all_features).value_counts()
            
            # Select features that appear in at least 2 selection methods
            ensemble_features = feature_counts[feature_counts >= 2].index.tolist()
            
            # Evaluate ensemble
            ensemble_metrics = self.evaluate_feature_subset(X, y, ensemble_features)
            
            results.append({
                'Method': 'Ensemble',
                'Features': ensemble_features,
                'Feature_Count': len(ensemble_features),
                **ensemble_metrics
            })
            
            feature_sets['Ensemble'] = ensemble_features
        except Exception as e:
            logger.error(f"Error creating ensemble feature set: {e}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Create visualization
        self._plot_ensemble_results(results_df)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'feature_ensemble_results.csv'), index=False)
        
        # Save Venn diagram of feature overlaps
        self._plot_feature_overlaps(feature_sets)
        
        logger.info(f"Feature ensemble analysis completed")
        logger.info(f"Best method: {results_df.loc[results_df['f1'].idxmax(), 'Method']} with F1 = {results_df['f1'].max():.4f}")
        
        return {
            'feature_sets': feature_sets,
            'results': results_df,
            'best_method': results_df.loc[results_df['f1'].idxmax(), 'Method'],
            'ensemble_features': feature_sets.get('Ensemble', [])
        }

    def _plot_ensemble_results(self, results_df: pd.DataFrame) -> None:
        """Plot performance metrics for different feature selection methods"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(len(metrics))
        width = 0.8 / len(results_df)
        
        for i, (idx, row) in enumerate(results_df.iterrows()):
            plt.bar(x + (i - len(results_df)/2 + 0.5) * width, 
                [row[m] for m in metrics], 
                width, 
                label=f"{row['Method']} ({row['Feature_Count']} features)")
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison of Feature Selection Methods')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_performance.png'), dpi=300)
        plt.close()
    
    def _plot_feature_overlaps(self, feature_sets: Dict[str, List[str]]) -> None:
        """Create visualization of feature set overlaps"""
        try:
            # This requires matplotlib_venn package
            from matplotlib_venn import venn2, venn3
            
            methods = list(feature_sets.keys())
            
            # If we have 2 or 3 methods (excluding Ensemble), create Venn diagram
            if 'Ensemble' in methods:
                methods.remove('Ensemble')
            
            if len(methods) == 2:
                plt.figure(figsize=(10, 8))
                set1 = set(feature_sets[methods[0]])
                set2 = set(feature_sets[methods[1]])
                
                venn2([set1, set2], (methods[0], methods[1]))
                plt.title('Feature Overlap Between Selection Methods')
                
                plt.savefig(os.path.join(self.output_dir, 'feature_overlap_venn2.png'), dpi=300)
                plt.close()
                
            elif len(methods) == 3:
                plt.figure(figsize=(10, 8))
                set1 = set(feature_sets[methods[0]])
                set2 = set(feature_sets[methods[1]])
                set3 = set(feature_sets[methods[2]])
                
                venn3([set1, set2, set3], (methods[0], methods[1], methods[2]))
                plt.title('Feature Overlap Between Selection Methods')
                
                plt.savefig(os.path.join(self.output_dir, 'feature_overlap_venn3.png'), dpi=300)
                plt.close()
            
            else:
                # For more than 3 methods, create an overlap matrix
                plt.figure(figsize=(10, 8))
                
                overlap_matrix = np.zeros((len(methods), len(methods)))
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i == j:
                            overlap_matrix[i, j] = 1.0
                        else:
                            set1 = set(feature_sets[method1])
                            set2 = set(feature_sets[method2])
                            
                            # Jaccard similarity
                            if len(set1.union(set2)) > 0:
                                overlap_matrix[i, j] = len(set1.intersection(set2)) / len(set1.union(set2))
                
                sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                            xticklabels=methods, yticklabels=methods)
                plt.title('Feature Overlap Between Selection Methods (Jaccard Similarity)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'feature_overlap_matrix.png'), dpi=300)
                plt.close()
                
        except ImportError:
            logger.warning("matplotlib-venn package not found. Creating alternative visualization.")
            
            # Create overlap matrix instead
            plt.figure(figsize=(10, 8))
            
            methods = list(feature_sets.keys())
            if 'Ensemble' in methods:
                methods.remove('Ensemble')
                
            overlap_matrix = np.zeros((len(methods), len(methods)))
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                    else:
                        set1 = set(feature_sets[method1])
                        set2 = set(feature_sets[method2])
                        
                        # Jaccard similarity
                        if len(set1.union(set2)) > 0:
                            overlap_matrix[i, j] = len(set1.intersection(set2)) / len(set1.union(set2))
            
            sns.heatmap(overlap_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
                    xticklabels=methods, yticklabels=methods)
            plt.title('Feature Overlap Between Selection Methods (Jaccard Similarity)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_overlap_matrix.png'), dpi=300)
            plt.close()
        
        except Exception as e:
            logger.error(f"Error creating feature overlap visualization: {e}")
    
    def enhanced_time_series_validation(self, X: pd.DataFrame, y: pd.Series, 
                                  features: List[str],
                                  n_splits: int = 5,
                                  expanding_window: bool = True) -> Dict[str, Any]:
        """
        Enhanced time-series validation with expanding/sliding windows
        
        Args:
            X: Feature DataFrame 
            y: Target Series
            features: List of features to use
            n_splits: Number of time splits
            expanding_window: Whether to use expanding window (vs sliding window)
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Performing enhanced time-series validation with {'expanding' if expanding_window else 'sliding'} window...")
        
        # Select features
        X_subset = X[features]
        
        # Get indices sorted by time
        indices = np.arange(len(X_subset))
        
        # Define split sizes
        if expanding_window:
            # Expanding window: initial_train_size increases while test_size stays constant
            test_size = len(indices) // (n_splits + 1)  # Reserve portion for testing
            initial_train_size = len(indices) // (n_splits + 1) * 2  # Start with double the test size
            remaining = len(indices) - initial_train_size - test_size * (n_splits - 1)
            
            splits = []
            train_start = 0
            
            for i in range(n_splits):
                if i == 0:
                    train_end = initial_train_size
                else:
                    train_end = initial_train_size + test_size * i + (remaining if i == n_splits - 1 else 0)
                    
                test_start = train_end
                test_end = test_start + test_size
                
                # Ensure we don't go beyond the dataset
                test_end = min(test_end, len(indices))
                
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                
                splits.append((train_indices, test_indices))
                
                if test_end >= len(indices):
                    break
        else:
            # Sliding window: train_size and test_size remain constant
            train_size = len(indices) // (n_splits + 1) * 2
            test_size = len(indices) // (n_splits + 1)
            
            splits = []
            
            for i in range(n_splits):
                train_start = i * test_size
                train_end = train_start + train_size
                
                test_start = train_end
                test_end = test_start + test_size
                
                # Ensure we don't go beyond the dataset
                test_end = min(test_end, len(indices))
                
                train_indices = indices[train_start:train_end]
                test_indices = indices[test_start:test_end]
                
                splits.append((train_indices, test_indices))
                
                if test_end >= len(indices):
                    break
        
        # Initialize metrics tracking
        metrics_by_split = []
        feature_importance_by_split = []
        
        # Train and evaluate across splits
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)} with {len(train_idx)} train samples and {len(test_idx)} test samples")
            
            X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            split_metrics = {
                'split': i+1,
                'train_start': X.index[train_idx[0]],
                'train_end': X.index[train_idx[-1]],
                'test_start': X.index[test_idx[0]],
                'test_end': X.index[test_idx[-1]],
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba)
            }
            
            metrics_by_split.append(split_metrics)
            
            # Track feature importance
            importance = pd.DataFrame({
                'Feature': X_subset.columns,
                f'Importance_Split_{i+1}': model.feature_importances_
            })
            
            feature_importance_by_split.append(importance)
        
        # Combine feature importance dataframes
        importance_df = feature_importance_by_split[0]
        for i in range(1, len(feature_importance_by_split)):
            importance_df = pd.merge(importance_df, feature_importance_by_split[i], on='Feature')
        
        # Calculate stability metrics
        importance_cols = [col for col in importance_df.columns if col.startswith('Importance_Split_')]
        importance_df['Mean_Importance'] = importance_df[importance_cols].mean(axis=1)
        importance_df['Std_Importance'] = importance_df[importance_cols].std(axis=1)
        importance_df['CV'] = importance_df['Std_Importance'] / (importance_df['Mean_Importance'] + 1e-10)
        importance_df['Stability'] = 1 - importance_df['CV'] 
        
        # Sort by mean importance
        importance_df = importance_df.sort_values('Mean_Importance', ascending=False)
        
        # Create visualizations
        self._plot_cv_metrics(pd.DataFrame(metrics_by_split))
        self._plot_feature_stability_across_splits(importance_df)
        
        # Save results
        pd.DataFrame(metrics_by_split).to_csv(os.path.join(self.output_dir, 'cv_metrics.csv'), index=False)
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance_cv.csv'), index=False)
        
        logger.info(f"Enhanced time-series validation completed")
        
        return {
            'metrics_by_split': metrics_by_split,
            'feature_importance': importance_df,
            'mean_metrics': {k: np.mean([split[k] for split in metrics_by_split]) 
                            for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}
        }

    def _plot_cv_metrics(self, df: pd.DataFrame) -> None:
        """Plot metrics across CV splits"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for metric in metrics:
            plt.plot(df['split'], df[metric], marker='o', label=metric)
        
        plt.xlabel('Split Number')
        plt.ylabel('Score')
        plt.title('Performance Metrics Across Time Splits')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cv_metrics.png'), dpi=300)
        plt.close()

    def _plot_feature_stability_across_splits(self, df: pd.DataFrame, top_n: int = 20) -> None:
        """Plot feature importance stability across CV splits"""
        # Get top features
        top_features = df.head(top_n)['Feature'].tolist()
        
        # Select importance columns
        importance_cols = [col for col in df.columns if col.startswith('Importance_Split_')]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        heatmap_data = df[df['Feature'].isin(top_features)].set_index('Feature')[importance_cols]
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.3f')
        plt.title(f'Top {top_n} Features: Importance Across Time Splits')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance_splits.png'), dpi=300)
        plt.close()
        
        # Create stability vs importance plot
        plt.figure(figsize=(12, 8))
        
        top_df = df.head(top_n)
        plt.scatter(top_df['Mean_Importance'], top_df['Stability'], alpha=0.7)
        
        for _, row in top_df.iterrows():
            plt.annotate(row['Feature'], 
                        (row['Mean_Importance'], row['Stability']),
                        xytext=(5, 5),
                        textcoords='offset points')
        
        plt.xlabel('Mean Importance')
        plt.ylabel('Stability (1 - CV)')
        plt.title('Feature Importance Stability vs Mean Importance Across Time Splits')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_stability_time.png'), dpi=300)
        plt.close()

    def _plot_divergent_features(self, df: pd.DataFrame) -> None:
        """Plot features with divergent Pearson and Spearman correlations"""
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df['Pearson_Corr_Target'], width, label='Pearson')
        plt.bar(x + width/2, df['Spearman_Corr_Target'], width, label='Spearman')
        
        plt.xlabel('Features')
        plt.ylabel('Correlation with Target')
        plt.title('Features with Divergent Correlation Types (Potential Non-Linear Relationships)')
        plt.xticks(x, df['Feature'], rotation=90)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'divergent_correlations.png'), dpi=300)
        plt.close()
    
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
            
            # 4. Enhanced Correlation Analysis (NEW)
            try:
                enhanced_corr_results = self.enhanced_correlation_analysis(X, y)
                results['enhanced_correlation'] = {
                    'top_pearson': enhanced_corr_results['target_correlation'].head(20)['Pearson_Corr_Target'].to_dict(),
                    'top_spearman': enhanced_corr_results['target_correlation'].head(20)['Spearman_Corr_Target'].to_dict(),
                    'divergent_features': enhanced_corr_results['divergent_features']['Feature'].tolist()
                }
            except Exception as e:
                logger.error(f"Error in enhanced correlation analysis: {e}", exc_info=True)
                results['enhanced_correlation'] = {'error': str(e)}
            
            # 5. Check multicollinearity (existing)
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
            
            # 6. Calculate feature importance with Random Forest
            try:
                rf_importance = self.feature_importance_rf(X, y)
                results['rf_importance'] = rf_importance
            except Exception as e:
                logger.error(f"Error in RF importance analysis: {e}", exc_info=True)
                rf_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.ones(len(X.columns))})
                results['rf_importance'] = {'error': str(e)}
            
            # 7. Calculate feature importance with LightGBM
            try:
                lgbm_importance = self.feature_importance_lgbm(X, y)
                results['lgbm_importance'] = lgbm_importance
            except Exception as e:
                logger.error(f"Error in LightGBM importance analysis: {e}", exc_info=True)
                lgbm_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.ones(len(X.columns))})
                results['lgbm_importance'] = {'error': str(e)}
            
            # 8. Combined feature importance
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
            
            # 9. Individual Feature Contribution Analysis (NEW)
            try:
                # Get top 20 features from combined ranking
                top_features = combined_ranks.head(20)['Feature'].tolist()
                self.analyze_individual_feature_contributions(X, y, top_features=top_features)
                results['individual_features'] = {'completed': True, 'top_features': top_features}
            except Exception as e:
                logger.error(f"Error in individual feature analysis: {e}", exc_info=True)
                results['individual_features'] = {'error': str(e)}
            
            # 10. Feature Ensemble Approach (NEW)
            try:
                ensemble_results = self.create_feature_ensemble(X, y)
                results['feature_ensemble'] = {
                    'best_method': ensemble_results['best_method'],
                    'ensemble_features': ensemble_results['ensemble_features']
                }
            except Exception as e:
                logger.error(f"Error in feature ensemble analysis: {e}", exc_info=True)
                results['feature_ensemble'] = {'error': str(e)}
            
            # 11. Market Regime Analysis (NEW) 
            try:
                regime_results = self.analyze_feature_stability_across_regimes(X, y)
                results['regime_analysis'] = {
                    'robust_features': regime_results.get('robust_features', [])
                }
            except Exception as e:
                logger.error(f"Error in market regime analysis: {e}", exc_info=True)
                results['regime_analysis'] = {'error': str(e)}
            
            # 12. Enhanced Feature Selection with Pruning (NEW)
            try:
                pruning_results = self.enhanced_feature_selection_with_pruning(X, y, combined_ranks[['Feature', 'Avg_Rank']])
                results['pruned_features'] = {
                    'features': pruning_results['selected_features'],
                    'feature_count': pruning_results['feature_count'],
                    'metrics': pruning_results['pruned_metrics']
                }
            except Exception as e:
                logger.error(f"Error in enhanced feature selection: {e}", exc_info=True)
                results['pruned_features'] = {'error': str(e)}
            
            # 13. Enhanced Time-Series Validation (NEW)
            try:
                # Use top features from either pruned set or combined ranking
                if 'pruned_features' in results and 'error' not in results['pruned_features']:
                    features_for_cv = results['pruned_features']['features']
                else:
                    features_for_cv = combined_ranks.head(30)['Feature'].tolist()
                    
                cv_results = self.enhanced_time_series_validation(X, y, features=features_for_cv, expanding_window=True)
                results['enhanced_cv'] = {
                    'mean_metrics': cv_results['mean_metrics']
                }
            except Exception as e:
                logger.error(f"Error in enhanced time-series validation: {e}", exc_info=True)
                results['enhanced_cv'] = {'error': str(e)}
            
            # Continue with existing analyses (PCA, SHAP, RFE)
            # 14. PCA analysis
            try:
                pca_results, explained_variance = self.pca_analysis(X)
                results['pca'] = {
                    'components_for_95_var': np.argmax(explained_variance >= 0.95) + 1,
                    'top_component_features': pca_results
                }
            except Exception as e:
                logger.error(f"Error in PCA analysis: {e}", exc_info=True)
                results['pca'] = {'error': str(e), 'components_for_95_var': X.shape[1]}
            
            # 15. SHAP analysis for deeper understanding
            try:
                self.shap_analysis(X, y)
                results['shap'] = {'completed': True}
            except Exception as e:
                logger.error(f"Error in SHAP analysis: {e}", exc_info=True)
                results['shap'] = {'error': str(e)}
            
            # 16. RFE for feature selection
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
            
            # 17. Find optimal feature subset (original method)
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
            
            # 18. Determine FINAL recommended feature set
            try:
                # Collect features from different methods
                feature_sets = {
                    'optimal': results['optimal_features']['features'],
                    'pruned': results.get('pruned_features', {}).get('features', []),
                    'ensemble': results.get('feature_ensemble', {}).get('ensemble_features', []),
                    'robust': results.get('regime_analysis', {}).get('robust_features', []),
                    'rfe': results['rfe']['selected_features']
                }
                
                # Count occurrences
                all_features = []
                for name, features in feature_sets.items():
                    if features:  # Only add non-empty feature sets
                        all_features.extend(features)
                
                feature_counts = pd.Series(all_features).value_counts()
                
                # Select features that appear in at least 2 selection methods
                final_features = feature_counts[feature_counts >= 2].index.tolist()
                
                # If we don't have enough features, use the optimal set
                if len(final_features) < 10:
                    final_features = results['optimal_features']['features']
                
                # Evaluate final feature set
                final_metrics = self.evaluate_feature_subset(X, y, final_features)
                
                results['final_features'] = {
                    'features': final_features,
                    'feature_count': len(final_features),
                    'metrics': final_metrics
                }
                
                # Save final feature set to text file
                with open(os.path.join(self.output_dir, 'final_features.txt'), 'w') as f:
                    for feature in final_features:
                        f.write(f"{feature}\n")
                        
                # Save final feature set as pickle
                joblib.dump(final_features, os.path.join(self.output_dir, 'final_features.pkl'))
                    
            except Exception as e:
                logger.error(f"Error determining final feature set: {e}", exc_info=True)
                results['final_features'] = {'error': str(e)}
                # Use optimal features as fallback
                final_features = optimal_features
            
            # 19. Save final report
            try:
                self.save_enhanced_report(results, rf_importance, lgbm_importance, combined_ranks, final_features)
            except Exception as e:
                logger.error(f"Error saving enhanced report: {e}", exc_info=True)
            
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
    
    def save_enhanced_report(self, results: Dict[str, Any], rf_importance: pd.DataFrame, 
                        lgbm_importance: pd.DataFrame, combined_ranks: pd.DataFrame,
                        final_features: List[str]) -> None:
        """
        Save enhanced analysis report with additional visualizations
        
        Args:
            results: Analysis results dictionary
            rf_importance: Random Forest importance DataFrame
            lgbm_importance: LightGBM importance DataFrame
            combined_ranks: Combined feature rankings DataFrame
            final_features: Final selected feature set
        """
        try:
            # Save CSV files
            rf_importance.to_csv(os.path.join(self.output_dir, 'rf_importance.csv'), index=False)
            lgbm_importance.to_csv(os.path.join(self.output_dir, 'lgbm_importance.csv'), index=False)
            combined_ranks.to_csv(os.path.join(self.output_dir, 'combined_rankings.csv'), index=False)
            
            # Save final features to text file
            with open(os.path.join(self.output_dir, 'final_features.txt'), 'w') as f:
                for feature in final_features:
                    f.write(f"{feature}\n")
            
            # Create comparison visualization of methods
            self._create_method_comparison_visualization(results)
            
            # Create HTML report (adapted from original save_report method)
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Feature Analysis Report</title>
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
                    .method-comparison { display: flex; flex-wrap: wrap; justify-content: space-between; }
                    .method-box { width: 48%; margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Enhanced Feature Analysis Report</h1>
                    <div class="summary">
                        <h2>Executive Summary</h2>
                        <p>This report presents the results of a comprehensive feature analysis for a trading prediction model using multiple selection techniques.</p>
                        <ul>
                            <li><strong>Data Coverage:</strong> {start_date} to {end_date}</li>
                            <li><strong>Symbols Analyzed:</strong> {symbol_count}</li>
                            <li><strong>Total Samples:</strong> {sample_count}</li>
                            <li><strong>Original Features:</strong> {feature_count}</li>
                            <li><strong>Final Feature Count:</strong> {final_feature_count}</li>
                            <li><strong>Top Feature Categories:</strong> Correlation features, Technical indicators, Temporal features</li>
                        </ul>
                    </div>
                    
                    <h2>Feature Selection Methods Comparison</h2>
                    <div class="figure">
                        <img src="methods_comparison.png" alt="Feature Selection Methods Comparison">
                        <p>Figure: Performance comparison of different feature selection methods</p>
                    </div>
                    
                    <div class="method-comparison">
                        <div class="method-box">
                            <h3>Raw Importance Rankings</h3>
                            <p>Features ranked by direct importance from tree-based models</p>
                            <ul>
                                <li><strong>RF Top Feature:</strong> {rf_top}</li>
                                <li><strong>LGBM Top Feature:</strong> {lgbm_top}</li>
                                <li><strong>Feature Count:</strong> {optimal_count}</li>
                                <li><strong>F1 Score:</strong> {optimal_f1:.4f}</li>
                            </ul>
                        </div>
                        
                        <div class="method-box">
                            <h3>Correlation Pruning</h3>
                            <p>Features selected with multicollinearity reduction</p>
                            <ul>
                                <li><strong>Top Feature:</strong> {pruned_top}</li>
                                <li><strong>Feature Count:</strong> {pruned_count}</li>
                                <li><strong>F1 Score:</strong> {pruned_f1:.4f}</li>
                                <li><strong>Redundancy Reduction:</strong> {redundancy_reduction}%</li>
                            </ul>
                        </div>
                        
                        <div class="method-box">
                            <h3>Feature Ensemble</h3>
                            <p>Features that appear important across multiple methods</p>
                            <ul>
                                <li><strong>Best Method:</strong> {best_ensemble_method}</li>
                                <li><strong>Feature Count:</strong> {ensemble_count}</li>
                                <li><strong>F1 Score:</strong> {ensemble_f1:.4f}</li>
                            </ul>
                        </div>
                        
                        <div class="method-box">
                            <h3>Market Regime Stability</h3>
                            <p>Features that maintain importance across market conditions</p>
                            <ul>
                                <li><strong>Top Stable Feature:</strong> {top_stable}</li>
                                <li><strong>Feature Count:</strong> {stable_count}</li>
                                <li><strong>High Stability Features:</strong> {high_stability_pct}%</li>
                            </ul>
                        </div>
                    </div>
                    
                    <h2>Final Feature Set</h2>
                    <p>The final recommended feature set combines insights from all methods, prioritizing features that show importance, stability, and non-redundancy.</p>
                    
                    <h3>Final Selected Features ({final_feature_count})</h3>
                    <table>
                        <tr>
                            <th>#</th>
                            <th>Feature</th>
                            <th>Selection Methods</th>
                            <th>Importance Rank</th>
                        </tr>
                        {final_feature_rows}
                    </table>
                    
                    <div class="figure">
                        <img src="feature_sets_comparison.png" alt="Feature Sets Comparison">
                        <p>Figure: Performance comparison between original and final feature sets</p>
                    </div>
                    
                    <h2>Feature Stability Analysis</h2>
                    <div class="figure">
                        <img src="stability_vs_importance.png" alt="Feature Stability vs Importance">
                        <p>Figure: Feature stability vs importance across market regimes</p>
                    </div>
                    
                    <h2>Individual Feature Analysis</h2>
                    <div class="figure">
                        <img src="individual_feature_f1.png" alt="Individual Feature F1 Scores">
                        <p>Figure: Predictive power of individual features</p>
                    </div>
                    
                    <h2>Time-Series Validation</h2>
                    <div class="figure">
                        <img src="cv_metrics.png" alt="Cross-Validation Metrics">
                        <p>Figure: Performance metrics across time periods</p>
                    </div>
                    
                    <h2>Correlation Analysis</h2>
                    <div class="figure">
                        <img src="pearson_vs_spearman.png" alt="Pearson vs Spearman Correlation">
                        <p>Figure: Comparison of linear vs non-linear relationships with target</p>
                    </div>
                    
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Use the final set of {final_feature_count} features for model training</li>
                        <li>Monitor performance of these features over time, especially {top_stable}</li>
                        <li>Consider feature engineering opportunities for {nonlinear_features}</li>
                        <li>For resource-constrained environments, the top 10 features provide good performance</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Fill in dynamic content
            data_summary = results['data_summary']
            
            # Fill template with basic info
            html_content = html_content.format(
                start_date=data_summary['date_range'][0].strftime('%Y-%m-%d'),
                end_date=data_summary['date_range'][1].strftime('%Y-%m-%d'),
                symbol_count=len(data_summary['symbols']),
                sample_count=data_summary['n_samples'],
                feature_count=data_summary['n_features'],
                final_feature_count=len(final_features),
                rf_top=rf_importance.iloc[0]['Feature'] if not rf_importance.empty else "N/A",
                lgbm_top=lgbm_importance.iloc[0]['Feature'] if not lgbm_importance.empty else "N/A",
                optimal_count=results['optimal_features']['feature_count'],
                optimal_f1=results['optimal_features'].get('metrics', {}).get('f1', 0.0),
                pruned_top=results.get('pruned_features', {}).get('features', ["N/A"])[0] if results.get('pruned_features', {}).get('features') else "N/A",
                pruned_count=results.get('pruned_features', {}).get('feature_count', 0),
                pruned_f1=results.get('pruned_features', {}).get('metrics', {}).get('f1', 0.0),
                redundancy_reduction=int((1 - (results.get('pruned_features', {}).get('feature_count', 0) / results['optimal_features']['feature_count'])) * 100) if results.get('pruned_features', {}).get('feature_count', 0) > 0 else 0,
                best_ensemble_method=results.get('feature_ensemble', {}).get('best_method', "N/A"),
                ensemble_count=len(results.get('feature_ensemble', {}).get('ensemble_features', [])),
                ensemble_f1=results.get('feature_ensemble', {}).get('metrics', {}).get('f1', 0.0),
                top_stable=results.get('regime_analysis', {}).get('robust_features', ["N/A"])[0] if results.get('regime_analysis', {}).get('robust_features') else "N/A",
                stable_count=len(results.get('regime_analysis', {}).get('robust_features', [])),
                high_stability_pct=int((len(results.get('regime_analysis', {}).get('robust_features', [])) / len(final_features)) * 100) if final_features else 0,
                final_feature_rows=self._generate_final_feature_rows(final_features, results, combined_ranks),
                nonlinear_features=", ".join(results.get('enhanced_correlation', {}).get('divergent_features', ["N/A"])[:3])
            )
            
            # Save HTML report
            with open(os.path.join(self.output_dir, 'enhanced_feature_analysis_report.html'), 'w') as f:
                f.write(html_content)
            
            logger.info(f"Enhanced analysis report saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced report: {e}", exc_info=True)
        
    
    def enhanced_feature_selection_with_early_stopping(self, X: pd.DataFrame, y: pd.Series, 
                                                importance_df: pd.DataFrame, 
                                                corr_threshold: float = 0.8,
                                                max_features: int = 30,
                                                patience: int = 3) -> Dict[str, Any]:
        """
        Enhanced feature selection with correlation pruning and early stopping
        
        Args:
            X: Feature DataFrame
            y: Target Series
            importance_df: DataFrame with feature importance scores
            corr_threshold: Correlation threshold for feature pruning
            max_features: Maximum number of features to select
            patience: Number of non-improving features to consider before stopping
            
        Returns:
            Dictionary with selection results and metrics
        """
        logger.info(f"Performing enhanced feature selection with early stopping...")
        
        # Get features sorted by importance
        sorted_features = importance_df['Feature'].tolist()
        
        # Calculate feature correlations
        corr_matrix = X[sorted_features].corr().abs()
        
        # Initialize selected features with the most important feature
        selected_features = [sorted_features[0]]
        excluded_features = []
        
        # Initialize for early stopping
        best_f1 = 0
        best_feature_count = 1
        no_improvement_count = 0
        
        # Evaluate initial feature
        initial_metrics = self.evaluate_feature_subset(X, y, selected_features)
        best_f1 = initial_metrics['f1']
        
        # Keep track of all metrics
        metrics_history = [{'features': 1, **initial_metrics}]
        
        # Iteratively add features that are not highly correlated with already selected ones
        for feature in sorted_features[1:]:
            # Check if feature is highly correlated with any selected feature
            is_correlated = False
            for selected in selected_features:
                if corr_matrix.loc[feature, selected] > corr_threshold:
                    is_correlated = True
                    excluded_features.append((feature, selected, corr_matrix.loc[feature, selected]))
                    break
            
            # If not highly correlated, add to selected features and evaluate
            if not is_correlated:
                selected_features.append(feature)
                
                # Evaluate current feature set
                current_metrics = self.evaluate_feature_subset(X, y, selected_features)
                metrics_history.append({'features': len(selected_features), **current_metrics})
                
                # Check for improvement
                if current_metrics['f1'] > best_f1 + 0.001:  # Small threshold to account for randomness
                    best_f1 = current_metrics['f1']
                    best_feature_count = len(selected_features)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Check for early stopping
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping triggered after {no_improvement_count} features without improvement")
                    # Revert to best feature set
                    selected_features = selected_features[:best_feature_count]
                    break
                
            # Stop when we have enough features
            if len(selected_features) >= max_features:
                break
        
        # Final evaluation
        final_metrics = self.evaluate_feature_subset(X, y, selected_features)
        
        # Create visualization of feature addition impact
        self._plot_feature_addition_impact(metrics_history)
        
        logger.info(f"Selected {len(selected_features)} features (best count: {best_feature_count})")
        logger.info(f"Final metrics: {final_metrics}")
        
        return {
            'selected_features': selected_features,
            'excluded_features': excluded_features,
            'metrics': final_metrics,
            'metrics_history': metrics_history,
            'feature_count': len(selected_features),
            'best_feature_count': best_feature_count
        }

    def _plot_feature_addition_impact(self, metrics_history: List[Dict[str, Any]]) -> None:
        """Plot impact of adding features on performance metrics"""
        plt.figure(figsize=(12, 8))
        
        # Convert to DataFrame
        history_df = pd.DataFrame(metrics_history)
        
        # Plot metrics
        plt.plot(history_df['features'], history_df['accuracy'], marker='o', label='Accuracy')
        plt.plot(history_df['features'], history_df['precision'], marker='s', label='Precision')
        plt.plot(history_df['features'], history_df['recall'], marker='^', label='Recall')
        plt.plot(history_df['features'], history_df['f1'], marker='d', label='F1', linewidth=2)
        plt.plot(history_df['features'], history_df['auc'], marker='*', label='AUC')
        
        # Find best point for F1
        best_idx = history_df['f1'].idxmax()
        best_features = history_df.iloc[best_idx]['features']
        best_f1 = history_df.iloc[best_idx]['f1']
        
        plt.axvline(x=best_features, color='r', linestyle='--', alpha=0.7,
                label=f'Best F1 at {best_features} features')
        
        plt.scatter([best_features], [best_f1], s=100, c='r', zorder=10)
        
        plt.xlabel('Number of Features')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs Number of Features')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_addition_impact.png'), dpi=300)
        plt.close()

    def _generate_final_feature_rows(self, final_features: List[str], results: Dict[str, Any], 
                                combined_ranks: pd.DataFrame) -> str:
        """Generate HTML rows for final features table"""
        rows = ""
        
        # Get methods for each feature
        feature_methods = {}
        
        # Check which methods selected each feature
        for feature in final_features:
            methods = []
            
            if feature in results['optimal_features']['features']:
                methods.append("Optimal")
                
            if feature in results.get('pruned_features', {}).get('features', []):
                methods.append("Pruned")
                
            if feature in results.get('feature_ensemble', {}).get('ensemble_features', []):
                methods.append("Ensemble")
                
            if feature in results.get('regime_analysis', {}).get('robust_features', []):
                methods.append("Regime-Stable")
                
            if feature in results['rfe']['selected_features']:
                methods.append("RFE")
                
            feature_methods[feature] = methods
        
        # Get ranks for each feature
        feature_ranks = {}
        for feature in final_features:
            rank_row = combined_ranks[combined_ranks['Feature'] == feature]
            if not rank_row.empty:
                feature_ranks[feature] = int(rank_row.iloc[0]['Avg_Rank'])
            else:
                feature_ranks[feature] = 999  # High rank for features not in combined ranking
        
        # Sort features by rank
        sorted_features = sorted(final_features, key=lambda f: feature_ranks.get(f, 999))
        
        # Generate rows
        for i, feature in enumerate(sorted_features):
            methods_str = ", ".join(feature_methods.get(feature, ["Unknown"]))
            rank = feature_ranks.get(feature, "N/A")
            
            rows += f"<tr><td>{i+1}</td><td>{feature}</td><td>{methods_str}</td><td>{rank}</td></tr>"
        
        return rows

    def _create_method_comparison_visualization(self, results: Dict[str, Any]) -> None:
        """Create visualization comparing different feature selection methods"""
        # Collect metrics from different methods
        methods = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        feature_counts = []
        
        # Original optimal features
        if 'optimal_features' in results:
            methods.append("Optimal")
            metrics = results['optimal_features'].get('metrics', {})
            if not metrics:
                # If metrics not stored, evaluate
                metrics = self.evaluate_feature_subset(X, y, results['optimal_features']['features'])
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(results['optimal_features']['feature_count'])
        
        # Pruned features
        if 'pruned_features' in results and 'error' not in results['pruned_features']:
            methods.append("Pruned")
            metrics = results['pruned_features'].get('metrics', {})
            if not metrics:
                # If metrics not stored, evaluate
                metrics = self.evaluate_feature_subset(X, y, results['pruned_features']['features'])
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(results['pruned_features']['feature_count'])
        
        # Ensemble features
        if 'feature_ensemble' in results and 'error' not in results['feature_ensemble']:
            methods.append("Ensemble")
            metrics = results['feature_ensemble'].get('metrics', {})
            if not metrics:
                # If metrics not stored, evaluate
                metrics = self.evaluate_feature_subset(X, y, results['feature_ensemble']['ensemble_features'])
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(len(results['feature_ensemble']['ensemble_features']))
        
        # Regime-stable features
        if 'regime_analysis' in results and 'error' not in results['regime_analysis']:
            methods.append("Regime-Stable")
            metrics = results['regime_analysis'].get('metrics', {})
            if not metrics:
                # If metrics not stored, evaluate
                metrics = self.evaluate_feature_subset(X, y, results['regime_analysis']['robust_features'])
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(len(results['regime_analysis']['robust_features']))
        
        # RFE features
        if 'rfe' in results:
            methods.append("RFE")
            metrics = results['rfe'].get('metrics', {})
            if not metrics:
                # If metrics not stored, evaluate
                metrics = self.evaluate_feature_subset(X, y, results['rfe']['selected_features'])
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(results['rfe']['feature_count'])
        
        # Final features
        if 'final_features' in results and 'error' not in results['final_features']:
            methods.append("Final")
            metrics = results['final_features'].get('metrics', {})
            accuracies.append(metrics.get('accuracy', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            feature_counts.append(results['final_features']['feature_count'])
        
        # Create plot
        if methods:
            plt.figure(figsize=(14, 10))
            
            # Create subplots for metrics and feature counts
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Metrics plot
            x = np.arange(len(methods))
            width = 0.2
            
            ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy')
            ax1.bar(x - 0.5*width, precisions, width, label='Precision')
            ax1.bar(x + 0.5*width, recalls, width, label='Recall')
            ax1.bar(x + 1.5*width, f1_scores, width, label='F1')
            
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics by Feature Selection Method')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods)
            ax1.legend()
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Feature count plot
            ax2.bar(x, feature_counts, width*2, color='gray', alpha=0.7)
            ax2.set_ylabel('Feature Count')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'methods_comparison.png'), dpi=300)
            plt.close()


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