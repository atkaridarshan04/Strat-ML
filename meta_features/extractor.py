import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from core.schemas import MetaFeatures, TaskType, FeatureType

class MetaFeatureExtractor:
    def extract(self, df: pd.DataFrame, target_column: str, task_type: TaskType, 
                feature_types: dict) -> MetaFeatures:
        """Extract meta-features describing dataset complexity"""
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Get numeric columns only
        numeric_cols = [col for col, ftype in feature_types.items() if ftype == FeatureType.NUMERIC]
        X_numeric = X[numeric_cols] if numeric_cols else pd.DataFrame()
        
        # Distribution features
        if len(numeric_cols) > 0:
            feature_variance_mean = float(X_numeric.var().mean())
            skewness_mean = float(np.mean([skew(X_numeric[col].dropna()) for col in numeric_cols]))
        else:
            feature_variance_mean = 0.0
            skewness_mean = 0.0
        
        # Entropy
        if len(numeric_cols) > 0:
            entropies = []
            for col in numeric_cols:
                hist, _ = np.histogram(X_numeric[col].dropna(), bins=10)
                hist = hist / hist.sum()
                entropies.append(entropy(hist + 1e-10))
            feature_entropy_mean = float(np.mean(entropies))
        else:
            feature_entropy_mean = 0.0
        
        # Correlation
        if len(numeric_cols) > 1:
            corr_matrix = X_numeric.corr().abs()
            feature_correlation_mean = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
        else:
            feature_correlation_mean = 0.0
        
        # Sparsity
        sparsity = float((X == 0).sum().sum() / (X.shape[0] * X.shape[1]))
        
        # Dimensionality ratio
        dimensionality_ratio = float(X.shape[1] / X.shape[0])
        
        return MetaFeatures(
            feature_entropy_mean=feature_entropy_mean,
            feature_correlation_mean=feature_correlation_mean,
            sparsity=sparsity,
            dimensionality_ratio=dimensionality_ratio,
            feature_variance_mean=feature_variance_mean,
            skewness_mean=skewness_mean
        )
