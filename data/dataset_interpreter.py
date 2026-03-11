import pandas as pd
import numpy as np
from typing import Optional
from core.schemas import DatasetInfo, TaskType, FeatureType

class DatasetInterpreter:
    def __init__(self, threshold_unique_ratio: float = 0.05):
        self.threshold_unique_ratio = threshold_unique_ratio
    
    def interpret(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DatasetInfo:
        """Automatically interpret dataset structure"""
        
        # Auto-detect target column if not provided
        if target_column is None:
            target_column = df.columns[-1]
        
        # Detect task type
        task_type = self._detect_task_type(df[target_column])
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        
        # Detect feature types
        numeric_count = 0
        categorical_count = 0
        
        for col in X.columns:
            if self._is_numeric_feature(X[col]):
                numeric_count += 1
            else:
                categorical_count += 1
        
        return DatasetInfo(
            task_type=task_type,
            samples=len(df),
            features=len(X.columns),
            numeric_features=numeric_count,
            categorical_features=categorical_count,
            target_column=target_column,
            feature_names=list(X.columns)
        )
    
    def _detect_task_type(self, target_series: pd.Series) -> TaskType:
        """Detect if classification or regression"""
        unique_ratio = len(target_series.unique()) / len(target_series)
        
        if unique_ratio < self.threshold_unique_ratio or target_series.dtype == 'object':
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION
    
    def _is_numeric_feature(self, series: pd.Series) -> bool:
        """Check if feature is numeric"""
        return pd.api.types.is_numeric_dtype(series)
