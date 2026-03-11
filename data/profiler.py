import pandas as pd
import numpy as np
from typing import Dict
from core.schemas import DatasetProfile, FeatureType, TaskType

class DatasetProfiler:
    def profile(self, df: pd.DataFrame, target_column: str, task_type: TaskType) -> DatasetProfile:
        """Profile dataset quality metrics"""
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Missing value ratio
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        # Class imbalance (only for classification)
        class_imbalance = None
        if task_type == TaskType.CLASSIFICATION:
            class_counts = y.value_counts()
            class_imbalance = class_counts.min() / class_counts.max()
        
        # Feature types
        feature_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[col] = FeatureType.NUMERIC
            else:
                feature_types[col] = FeatureType.CATEGORICAL
        
        return DatasetProfile(
            missing_ratio=float(missing_ratio),
            class_imbalance=float(class_imbalance) if class_imbalance is not None else None,
            num_features=X.shape[1],
            num_samples=X.shape[0],
            feature_types=feature_types
        )
