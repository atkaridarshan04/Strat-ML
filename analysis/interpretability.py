import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from core.schemas import TaskType

class InterpretabilityAnalyzer:
    def analyze(self, trained_pipeline, X_test, y_test, original_feature_names: list):
        """Analyze model interpretability using trained pipeline"""
        
        model = trained_pipeline.named_steps['model']
        feature_importance = {}
        
        # Tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Use original feature names (simplified mapping)
            for i, name in enumerate(original_feature_names):
                if i < len(importances):
                    feature_importance[name] = float(importances[i])
        
        # Permutation importance (works for all models)
        else:
            perm_importance = permutation_importance(
                trained_pipeline, X_test, y_test, n_repeats=5, random_state=42
            )
            
            for i, name in enumerate(original_feature_names):
                feature_importance[name] = float(perm_importance.importances_mean[i])
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        return feature_importance
