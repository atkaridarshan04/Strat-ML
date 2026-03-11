from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from core.schemas import DatasetProfile, FeatureType

class PreprocessingPlanner:
    def build_pipeline(self, profile: DatasetProfile) -> ColumnTransformer:
        """Build adaptive preprocessing pipeline based on dataset profile"""
        
        numeric_features = [col for col, ftype in profile.feature_types.items() 
                           if ftype == FeatureType.NUMERIC]
        categorical_features = [col for col, ftype in profile.feature_types.items() 
                               if ftype == FeatureType.CATEGORICAL]
        
        transformers = []
        
        # Numeric pipeline
        if numeric_features:
            numeric_steps = []
            if profile.missing_ratio > 0:
                numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            numeric_steps.append(('scaler', StandardScaler()))
            
            transformers.append(('numeric', Pipeline(numeric_steps), numeric_features))
        
        # Categorical pipeline
        if categorical_features:
            categorical_steps = []
            if profile.missing_ratio > 0:
                categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            
            transformers.append(('categorical', Pipeline(categorical_steps), categorical_features))
        
        return ColumnTransformer(transformers=transformers)
