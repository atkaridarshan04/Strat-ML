from sklearn.model_selection import GridSearchCV
from core.schemas import TaskType

class HyperparameterTuner:
    def __init__(self, cv: int = 2):  # Reduced CV folds
        self.cv = cv
    
    def get_param_grid(self, model_name: str, task_type: TaskType) -> dict:
        """Get parameter grid for model - minimal for speed"""
        
        if model_name == 'LogisticRegression':
            return {
                'C': [0.1, 1.0],
                'solver': ['lbfgs']
            }
        elif model_name == 'RandomForest':
            # Minimal grid for speed
            return {
                'n_estimators': [100, 150],
                'max_depth': [10, None]
            }
        elif model_name == 'LinearRegression':
            return {}
        
        return {}
    
    def tune(self, model, param_grid: dict, X_train, y_train) -> tuple:
        """Tune hyperparameters using GridSearchCV"""
        
        if not param_grid:
            return model, model.get_params()
        
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv, 
            scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
            n_jobs=-1,
            verbose=1  # Show progress
        )
        
        print(f"    Searching {len(param_grid)} parameters with {self.cv}-fold CV...")
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
