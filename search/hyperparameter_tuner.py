from sklearn.model_selection import RandomizedSearchCV
from core.schemas import TaskType

class HyperparameterTuner:
    def __init__(self, cv: int = 3, n_iter: int = 20):
        self.cv = cv
        self.n_iter = n_iter  # Number of random combinations to try
    
    def get_param_grid(self, model_name: str, task_type: TaskType) -> dict:
        """Get parameter grid for model - expanded for better tuning"""
        
        if model_name == 'LogisticRegression':
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [500, 1000, 2000]
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif model_name == 'LinearRegression':
            return {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        elif model_name == 'SVM' or model_name == 'SVR':
            return {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4]  # for poly kernel
            }
        
        return {}
    
    def tune(self, model, param_grid: dict, X_train, y_train) -> tuple:
        """Tune hyperparameters using RandomizedSearchCV"""
        
        if not param_grid:
            return model, model.get_params()
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        # Use GridSearch if combinations are small, otherwise RandomizedSearch
        if total_combinations <= 30:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid, cv=self.cv, 
                scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
                n_jobs=-1,
                verbose=0
            )
            print(f"    Grid search: {total_combinations} combinations, {self.cv}-fold CV")
        else:
            search = RandomizedSearchCV(
                model, param_grid, 
                n_iter=min(self.n_iter, total_combinations),
                cv=self.cv, 
                scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
            print(f"    Randomized search: {min(self.n_iter, total_combinations)} of {total_combinations} combinations, {self.cv}-fold CV")
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_
