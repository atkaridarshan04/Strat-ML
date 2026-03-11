from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from core.schemas import TaskType

class ModelSearchSpace:
    @staticmethod
    def get_models(task_type: TaskType):
        """Get available models for task type"""
        
        if task_type == TaskType.CLASSIFICATION:
            return {
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
        else:
            return {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
    
    @staticmethod
    def get_model_names(task_type: TaskType):
        """Get list of model names"""
        return list(ModelSearchSpace.get_models(task_type).keys())
