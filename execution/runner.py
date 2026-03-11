import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from core.schemas import ExperimentResult, TaskType

class ExperimentRunner:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipeline = None
    
    def run(self, df: pd.DataFrame, target_column: str, task_type: TaskType,
            preprocessor, model, model_name: str, iteration: int, 
            is_tuned: bool = False) -> ExperimentResult:
        """Run single experiment"""
        
        # Split data once
        if self.X_train is None:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Build pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train and measure time
        start_time = time.time()
        self.pipeline.fit(self.X_train, self.y_train)
        runtime = time.time() - start_time
        
        # Evaluate
        y_pred = self.pipeline.predict(self.X_test)
        
        if task_type == TaskType.CLASSIFICATION:
            score = accuracy_score(self.y_test, y_pred)
        else:
            score = r2_score(self.y_test, y_pred)
        
        return ExperimentResult(
            iteration=iteration,
            model_name=model_name,
            accuracy=float(score),
            runtime=float(runtime),
            parameters=model.get_params(),
            is_tuned=is_tuned
        )
    
    def get_train_data(self):
        """Get cached training data"""
        return self.X_train, self.y_train
    
    def get_trained_pipeline(self):
        """Get the last trained pipeline"""
        return self.pipeline
