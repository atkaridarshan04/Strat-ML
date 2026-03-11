from typing import List, Optional
from core.schemas import ExperimentResult

class ExperimentMemory:
    def __init__(self):
        self.experiments: List[ExperimentResult] = []
    
    def add(self, result: ExperimentResult):
        """Add experiment result to memory"""
        self.experiments.append(result)
    
    def best_model(self) -> Optional[ExperimentResult]:
        """Get best performing model"""
        if not self.experiments:
            return None
        return max(self.experiments, key=lambda x: x.accuracy)
    
    def accuracy_trend(self) -> float:
        """Calculate accuracy trend"""
        if len(self.experiments) < 2:
            return 0.0
        return self.experiments[-1].accuracy - self.experiments[-2].accuracy
    
    def experiment_count(self) -> int:
        """Get total experiment count"""
        return len(self.experiments)
    
    def get_last_result(self) -> Optional[ExperimentResult]:
        """Get last experiment result"""
        return self.experiments[-1] if self.experiments else None
    
    def get_all(self) -> List[ExperimentResult]:
        """Get all experiments"""
        return self.experiments
