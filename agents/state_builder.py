from core.schemas import AgentState, MetaFeatures
from typing import Optional

class StateBuilder:
    def build_state(self, current_accuracy: float, previous_accuracy: Optional[float],
                   runtime: float, iteration: int, meta_features: MetaFeatures) -> AgentState:
        """Build agent state from experiment results"""
        
        accuracy_trend = 0.0
        if previous_accuracy is not None:
            accuracy_trend = current_accuracy - previous_accuracy
        
        return AgentState(
            accuracy=current_accuracy,
            previous_accuracy=previous_accuracy,
            accuracy_trend=accuracy_trend,
            runtime=runtime,
            iteration=iteration,
            dataset_meta_features=meta_features
        )
