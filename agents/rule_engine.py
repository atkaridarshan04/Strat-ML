from core.schemas import AgentState, AgentDecision, AgentAction
from search.model_space import ModelSearchSpace

class RuleEngine:
    def __init__(self, improvement_threshold: float = 0.01, 
                 max_iterations: int = 5, runtime_threshold: float = 5.0,
                 tune_threshold: float = -0.001):
        self.improvement_threshold = improvement_threshold
        self.max_iterations = max_iterations
        self.runtime_threshold = runtime_threshold
        self.tune_threshold = tune_threshold
    
    def decide(self, state: AgentState, current_model: str, 
               available_models: list, tried_models: set, 
               current_model_tuned: bool = False) -> AgentDecision:
        """Make decision based on interpretable rules"""
        
        # Rule 1: Max iterations reached
        if state.iteration >= self.max_iterations:
            return AgentDecision(
                action=AgentAction.TERMINATE,
                reason=f"Maximum iterations ({self.max_iterations}) reached",
                rule_triggered="max_iteration_limit"
            )
        
        # Rule 2: First iteration - continue with same model
        if state.iteration == 1:
            return AgentDecision(
                action=AgentAction.CONTINUE,
                reason="First iteration completed, continue exploration",
                rule_triggered="initial_exploration"
            )
        
        # Rule 3: Small improvement - try tuning before switching
        if (self.tune_threshold < state.accuracy_trend < self.improvement_threshold 
            and not current_model_tuned):
            return AgentDecision(
                action=AgentAction.TUNE_HYPERPARAMETERS,
                reason=f"Small improvement {state.accuracy_trend:.4f}, try hyperparameter tuning",
                rule_triggered="plateau_tune_hyperparameters"
            )
        
        # Rule 4: Low improvement - switch model
        if state.accuracy_trend < self.improvement_threshold:
            untried_models = [m for m in available_models if m not in tried_models]
            
            if untried_models:
                next_model = untried_models[0]
                return AgentDecision(
                    action=AgentAction.SWITCH_MODEL,
                    reason=f"Improvement {state.accuracy_trend:.4f} below threshold {self.improvement_threshold}",
                    rule_triggered="low_improvement_threshold",
                    next_model=next_model
                )
            else:
                return AgentDecision(
                    action=AgentAction.TERMINATE,
                    reason="All models tried, no further improvement",
                    rule_triggered="exhausted_search_space"
                )
        
        # Rule 5: Good improvement - continue
        return AgentDecision(
            action=AgentAction.CONTINUE,
            reason=f"Good improvement {state.accuracy_trend:.4f}, continue with {current_model}",
            rule_triggered="positive_trend"
        )
