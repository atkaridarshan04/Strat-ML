from typing import List
from core.schemas import AgentDecision

class DecisionTracker:
    def __init__(self):
        self.decisions: List[AgentDecision] = []
    
    def log(self, decision: AgentDecision):
        """Log agent decision"""
        self.decisions.append(decision)
    
    def get_all(self) -> List[AgentDecision]:
        """Get all decisions"""
        return self.decisions
