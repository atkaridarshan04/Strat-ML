from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"

@dataclass
class DatasetInfo:
    task_type: TaskType
    samples: int
    features: int
    numeric_features: int
    categorical_features: int
    target_column: str
    feature_names: List[str]

@dataclass
class DatasetProfile:
    missing_ratio: float
    class_imbalance: Optional[float]
    num_features: int
    num_samples: int
    feature_types: Dict[str, FeatureType]

@dataclass
class MetaFeatures:
    feature_entropy_mean: float
    feature_correlation_mean: float
    sparsity: float
    dimensionality_ratio: float
    feature_variance_mean: float
    skewness_mean: float

@dataclass
class ExperimentResult:
    iteration: int
    model_name: str
    accuracy: float
    runtime: float
    parameters: Dict[str, Any]
    is_tuned: bool = False

@dataclass
class AgentState:
    accuracy: float
    previous_accuracy: Optional[float]
    accuracy_trend: float
    runtime: float
    iteration: int
    dataset_meta_features: MetaFeatures
    experiment_history: List[ExperimentResult] = field(default_factory=list)

class AgentAction(Enum):
    CONTINUE = "continue"
    SWITCH_MODEL = "switch_model"
    TUNE_HYPERPARAMETERS = "tune_hyperparameters"
    TERMINATE = "terminate"

@dataclass
class AgentDecision:
    action: AgentAction
    reason: str
    rule_triggered: str
    next_model: Optional[str] = None
