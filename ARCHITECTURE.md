# AutoML System Architecture & Component Interaction

## System Overview

This is a research-oriented AutoML system that automatically analyzes datasets, selects models, and makes transparent decisions through a rule-based agent. The system emphasizes interpretability and traceability.

## High-Level Flow

```
User Input (CSV) 
    ↓
CLI Entry Point (cli/main.py)
    ↓
Orchestrator (orchestration/orchestrator.py) - Main Controller
    ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Dataset Analysis                               │
│  1. Dataset Interpreter → Detect task & features        │
│  2. Dataset Profiler → Evaluate data quality            │
│  3. Meta-Feature Extractor → Compute complexity metrics │
│  4. Preprocessing Planner → Build pipeline              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: Agent-Controlled Experiments (Loop)            │
│  1. Experiment Runner → Train & evaluate model          │
│  2. Experiment Memory → Store results                   │
│  3. State Builder → Create agent state                  │
│  4. Rule Engine → Make decision                         │
│  5. Decision Tracker → Log decision                     │
│  6. Execute action (continue/tune/switch/terminate)     │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: Analysis & Reporting                           │
│  1. Interpretability Analyzer → Feature importance      │
│  2. Report Generator → Create PDF report                │
└─────────────────────────────────────────────────────────┘
    ↓
Output (PDF Report + Terminal Visualizations)
```

---

## Component Details

### 1. **CLI Entry Point** (`cli/main.py`)

**Purpose**: Parse command-line arguments and start the system

**Key Code**:
```python
parser.add_argument('dataset', type=str)
parser.add_argument('--output', default='experiment_report.pdf')
parser.add_argument('--max-iterations', type=int, default=5)
parser.add_argument('--enable-tuning', action='store_true')

orchestrator = Orchestrator(max_iterations=args.max_iterations, 
                           enable_tuning=args.enable_tuning)
orchestrator.run(args.dataset, args.output)
```

**Interaction**: Creates Orchestrator and delegates control

---

### 2. **Orchestrator** (`orchestration/orchestrator.py`)

**Purpose**: Main controller that coordinates all components

**Key Responsibilities**:
- Initialize all components
- Execute 3-phase pipeline
- Manage experiment loop
- Display terminal output

**Key Components Initialized**:
```python
self.interpreter = DatasetInterpreter()
self.profiler = DatasetProfiler()
self.meta_extractor = MetaFeatureExtractor()
self.preprocessing_planner = PreprocessingPlanner()
self.runner = ExperimentRunner()
self.tuner = HyperparameterTuner()
self.memory = ExperimentMemory()
self.state_builder = StateBuilder()
self.rule_engine = RuleEngine(max_iterations, enable_tuning)
self.decision_tracker = DecisionTracker()
self.interpretability = InterpretabilityAnalyzer()
```

**Interaction**: Orchestrates all components in sequence

---

## PHASE 1: Dataset Analysis

### 3. **Dataset Interpreter** (`data/dataset_interpreter.py`)

**Purpose**: Automatically understand dataset structure

**What it does**:
- Detects task type (classification/regression)
- Identifies target column (last column by default)
- Classifies features as numeric/categorical
- Counts feature types

**Key Method**:
```python
def interpret(self, df: pd.DataFrame) -> DatasetInfo:
    # Detect if classification or regression
    task_type = self._detect_task_type(df, target_column)
    
    # Classify each feature
    for col in feature_columns:
        if self._is_numeric_feature(df[col]):
            numeric_features += 1
        else:
            categorical_features += 1
    
    return DatasetInfo(...)
```

**Output**: `DatasetInfo` object with task type, target, feature counts

**Interaction**: Feeds info to Profiler and Meta-Feature Extractor

---

### 4. **Dataset Profiler** (`data/profiler.py`)

**Purpose**: Evaluate data quality

**What it does**:
- Calculates missing value ratio
- Detects class imbalance (for classification)
- Identifies feature types (numeric/categorical)

**Key Method**:
```python
def profile(self, df, target_column, task_type) -> DatasetProfile:
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    
    if task_type == TaskType.CLASSIFICATION:
        class_counts = df[target_column].value_counts()
        class_imbalance = 1 - (class_counts.min() / class_counts.max())
    
    return DatasetProfile(...)
```

**Output**: `DatasetProfile` with quality metrics

**Interaction**: Used by Preprocessing Planner to build pipeline

---

### 5. **Meta-Feature Extractor** (`meta_features/extractor.py`)

**Purpose**: Compute dataset complexity metrics used in AutoML research

**What it does**:
- Feature entropy (information content)
- Feature correlation (redundancy)
- Dimensionality ratio (samples/features)
- Statistical moments (skewness, kurtosis)

**Key Method**:
```python
def extract(self, df, target_column, task_type, feature_types) -> MetaFeatures:
    # Compute entropy for each feature
    entropies = [self._compute_entropy(df[col]) for col in numeric_cols]
    
    # Compute correlation matrix
    correlations = df[numeric_cols].corr().abs().values
    
    # Dimensionality ratio
    dimensionality_ratio = df.shape[1] / df.shape[0]
    
    return MetaFeatures(...)
```

**Output**: `MetaFeatures` object with complexity metrics

**Interaction**: Used by State Builder to inform agent decisions

---

### 6. **Preprocessing Planner** (`data/preprocessing_planner.py`)

**Purpose**: Build adaptive preprocessing pipeline based on data quality

**What it does**:
- Handles missing values (imputation)
- Encodes categorical features
- Scales numeric features

**Key Method**:
```python
def build_pipeline(self, profile: DatasetProfile) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
```

**Output**: Scikit-learn `ColumnTransformer` pipeline

**Interaction**: Used by Experiment Runner to preprocess data

---

## PHASE 2: Agent-Controlled Experiments

### 7. **Model Search Space** (`search/model_space.py`)

**Purpose**: Define available models for each task type

**What it does**:
- Provides model instances for classification/regression
- Returns model names for exploration

**Key Methods**:
```python
@staticmethod
def get_models(task_type: TaskType) -> dict:
    if task_type == TaskType.CLASSIFICATION:
        return {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'SVM': SVC(kernel='rbf')
        }
    else:  # Regression
        return {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'SVR': SVR(kernel='rbf')
        }
```

**Output**: Dictionary of model instances

**Interaction**: Orchestrator gets models, Runner uses them for training

---

### 8. **Experiment Runner** (`execution/runner.py`)

**Purpose**: Execute model training and evaluation

**What it does**:
- Splits data (80/20 train/test)
- Fits preprocessing + model pipeline
- Evaluates on test set
- Measures runtime

**Key Method**:
```python
def run(self, df, target_column, task_type, preprocessor, 
        model, model_name, iteration, is_tuned=False) -> ExperimentResult:
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train and time
    start = time.time()
    pipeline.fit(X_train, y_train)
    runtime = time.time() - start
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # or r2_score for regression
    
    return ExperimentResult(iteration, model_name, accuracy, runtime, is_tuned)
```

**Output**: `ExperimentResult` with metrics

**Interaction**: Results stored in Experiment Memory

---

### 9. **Experiment Memory** (`memory/experiment_memory.py`)

**Purpose**: Track all experiment results

**What it does**:
- Stores experiment history
- Computes accuracy trends
- Identifies best model

**Key Methods**:
```python
def add(self, result: ExperimentResult):
    self.experiments.append(result)

def accuracy_trend(self) -> float:
    if len(self.experiments) < 2:
        return 0.0
    return self.experiments[-1].accuracy - self.experiments[-2].accuracy

def best_model(self) -> ExperimentResult:
    return max(self.experiments, key=lambda x: x.accuracy)
```

**Output**: Historical data and trends

**Interaction**: State Builder uses trends, Orchestrator gets best model

---

### 10. **State Builder** (`agents/state_builder.py`)

**Purpose**: Create agent state from experiment data

**What it does**:
- Computes accuracy trend
- Packages current metrics
- Includes meta-features

**Key Method**:
```python
def build_state(self, current_accuracy, previous_accuracy, 
                runtime, iteration, meta_features) -> AgentState:
    
    accuracy_trend = 0.0
    if previous_accuracy is not None:
        accuracy_trend = current_accuracy - previous_accuracy
    
    return AgentState(
        iteration=iteration,
        current_accuracy=current_accuracy,
        accuracy_trend=accuracy_trend,
        runtime=runtime,
        meta_features=meta_features
    )
```

**Output**: `AgentState` object

**Interaction**: Rule Engine uses state to make decisions

---

### 11. **Rule Engine** (`agents/rule_engine.py`)

**Purpose**: Make transparent, rule-based decisions

**What it does**:
- Evaluates interpretable rules
- Decides next action (continue/tune/switch/terminate)
- Provides reasoning for each decision

**Key Rules**:
```python
def decide(self, state, current_model, available_models, 
           tried_models, current_model_tuned) -> AgentDecision:
    
    # Rule 1: Max iterations reached
    if state.iteration >= self.max_iterations:
        return AgentDecision(action=TERMINATE, 
                           reason="Max iterations reached",
                           rule_triggered="max_iteration_limit")
    
    # Rule 2: First iteration - continue
    if state.iteration == 1:
        return AgentDecision(action=CONTINUE,
                           reason="First iteration, continue exploration",
                           rule_triggered="initial_exploration")
    
    # Rule 3: Small improvement - try tuning (if enabled)
    if (self.enable_tuning and 
        self.tune_threshold < state.accuracy_trend < self.improvement_threshold 
        and not current_model_tuned):
        return AgentDecision(action=TUNE_HYPERPARAMETERS,
                           reason=f"Small improvement {state.accuracy_trend:.4f}",
                           rule_triggered="plateau_tune_hyperparameters")
    
    # Rule 4: Low improvement - switch model
    if state.accuracy_trend < self.improvement_threshold:
        untried_models = [m for m in available_models if m not in tried_models]
        if untried_models:
            return AgentDecision(action=SWITCH_MODEL,
                               next_model=untried_models[0],
                               reason=f"Low improvement {state.accuracy_trend:.4f}",
                               rule_triggered="low_improvement_threshold")
        else:
            return AgentDecision(action=TERMINATE,
                               reason="All models tried",
                               rule_triggered="exhausted_search_space")
    
    # Rule 5: Good improvement - continue
    return AgentDecision(action=CONTINUE,
                       reason=f"Good improvement {state.accuracy_trend:.4f}",
                       rule_triggered="positive_trend")
```

**Output**: `AgentDecision` with action, reason, and rule

**Interaction**: Orchestrator executes the decision

---

### 12. **Hyperparameter Tuner** (`search/hyperparameter_tuner.py`)

**Purpose**: Optimize model hyperparameters

**What it does**:
- Defines search grids for each model
- Uses GridSearchCV with cross-validation
- Returns tuned model

**Key Methods**:
```python
def get_param_grid(self, model_name, task_type) -> dict:
    if model_name == 'LogisticRegression':
        return {'C': [0.1, 1.0, 10.0]}
    elif model_name == 'RandomForest':
        return {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    # ... more models

def tune(self, model, param_grid, X_train, y_train) -> tuple:
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
```

**Output**: Tuned model and best parameters

**Interaction**: Orchestrator calls when Rule Engine decides to tune

---

### 13. **Decision Tracker** (`tracking/decision_tracker.py`)

**Purpose**: Log all agent decisions for traceability

**What it does**:
- Stores decision history
- Provides audit trail

**Key Methods**:
```python
def log(self, decision: AgentDecision):
    self.decisions.append(decision)

def get_all(self) -> list:
    return self.decisions
```

**Output**: List of all decisions

**Interaction**: Report Generator includes decision trace

---

## PHASE 3: Analysis & Reporting

### 14. **Interpretability Analyzer** (`analysis/interpretability.py`)

**Purpose**: Explain model predictions

**What it does**:
- Computes feature importance using permutation
- Ranks features by impact

**Key Method**:
```python
def analyze(self, pipeline, X_test, y_test, feature_names) -> dict:
    # Use permutation importance
    result = permutation_importance(pipeline, X_test, y_test, 
                                   n_repeats=10, random_state=42)
    
    # Sort by importance
    importance_dict = dict(zip(feature_names, result.importances_mean))
    return dict(sorted(importance_dict.items(), 
                      key=lambda x: x[1], reverse=True))
```

**Output**: Dictionary of feature → importance

**Interaction**: Displayed in terminal and PDF report

---

### 15. **Report Generator** (`reporting/final_report.py`)

**Purpose**: Create comprehensive PDF report

**What it does**:
- Dataset overview
- Meta-features table
- Experiment results table
- Decision trace
- Feature importance
- Visualizations (using matplotlib)

**Key Method**:
```python
def generate(self, dataset_info, profile, meta_features, 
             experiments, decisions, feature_importance, best_result):
    
    self._add_title("AutoML Experiment Report")
    self._add_dataset_overview(dataset_info)
    self._add_meta_features(meta_features)
    self._add_data_quality(profile)
    self._add_experiment_table(experiments)
    self._add_decisions(decisions)
    self._add_feature_importance(feature_importance)
    self._add_best_model(best_result)
    
    # Add visualizations
    visualizer = ReportVisualizer()
    accuracy_plot = visualizer.plot_accuracy_trend(experiments)
    self._add_image(accuracy_plot)
    
    self.doc.build(self.story)
```

**Output**: PDF file

**Interaction**: Final output for user

---

## Data Flow Example

Let's trace a complete execution:

### Iteration 1:
```
1. User runs: python cli/main.py iris.csv --max-iterations 3

2. CLI → Orchestrator.run()

3. PHASE 1:
   - Interpreter: "Classification task, 4 numeric features"
   - Profiler: "No missing values, balanced classes"
   - Meta-Extractor: "Low dimensionality, moderate correlation"
   - Preprocessing: "StandardScaler for numeric features"

4. PHASE 2 - Iteration 1:
   - Runner: Train LogisticRegression → Accuracy: 0.95
   - Memory: Store result
   - State Builder: accuracy_trend = 0.0 (first iteration)
   - Rule Engine: "First iteration" → CONTINUE
   - Decision Tracker: Log decision
   - Display: Table + bar chart

5. PHASE 2 - Iteration 2:
   - Runner: Train LogisticRegression again → Accuracy: 0.95
   - Memory: Store result
   - State Builder: accuracy_trend = 0.0 (no improvement)
   - Rule Engine: "Low improvement" → SWITCH_MODEL to RandomForest
   - Decision Tracker: Log decision
   - Display: Table shows "Continued" in Change column

6. PHASE 2 - Iteration 3:
   - Runner: Train RandomForest → Accuracy: 0.97
   - Memory: Store result
   - State Builder: accuracy_trend = 0.02 (good improvement)
   - Rule Engine: "Max iterations reached" → TERMINATE
   - Decision Tracker: Log decision
   - Display: Table shows "Switched from LogisticRegression"

7. PHASE 3:
   - Memory: Best model = RandomForest (0.97)
   - Interpretability: Compute feature importance
   - Display: Feature importance table + bar chart
   - Report Generator: Create PDF with all results
```

---

## Key Design Patterns

### 1. **Pipeline Pattern**
- Data flows through sequential stages
- Each component has single responsibility

### 2. **Strategy Pattern**
- Different models/algorithms swappable
- Rule Engine uses different strategies

### 3. **Observer Pattern**
- Memory tracks all experiments
- Decision Tracker logs all decisions

### 4. **Builder Pattern**
- State Builder constructs agent state
- Preprocessing Planner builds pipelines

---

## Configuration Points

### Thresholds (in RuleEngine):
- `improvement_threshold = 0.01`: Minimum accuracy gain to continue
- `tune_threshold = -0.001`: Trigger tuning for small improvements
- `max_iterations = 5`: Maximum experiment iterations
- `enable_tuning = False`: Enable/disable hyperparameter tuning

### Model Search Space:
- Defined in `ModelSearchSpace.get_models()`
- Easy to add new models

### Hyperparameter Grids:
- Defined in `HyperparameterTuner.get_param_grid()`
- Customizable per model

---

## Terminal Output Components

### Tables (using tabulate):
1. Dataset info (samples, features, task type)
2. Meta-features (entropy, correlation, dimensionality)
3. Iteration results (model, accuracy, runtime, decision, change)
4. Feature importance (top 10 features)

### Graphs (using plotext):
1. Model accuracy comparison (bar chart)
2. Feature importance (bar chart)

---

## Summary

The system is a **closed-loop AutoML pipeline**:

1. **Analyze** dataset automatically
2. **Experiment** with models using agent decisions
3. **Learn** from results and adapt strategy
4. **Report** findings with full traceability

Each component is **modular** and **testable**, with clear inputs/outputs defined by schemas in `core/schemas.py`. The agent makes **transparent decisions** based on interpretable rules, not black-box optimization.
