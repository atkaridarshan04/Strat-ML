# AutoML Research Prototype

A research-oriented AutoML system emphasizing interpretability, dataset understanding, and experiment traceability.

## Features

- **Automatic Dataset Interpretation**: Detects task type, feature types, and structure
- **Meta-Feature Extraction**: Computes dataset complexity metrics used in AutoML research
- **Adaptive Preprocessing**: Builds preprocessing pipelines based on data quality
- **Agent-Controlled Experiments**: Rule-based agent makes transparent decisions
- **Experiment Memory**: Tracks all experiments and trends
- **Model Interpretability**: Feature importance analysis
- **Decision Traceability**: Complete audit trail of agent decisions
- **Comprehensive Reports**: PDF research reports with all findings
- **Terminal Visualization**: Real-time graphs and tables using plotext
- **Structured Output**: Clean tabular display of results in terminal

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python cli/main.py <dataset.csv> --output report.pdf --max-iterations 5 [--enable-tuning]
```

Example:
```bash
# Basic run without hyperparameter tuning
python cli/main.py data/iris.csv --output iris_report.pdf

# With hyperparameter tuning enabled
python cli/main.py data/iris.csv --output iris_report.pdf --enable-tuning
```

### CLI Options

- `dataset`: Path to CSV dataset (required)
- `--output`: Output PDF report path (default: experiment_report.pdf)
- `--max-iterations`: Maximum experiment iterations (default: 5)
- `--enable-tuning`: Enable hyperparameter tuning (disabled by default)

## Architecture

```
Dataset → Interpreter → Profiler → Meta-Features → Preprocessing
    ↓
Agent Controller → Experiments → Memory → Interpretability → Report
```

## System Components

1. **Dataset Interpreter**: Auto-detects structure and task type
2. **Dataset Profiler**: Evaluates data quality
3. **Meta-Feature Extractor**: Computes complexity metrics
4. **Preprocessing Planner**: Builds adaptive pipelines
5. **Model Search Space**: Task-specific model selection
6. **Experiment Runner**: Executes training and evaluation
7. **Experiment Memory**: Tracks history and trends
8. **Rule Engine**: Interpretable decision logic
9. **Decision Tracker**: Logs all agent decisions
10. **Interpretability Analyzer**: Feature importance
11. **Report Generator**: Comprehensive PDF reports

## Output

The system generates a PDF report containing:
- Dataset overview and structure
- Dataset meta-features
- Data quality analysis
- Experiment results table
- Agent decision trace
- Model interpretability analysis
- Best model summary
