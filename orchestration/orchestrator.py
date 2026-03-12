import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import plotext as plt
from data.dataset_interpreter import DatasetInterpreter
from data.profiler import DatasetProfiler
from data.preprocessing_planner import PreprocessingPlanner
from meta_features.extractor import MetaFeatureExtractor
from search.model_space import ModelSearchSpace
from search.hyperparameter_tuner import HyperparameterTuner
from execution.runner import ExperimentRunner
from memory.experiment_memory import ExperimentMemory
from agents.state_builder import StateBuilder
from agents.rule_engine import RuleEngine
from tracking.decision_tracker import DecisionTracker
from analysis.interpretability import InterpretabilityAnalyzer
from reporting.final_report import FinalReportGenerator
from core.schemas import AgentAction

class Orchestrator:
    def __init__(self, max_iterations: int = 5, enable_tuning: bool = False):
        self.interpreter = DatasetInterpreter()
        self.profiler = DatasetProfiler()
        self.meta_extractor = MetaFeatureExtractor()
        self.preprocessing_planner = PreprocessingPlanner()
        self.runner = ExperimentRunner()
        self.tuner = HyperparameterTuner()
        self.memory = ExperimentMemory()
        self.state_builder = StateBuilder()
        self.rule_engine = RuleEngine(max_iterations=max_iterations, enable_tuning=enable_tuning)
        self.decision_tracker = DecisionTracker()
        self.interpretability = InterpretabilityAnalyzer()
    
    def run(self, dataset_path: str, output_report_path: str = "experiment_report.pdf"):
        """Run complete AutoML pipeline"""
        
        print("=" * 60)
        print("AutoML Research Prototype")
        print("=" * 60)
        
        # Load dataset
        print("\n[1/3] Dataset Analysis Phase")
        df = pd.read_csv(dataset_path)
        
        # Interpret dataset
        dataset_info = self.interpreter.interpret(df)
        
        # Profile dataset
        profile = self.profiler.profile(df, dataset_info.target_column, dataset_info.task_type)
        
        # Extract meta-features
        meta_features = self.meta_extractor.extract(
            df, dataset_info.target_column, dataset_info.task_type, profile.feature_types
        )
        
        # Display dataset info table
        dataset_table = [
            ["Samples", df.shape[0]],
            ["Features", df.shape[1]],
            ["Task Type", dataset_info.task_type.value],
            ["Numeric Features", dataset_info.numeric_features],
            ["Categorical Features", dataset_info.categorical_features],
            ["Missing Ratio", f"{profile.missing_ratio:.4f}"],
        ]
        if profile.class_imbalance:
            dataset_table.append(["Class Imbalance", f"{profile.class_imbalance:.4f}"])
        
        print("\n" + tabulate(dataset_table, headers=["Property", "Value"], tablefmt="grid"))
        
        # Display meta-features table
        meta_table = [
            ["Entropy (mean)", f"{meta_features.feature_entropy_mean:.4f}"],
            ["Correlation (mean)", f"{meta_features.feature_correlation_mean:.4f}"],
            ["Dimensionality Ratio", f"{meta_features.dimensionality_ratio:.4f}"],
        ]
        print("\n" + tabulate(meta_table, headers=["Meta-Feature", "Value"], tablefmt="grid"))
        
        # Build preprocessing pipeline
        preprocessor = self.preprocessing_planner.build_pipeline(profile)
        
        # Get model search space
        available_models = ModelSearchSpace.get_model_names(dataset_info.task_type)
        print(f"\n✓ Model search space: {', '.join(available_models)}")
        
        # Agent-controlled experiment loop
        print(f"\n[2/3] Agent-Controlled Experiment Phase")
        print("-" * 60)
        
        current_model_name = available_models[0]
        tried_models = set()
        iteration = 0
        current_model_tuned = False
        baseline_accuracy = None
        
        # Cache models to persist tuning
        models = ModelSearchSpace.get_models(dataset_info.task_type)
        
        while iteration < self.rule_engine.max_iterations:
            iteration += 1
            tried_models.add(current_model_name)
            
            # Get model (from cached dict)
            model = models[current_model_name]
            
            # Run experiment
            result = self.runner.run(
                df, dataset_info.target_column, dataset_info.task_type,
                preprocessor, model, current_model_name, iteration, is_tuned=current_model_tuned
            )
            self.memory.add(result)
            
            # Build agent state
            previous_result = self.memory.get_all()[-2] if len(self.memory.get_all()) > 1 else None
            previous_accuracy = previous_result.accuracy if previous_result else None
            
            state = self.state_builder.build_state(
                result.accuracy, previous_accuracy, result.runtime, iteration, meta_features
            )
            
            # Agent decision
            decision = self.rule_engine.decide(state, current_model_name, available_models, 
                                              tried_models, current_model_tuned)
            self.decision_tracker.log(decision)
            
            # Display iteration result in table
            iter_table = [
                ["Iteration", iteration],
                ["Model", f"{current_model_name}{' (tuned)' if current_model_tuned else ''}"],
                ["Accuracy", f"{result.accuracy:.4f}"],
                ["Runtime", f"{result.runtime:.2f}s"],
                ["Decision", decision.action.value],
                ["Rule", decision.rule_triggered],
            ]
            print("\n" + tabulate(iter_table, tablefmt="simple"))
            print(f"Reason: {decision.reason}")
            
            # Execute decision
            if decision.action == AgentAction.TERMINATE:
                print("\n✓ Agent terminated search")
                break
            
            elif decision.action == AgentAction.TUNE_HYPERPARAMETERS:
                print(f"→ Tuning hyperparameters for {current_model_name}")
                
                # Store baseline
                baseline_accuracy = result.accuracy
                
                try:
                    # Get training data and tune
                    X_train, y_train = self.runner.get_train_data()
                    X_train_transformed = preprocessor.fit_transform(X_train)
                    
                    param_grid = self.tuner.get_param_grid(current_model_name, dataset_info.task_type)
                    tuned_model, best_params = self.tuner.tune(model, param_grid, X_train_transformed, y_train)
                    
                    # Update model in cache
                    models[current_model_name] = tuned_model
                    current_model_tuned = True
                    print(f"→ Tuning complete: {best_params}")
                except Exception as e:
                    print(f"⚠ Tuning failed: {str(e)}")
                    print(f"→ Continuing with baseline model")
                    current_model_tuned = False
            
            elif decision.action == AgentAction.SWITCH_MODEL and decision.next_model:
                # Check if we should rollback from tuning
                if current_model_tuned and baseline_accuracy and result.accuracy < baseline_accuracy:
                    print(f"⚠ Tuning degraded performance ({result.accuracy:.4f} < {baseline_accuracy:.4f})")
                    print(f"→ Rolling back to baseline model")
                
                current_model_name = decision.next_model
                current_model_tuned = False
                baseline_accuracy = None
                print(f"→ Switching to {current_model_name}")
        
        # Display experiment summary table with changes
        all_experiments = self.memory.get_all()
        exp_table = []
        for i, exp in enumerate(all_experiments):
            change = ""
            if i > 0:
                prev = all_experiments[i-1]
                if exp.model_name != prev.model_name:
                    change = f"Switched from {prev.model_name}"
                elif exp.is_tuned and not prev.is_tuned:
                    change = "Hyperparameter tuned"
                else:
                    change = "Continued"
            else:
                change = "Initial model"
            
            exp_table.append([
                exp.iteration, 
                exp.model_name, 
                "Y" if exp.is_tuned else "N", 
                f"{exp.accuracy:.4f}", 
                f"{exp.runtime:.2f}",
                change
            ])
        
        print("\n" + tabulate(exp_table, headers=["Iter", "Model", "Tuned", "Accuracy", "Time(s)", "Change"], tablefmt="grid"))
        
        # Plot model comparison bar chart
        model_labels = [f"{exp.model_name}{'*' if exp.is_tuned else ''} (#{exp.iteration})" 
                       for exp in all_experiments]
        accuracies = [exp.accuracy for exp in all_experiments]
        
        plt.clear_figure()
        plt.plotsize(60, 15)
        plt.bar(model_labels, accuracies)
        plt.title("Model Accuracy Comparison")
        plt.xlabel("Model (Iteration)")
        plt.ylabel("Accuracy")
        plt.theme("clear")
        plt.show()
        
        # Interpretability analysis
        print(f"\n[3/3] Analysis & Reporting Phase")
        print("-" * 60)
        
        best_result = self.memory.best_model()
        print(f"\n✓ Best model: {best_result.model_name} (Accuracy: {best_result.accuracy:.4f})")
        
        # Use a simpler model for interpretability if best model is too complex
        print(f"✓ Computing feature importance...")
        best_model_obj = models[best_result.model_name]
        
        # For RandomForest, use a lighter version for interpretability
        if best_result.model_name == 'RandomForest' and hasattr(best_model_obj, 'n_estimators'):
            if best_model_obj.n_estimators > 100:
                print(f"  (Using simplified model for faster analysis)")
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                if dataset_info.task_type.value == 'classification':
                    best_model_obj = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                else:
                    best_model_obj = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        
        best_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model_obj)
        ])
        
        # Train on subset for speed
        X = df.drop(columns=[dataset_info.target_column])
        y = df[dataset_info.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        best_pipeline.fit(X_train, y_train)
        
        # Analyze
        feature_importance = self.interpretability.analyze(
            best_pipeline, X_test, y_test, list(X.columns)
        )
        
        # Display feature importance table
        feat_table = [[feat, f"{imp:.4f}"] for feat, imp in list(feature_importance.items())[:10]]
        print("\n" + tabulate(feat_table, headers=["Feature", "Importance"], tablefmt="grid"))
        
        # Plot feature importance
        top_features = list(feature_importance.items())[:8]
        feat_names = [f[:15] for f, _ in top_features]  # Truncate long names
        feat_values = [imp for _, imp in top_features]
        
        plt.clear_figure()
        plt.plotsize(60, 15)
        plt.bar(feat_names, feat_values)
        plt.title("Top Feature Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.theme("clear")
        plt.show()
        
        # Generate report
        print(f"\n✓ Generating PDF report...")
        report_gen = FinalReportGenerator(output_report_path)
        report_gen.generate(
            dataset_info, profile, meta_features,
            self.memory.get_all(), self.decision_tracker.get_all(),
            feature_importance, best_result
        )
        
        print(f"✓ Report saved: {output_report_path}")
        print("\n" + "=" * 60)
        print("Experiment Complete")
        print("=" * 60)
