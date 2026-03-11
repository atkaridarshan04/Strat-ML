import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import io
from typing import List
from core.schemas import ExperimentResult

class ReportVisualizer:
    def __init__(self):
        self.fig_width = 6
        self.fig_height = 4
    
    def plot_accuracy_trend(self, experiments: List[ExperimentResult]) -> io.BytesIO:
        """Plot accuracy over iterations"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        iterations = [exp.iteration for exp in experiments]
        accuracies = [exp.accuracy for exp in experiments]
        colors = ['blue' if not exp.is_tuned else 'green' for exp in experiments]
        
        ax.plot(iterations, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
        for i, (it, acc, color) in enumerate(zip(iterations, accuracies, colors)):
            ax.scatter(it, acc, color=color, s=100, zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Accuracy Progression', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Baseline'),
            Patch(facecolor='green', label='Tuned')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_model_comparison(self, experiments: List[ExperimentResult]) -> io.BytesIO:
        """Plot model performance comparison"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Group by model
        model_best = {}
        for exp in experiments:
            key = f"{exp.model_name}{'*' if exp.is_tuned else ''}"
            if key not in model_best or exp.accuracy > model_best[key]:
                model_best[key] = exp.accuracy
        
        models = list(model_best.keys())
        accuracies = list(model_best.values())
        colors = ['green' if '*' in m else 'steelblue' for m in models]
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.02])
        plt.xticks(rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_feature_importance(self, importance: dict, top_n: int = 10) -> io.BytesIO:
        """Plot top feature importances"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Get top N features
        top_features = list(importance.items())[:top_n]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        # Horizontal bar chart
        y_pos = range(len(features))
        ax.barh(y_pos, importances, color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title('Top Feature Importance', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_runtime_comparison(self, experiments: List[ExperimentResult]) -> io.BytesIO:
        """Plot runtime per iteration"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        iterations = [exp.iteration for exp in experiments]
        runtimes = [exp.runtime for exp in experiments]
        colors = ['orange' if not exp.is_tuned else 'red' for exp in experiments]
        
        ax.bar(iterations, runtimes, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Runtime (seconds)', fontsize=11)
        ax.set_title('Experiment Runtime', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
