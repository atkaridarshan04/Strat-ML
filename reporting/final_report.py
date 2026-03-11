from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from typing import List
from core.schemas import DatasetInfo, DatasetProfile, MetaFeatures, ExperimentResult, AgentDecision

class FinalReportGenerator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.story = []
    
    def generate(self, dataset_info: DatasetInfo, profile: DatasetProfile,
                meta_features: MetaFeatures, experiments: List[ExperimentResult],
                decisions: List[AgentDecision], feature_importance: dict,
                best_model: ExperimentResult):
        """Generate comprehensive research report"""
        
        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        
        # Title
        self._add_title("AutoML Research Prototype - Experiment Report")
        self._add_spacer()
        
        # 1. Dataset Overview
        self._add_section("1. Dataset Overview")
        self._add_dataset_overview(dataset_info)
        self._add_spacer()
        
        # 2. Dataset Meta-Features
        self._add_section("2. Dataset Meta-Features")
        self._add_meta_features(meta_features)
        self._add_spacer()
        
        # 3. Data Quality Analysis
        self._add_section("3. Data Quality Analysis")
        self._add_data_quality(profile)
        self._add_spacer()
        
        # 4. Experiment Summary
        self._add_section("4. Experiment Summary")
        self._add_experiment_table(experiments)
        self._add_spacer()
        
        # 5. Agent Decisions
        self._add_section("5. Agent Decision Trace")
        self._add_decisions(decisions)
        self._add_spacer()
        
        # 6. Model Interpretability
        self._add_section("6. Model Interpretability")
        self._add_feature_importance(feature_importance)
        self._add_spacer()
        
        # 7. Best Model
        self._add_section("7. Best Model")
        self._add_best_model(best_model)
        
        # Build PDF
        doc.build(self.story)
    
    def _add_title(self, text: str):
        self.story.append(Paragraph(text, self.styles['Title']))
    
    def _add_section(self, text: str):
        self.story.append(Paragraph(text, self.styles['Heading2']))
    
    def _add_text(self, text: str):
        self.story.append(Paragraph(text, self.styles['Normal']))
    
    def _add_spacer(self, height: float = 0.2):
        self.story.append(Spacer(1, height * inch))
    
    def _add_dataset_overview(self, info: DatasetInfo):
        data = [
            ['Property', 'Value'],
            ['Task Type', info.task_type.value],
            ['Samples', str(info.samples)],
            ['Features', str(info.features)],
            ['Numeric Features', str(info.numeric_features)],
            ['Categorical Features', str(info.categorical_features)],
            ['Target Column', info.target_column]
        ]
        self._add_table(data)
    
    def _add_meta_features(self, meta: MetaFeatures):
        data = [
            ['Meta-Feature', 'Value'],
            ['Feature Entropy Mean', f'{meta.feature_entropy_mean:.4f}'],
            ['Feature Correlation Mean', f'{meta.feature_correlation_mean:.4f}'],
            ['Sparsity', f'{meta.sparsity:.4f}'],
            ['Dimensionality Ratio', f'{meta.dimensionality_ratio:.4f}'],
            ['Feature Variance Mean', f'{meta.feature_variance_mean:.4f}'],
            ['Skewness Mean', f'{meta.skewness_mean:.4f}']
        ]
        self._add_table(data)
    
    def _add_data_quality(self, profile: DatasetProfile):
        data = [
            ['Metric', 'Value'],
            ['Missing Ratio', f'{profile.missing_ratio:.4f}'],
            ['Class Imbalance', f'{profile.class_imbalance:.4f}' if profile.class_imbalance else 'N/A'],
            ['Samples', str(profile.num_samples)],
            ['Features', str(profile.num_features)]
        ]
        self._add_table(data)
    
    def _add_experiment_table(self, experiments: List[ExperimentResult]):
        data = [['Iteration', 'Model', 'Tuned', 'Accuracy', 'Runtime (s)']]
        for exp in experiments:
            data.append([
                str(exp.iteration),
                exp.model_name,
                'Yes' if exp.is_tuned else 'No',
                f'{exp.accuracy:.4f}',
                f'{exp.runtime:.2f}'
            ])
        self._add_table(data)
    
    def _add_decisions(self, decisions: List[AgentDecision]):
        for i, dec in enumerate(decisions, 1):
            text = f"<b>Iteration {i}:</b> Action={dec.action.value}, Rule={dec.rule_triggered}, Reason={dec.reason}"
            self._add_text(text)
            self._add_spacer(0.1)
    
    def _add_feature_importance(self, importance: dict):
        data = [['Feature', 'Importance']]
        for feat, imp in list(importance.items())[:10]:  # Top 10
            data.append([feat, f'{imp:.4f}'])
        self._add_table(data)
    
    def _add_best_model(self, best: ExperimentResult):
        data = [
            ['Property', 'Value'],
            ['Model', best.model_name],
            ['Accuracy', f'{best.accuracy:.4f}'],
            ['Runtime', f'{best.runtime:.2f}s'],
            ['Iteration', str(best.iteration)]
        ]
        self._add_table(data)
    
    def _add_table(self, data: List[List[str]]):
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(table)
