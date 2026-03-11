from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from typing import List
import io
from core.schemas import DatasetInfo, DatasetProfile, MetaFeatures, ExperimentResult, AgentDecision
from reporting.visualizer import ReportVisualizer

class FinalReportGenerator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.story = []
        self.visualizer = ReportVisualizer()
    
    def generate(self, dataset_info: DatasetInfo, profile: DatasetProfile,
                meta_features: MetaFeatures, experiments: List[ExperimentResult],
                decisions: List[AgentDecision], feature_importance: dict,
                best_model: ExperimentResult):
        """Generate comprehensive research report with visualizations"""
        
        doc = SimpleDocTemplate(self.output_path, pagesize=letter,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Title
        self._add_title("AutoML Research Prototype")
        self._add_subtitle("Experiment Report")
        self._add_spacer(0.3)
        
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
        
        # 4. Experiment Results
        self._add_section("4. Experiment Results")
        
        # Accuracy trend graph
        self._add_text("<b>Accuracy Progression:</b>")
        self._add_spacer(0.1)
        accuracy_plot = self.visualizer.plot_accuracy_trend(experiments)
        self._add_image(accuracy_plot, width=5*inch)
        self._add_spacer()
        
        # Experiment table
        self._add_text("<b>Detailed Results:</b>")
        self._add_spacer(0.1)
        self._add_experiment_table(experiments)
        self._add_spacer()
        
        # Model comparison graph
        self._add_text("<b>Model Comparison:</b>")
        self._add_spacer(0.1)
        model_plot = self.visualizer.plot_model_comparison(experiments)
        self._add_image(model_plot, width=5*inch)
        self._add_spacer()
        
        # Runtime graph
        self._add_text("<b>Runtime Analysis:</b>")
        self._add_spacer(0.1)
        runtime_plot = self.visualizer.plot_runtime_comparison(experiments)
        self._add_image(runtime_plot, width=5*inch)
        
        self.story.append(PageBreak())
        
        # 5. Agent Decisions
        self._add_section("5. Agent Decision Trace")
        self._add_decisions(decisions)
        self._add_spacer()
        
        # 6. Model Interpretability
        self._add_section("6. Model Interpretability")
        
        # Feature importance graph
        self._add_text("<b>Feature Importance Visualization:</b>")
        self._add_spacer(0.1)
        importance_plot = self.visualizer.plot_feature_importance(feature_importance)
        self._add_image(importance_plot, width=5*inch)
        self._add_spacer()
        
        # Feature importance table
        self._add_text("<b>Top 10 Features:</b>")
        self._add_spacer(0.1)
        self._add_feature_importance(feature_importance)
        self._add_spacer()
        
        # 7. Best Model Summary
        self._add_section("7. Best Model Summary")
        self._add_best_model(best_model)
        
        # Build PDF
        doc.build(self.story)
    
    def _add_title(self, text: str):
        self.story.append(Paragraph(text, self.styles['Title']))
    
    def _add_subtitle(self, text: str):
        style = ParagraphStyle('subtitle', parent=self.styles['Normal'], 
                              fontSize=14, textColor=colors.grey, alignment=1)
        self.story.append(Paragraph(text, style))
    
    def _add_section(self, text: str):
        self.story.append(Paragraph(text, self.styles['Heading2']))
    
    def _add_text(self, text: str):
        self.story.append(Paragraph(text, self.styles['Normal']))
    
    def _add_spacer(self, height: float = 0.2):
        self.story.append(Spacer(1, height * inch))
    
    def _add_image(self, img_buffer: io.BytesIO, width: float = 4*inch):
        img = Image(img_buffer, width=width, height=width*0.67)
        self.story.append(img)
    
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
        data = [['Iter', 'Model', 'Tuned', 'Accuracy', 'Time(s)']]
        for exp in experiments:
            data.append([
                str(exp.iteration),
                exp.model_name,
                'Y' if exp.is_tuned else 'N',
                f'{exp.accuracy:.4f}',
                f'{exp.runtime:.2f}'
            ])
        # Custom widths for 5 columns
        self._add_table_custom(data, [0.6*inch, 2*inch, 0.8*inch, 1.3*inch, 1*inch])
    
    def _add_decisions(self, decisions: List[AgentDecision]):
        for i, dec in enumerate(decisions, 1):
            text = f"<b>Iteration {i}:</b> Action={dec.action.value}, Rule={dec.rule_triggered}<br/>Reason: {dec.reason}"
            self._add_text(text)
            self._add_spacer(0.15)
    
    def _add_feature_importance(self, importance: dict):
        data = [['Feature', 'Importance']]
        for feat, imp in list(importance.items())[:10]:
            data.append([feat, f'{imp:.4f}'])
        self._add_table(data)
    
    def _add_best_model(self, best: ExperimentResult):
        data = [
            ['Property', 'Value'],
            ['Model', best.model_name],
            ['Tuned', 'Yes' if best.is_tuned else 'No'],
            ['Accuracy', f'{best.accuracy:.4f}'],
            ['Runtime', f'{best.runtime:.2f}s'],
            ['Iteration', str(best.iteration)]
        ]
        self._add_table(data)
    
    def _add_table(self, data: List[List[str]]):
        """Add table with automatic width adjustment"""
        num_cols = len(data[0])
        available_width = 6.5 * inch  # Safe width within margins
        col_widths = [available_width / num_cols] * num_cols
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        self.story.append(table)
    
    def _add_table_custom(self, data: List[List[str]], col_widths: List):
        """Add table with custom column widths"""
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        self.story.append(table)

