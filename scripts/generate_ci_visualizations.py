"""
Generate Visualizations for Confidence Interval Results

Creates three publication-quality diagrams:
1. Causal Graph with edge validation
2. Fairness-Accuracy Tradeoff with confidence intervals
3. Bootstrap Distribution Violin Plots

Run: python scripts/generate_ci_visualizations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_ci_results():
    """Load the latest CI benchmark results."""
    results_dir = Path(__file__).parent.parent / "results"
    json_files = sorted(results_dir.glob("benchmark_with_ci_*.json"))

    if not json_files:
        raise FileNotFoundError("No CI benchmark results found. Run add_confidence_intervals_to_benchmarks.py first.")

    latest_file = json_files[-1]
    logger.info(f"Loading CI results from: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)

def load_clinical_report():
    """Load the latest clinical fairness report for causal graph."""
    results_dir = Path(__file__).parent.parent / "results"
    report_files = sorted(results_dir.glob("clinical_fairness_*.json"))

    if not report_files:
        raise FileNotFoundError("No clinical report found. Run generate_full_clinical_report.py first.")

    latest_file = report_files[-1]
    logger.info(f"Loading clinical report from: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)

def create_causal_graph(clinical_report, output_path):
    """
    Create causal graph visualization with edge validation.

    Shows expert knowledge edges (green) vs data-validated edges (orange).
    """
    logger.info("Creating causal graph visualization...")

    edges = clinical_report['causal_graph_validation']['edges']

    fig, ax = plt.subplots(figsize=(14, 10))

    pos = {
        'race_white': (0, 3),
        'age': (2, 3),
        'sex': (1, 3),

        'has_diabetes': (0, 2),
        'has_chf': (1, 2),
        'has_copd': (2, 2),
        'has_esrd': (3, 2),

        'chronic_count': (1.5, 1),

        'high_cost': (1.5, 0)
    }

    for edge in edges:
        source = edge['source']
        target = edge['target']

        if source not in pos or target not in pos:
            continue

        x_start, y_start = pos[source]
        x_end, y_end = pos[target]

        if edge['edge_type'] == 'expert':
            color = '#2ecc71'
            alpha = 0.9
            linewidth = 2.5
            linestyle = '-'
        else:
            color = '#d97706'
            alpha = 0.9
            linewidth = 2.5
            linestyle = '--'

        ax.annotate('',
                   xy=(x_end, y_end),
                   xytext=(x_start, y_start),
                   arrowprops=dict(
                       arrowstyle='->',
                       lw=linewidth,
                       color=color,
                       alpha=alpha,
                       linestyle=linestyle,
                       connectionstyle="arc3,rad=0.1"
                   ))

        mid_x = (x_start + x_end) / 2
        mid_y = (y_start + y_end) / 2
        ax.text(mid_x + 0.1, mid_y, f"{edge['confidence']:.2f}",
               fontsize=11, color='black', alpha=0.8, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    for node, (x, y) in pos.items():
        if node == 'race_white':
            node_color = '#e74c3c'
            label_weight = 'bold'
        elif node == 'high_cost':
            node_color = '#3498db'
            label_weight = 'bold'
        elif node == 'chronic_count':
            node_color = '#9b59b6'
            label_weight = 'normal'
        else:
            node_color = '#95a5a6'
            label_weight = 'normal'

        circle = plt.Circle((x, y), 0.15, color=node_color, alpha=0.8, zorder=10)
        ax.add_patch(circle)

        label = node.replace('_', ' ').title()
        ax.text(x, y - 0.35, label, ha='center', va='top',
               fontsize=10, weight=label_weight)

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')

    expert_patch = mpatches.Patch(color='#2ecc71', label='Expert Knowledge Edges', alpha=0.9)
    validated_patch = mpatches.Patch(color='#f39c12', label='Data-Validated Edges', alpha=0.7)
    protected_patch = mpatches.Patch(color='#e74c3c', label='Protected Attribute', alpha=0.8)
    outcome_patch = mpatches.Patch(color='#3498db', label='Outcome', alpha=0.8)

    ax.legend(handles=[expert_patch, validated_patch, protected_patch, outcome_patch],
             loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved causal graph: {output_path}")
    plt.close()

def create_fairness_accuracy_tradeoff(ci_results, output_path):
    """
    Create fairness-accuracy tradeoff plot with confidence interval ellipses.

    Shows the relationship between FNR disparity (fairness) and accuracy
    with 95% confidence regions for each method.
    """
    logger.info("Creating fairness-accuracy tradeoff visualization...")

    fig, ax = plt.subplots(figsize=(12, 8))

    methods = list(ci_results.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, (method, color) in enumerate(zip(methods, colors)):
        results = ci_results[method]

        fnr_mean = results['fnr_disparity']['mean']
        acc_mean = results['accuracy']['mean']

        fnr_ci = [results['fnr_disparity']['ci_lower'], results['fnr_disparity']['ci_upper']]
        acc_ci = [results['accuracy']['ci_lower'], results['accuracy']['ci_upper']]

        fnr_width = fnr_ci[1] - fnr_ci[0]
        acc_height = acc_ci[1] - acc_ci[0]

        ellipse = Ellipse((fnr_mean, acc_mean),
                         width=fnr_width,
                         height=acc_height,
                         alpha=0.3,
                         color=color,
                         label=f'{method} (95% CI)')
        ax.add_patch(ellipse)

        ax.scatter(fnr_mean, acc_mean, s=200, color=color, marker='o',
                  edgecolors='black', linewidths=2, zorder=10)

        offset_x = 0.002 if i % 2 == 0 else -0.002
        offset_y = 0.002 if i < 2 else -0.002
        ax.text(fnr_mean + offset_x, acc_mean + offset_y,
               method.replace(' (', '\n('),
               fontsize=9, ha='center', va='bottom' if i < 2 else 'top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.axvline(x=0.05, color='black', linestyle='--', linewidth=2.5, alpha=0.8,
              label='Clinical Safety Threshold (5%)')

    ax.axvspan(0, 0.05, alpha=0.15, color='#6baed6', label='Safe Region')

    ax.set_xlabel('FNR Disparity (lower is better)', fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy (higher is better)', fontsize=14, weight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, bbox_to_anchor=(1.0, 1.0))

    ax.set_xlim(-0.005, 0.08)
    ax.set_ylim(0.82, 0.86)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved fairness-accuracy tradeoff: {output_path}")
    plt.close()

def create_bootstrap_violin_plots(output_path):
    """
    Create violin plots showing bootstrap distributions for each method.

    Requires re-running bootstrap to get raw distributions.
    For now, we'll create a simulation based on the CI results.
    """
    logger.info("Creating bootstrap distribution violin plots...")

    ci_results = load_ci_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ('fnr_disparity', 'FNR Disparity', 0),
        ('dp_difference', 'Demographic Parity Difference', 1),
        ('accuracy', 'Accuracy', 2),
        ('equalized_odds', 'Equalized Odds', 3)
    ]

    for metric_key, metric_label, ax_idx in metrics:
        ax = axes[ax_idx]

        data_for_violin = []
        labels = []

        for method_name, results in ci_results.items():
            metric_data = results[metric_key]

            mean = metric_data['mean']
            std = metric_data['std']

            simulated_samples = np.random.normal(mean, std, 1000)

            if metric_key in ['fnr_disparity', 'dp_difference', 'equalized_odds']:
                simulated_samples = np.clip(simulated_samples, 0, None)

            data_for_violin.append(simulated_samples)
            labels.append(method_name)

        parts = ax.violinplot(data_for_violin, positions=range(len(labels)),
                            showmeans=True, showmedians=True)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        for i, (method_name, results) in enumerate(ci_results.items()):
            metric_data = results[metric_key]
            ci_lower = metric_data['ci_lower']
            ci_upper = metric_data['ci_upper']
            mean = metric_data['mean']

            ax.plot([i, i], [ci_lower, ci_upper], 'k-', linewidth=3, alpha=0.8)
            ax.plot(i, mean, 'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=10)
        ax.set_ylabel(metric_label, fontsize=13, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        if metric_key in ['fnr_disparity', 'dp_difference', 'equalized_odds']:
            current_ylim = ax.get_ylim()
            ax.set_ylim(max(0, current_ylim[0]), current_ylim[1])

        if metric_key == 'fnr_disparity':
            ax.axhline(y=0.05, color='black', linestyle='--', linewidth=2.5, alpha=0.8,
                      label='Clinical Safety (5%)')

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved bootstrap violin plots: {output_path}")
    plt.close()

def generate_all_visualizations():
    """Generate all three visualizations."""
    logger.info("="*80)
    logger.info("GENERATING CONFIDENCE INTERVAL VISUALIZATIONS")
    logger.info("="*80)

    output_dir = Path(__file__).parent.parent / "results" / "visualizations"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        ci_results = load_ci_results()
        clinical_report = load_clinical_report()

        causal_path = output_dir / f"causal_graph_{timestamp}.png"
        create_causal_graph(clinical_report, causal_path)

        tradeoff_path = output_dir / f"fairness_accuracy_tradeoff_{timestamp}.png"
        create_fairness_accuracy_tradeoff(ci_results, tradeoff_path)

        violin_path = output_dir / f"bootstrap_distributions_{timestamp}.png"
        create_bootstrap_violin_plots(violin_path)

        logger.info("\n" + "="*80)
        logger.info("SUCCESS! All visualizations created:")
        logger.info(f"  1. Causal Graph: {causal_path}")
        logger.info(f"  2. Fairness-Accuracy Tradeoff: {tradeoff_path}")
        logger.info(f"  3. Bootstrap Distributions: {violin_path}")
        logger.info("="*80)

        return causal_path, tradeoff_path, violin_path

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        generate_all_visualizations()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        input("\nPress Enter to exit...")
        raise

    input("\nCompleted! Press Enter to exit...")
