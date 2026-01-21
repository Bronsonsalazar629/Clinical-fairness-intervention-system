"""
Confidence Intervals for Fairness Metrics

Provides bootstrap confidence intervals to determine if fairness metric
differences are statistically meaningful.

Example: Is FNR disparity of 2.2% significantly different from 2.6%?
"""

import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CORE METRIC COMPUTATION
# =============================================================================

def compute_fnr_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Tuple[float, float]:
    """
    Compute False Negative Rate for each group.

    FNR = FN / (FN + TP) = FN / Actual Positives

    Returns:
        (fnr_group_0, fnr_group_1)
    """
    fnr_by_group = []

    for group in [0, 1]:
        mask = protected == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        # Actual positives in this group
        actual_positives = (y_true_g == 1).sum()

        if actual_positives == 0:
            fnr_by_group.append(0.0)
            continue

        # False negatives: actual=1, predicted=0
        false_negatives = ((y_true_g == 1) & (y_pred_g == 0)).sum()

        fnr = false_negatives / actual_positives
        fnr_by_group.append(fnr)

    return fnr_by_group[0], fnr_by_group[1]


def compute_dp_by_group(
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Tuple[float, float]:
    """
    Compute positive prediction rate for each group.

    Returns:
        (rate_group_0, rate_group_1)
    """
    rates = []

    for group in [0, 1]:
        mask = protected == group
        y_pred_g = y_pred[mask]

        if len(y_pred_g) == 0:
            rates.append(0.0)
            continue

        rate = y_pred_g.mean()
        rates.append(rate)

    return rates[0], rates[1]


def compute_tpr_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Tuple[float, float]:
    """
    Compute True Positive Rate for each group.

    TPR = TP / (TP + FN) = TP / Actual Positives

    Returns:
        (tpr_group_0, tpr_group_1)
    """
    tpr_by_group = []

    for group in [0, 1]:
        mask = protected == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        actual_positives = (y_true_g == 1).sum()

        if actual_positives == 0:
            tpr_by_group.append(0.0)
            continue

        true_positives = ((y_true_g == 1) & (y_pred_g == 1)).sum()
        tpr = true_positives / actual_positives
        tpr_by_group.append(tpr)

    return tpr_by_group[0], tpr_by_group[1]


def compute_fpr_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Tuple[float, float]:
    """
    Compute False Positive Rate for each group.

    FPR = FP / (FP + TN) = FP / Actual Negatives

    Returns:
        (fpr_group_0, fpr_group_1)
    """
    fpr_by_group = []

    for group in [0, 1]:
        mask = protected == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        actual_negatives = (y_true_g == 0).sum()

        if actual_negatives == 0:
            fpr_by_group.append(0.0)
            continue

        false_positives = ((y_true_g == 0) & (y_pred_g == 1)).sum()
        fpr = false_positives / actual_negatives
        fpr_by_group.append(fpr)

    return fpr_by_group[0], fpr_by_group[1]


def compute_metrics_single_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray
) -> Dict[str, float]:
    """
    Compute all fairness metrics for a single sample.

    Returns:
        Dictionary with:
        - fnr_disparity: |FNR_group0 - FNR_group1|
        - dp_difference: |PositiveRate_group0 - PositiveRate_group1|
        - accuracy: overall accuracy
        - equalized_odds: average of |TPR_diff| and |FPR_diff|
        - fnr_group_0: FNR for group 0
        - fnr_group_1: FNR for group 1
        - tpr_group_0: TPR for group 0
        - tpr_group_1: TPR for group 1
        - fpr_group_0: FPR for group 0
        - fpr_group_1: FPR for group 1
    """
    # FNR Disparity
    fnr_0, fnr_1 = compute_fnr_by_group(y_true, y_pred, protected)
    fnr_disparity = abs(fnr_0 - fnr_1)

    # Demographic Parity Difference
    dp_0, dp_1 = compute_dp_by_group(y_pred, protected)
    dp_difference = abs(dp_0 - dp_1)

    # TPR and FPR for Equalized Odds
    tpr_0, tpr_1 = compute_tpr_by_group(y_true, y_pred, protected)
    fpr_0, fpr_1 = compute_fpr_by_group(y_true, y_pred, protected)

    tpr_diff = abs(tpr_0 - tpr_1)
    fpr_diff = abs(fpr_0 - fpr_1)
    equalized_odds = (tpr_diff + fpr_diff) / 2

    # Accuracy
    accuracy = (y_true == y_pred).mean()

    return {
        'fnr_disparity': fnr_disparity,
        'dp_difference': dp_difference,
        'accuracy': accuracy,
        'equalized_odds': equalized_odds,
        'fnr_group_0': fnr_0,
        'fnr_group_1': fnr_1,
        'tpr_group_0': tpr_0,
        'tpr_group_1': tpr_1,
        'fpr_group_0': fpr_0,
        'fpr_group_1': fpr_1,
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def compute_fairness_with_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute fairness metrics with bootstrap confidence intervals.

    This is the MAIN FUNCTION to call.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        protected: Protected attribute values (n_samples,)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with structure:
        {
            'fnr_disparity': {
                'mean': 0.022,
                'ci_lower': 0.018,
                'ci_upper': 0.026,
                'std': 0.002
            },
            'dp_difference': {...},
            'accuracy': {...},
            'equalized_odds': {...}
        }

    Example usage:
        >>> results = compute_fairness_with_confidence_intervals(
        ...     y_true, y_pred, protected_attr
        ... )
        >>> print(f"FNR Disparity: {results['fnr_disparity']['mean']:.1%} "
        ...       f"[95% CI: {results['fnr_disparity']['ci_lower']:.1%}-"
        ...       f"{results['fnr_disparity']['ci_upper']:.1%}]")
        FNR Disparity: 2.2% [95% CI: 1.8%-2.6%]
    """
    logger.info(f"Computing bootstrap CIs with {n_bootstrap} iterations...")

    np.random.seed(random_seed)

    n_samples = len(y_true)
    alpha = 1 - confidence_level

    # Storage for bootstrap samples
    bootstrap_results = {
        'fnr_disparity': [],
        'dp_difference': [],
        'accuracy': [],
        'equalized_odds': [],
        'fnr_group_0': [],
        'fnr_group_1': [],
        'tpr_group_0': [],
        'tpr_group_1': [],
        'fpr_group_0': [],
        'fpr_group_1': [],
    }

    # Bootstrap resampling
    for i in range(n_bootstrap):
        if i % 100 == 0:
            logger.debug(f"Bootstrap iteration {i}/{n_bootstrap}")

        # Sample WITH replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        protected_boot = protected[indices]

        # Compute metrics on bootstrap sample
        metrics = compute_metrics_single_sample(
            y_true_boot, y_pred_boot, protected_boot
        )

        # Store results
        for key in bootstrap_results:
            bootstrap_results[key].append(metrics[key])

    # Compute statistics from bootstrap distribution
    final_results = {}

    for metric_name, values in bootstrap_results.items():
        values = np.array(values)

        final_results[metric_name] = {
            'mean': np.mean(values),
            'ci_lower': np.percentile(values, (alpha / 2) * 100),
            'ci_upper': np.percentile(values, (1 - alpha / 2) * 100),
            'std': np.std(values),
        }

    logger.info("Bootstrap CI computation complete")

    return final_results



def format_metric_with_ci(
    results: Dict[str, Dict[str, float]],
    metric_name: str,
    as_percentage: bool = True
) -> str:
    """
    Format a metric with CI for paper reporting.

    Args:
        results: Output from compute_fairness_with_confidence_intervals
        metric_name: Key in results dict
        as_percentage: If True, format as percentage

    Returns:
        Formatted string like "2.2% [95% CI: 1.8%-2.6%]"
    """
    data = results[metric_name]

    if as_percentage:
        return (
            f"{data['mean']:.1%} "
            f"[95% CI: {data['ci_lower']:.1%}-{data['ci_upper']:.1%}]"
        )
    else:
        return (
            f"{data['mean']:.3f} "
            f"[95% CI: {data['ci_lower']:.3f}-{data['ci_upper']:.3f}]"
        )


def compare_methods_with_ci(
    methods_results: Dict[str, Dict],
    metric_name: str = 'fnr_disparity'
) -> Dict[str, Dict[str, any]]:
    """
    Determine if differences between methods are statistically significant.

    Two methods are significantly different if their 95% CIs don't overlap.

    Args:
        methods_results: Dict mapping method name to CI results
        metric_name: Which metric to compare

    Returns:
        Dictionary of pairwise comparisons with significance and effect size

    Example:
        >>> comparisons = compare_methods_with_ci({
        ...     'Baseline': baseline_results,
        ...     'Fairlearn_EO': fairlearn_results,
        ... })
        >>> print(comparisons['Baseline_vs_Fairlearn_EO'])
        {
            'significant': True,
            'difference': 0.004,
            'method_a_ci': '2.6% [2.2%-3.0%]',
            'method_b_ci': '2.2% [1.8%-2.6%]'
        }
    """
    method_names = list(methods_results.keys())
    comparisons = {}

    for i, method_a in enumerate(method_names):
        for method_b in method_names[i+1:]:
            ci_a = methods_results[method_a][metric_name]
            ci_b = methods_results[method_b][metric_name]

            # Check if CIs overlap
            # No overlap if: lower_a > upper_b OR lower_b > upper_a
            no_overlap = (
                ci_a['ci_lower'] > ci_b['ci_upper'] or
                ci_b['ci_lower'] > ci_a['ci_upper']
            )

            # Compute difference in means
            difference = ci_a['mean'] - ci_b['mean']

            comparisons[f"{method_a}_vs_{method_b}"] = {
                'significant': no_overlap,
                'difference': difference,
                'abs_difference': abs(difference),
                'method_a_mean': ci_a['mean'],
                'method_b_mean': ci_b['mean'],
                'method_a_ci': f"{ci_a['ci_lower']:.1%}-{ci_a['ci_upper']:.1%}",
                'method_b_ci': f"{ci_b['ci_lower']:.1%}-{ci_b['ci_upper']:.1%}",
            }

    return comparisons


def generate_ci_summary_table(
    methods_results: Dict[str, Dict],
    metrics: List[str] = None
) -> str:
    """
    Generate a formatted table showing all methods with CIs.

    Args:
        methods_results: Dict mapping method name to CI results
        metrics: List of metrics to include (default: all main metrics)

    Returns:
        Formatted markdown table string
    """
    if metrics is None:
        metrics = ['fnr_disparity', 'dp_difference', 'accuracy', 'equalized_odds']

    # Header
    table = "| Method | " + " | ".join([m.replace('_', ' ').title() for m in metrics]) + " |\n"
    table += "|" + "---|" * (len(metrics) + 1) + "\n"

    # Rows
    for method_name, results in methods_results.items():
        row = f"| {method_name} |"
        for metric in metrics:
            if metric in results:
                formatted = format_metric_with_ci(results, metric, as_percentage=True)
                row += f" {formatted} |"
            else:
                row += " N/A |"
        table += row + "\n"

    return table



if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example with simulated data
    print("="*80)
    print("FAIRNESS METRICS WITH CONFIDENCE INTERVALS - DEMO")
    print("="*80)

    # Simulate Medicare-like data
    np.random.seed(42)
    n = 116352

    # Simulate outcomes
    y_true = np.random.binomial(1, 0.25, n)  # 25% high-cost
    protected = np.random.binomial(1, 0.17, n)  # 17% Non-White

    # Simulate different models
    # Baseline: slight bias
    y_pred_baseline = y_true.copy()
    bias_indices = (protected == 1) & (y_true == 1)
    y_pred_baseline[bias_indices] = np.random.binomial(1, 0.95, bias_indices.sum())

    # Fairlearn EO: less bias
    y_pred_fairlearn = y_true.copy()
    y_pred_fairlearn[bias_indices] = np.random.binomial(1, 0.97, bias_indices.sum())

    # Compute CIs for each method
    methods_results = {}

    for method_name, y_pred in [
        ('Baseline', y_pred_baseline),
        ('Fairlearn_EO', y_pred_fairlearn),
    ]:
        print(f"\nComputing CIs for {method_name}...")

        results = compute_fairness_with_confidence_intervals(
            y_true, y_pred, protected, n_bootstrap=1000
        )

        methods_results[method_name] = results

        print(f"\n{method_name} Results:")
        print(f"  FNR Disparity: {format_metric_with_ci(results, 'fnr_disparity')}")
        print(f"  DP Difference: {format_metric_with_ci(results, 'dp_difference')}")
        print(f"  Accuracy: {format_metric_with_ci(results, 'accuracy')}")
        print(f"  Equalized Odds: {format_metric_with_ci(results, 'equalized_odds')}")

    # Compare methods
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    comparisons = compare_methods_with_ci(methods_results, 'fnr_disparity')

    for comparison_name, comparison_data in comparisons.items():
        print(f"\n{comparison_name}:")
        print(f"  Significant: {'YES' if comparison_data['significant'] else 'NO'}")
        print(f"  Difference: {comparison_data['difference']:.1%}")
        print(f"  Method A CI: {comparison_data['method_a_ci']}")
        print(f"  Method B CI: {comparison_data['method_b_ci']}")

    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(generate_ci_summary_table(methods_results))
