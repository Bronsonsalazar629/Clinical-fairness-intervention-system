"""
Add Confidence Intervals to Benchmark Results

Recomputes benchmarks with bootstrap confidence intervals to answer:
"Is the 2.2% FNR disparity from Fairlearn EO statistically different from
the 2.6% baseline disparity?"

Run: python scripts/add_confidence_intervals_to_benchmarks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging

from src.confidence_intervals import (
    compute_fairness_with_confidence_intervals,
    compare_methods_with_ci,
    generate_ci_summary_table,
    format_metric_with_ci,
    permutation_test_methods
)
from src.pc_algorithm_clinical import PCAlgorithmClinical

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_medicare_data():
    """Load Medicare patient summary dataset."""
    data_path = Path(__file__).parent.parent / "data" / "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Medicare data not found at {data_path}")

    logger.info("Loading Medicare data...")
    df = pd.read_csv(data_path)

    df['age'] = 2008 - (df['BENE_BIRTH_DT'] // 10000)
    df['sex'] = (df['BENE_SEX_IDENT_CD'] == 1).astype(int)
    df['race_white'] = (df['BENE_RACE_CD'] == 1).astype(int)
    df['has_esrd'] = (df['BENE_ESRD_IND'] == 'Y').astype(int)
    df['has_diabetes'] = (df['SP_DIABETES'] == 1).astype(int)
    df['has_chf'] = (df['SP_CHF'] == 1).astype(int)
    df['has_copd'] = (df['SP_COPD'] == 1).astype(int)
    df['chronic_count'] = (
        (df['SP_ALZHDMTA'] == 1).astype(int) +
        (df['SP_CHF'] == 1).astype(int) +
        (df['SP_CHRNKIDN'] == 1).astype(int) +
        (df['SP_CNCR'] == 1).astype(int) +
        (df['SP_COPD'] == 1).astype(int) +
        (df['SP_DEPRESSN'] == 1).astype(int) +
        (df['SP_DIABETES'] == 1).astype(int)
    )
    df['total_cost'] = (
        df['MEDREIMB_IP'].fillna(0) +
        df['MEDREIMB_OP'].fillna(0) +
        df['MEDREIMB_CAR'].fillna(0)
    )

    cost_threshold = df['total_cost'].quantile(0.75)
    df['high_cost'] = (df['total_cost'] > cost_threshold).astype(int)

    feature_cols = ['age', 'sex', 'has_esrd', 'has_diabetes', 'has_chf', 'has_copd', 'chronic_count']

    df_clean = df[feature_cols + ['high_cost', 'race_white']].dropna()

    logger.info(f"Loaded Medicare: {len(df_clean):,} patients")
    logger.info(f"  High-cost rate: {df_clean['high_cost'].mean():.1%}")
    logger.info(f"  White patients: {df_clean['race_white'].mean():.1%}")

    return df_clean

def train_causal_discovery_method(X_train, y_train, X_test, protected_train, feature_cols):
    """
    Train a causally-fair model using PC algorithm causal discovery.

    Discovers causal structure, identifies features that are causal descendants
    of the protected attribute (race_white), and trains a model excluding those
    features to avoid proxy discrimination.

    Returns:
        y_pred: Predictions on test set using only non-descendant features
        causal_info: Dict with kept/excluded features, bias pathways, edges
    """
    import networkx as nx
    from sklearn.linear_model import LogisticRegression

    # Build DataFrame for causal discovery
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['race_white'] = protected_train
    train_df['high_cost'] = y_train

    # Define temporal ordering for clinical domain knowledge
    temporal_order = {
        'race_white': 0, 'age': 0, 'sex': 0,
        'has_esrd': 1, 'has_diabetes': 1, 'has_chf': 1, 'has_copd': 1,
        'chronic_count': 2,
        'high_cost': 3
    }

    clinical_variables = {
        'conditions': ['has_esrd', 'has_diabetes', 'has_chf', 'has_copd', 'chronic_count']
    }

    logger.info("Running PC Algorithm for causal discovery...")
    pc_algo = PCAlgorithmClinical(
        data=train_df,
        protected_attr='race_white',
        outcome='high_cost',
        temporal_order=temporal_order,
        clinical_variables=clinical_variables,
        alpha=0.001,
        n_bootstrap=50
    )
    result = pc_algo.run()

    # Find features causally downstream of race_white
    causal_graph = result['causal_graph']
    if 'race_white' in causal_graph:
        descendants = nx.descendants(causal_graph, 'race_white')
        # If all features would be excluded (dense graph), fall back to direct children only
        excluded_features = descendants & set(feature_cols)
        if len(excluded_features) >= len(feature_cols):
            logger.warning("  All features are descendants of race_white (dense graph). "
                           "Falling back to direct children only.")
            direct_children = set(causal_graph.successors('race_white'))
            excluded_features = direct_children & set(feature_cols)
    else:
        descendants = set()
        excluded_features = set()

    kept_features = [f for f in feature_cols if f not in excluded_features]
    kept_indices = [feature_cols.index(f) for f in kept_features]

    logger.info(f"  Causal discovery: {len(kept_features)} features kept, {len(excluded_features)} excluded")
    logger.info(f"  Kept: {kept_features}")
    logger.info(f"  Excluded (descendants of race): {sorted(excluded_features)}")

    # Train model on non-descendant features only
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train[:, kept_indices], y_train)
    y_pred = model.predict(X_test[:, kept_indices])

    # Build causal info dict
    causal_info = {
        'kept_features': kept_features,
        'excluded_features': sorted(excluded_features),
        'n_edges': len(result['directed_edges']),
        'directed_edges': [list(e) for e in result['directed_edges']],
        'bias_pathways': [
            {
                'path': p.path,
                'pathway_type': p.pathway_type,
                'intervention_point': p.intervention_point,
                'rationale': p.rationale,
                'confidence': p.confidence,
                'sensitivity_robustness': p.sensitivity_robustness
            }
            for p in result['bias_pathways']
        ]
    }

    return y_pred, causal_info


def train_and_evaluate_methods_with_ci(df, n_bootstrap=1000):
    """
    Train all fairness methods and compute metrics with CIs.

    Returns:
        Dictionary mapping method name to CI results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing

    feature_cols = ['age', 'sex', 'has_esrd', 'has_diabetes', 'has_chf', 'has_copd', 'chronic_count']

    X = df[feature_cols].values
    y = df['high_cost'].values
    protected = df['race_white'].values

    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected, test_size=0.3, random_state=42
    )

    methods_results = {}

    logger.info("Training Unmitigated Baseline...")
    model_baseline = LogisticRegression(random_state=42, max_iter=1000)
    model_baseline.fit(X_train, y_train)
    y_pred_baseline = model_baseline.predict(X_test)

    logger.info("Computing CIs for Unmitigated Baseline...")
    methods_results['Unmitigated Baseline'] = compute_fairness_with_confidence_intervals(
        y_test, y_pred_baseline, protected_test, n_bootstrap=n_bootstrap,
        return_raw_samples=True
    )

    logger.info("Training Fairlearn (Demographic Parity)...")
    model_dp = ExponentiatedGradient(
        LogisticRegression(random_state=42, max_iter=1000),
        constraints=DemographicParity()
    )
    model_dp.fit(X_train, y_train, sensitive_features=protected_train)
    y_pred_dp = model_dp.predict(X_test)

    logger.info("Computing CIs for Fairlearn (Demographic Parity)...")
    methods_results['Fairlearn (Demographic Parity)'] = compute_fairness_with_confidence_intervals(
        y_test, y_pred_dp, protected_test, n_bootstrap=n_bootstrap,
        return_raw_samples=True
    )

    logger.info("Training Fairlearn (Equalized Odds)...")
    model_eo = ExponentiatedGradient(
        LogisticRegression(random_state=42, max_iter=1000),
        constraints=EqualizedOdds()
    )
    model_eo.fit(X_train, y_train, sensitive_features=protected_train)
    y_pred_eo = model_eo.predict(X_test)

    logger.info("Computing CIs for Fairlearn (Equalized Odds)...")
    methods_results['Fairlearn (Equalized Odds)'] = compute_fairness_with_confidence_intervals(
        y_test, y_pred_eo, protected_test, n_bootstrap=n_bootstrap,
        return_raw_samples=True
    )

    logger.info("Training AIF360 Reweighing...")

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['high_cost'] = y_train
    train_df['race_white'] = protected_train

    aif_dataset = BinaryLabelDataset(
        df=train_df,
        label_names=['high_cost'],
        protected_attribute_names=['race_white']
    )

    rw = Reweighing(unprivileged_groups=[{'race_white': 0}],
                   privileged_groups=[{'race_white': 1}])
    aif_dataset_transformed = rw.fit_transform(aif_dataset)

    model_aif = LogisticRegression(random_state=42, max_iter=1000)
    model_aif.fit(X_train, y_train, sample_weight=aif_dataset_transformed.instance_weights)
    y_pred_aif = model_aif.predict(X_test)

    logger.info("Computing CIs for AIF360 Reweighing...")
    methods_results['AIF360 Reweighing'] = compute_fairness_with_confidence_intervals(
        y_test, y_pred_aif, protected_test, n_bootstrap=n_bootstrap,
        return_raw_samples=True
    )

    # 5th method: Causal Discovery (PC Algorithm)
    logger.info("Training Causal Discovery (PC Algorithm)...")
    y_pred_causal, causal_info = train_causal_discovery_method(
        X_train, y_train, X_test, protected_train, feature_cols
    )

    logger.info("Computing CIs for Causal Discovery (PC Algorithm)...")
    methods_results['Causal Discovery (PC Algorithm)'] = compute_fairness_with_confidence_intervals(
        y_test, y_pred_causal, protected_test, n_bootstrap=n_bootstrap,
        return_raw_samples=True
    )
    methods_results['Causal Discovery (PC Algorithm)']['causal_info'] = causal_info

    return methods_results

def save_results_with_ci(methods_results):
    """Save results with confidence intervals."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results"

    json_path = results_dir / f"benchmark_with_ci_{timestamp}.json"

    # Strip non-serializable fields (bootstrap_samples arrays, causal_info)
    serializable_results = {}
    for method_name, results in methods_results.items():
        serializable_results[method_name] = {
            k: v for k, v in results.items()
            if k not in ('bootstrap_samples', 'causal_info')
        }
        # Include causal_info if present (already serializable)
        if 'causal_info' in results:
            serializable_results[method_name]['causal_info'] = results['causal_info']

    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Saved JSON results: {json_path}")

    rows = []

    for method_name, results in methods_results.items():
        row = {
            'Method': method_name,
            'FNR Disparity (Mean)': results['fnr_disparity']['mean'],
            'FNR Disparity (CI Lower)': results['fnr_disparity']['ci_lower'],
            'FNR Disparity (CI Upper)': results['fnr_disparity']['ci_upper'],
            'Accuracy (Mean)': results['accuracy']['mean'],
            'Accuracy (CI Lower)': results['accuracy']['ci_lower'],
            'Accuracy (CI Upper)': results['accuracy']['ci_upper'],
            'DP Difference (Mean)': results['dp_difference']['mean'],
            'DP Difference (CI Lower)': results['dp_difference']['ci_lower'],
            'DP Difference (CI Upper)': results['dp_difference']['ci_upper'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = results_dir / f"benchmark_with_ci_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved CSV table: {csv_path}")

    return json_path, csv_path

def generate_report(methods_results):
    """Generate human-readable report."""
    report = []

    report.append("="*80)
    report.append("FAIRNESS BENCHMARK WITH CONFIDENCE INTERVALS")
    report.append("="*80)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("SUMMARY TABLE")
    report.append("-"*80)
    report.append("")
    report.append(generate_ci_summary_table(methods_results))
    report.append("")

    report.append("DETAILED RESULTS")
    report.append("-"*80)
    report.append("")

    for method_name, results in methods_results.items():
        report.append(f"\n{method_name}:")
        report.append(f"  FNR Disparity: {format_metric_with_ci(results, 'fnr_disparity')}")
        report.append(f"  DP Difference: {format_metric_with_ci(results, 'dp_difference')}")
        report.append(f"  Accuracy: {format_metric_with_ci(results, 'accuracy')}")
        report.append(f"  Equalized Odds: {format_metric_with_ci(results, 'equalized_odds')}")

    report.append("\n" + "="*80)
    report.append("STATISTICAL SIGNIFICANCE TESTS")
    report.append("="*80)
    report.append("")

    comparisons = compare_methods_with_ci(methods_results, 'fnr_disparity')

    for comparison_name, comparison_data in comparisons.items():
        report.append(f"\n{comparison_name}:")
        report.append(f"  Statistically Significant: {'YES' if comparison_data['significant'] else 'NO'}")
        report.append(f"  Difference in FNR Disparity: {comparison_data['difference']:.3%}")
        report.append(f"  {comparison_name.split('_vs_')[0]} CI: {comparison_data['method_a_ci']}")
        report.append(f"  {comparison_name.split('_vs_')[1]} CI: {comparison_data['method_b_ci']}")

        if comparison_data['significant']:
            winner = comparison_name.split('_vs_')[0] if comparison_data['difference'] > 0 else comparison_name.split('_vs_')[1]
            report.append(f"  Winner: {winner} (lower FNR disparity)")

    # Permutation tests
    report.append("\n" + "="*80)
    report.append("PERMUTATION TESTS (FNR DISPARITY)")
    report.append("="*80)
    report.append("")

    try:
        perm_results = permutation_test_methods(
            methods_results, metric_name='fnr_disparity',
            n_permutations=10000, correction='bonferroni'
        )

        for comparison_name, perm_data in perm_results.items():
            sig_marker = " ***" if perm_data['significant'] else ""
            report.append(f"\n{comparison_name}:")
            report.append(f"  Raw p-value: {perm_data['p_value']:.6f}")
            report.append(f"  Corrected p-value (Bonferroni): {perm_data['p_value_corrected']:.6f}")
            report.append(f"  Effect size (Cohen's d): {perm_data['effect_size']:.3f}")
            report.append(f"  Significant: {'YES' if perm_data['significant'] else 'NO'}{sig_marker}")
    except Exception as e:
        report.append(f"  Permutation tests skipped: {e}")

    # Causal discovery details
    causal_method_key = 'Causal Discovery (PC Algorithm)'
    if causal_method_key in methods_results and 'causal_info' in methods_results[causal_method_key]:
        causal_info = methods_results[causal_method_key]['causal_info']

        report.append("\n" + "="*80)
        report.append("CAUSAL DISCOVERY DETAILS")
        report.append("="*80)
        report.append("")

        report.append(f"Features kept ({len(causal_info['kept_features'])}): {', '.join(causal_info['kept_features'])}")
        report.append(f"Features excluded as descendants of race ({len(causal_info['excluded_features'])}): "
                      f"{', '.join(causal_info['excluded_features']) if causal_info['excluded_features'] else 'None'}")
        report.append(f"Total directed edges discovered: {causal_info['n_edges']}")
        report.append("")

        pathways = causal_info['bias_pathways']
        report.append(f"Bias Pathways ({len(pathways)} detected):")
        for i, pathway in enumerate(pathways, 1):
            report.append(f"  {i}. {' -> '.join(pathway['path'])}")
            report.append(f"     Type: {pathway['pathway_type']}")
            report.append(f"     Intervention point: {pathway['intervention_point']}")
            report.append(f"     Robustness: {pathway['sensitivity_robustness']:.3f}")

    report.append("\n" + "="*80)
    report.append("KEY FINDINGS")
    report.append("="*80)
    report.append("")

    best_method = min(methods_results.items(), key=lambda x: x[1]['fnr_disparity']['mean'])

    report.append(f"1. Best Performing Method: {best_method[0]}")
    report.append(f"   FNR Disparity: {format_metric_with_ci(best_method[1], 'fnr_disparity')}")
    report.append("")

    sig_count = sum(1 for c in comparisons.values() if c['significant'])
    total_comparisons = len(comparisons)

    report.append(f"2. Statistical Significance: {sig_count}/{total_comparisons} pairwise comparisons show")
    report.append(f"   non-overlapping confidence intervals (statistically significant difference)")
    report.append("")

    report.append(f"3. Clinical Safety: All methods with FNR disparity < 5% are considered clinically safe.")

    safe_methods = [name for name, results in methods_results.items()
                   if results['fnr_disparity']['ci_upper'] < 0.05]

    report.append(f"   Safe methods (upper CI < 5%): {', '.join(safe_methods) if safe_methods else 'None'}")

    return "\n".join(report)

def main():
    """Main execution."""
    logger.info("Starting Medicare benchmark with confidence intervals...")

    logger.info("Loading Medicare patient summary data...")
    df = load_medicare_data()

    logger.info("Training and evaluating all fairness methods with bootstrap CIs...")
    logger.info("This will take several minutes (1000 bootstrap iterations per method)...")

    methods_results = train_and_evaluate_methods_with_ci(df, n_bootstrap=1000)

    logger.info("Saving results...")
    json_path, csv_path = save_results_with_ci(methods_results)

    report = generate_report(methods_results)

    results_dir = Path(__file__).parent.parent / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"benchmark_with_ci_{timestamp}_report.txt"

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved report: {report_path}")

    print("\n" + report)

    logger.info("\nDone! Files created:")
    logger.info(f"  - {json_path}")
    logger.info(f"  - {csv_path}")
    logger.info(f"  - {report_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        input("\nPress Enter to exit...")
        raise

    input("\nCompleted! Press Enter to exit...")
