"""
Generate Full Clinical Fairness Report

Integrates all 4 tiers of Gemini LLM to produce a comprehensive clinical fairness report:
1. Causal graph validation
2. Clinical harm narratives
3. Intervention safety rationales
4. Auto-generated intervention code

Run with: python scripts/generate_full_clinical_report.py
"""

import logging
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Generate comprehensive clinical fairness report."""
    logger.info("="*80)
    logger.info("CLINICAL FAIRNESS REPORT GENERATION")
    logger.info("="*80)

    logger.info("\nStep 1: Loading benchmark results...")

    results_dir = Path("results")
    benchmark_files = sorted(results_dir.glob("benchmark_with_ci_*.json"))

    if not benchmark_files:
        logger.error("   No Medicare benchmark files found!")
        logger.info("\nPlease run the confidence intervals script first:")
        logger.info("  python scripts/add_confidence_intervals_to_benchmarks.py")
        return

    benchmark_path = benchmark_files[-1]
    logger.info(f"  Loading: {benchmark_path.name}")

    with open(benchmark_path, 'r') as f:
        benchmark_results = json.load(f)

    for method_name, results in benchmark_results.items():
        if 'accuracy' in results and isinstance(results['accuracy'], dict):
            benchmark_results[method_name]['accuracy_mean'] = results['accuracy']['mean']
            benchmark_results[method_name]['fnr_disparity_mean'] = results['fnr_disparity']['mean']
            benchmark_results[method_name]['dp_difference'] = results['dp_difference']['mean']

    logger.info(f"   Loaded Medicare benchmark with {len(benchmark_results)} methods")

    logger.info("\nStep 2: Loading Medicare data...")
    data_path = "data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"

    if not Path(data_path).exists():
        logger.error(f"   Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"   Loaded {len(df)} patient records")

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

    df_clean = df[[
        'age', 'sex', 'has_esrd', 'has_diabetes',
        'has_chf', 'has_copd', 'chronic_count',
        'race_white', 'high_cost'
    ]].dropna()

    logger.info(f"   Prepared {len(df_clean)} samples for analysis")

    logger.info("\nStep 3: Initializing Gemini client...")

    gemini_client = None

    try:
        from src.gemini_client import create_smart_llm_client
        llm_client = create_smart_llm_client()
        logger.info(f"   LLM client initialized: {llm_client.provider_name}")
        logger.info(f"   Model: {llm_client.model}")
        logger.info("   All 4 LLM tiers are now ACTIVE!")
    except ImportError as e:
        logger.warning(f"   LLM libraries not installed: {e}")
        logger.info("  ℹ Install with: pip install google-genai requests")
        logger.info("  ℹ Running in FALLBACK MODE")
        llm_client = None
    except Exception as e:
        logger.warning(f"   Could not initialize LLM client: {e}")
        logger.info("  ℹ Running in FALLBACK MODE")
        llm_client = None

    if llm_client is None:
        logger.info("  ℹ Using correlation-based validation and template narratives")

    logger.info("\nStep 4: Initializing report generator...")
    from src.clinical_fairness_report import ClinicalFairnessReportGenerator
    from src.causal_graph_refiner import CausalEdge

    report_gen = ClinicalFairnessReportGenerator(llm_client)
    logger.info(" Report generator ready")

    logger.info("\nStep 5: Defining expert causal edges...")
    expert_edges = [
        CausalEdge("age", "has_diabetes", 0.9, "expert", "Age increases diabetes risk"),
        CausalEdge("age", "has_chf", 0.85, "expert", "Age increases CHF risk"),
        CausalEdge("age", "has_copd", 0.80, "expert", "Age increases COPD risk"),
        CausalEdge("has_diabetes", "chronic_count", 0.95, "expert", "Diabetes contributes to chronic disease count"),
        CausalEdge("has_chf", "chronic_count", 0.95, "expert", "CHF contributes to chronic disease count"),
        CausalEdge("has_copd", "chronic_count", 0.95, "expert", "COPD contributes to chronic disease count"),
        CausalEdge("chronic_count", "high_cost", 0.90, "expert", "Chronic conditions increase healthcare costs"),
        CausalEdge("has_esrd", "high_cost", 0.95, "expert", "ESRD significantly increases costs"),
    ]
    logger.info(f"   Defined {len(expert_edges)} expert edges")

    discovered_edges = [
        ("race_white", "age"),
        ("age", "chronic_count"),
        ("chronic_count", "high_cost"),
        ("has_chf", "high_cost"),
        ("has_diabetes", "high_cost"),
        ("has_esrd", "high_cost"),
        ("age", "has_diabetes"),
        ("age", "has_chf"),
    ]
    logger.info(f"   Using {len(discovered_edges)} discovered edges")

    logger.info("\nStep 7: Generating comprehensive clinical fairness report...")
    logger.info("  (This may take 2-3 minutes with fallbacks, or longer with LLM)")

    report = report_gen.generate_report(
        data=df_clean,
        protected_attr="race_white",
        outcome="high_cost",
        expert_edges=expert_edges,
        discovered_edges=discovered_edges,
        benchmark_results=benchmark_results,
        context="medicare_high_cost"
    )

    report_files = list(Path("results").glob("clinical_fairness_*.json"))
    summary_files = list(Path("results").glob("clinical_fairness_*_summary.txt"))

    logger.info(f"\n   Report saved: {report_files[-1] if report_files else 'N/A'}")
    logger.info(f"   Summary saved: {summary_files[-1] if summary_files else 'N/A'}")

    logger.info("\n" + "="*80)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("="*80)

    code_files = list(Path("results/generated_code").glob("*.py")) if Path("results/generated_code").exists() else []
    llm_logs = len(list(Path("llm_logs").glob("*.json"))) if Path("llm_logs").exists() else 0
    llm_cache = len(list(Path("llm_cache").glob("*.json"))) if Path("llm_cache").exists() else 0

    logger.info(f"\nOutputs:")
    logger.info(f"  1. JSON report: {report_files[-1] if report_files else 'N/A'}")
    logger.info(f"  2. Text summary: {summary_files[-1] if summary_files else 'N/A'}")
    logger.info(f"  3. Generated code: {len(code_files)} files in results/generated_code/")
    logger.info(f"  4. LLM logs: {llm_logs} files in llm_logs/")
    logger.info(f"  5. LLM cache: {llm_cache} files in llm_cache/")

    logger.info("\nNext steps:")
    logger.info("  1. Review the summary file for clinical insights")
    logger.info("  2. Inspect generated code in results/generated_code/")
    logger.info("  3. Check llm_logs/ for reproducibility verification")
    logger.info("  4. Share results with clinical stakeholders")

    if gemini_client is None:
        logger.info("\n NOTE: This report was generated using fallback mode")
        logger.info("  To get LLM-enhanced narratives:")
        logger.info("  - Use Python 3.11 or 3.12 (currently using Python 3.13)")
        logger.info("  - Or wait for google-generativeai to support Python 3.13")

if __name__ == "__main__":
    main()
