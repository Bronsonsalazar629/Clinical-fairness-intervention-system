"""
Interactive Visualization App for Causal Graph and Fairness-Accuracy Tradeoff
Uses real data from clinical-fairness-hack results
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Clinical Fairness Visualization",
    page_icon="x",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color:
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color:
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .safe-badge {
        background-color:
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .conditional-badge {
        background-color:
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .not-safe-badge {
        background-color:
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_results_data():
    """Load real experimental results from JSON file"""
    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent
    results_dir = project_dir / "results"

    if not results_dir.exists():
        st.error(f"Results directory not found at: {results_dir}")
        st.info(f"App directory: {app_dir}")
        st.info(f"Project directory: {project_dir}")
        return None

    json_files = list(results_dir.glob("clinical_fairness_*.json"))
    if not json_files:
        st.error(f"No results files found in results directory: {results_dir}")
        st.info(f"Files in directory: {list(results_dir.glob('*'))}")
        return None

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return data

@st.cache_data
def load_benchmark_data():
    """Load benchmark results with confidence intervals"""
    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent
    results_dir = project_dir / "results"

    csv_files = list(results_dir.glob("benchmark_with_ci_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        return pd.read_csv(latest_csv)

    return None

def create_interactive_causal_graph(causal_data):
    """Create interactive causal graph using Pyvis"""

    net = Network(
        height="100%",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        directed=True,
        notebook=False,
        cdn_resources='in_line'
    )

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 200
            },
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    edges_data = causal_data.get('causal_graph_validation', {}).get('edges', [])

    nodes = set()
    for edge in edges_data:
        nodes.add(edge['source'])
        nodes.add(edge['target'])

    node_colors = {
        'race_white': '#ff6b6b',
        'high_cost': '#4ecdc4',
        'age': '#95e1d3',
        'sex': '#95e1d3',
        'has_esrd': '#f38181',
        'has_diabetes': '#f38181',
        'has_chf': '#f38181',
        'has_copd': '#f38181',
        'chronic_count': '#aa96da'
    }

    for node in nodes:
        color = node_colors.get(node, '#lightblue')
        size = 25 if node in ['race_white', 'high_cost'] else 20

        if node == 'race_white':
            title = "Protected Attribute: Race (White/Non-White)"
        elif node == 'high_cost':
            title = "Outcome: High Healthcare Cost Prediction"
        elif node.startswith('has_'):
            title = f"Clinical Condition: {node.replace('has_', '').upper()}"
        elif node == 'chronic_count':
            title = "Total Chronic Conditions Count"
        else:
            title = f"Feature: {node.title()}"

        net.add_node(
            node,
            label=node.replace('_', ' ').title(),
            color=color,
            size=size,
            title=title,
            font={'size': 14}
        )

    for edge in edges_data:
        source = edge['source']
        target = edge['target']
        edge_type = edge.get('edge_type', 'discovered')
        confidence = edge.get('confidence', 0.5)

        if edge_type == 'validated':
            color = '#28a745'
            width = 3
        elif edge_type == 'expert':
            color = '#007bff'
            width = 2.5
        else:
            color = '#6c757d'
            width = 2

        title = f"{edge_type.upper()}<br>"
        title += f"Confidence: {confidence:.2f}<br>"
        if edge.get('literature_support'):
            title += f"Literature: {edge['literature_support']}"
        if edge.get('rationale'):
            title += f"<br>Rationale: {edge['rationale']}"

        net.add_edge(
            source,
            target,
            color=color,
            width=width,
            title=title,
            arrows={'to': {'enabled': True, 'scaleFactor': 1.2}}
        )

    html_content = net.generate_html()

    full_width_style = """
    <style>
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
            width: 100% !important;
            height: 100% !important;
            outline: none;
        }
        /* This targets the canvas specifically */
        canvas {
            width: 100% !important;
            height: 100% !important;
        }
    </style>
    """

    html_content = html_content.replace('</head>', full_width_style + '</head>')

    return html_content

def create_fairness_accuracy_tradeoff(benchmark_df, results_data):
    """Create interactive Plotly scatter plot for fairness-accuracy tradeoff"""

    if benchmark_df is None:
        st.warning("Loading from primary results file")
        methods_data = []

        if 'bias_detection_results' in results_data:
            bias_results = results_data['bias_detection_results']
            methods_data.append({
                'Method': bias_results.get('method', 'Unknown'),
                'FNR Disparity': bias_results.get('fnr_disparity', 0) * 100,
                'Accuracy': bias_results.get('accuracy', 0) * 100,
                'Safety': bias_results.get('deployment_verdict', 'Unknown')
            })

        df = pd.DataFrame(methods_data)
    else:
        df = benchmark_df.copy()

        if 'FNR Disparity (Mean)' in df.columns:
            df['FNR Disparity'] = df['FNR Disparity (Mean)'] * 100
        if 'Accuracy (Mean)' in df.columns:
            df['Accuracy'] = df['Accuracy (Mean)'] * 100

        df['Safety'] = df['FNR Disparity'].apply(
            lambda x: 'SAFE' if x < 5 else ('CONDITIONAL' if x < 10 else 'NOT_SAFE')
        )

    color_map = {
        'SAFE': '#28a745',
        'CONDITIONAL': '#ffc107',
        'NOT_SAFE': '#dc3545'
    }

    fig = go.Figure()

    fig.add_shape(
        type="line",
        x0=0, x1=12,
        y0=82, y1=82,
        line=dict(color="gray", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=11, y=82.5,
        text="Min Acceptable Accuracy",
        showarrow=False,
        font=dict(size=10, color="gray")
    )

    fig.add_shape(
        type="line",
        x0=5, x1=5,
        y0=80, y1=88,
        line=dict(color="#28a745", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=5.5, y=87.5,
        text="Safety Threshold<br>(5% FNR)",
        showarrow=False,
        font=dict(size=10, color="#28a745")
    )

    for idx, row in df.iterrows():
        method = row['Method']
        fnr = row['FNR Disparity']
        acc = row['Accuracy']
        safety = row['Safety']

        fnr_lower = row.get('FNR Disparity (CI Lower)', fnr) * 100 if 'FNR Disparity (CI Lower)' in df.columns else None
        fnr_upper = row.get('FNR Disparity (CI Upper)', fnr) * 100 if 'FNR Disparity (CI Upper)' in df.columns else None
        acc_lower = row.get('Accuracy (CI Lower)', acc) * 100 if 'Accuracy (CI Lower)' in df.columns else None
        acc_upper = row.get('Accuracy (CI Upper)', acc) * 100 if 'Accuracy (CI Upper)' in df.columns else None

        hover_text = f"<b>{method}</b><br>"
        hover_text += f"FNR Disparity: {fnr:.2f}%<br>"
        hover_text += f"Accuracy: {acc:.2f}%<br>"
        hover_text += f"Safety: {safety}"

        fig.add_trace(go.Scatter(
            x=[fnr],
            y=[acc],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color_map[safety],
                line=dict(color='white', width=2)
            ),
            text=[method.replace(' ', '<br>')],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_text,
            hoverinfo='text',
            name=method,
            error_x=dict(
                type='data',
                symmetric=False,
                array=[fnr_upper - fnr] if fnr_upper else None,
                arrayminus=[fnr - fnr_lower] if fnr_lower else None,
                visible=True if fnr_upper else False
            ) if fnr_upper else None,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[acc_upper - acc] if acc_upper else None,
                arrayminus=[acc - acc_lower] if acc_lower else None,
                visible=True if acc_upper else False
            ) if acc_upper else None
        ))

    fig.update_layout(
        title="Fairness-Accuracy Tradeoff (Clinical Safety Focus)",
        xaxis_title="FNR Disparity (%) - Lower is Better",
        yaxis_title="Accuracy (%) - Higher is Better",
        xaxis=dict(range=[0, max(df['FNR Disparity'].max() + 2, 12)]),
        yaxis=dict(range=[min(df['Accuracy'].min() - 2, 80), 88]),
        hovermode='closest',
        showlegend=False,
        height=800,
        template='plotly_white'
    )

    return fig

def main():
    st.markdown('<div class="main-header">Clinical Fairness Interactive Visualization</div>', unsafe_allow_html=True)
    st.markdown("**Real-time exploration of causal relationships and fairness-accuracy tradeoffs in Medicare high-cost prediction**")

    with st.spinner("Loading experimental results..."):
        results_data = load_results_data()
        benchmark_df = load_benchmark_data()

    if results_data is None:
        st.error("Failed to load results data. Please check that results files exist in the results/ directory.")
        return

    st.sidebar.header("Dataset Information")
    dataset_info = results_data.get('dataset_info', {})

    st.sidebar.metric("Total Patients", f"{dataset_info.get('n_samples', 0):,}")
    st.sidebar.metric("Features", dataset_info.get('n_features', 0))
    st.sidebar.metric("Outcome Prevalence", f"{dataset_info.get('outcome_prevalence', 0)*100:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Protected Attribute:** " + dataset_info.get('protected_attr', 'N/A').replace('_', ' ').title())
    st.sidebar.markdown("**Outcome:** " + dataset_info.get('outcome', 'N/A').replace('_', ' ').title())

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Causal Graph",
        "Fairness-Accuracy Tradeoff",
        "Detailed Metrics"
    ])

    with tab1:
        st.header("Executive Summary")

        col1, col2, col3 = st.columns(3)

        causal_summary = results_data.get('causal_graph_validation', {}).get('summary', {})

        with col1:
            st.metric(
                "Total Causal Edges",
                causal_summary.get('total_edges', 0),
                delta=f"{causal_summary.get('validated_discovered_edges', 0)} LLM-validated"
            )

        with col2:
            st.metric(
                "Expert Knowledge Edges",
                causal_summary.get('expert_edges', 0)
            )

        with col3:
            st.metric(
                "Cycles Detected",
                causal_summary.get('removed_due_to_cycles', 0),
                delta="Valid DAG" if causal_summary.get('removed_due_to_cycles', 0) == 0 else "Issues found",
                delta_color="normal" if causal_summary.get('removed_due_to_cycles', 0) == 0 else "inverse"
            )

        st.markdown("---")

        st.subheader("Recommended Intervention")

        if benchmark_df is not None and len(benchmark_df) > 0:
            best_method = benchmark_df.loc[benchmark_df['FNR Disparity (Mean)'].idxmin()]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {best_method['Method']}")
                st.markdown(f"**FNR Disparity:** {best_method['FNR Disparity (Mean)']*100:.2f}% "
                           f"[95% CI: {best_method['FNR Disparity (CI Lower)']*100:.1f}%-{best_method['FNR Disparity (CI Upper)']*100:.1f}%]")
                st.markdown(f"**Accuracy:** {best_method['Accuracy (Mean)']*100:.2f}% "
                           f"[95% CI: {best_method['Accuracy (CI Lower)']*100:.1f}%-{best_method['Accuracy (CI Upper)']*100:.1f}%]")

                fnr_disparity = best_method['FNR Disparity (Mean)'] * 100
                if fnr_disparity < 5:
                    badge_class = "safe-badge"
                    badge_text = "SAFE"
                elif fnr_disparity < 10:
                    badge_class = "conditional-badge"
                    badge_text = "CONDITIONAL"
                else:
                    badge_class = "not-safe-badge"
                    badge_text = "NOT SAFE"

                st.markdown(f'<span class="{badge_class}">{badge_text} FOR DEPLOYMENT</span>', unsafe_allow_html=True)

            with col2:
                if 'Unmitigated Baseline' in benchmark_df['Method'].values:
                    baseline = benchmark_df[benchmark_df['Method'] == 'Unmitigated Baseline'].iloc[0]

                    fnr_improvement = (baseline['FNR Disparity (Mean)'] - best_method['FNR Disparity (Mean)']) * 100
                    acc_change = (best_method['Accuracy (Mean)'] - baseline['Accuracy (Mean)']) * 100

                    st.metric("FNR Improvement", f"{fnr_improvement:+.2f}%",
                             delta="Better" if fnr_improvement > 0 else "Worse")
                    st.metric("Accuracy Change", f"{acc_change:+.2f}%",
                             delta="Higher" if acc_change > 0 else "Lower",
                             delta_color="normal" if acc_change > -1 else "inverse")

    with tab2:
        st.header("Interactive Causal Graph")
        st.markdown("**Drag nodes to rearrange | Hover for details | Zoom and pan enabled**")

        with st.spinner("Generating interactive causal graph..."):
            graph_html = create_interactive_causal_graph(results_data)
            components.html(graph_html, height=900, scrolling=False)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Edge Types:**")
            st.markdown("**Green:** LLM-validated (literature support)")
            st.markdown("**Blue:** Expert knowledge (Obermeyer pathway)")
            st.markdown("**Gray:** Discovered by PC algorithm")

        with col2:
            st.markdown("**Node Types:**")
            st.markdown("**Red:** Protected attribute (race)")
            st.markdown("**Teal:** Outcome (high cost)")
            st.markdown("**Purple:** Derived features")
            st.markdown("**Green:** Demographics")
            st.markdown("**Pink:** Clinical conditions")

    with tab3:
        st.header("Fairness-Accuracy Pareto Frontier")

        with st.spinner("Creating interactive plot..."):
            fig = create_fairness_accuracy_tradeoff(benchmark_df, results_data)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Interpretation Guide:**")
        st.markdown("- **X-axis (FNR Disparity):** Measures unfairness - how much more likely non-white patients are misclassified as low-risk")
        st.markdown("- **Y-axis (Accuracy):** Overall model performance")
        st.markdown("- **Green zone (left of 5%):** Clinically safe for deployment")
        st.markdown("- **Error bars:** 95% confidence intervals from bootstrap resampling")

    with tab4:
        st.header("Detailed Performance Metrics")

        if benchmark_df is not None:
            st.dataframe(
                benchmark_df.style.format({
                    'FNR Disparity (Mean)': '{:.4f}',
                    'FNR Disparity (CI Lower)': '{:.4f}',
                    'FNR Disparity (CI Upper)': '{:.4f}',
                    'Accuracy (Mean)': '{:.4f}',
                    'Accuracy (CI Lower)': '{:.4f}',
                    'Accuracy (CI Upper)': '{:.4f}',
                }),
                use_container_width=True
            )

        st.markdown("---")

        if 'group_metrics' in results_data:
            st.subheader("Per-Group Performance")

            group_metrics = results_data['group_metrics']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**White Patients**")
                white_metrics = group_metrics.get('white', {})
                st.metric("Accuracy", f"{white_metrics.get('accuracy', 0)*100:.2f}%")
                st.metric("True Positive Rate", f"{white_metrics.get('tpr', 0)*100:.2f}%")
                st.metric("False Negative Rate", f"{white_metrics.get('fnr', 0)*100:.2f}%")

            with col2:
                st.markdown("**Non-White Patients**")
                nonwhite_metrics = group_metrics.get('non_white', {})
                st.metric("Accuracy", f"{nonwhite_metrics.get('accuracy', 0)*100:.2f}%")
                st.metric("True Positive Rate", f"{nonwhite_metrics.get('tpr', 0)*100:.2f}%")
                st.metric("False Negative Rate", f"{nonwhite_metrics.get('fnr', 0)*100:.2f}%")

    st.markdown("---")
    st.markdown(f"**Report ID:** {results_data.get('report_id', 'N/A')}")
    st.markdown(f"**Generated:** {results_data.get('timestamp', 'N/A')}")

if __name__ == "__main__":
    main()
