"""
dashboard/app.py — Streamlit dashboard entry point.

Run with: streamlit run dashboard/app.py

Architecture:
  Streamlit (port 8501) ──HTTP──▶ FastAPI (port 8000) ──▶ SQLite
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Reads from env var for deployment, falls back to localhost for dev
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="LLM-as-a-Judge Dashboard",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_results(limit: int = 200) -> dict | None:
    """Fetch all evaluations from the API."""
    try:
        resp = httpx.get(f"{API_BASE}/results", params={"limit": limit}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=30)
def fetch_agreement() -> dict | None:
    """Fetch Cohen's Kappa agreement metrics."""
    try:
        resp = httpx.get(f"{API_BASE}/metrics/agreement", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=30)
def fetch_health() -> bool:
    """Check if the API is reachable."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_data(ttl=30)
def fetch_bias() -> dict | None:
    """Fetch bias analysis metrics."""
    try:
        resp = httpx.get(f"{API_BASE}/metrics/bias", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


# ---------------------------------------------------------------------------
# Data transformation helpers
# ---------------------------------------------------------------------------

def results_to_dataframe(data: dict) -> pd.DataFrame:
    """Flatten evaluations + judge results into a single DataFrame."""
    rows = []
    for ev in data.get("evaluations", []):
        for r in ev.get("results", []):
            rows.append({
                "evaluation_id": ev["evaluation_id"][:8],
                "problem": ev["problem"][:60],
                "created_at": ev.get("created_at", ""),
                "model": r["model"].split("/")[-1],
                "model_full": r["model"],
                "correctness": r["correctness"],
                "code_quality": r["code_quality"],
                "efficiency": r["efficiency"],
                "latency_ms": r.get("latency_ms", 0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "created_at" in df.columns and df["created_at"].notna().any():
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def agreement_to_matrix(data: dict) -> tuple[pd.DataFrame, list[str]]:
    """Convert pairwise Kappa results into a symmetric matrix for heatmap."""
    models = data.get("models", [])
    short_names = [m.split("/")[-1] for m in models]
    n = len(models)

    matrix = pd.DataFrame(1.0, index=short_names, columns=short_names)
    for pair in data.get("pairs", []):
        if pair.get("mean_kappa") is None:
            continue
        a = pair["model_a"].split("/")[-1]
        b = pair["model_b"].split("/")[-1]
        kappa = pair["mean_kappa"]
        matrix.loc[a, b] = kappa
        matrix.loc[b, a] = kappa

    return matrix, short_names


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

st.title("⚖️ LLM-as-a-Judge Dashboard")

# Connection status
api_ok = fetch_health()
if not api_ok:
    st.error(
        f"**Cannot connect to API** at `{API_BASE}`. "
        "Start the server with: `uvicorn app.main:app --reload`"
    )
    st.stop()

st.success("Connected to API")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Fetch data
results_data = fetch_results()
agreement_data = fetch_agreement()
bias_data = fetch_bias()

if not results_data or results_data.get("total", 0) == 0:
    st.warning("No evaluations found. Submit some via `POST /evaluate` first.")
    st.stop()

df = results_to_dataframe(results_data)

# Top-level metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Evaluations", results_data["total"])
with col2:
    st.metric("Judge Models", df["model"].nunique())
with col3:
    st.metric("Avg Correctness", f"{df['correctness'].mean():.2f}")
with col4:
    if agreement_data and agreement_data.get("overall_mean_kappa") is not None:
        kappa = agreement_data["overall_mean_kappa"]
        st.metric("Overall Kappa", f"{kappa:.3f}")
    else:
        st.metric("Overall Kappa", "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Score Distributions",
    "🤝 Model Agreement",
    "📈 Trends Over Time",
    "📋 Evaluation Browser",
    "⚠️ Bias Analysis",
])

# ---- Tab 1: Score Distributions ----
with tab1:
    st.subheader("Score Distributions by Model")

    metric_choice = st.selectbox(
        "Select metric",
        ["correctness", "code_quality", "efficiency"],
        key="dist_metric",
    )

    fig = px.histogram(
        df,
        x=metric_choice,
        color="model",
        barmode="group",
        nbins=5,
        range_x=[0.5, 5.5],
        title=f"Distribution of {metric_choice.replace('_', ' ').title()} Scores",
        labels={metric_choice: f"{metric_choice.replace('_', ' ').title()} (1-5)", "count": "Count"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(bargap=0.1, xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    # Per-model summary stats
    st.subheader("Per-Model Score Summary")
    summary = df.groupby("model")[["correctness", "code_quality", "efficiency"]].agg(
        ["mean", "std", "count"]
    ).round(2)
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    st.dataframe(summary, use_container_width=True)

    # Radar chart comparing models across all 3 metrics
    st.subheader("Model Comparison Radar")
    means = df.groupby("model")[["correctness", "code_quality", "efficiency"]].mean()
    fig_radar = go.Figure()
    for model_name in means.index:
        values = means.loc[model_name].tolist()
        values.append(values[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=["Correctness", "Code Quality", "Efficiency", "Correctness"],
            fill="toself",
            name=model_name,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="Mean Scores by Model",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ---- Tab 2: Model Agreement ----
with tab2:
    st.subheader("Inter-Judge Agreement (Cohen's Kappa)")

    if agreement_data and agreement_data.get("pairs"):
        matrix, labels = agreement_to_matrix(agreement_data)

        fig_heatmap = px.imshow(
            matrix.values,
            x=labels,
            y=labels,
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1,
            text_auto=".3f",
            title="Pairwise Cohen's Kappa (Mean Across Metrics)",
            labels=dict(color="Kappa"),
        )
        fig_heatmap.update_layout(width=600, height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Per-metric breakdown
        st.subheader("Per-Metric Kappa Breakdown")
        kappa_rows = []
        for pair in agreement_data["pairs"]:
            if not pair.get("metrics"):
                continue
            for metric_name, metric_data in pair["metrics"].items():
                kappa_rows.append({
                    "Model A": pair["model_a"].split("/")[-1],
                    "Model B": pair["model_b"].split("/")[-1],
                    "Metric": metric_name.replace("_", " ").title(),
                    "Kappa": metric_data["kappa"],
                    "Interpretation": metric_data["interpretation"],
                })
        if kappa_rows:
            kappa_df = pd.DataFrame(kappa_rows)
            st.dataframe(kappa_df, use_container_width=True, hide_index=True)

            fig_kappa_bar = px.bar(
                kappa_df,
                x="Metric",
                y="Kappa",
                color="Model A",
                barmode="group",
                title="Kappa by Metric and Model Pair",
                pattern_shape="Model B",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_kappa_bar.update_layout(yaxis=dict(range=[-1, 1]))
            st.plotly_chart(fig_kappa_bar, use_container_width=True)
    else:
        st.info("Need at least 2 evaluations with shared models to compute agreement.")

# ---- Tab 3: Trends Over Time ----
with tab3:
    st.subheader("Score Trends Over Time")

    if "created_at" in df.columns and df["created_at"].notna().any():
        time_metric = st.selectbox(
            "Select metric",
            ["correctness", "code_quality", "efficiency"],
            key="trend_metric",
        )

        fig_trend = px.line(
            df.sort_values("created_at"),
            x="created_at",
            y=time_metric,
            color="model",
            markers=True,
            title=f"{time_metric.replace('_', ' ').title()} Over Time",
            labels={"created_at": "Date", time_metric: f"{time_metric.replace('_', ' ').title()} (1-5)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_trend.update_layout(yaxis=dict(range=[0.5, 5.5], dtick=1))
        st.plotly_chart(fig_trend, use_container_width=True)

        # Aggregate trend (mean across models per evaluation)
        st.subheader("Aggregate Trend (Mean Across Models)")
        agg_trend = df.groupby("created_at")[["correctness", "code_quality", "efficiency"]].mean().reset_index()
        fig_agg = px.line(
            agg_trend.sort_values("created_at"),
            x="created_at",
            y=["correctness", "code_quality", "efficiency"],
            markers=True,
            title="Mean Scores Over Time (All Models)",
            labels={"created_at": "Date", "value": "Score (1-5)", "variable": "Metric"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_agg.update_layout(yaxis=dict(range=[0.5, 5.5], dtick=1))
        st.plotly_chart(fig_agg, use_container_width=True)
    else:
        st.info("No timestamp data available for trend analysis.")

    # Latency trend
    st.subheader("Judge Latency Over Time")
    if "latency_ms" in df.columns and df["latency_ms"].sum() > 0:
        fig_latency = px.scatter(
            df.sort_values("created_at") if "created_at" in df.columns else df,
            x="created_at" if "created_at" in df.columns else df.index,
            y="latency_ms",
            color="model",
            title="Judge Response Latency",
            labels={"latency_ms": "Latency (ms)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_latency, use_container_width=True)

# ---- Tab 4: Evaluation Browser ----
with tab4:
    st.subheader("All Evaluations")

    for ev in results_data.get("evaluations", []):
        eval_id = ev["evaluation_id"][:8]
        problem = ev["problem"][:80]
        created = ev.get("created_at", "unknown")
        agg = ev.get("aggregate", {})
        n_judges = agg.get("num_judges", 0)

        with st.expander(f"🔹 {eval_id}... — {problem}  ({created})"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Correctness", f"{agg.get('mean_correctness', 'N/A')}")
            with col_b:
                st.metric("Code Quality", f"{agg.get('mean_code_quality', 'N/A')}")
            with col_c:
                st.metric("Efficiency", f"{agg.get('mean_efficiency', 'N/A')}")

            st.markdown("**Problem:**")
            st.code(ev["problem"], language="text")
            st.markdown("**Code:**")
            st.code(ev["code"], language="python")

            st.markdown(f"**Judge Results ({n_judges} models):**")
            for r in ev.get("results", []):
                model_short = r["model"].split("/")[-1]
                st.markdown(
                    f"- **{model_short}**: "
                    f"correctness={r['correctness']}, "
                    f"quality={r['code_quality']}, "
                    f"efficiency={r['efficiency']} "
                    f"({r.get('latency_ms', 0):.0f}ms)"
                )
                st.markdown(f"  > {r['explanation']}")

# ---- Tab 5: Bias Analysis ----
with tab5:
    st.subheader("Judge Model Bias Detection")
    st.markdown(
        "Compares each model's mean scores to the grand mean across all models. "
        "A model is flagged as **biased** if it deviates by ≥0.5 points on any metric."
    )

    if bias_data and bias_data.get("models"):
        # Grand means
        st.markdown("#### Grand Mean Scores (All Models)")
        gm = bias_data["grand_means"]
        gcol1, gcol2, gcol3, gcol4 = st.columns(4)
        with gcol1:
            st.metric("Correctness", f"{gm.get('correctness', 0):.3f}")
        with gcol2:
            st.metric("Code Quality", f"{gm.get('code_quality', 0):.3f}")
        with gcol3:
            st.metric("Efficiency", f"{gm.get('efficiency', 0):.3f}")
        with gcol4:
            st.metric("Overall", f"{bias_data.get('grand_overall_mean', 0):.3f}")

        st.markdown("---")

        # Per-model deviation bar chart
        st.markdown("#### Deviation from Grand Mean")
        dev_rows = []
        for model_name, info in bias_data["models"].items():
            short = model_name.split("/")[-1]
            for metric in ["correctness", "code_quality", "efficiency"]:
                dev_rows.append({
                    "Model": short,
                    "Metric": metric.replace("_", " ").title(),
                    "Deviation": info["deviation_from_grand_mean"].get(metric, 0),
                })
        if dev_rows:
            dev_df = pd.DataFrame(dev_rows)
            fig_dev = px.bar(
                dev_df,
                x="Model",
                y="Deviation",
                color="Metric",
                barmode="group",
                title="Score Deviation from Grand Mean (per model, per metric)",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_dev.add_hline(y=0.5, line_dash="dash", line_color="red",
                              annotation_text="Bias threshold (+0.5)")
            fig_dev.add_hline(y=-0.5, line_dash="dash", line_color="red",
                              annotation_text="Bias threshold (-0.5)")
            fig_dev.update_layout(yaxis_title="Deviation (points)")
            st.plotly_chart(fig_dev, use_container_width=True)

        # Per-model cards
        st.markdown("#### Per-Model Bias Summary")
        for model_name, info in bias_data["models"].items():
            short = model_name.split("/")[-1]
            badge = "🟢 Neutral" if not info["is_biased"] else (
                "🔴 Lenient" if info["bias_direction"] == "lenient" else
                "🔴 Severe" if info["bias_direction"] == "severe" else "🟡 Biased"
            )
            with st.expander(f"{short} — {badge} (n={info['n_evaluations']})"):
                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    ms = info["mean_scores"]
                    st.markdown("**Mean Scores**")
                    for m, v in ms.items():
                        dev = info["deviation_from_grand_mean"][m]
                        arrow = "↑" if dev > 0 else "↓" if dev < 0 else "→"
                        st.markdown(f"- {m.replace('_', ' ').title()}: **{v}** ({arrow}{dev:+.3f})")
                with bcol2:
                    st.markdown("**Overall**")
                    st.markdown(f"- Mean: **{info['overall_mean']}**")
                    st.markdown(f"- Deviation: **{info['overall_deviation']:+.3f}**")
                    st.markdown(f"- Direction: **{info['bias_direction']}**")
                with bcol3:
                    if info["bias_flags"]:
                        st.markdown("**⚠️ Flagged Metrics**")
                        for f in info["bias_flags"]:
                            st.markdown(f"- {f.replace('_', ' ').title()}")
                    else:
                        st.markdown("**✅ No flags**")

                # Score distribution mini-chart
                if info.get("score_distribution"):
                    dist_rows = []
                    for metric, dist in info["score_distribution"].items():
                        for score, count in dist.items():
                            dist_rows.append({
                                "Metric": metric.replace("_", " ").title(),
                                "Score": int(score),
                                "Count": count,
                            })
                    if dist_rows:
                        dist_df = pd.DataFrame(dist_rows)
                        fig_dist = px.bar(
                            dist_df,
                            x="Score",
                            y="Count",
                            color="Metric",
                            barmode="group",
                            title=f"Score Distribution — {short}",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig_dist.update_layout(xaxis=dict(dtick=1))
                        st.plotly_chart(fig_dist, use_container_width=True)

        # Pairwise bias
        if bias_data.get("pairwise_bias"):
            st.markdown("---")
            st.markdown("#### Pairwise Directional Bias")
            st.markdown(
                "Shows which model scores higher when both judge the same code. "
                "Based on mean score difference on shared evaluations."
            )
            pair_rows = []
            for pair in bias_data["pairwise_bias"]:
                pair_rows.append({
                    "Model A": pair["model_a"].split("/")[-1],
                    "Model B": pair["model_b"].split("/")[-1],
                    "Shared Evals": pair["n_shared"],
                    "Overall Diff (A−B)": pair["overall_mean_diff"],
                    "Direction": pair["direction"],
                })
            if pair_rows:
                pair_df = pd.DataFrame(pair_rows)
                st.dataframe(pair_df, use_container_width=True, hide_index=True)
    else:
        st.info("Need at least 2 judge results to compute bias analysis.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "LLM-as-a-Judge Dashboard • "
    f"Data: {results_data.get('total', 0)} evaluations • "
    f"API: {API_BASE}"
)
st.write("Detect systematic scoring biases across models.")
