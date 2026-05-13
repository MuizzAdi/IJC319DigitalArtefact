from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app import load_data
from utils.helpers import plotly_defaults, section_header, reliability_pill

_, _, rq3, *_ = load_data()

_COLORSCALE = [[0.0, "#0f1c2e"], [0.5, "#2563eb"], [1.0, "#06b6d4"]]

# Sort once; every section uses this order
rq3 = rq3.sort_values("event_auc", ascending=False).reset_index(drop=True)

sectors = rq3["sector"].tolist()

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="background:#0f1c2e;border-left:3px solid #2563eb;border-radius:12px;
                padding:1.2rem;margin-bottom:1.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;
                    color:#2563eb;margin-bottom:0.35rem;">
            Sector Analysis
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#ffffff;">
            Sector-stratified ROC-AUC for event-based and sentiment-based representations
            at the 30-day horizon. Reliability tiers are derived from class prevalence shift
            between training and test partitions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Section 1: Side-by-side heatmaps ──────────────────────────────────────────

st.markdown(section_header("AUC by Sector"), unsafe_allow_html=True)

fig_heat = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Event-Based AUC", "Sentiment-Based AUC"],
    horizontal_spacing=0.08,
    column_widths=[0.5, 0.5],
)

for col_idx, auc_col in enumerate(["event_auc", "sentiment_auc"], start=1):
    values = rq3[auc_col].tolist()
    fig_heat.add_trace(
        go.Heatmap(
            z=[[v] for v in values],
            y=sectors,
            x=["AUC"],
            coloraxis="coloraxis",
            showscale=(col_idx == 2),
            text=[[f"{v:.3f}"] for v in values],
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=11, color="#dbeafe"),
            hovertemplate="%{y}: %{text}<extra></extra>",
        ),
        row=1,
        col=col_idx,
    )

fig_heat.update_layout(
    **plotly_defaults(),
    height=520,
    margin=dict(l=200, r=40, t=50, b=20),
    coloraxis=dict(colorscale=_COLORSCALE, cmin=0.20, cmax=0.75),
)
fig_heat.update_xaxes(showticklabels=False, showgrid=False)
fig_heat.update_yaxes(showgrid=False)
fig_heat.update_yaxes(showticklabels=False, row=1, col=2)

st.plotly_chart(fig_heat, use_container_width=True)

# ── Section 2: Advantage bar chart ────────────────────────────────────────────

st.markdown(section_header("Event Advantage over Sentiment (Delta AUC)"), unsafe_allow_html=True)

adv_sorted = rq3.sort_values("event_advantage", ascending=True)
bar_colors = [
    "#06b6d4" if v >= 0 else "#ef4444"
    for v in adv_sorted["event_advantage"]
]

fig_adv = go.Figure(go.Bar(
    x=adv_sorted["event_advantage"].tolist(),
    y=adv_sorted["sector"].tolist(),
    orientation="h",
    marker_color=bar_colors,
    text=adv_sorted["reliability"].tolist(),
    textposition="outside",
    textfont=dict(family="IBM Plex Mono", size=10, color="#ffffff"),
))

fig_adv.add_vline(
    x=0,
    line=dict(color="#475569", width=1, dash="dash"),
)

fig_adv.update_layout(
    **plotly_defaults(),
    height=480,
    margin=dict(l=180, r=150, t=50, b=60),
    xaxis=dict(
        title="Event AUC - Sentiment AUC",
        showgrid=True,
        gridcolor="#1e3a5f",
        zeroline=False,
    ),
    yaxis=dict(showgrid=False),
)

st.plotly_chart(fig_adv, use_container_width=True)

# ── Table + reliability legend ─────────────────────────────────────────────────

col_tbl, col_legend = st.columns([3, 2], gap="large")

with col_tbl:
    tbl = rq3[["sector", "event_auc", "sentiment_auc", "event_advantage", "reliability"]].copy()
    tbl["delta_auc"] = tbl["event_advantage"]
    tbl_display = tbl[["sector", "event_auc", "sentiment_auc", "delta_auc", "reliability"]].copy()

    col_labels = ["Sector", "Event AUC", "Sentiment AUC", "Delta AUC", "Reliability"]
    n_rows = len(tbl_display)
    row_fills = [
        ["#0f1c2e" if i % 2 == 0 else "#0d1829" for i in range(n_rows)]
        for _ in col_labels
    ]

    cell_values = [
        tbl_display["sector"].tolist(),
        [f"{v:.3f}" for v in tbl_display["event_auc"]],
        [f"{v:.3f}" for v in tbl_display["sentiment_auc"]],
        [f"{'+' if v >= 0 else ''}{v:.3f}" for v in tbl_display["delta_auc"]],
        tbl_display["reliability"].tolist(),
    ]

    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in col_labels],
            fill_color="#1e3a5f",
            font=dict(color="#dbeafe", family="IBM Plex Mono", size=11),
            align="left",
            height=32,
        ),
        cells=dict(
            values=cell_values,
            fill_color=row_fills,
            font=dict(color="#dbeafe", family="IBM Plex Mono", size=11),
            align="left",
            height=28,
        ),
    ))

    fig_tbl.update_layout(
        **plotly_defaults(),
        height=40 + 32 + n_rows * 28,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig_tbl, use_container_width=True)

with col_legend:
    tiers = [
        (
            "#10b981",
            "High",
            "Prevalence shift below 5 percent. Class distribution is stable between "
            "train and test. AUC reflects genuine signal quality.",
        ),
        (
            "#f59e0b",
            "Medium",
            "Prevalence shift between 5 and 10 percent. Moderate label imbalance "
            "shift. Interpret AUC with some caution.",
        ),
        (
            "#ef4444",
            "Low",
            "Prevalence shift above 10 percent. Substantial distribution shift. "
            "AUC may overstate or understate true predictive power.",
        ),
    ]
    for border_col, label, explanation in tiers:
        st.markdown(
            f"""
            <div style="border-left:3px solid {border_col};background:#0d1829;
                        border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:0.75rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                            font-weight:600;color:{border_col};margin-bottom:0.35rem;
                            text-transform:uppercase;letter-spacing:0.06em;">
                    {label} Reliability
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:0.82rem;
                            color:#ffffff;line-height:1.6;">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Section 3: ANOVA callout ───────────────────────────────────────────────────

st.markdown(section_header("Statistical Finding"), unsafe_allow_html=True)

st.markdown(
    """
    <div style="border-left:3px solid #06b6d4;background:#0d1829;border-radius:8px;
                padding:1rem 1.3rem;font-family:'IBM Plex Mono',monospace;
                font-size:0.82rem;line-height:1.75;color:#dbeafe;">
        One-way ANOVA on event-based AUC across sectors: F = 4.358, p &lt; 0.001,
        eta-squared = 0.017. Sector membership explains a statistically significant
        but practically small share of variance in event-based performance.
        No equivalent sector effect was found for sentiment-based features
        (F = 1.535, p = 0.121).
        Financials achieves the strongest event AUC (0.714, High reliability).
        Real Estate is the weakest sector across both representations
        (event AUC = 0.452, Medium reliability).
    </div>
    """,
    unsafe_allow_html=True,
)
