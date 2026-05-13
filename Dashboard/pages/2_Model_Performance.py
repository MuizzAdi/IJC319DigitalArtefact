from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import load_data
from utils.helpers import metric_card, plotly_defaults, section_header

_, rq1, *_ = load_data()

# Bootstrap 95% CI (hardcoded from report) 
# (centre, half-width) keyed by (horizon, feature_set)
_CI: dict[tuple[str, str], tuple[float, float]] = {
    ("1d",  "sentiment_only"): (0.532, 0.037),
    ("7d",  "sentiment_only"): (0.488, 0.032),
    ("30d", "sentiment_only"): (0.510, 0.030),
    ("1d",  "event_only"):     (0.515, 0.032),
    ("7d",  "event_only"):     (0.524, 0.025),
    ("30d", "event_only"):     (0.586, 0.024),
}

_HORIZONS    = ["1d", "7d", "30d"]
_FEAT_COLOURS = {"sentiment_only": "#2563eb", "event_only": "#06b6d4"}
_FEAT_NAMES   = {"sentiment_only": "Sentiment Only", "event_only": "Event Only"}

# Hero 

st.markdown(
    """
    <div style="background:#0f1c2e;border-left:3px solid #2563eb;border-radius:12px;
                padding:1.2rem;margin-bottom:1.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;
                    color:#2563eb;margin-bottom:0.35rem;">
            Model Performance
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#ffffff;">
            ROC-AUC across feature representations, classifiers, and prediction horizons.
            All values computed from the trained pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section 1: Grouped bar chart

st.markdown(section_header("Best AUC per Horizon and Feature Set"), unsafe_allow_html=True)

best = (
    rq1.groupby(["horizon", "feature_set"], sort=False)["roc_auc"]
    .max()
    .reset_index()
)

fig_bar = go.Figure()

for feat in ("sentiment_only", "event_only"):
    subset = best[best["feature_set"] == feat].set_index("horizon")
    y_vals = [subset.loc[h, "roc_auc"] if h in subset.index else None for h in _HORIZONS]
    errors = [_CI[(h, feat)][1] for h in _HORIZONS]

    fig_bar.add_trace(go.Bar(
        name=_FEAT_NAMES[feat],
        x=_HORIZONS,
        y=y_vals,
        marker_color=_FEAT_COLOURS[feat],
        error_y=dict(
            type="data",
            array=errors,
            visible=True,
            color="#ffffff",
            thickness=1.0,
            width=6,
        ),
        customdata=[_CI[(h, feat)][1] for h in _HORIZONS],
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Horizon: %{x}<br>"
            "AUC = %{y:.3f} ± %{customdata:.3f}"
            "<extra></extra>"
        ),
    ))

fig_bar.add_hline(
    y=0.5,
    line=dict(color="#ffffff", width=1, dash="dash"),
    annotation_text="Random baseline",
    annotation_font=dict(color="#ffffff", size=10, family="IBM Plex Mono"),
    annotation_position="right",
    annotation_xshift=10,
)

fig_bar.update_layout(
    **plotly_defaults(),
    barmode="group",
    height=450,
    yaxis=dict(
        range=[0.40, 0.78],
        title="ROC-AUC",
        showgrid=True,
        gridcolor="#1e3a5f",
        zeroline=False,
    ),
    xaxis=dict(showgrid=False, tickfont=dict(size=12), range=[-0.5, 2.8]),
    legend=dict(
        orientation="v",
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(15,28,46,0.8)",
        bordercolor="#1e3a5f",
        borderwidth=1,
    ),
    margin=dict(l=60, r=120, t=50, b=70),
)

st.plotly_chart(fig_bar, use_container_width=True)

# Section 2: Full results table 

st.markdown(section_header("Full Results"), unsafe_allow_html=True)

display_cols = ["horizon", "feature_set", "model", "roc_auc", "f1", "accuracy", "n_test"]
tbl = rq1[display_cols].copy()
tbl["roc_auc"]  = tbl["roc_auc"].map("{:.3f}".format)
tbl["f1"]       = tbl["f1"].map("{:.3f}".format)
tbl["accuracy"] = tbl["accuracy"].map("{:.3f}".format)
tbl["n_test"]   = tbl["n_test"].astype(int)

col_labels = ["Horizon", "Feature Set", "Model", "ROC-AUC", "F1", "Accuracy", "N Test"]
n_rows = len(tbl)
row_fills = [
    ["#0f1c2e" if i % 2 == 0 else "#0d1829" for i in range(n_rows)]
    for _ in col_labels
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
        values=[tbl[c].tolist() for c in display_cols],
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

#  Section 3: Key finding callout 

st.markdown(section_header("Key Finding"), unsafe_allow_html=True)

st.markdown(
    """
    <div style="border-left:3px solid #06b6d4;background:#0d1829;border-radius:8px;
                padding:1rem 1.3rem;font-family:'IBM Plex Mono',monospace;
                font-size:0.82rem;line-height:1.75;color:#dbeafe;">
        At the 30-day horizon, event-based features (LR, AUC = 0.586) outperform
        sentiment-based features (SVM, AUC = 0.510), delta AUC = +0.077
        [95% CI: +0.039, +0.114].
        At 1-day and 7-day horizons, confidence intervals span zero, indicating
        that neither representation reliably dominates.
        This is consistent with the semi-strong Efficient Market Hypothesis: short-term
        price movements absorb news information too rapidly for aggregated features to
        carry reliable directional signal.
    </div>
    """,
    unsafe_allow_html=True,
)
