from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import load_data
from utils.helpers import COLOURS, plotly_defaults, section_header

_, _, _, shap_e_1d, shap_s_1d, shap_e_7d, shap_s_7d, shap_e_30d, shap_s_30d, _ = load_data()

shap_event = {"1d": shap_e_1d, "7d": shap_e_7d, "30d": shap_e_30d}
shap_sent  = {"1d": shap_s_1d, "7d": shap_s_7d, "30d": shap_s_30d}


def _clean_feature(name: str) -> str:
    name = name.removeprefix("event_")
    name = name.removeprefix("finetuned_")
    return name.replace("_", " ").title()


#  Hero 

st.markdown(
    """
    <div style="background:#0f1c2e;border-left:3px solid #2563eb;border-radius:12px;
                padding:1.2rem;margin-bottom:1.8rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;
                    color:#2563eb;margin-bottom:0.35rem;">
            SHAP Feature Attribution
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#ffffff;">
            Mean absolute SHAP values showing which features drive predictions at each
            horizon. Computed using TreeExplainer, LinearExplainer, and KernelExplainer
            depending on the classifier.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Horizon selector ───────────────────────────────────────────────────────────

horizon = st.radio(
    "Prediction horizon",
    ["1d", "7d", "30d"],
    horizontal=True,
    index=2,
)

# ── Section 1: Side-by-side importance charts ──────────────────────────────────

st.markdown(section_header("Feature Importance"), unsafe_allow_html=True)

col_ev, col_sent = st.columns(2)

for col, df, colour, title in (
    (col_ev,   shap_event[horizon], "#06b6d4", "Event Features"),
    (col_sent, shap_sent[horizon],  "#2563eb", "Sentiment Features"),
):
    top15 = df.nlargest(15, "importance").sort_values("importance", ascending=True)
    labels = [_clean_feature(f) for f in top15["feature"]]
    values = top15["importance"].tolist()
    max_val = max(values) if values else 1.0

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colour,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#dbeafe"),
    ))
    fig.update_layout(
        **plotly_defaults(),
        title=dict(text=title, font=dict(family="IBM Plex Mono", size=13, color="#dbeafe")),
        height=580,
        margin=dict(l=230, r=80, t=50, b=60),
        xaxis=dict(
            range=[0, max_val * 1.15],
            showgrid=True,
            gridcolor="#1e3a5f",
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False),
    )
    col.plotly_chart(fig, use_container_width=True)

# ── Section 2: Scale gap callout ───────────────────────────────────────────────

st.markdown(section_header("Scale Gap"), unsafe_allow_html=True)

max_event = shap_event[horizon]["importance"].max()
max_sent  = shap_sent[horizon]["importance"].max()
ratio     = max_event / max_sent

st.markdown(
    f"""
    <div style="border-left:3px solid #f59e0b;background:#0d1829;border-radius:8px;
                padding:1rem 1.3rem;font-family:'IBM Plex Mono',monospace;
                font-size:0.82rem;line-height:1.75;color:#dbeafe;">
        At the {horizon} horizon, the top event feature carries a mean absolute SHAP
        of {max_event:.4f} versus the top sentiment feature at {max_sent:.4f}, a
        {ratio:.0f}x ratio. This gap partly reflects that event cosine similarities
        span a larger numeric range than FinBERT softmax posteriors, so absolute SHAP
        values are not strictly comparable across separately-trained models. Within
        each model the rank ordering is interpretable, and the ROC-AUC comparison in
        Model Performance provides the rigorous statistical test.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Section 3: Lag pattern note ────────────────────────────────────────────────

st.markdown(section_header("Lag Pattern"), unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="background:{COLOURS['surface']};border:1px solid {COLOURS['border']};
                border-radius:10px;padding:1rem;
                font-family:'Inter',sans-serif;font-size:0.85rem;
                line-height:1.7;color:{COLOURS['text']};">
        The dominance of lagged event features (lag1, lag2, lag3) over contemporaneous
        ones reflects the temporal structure of market reactions. News from three days
        prior continues to carry predictive signal at the 30-day horizon, suggesting
        gradual information absorption rather than instantaneous pricing -- consistent
        with the gradual-diffusion hypothesis in behavioural finance literature.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Section 4: Methodology card ────────────────────────────────────────────────

st.markdown(section_header("Methodology"), unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="background:{COLOURS['surface']};border:1px solid {COLOURS['border']};
                border-radius:10px;padding:1rem;
                font-family:'Inter',sans-serif;font-size:0.85rem;
                line-height:1.7;color:{COLOURS['text']};">
        SHAP computation: TreeExplainer for XGBoost, LinearExplainer for Logistic
        Regression, KernelExplainer for SVM-RBF with 100-point background sample.
        Test subsample: 200 observations per model. Values shown are mean absolute
        SHAP across the subsample which represent average contribution magnitude,
        not directionality.
    </div>
    """,
    unsafe_allow_html=True,
)
