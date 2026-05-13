from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="S&P 500 NLP Explorer",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

    html, body {
        background-color: #080f1e !important;
        color: #dbeafe;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"],
    .stApp {
        background-color: #080f1e !important;
    }

    .stMarkdown {
        background: transparent !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Mono', monospace;
        color: #dbeafe;
    }

    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        background-color: #0f1c2e !important;
        border-right: 1px solid #1e3a5f;
    }

    [data-testid="stSidebar"] * {
        color: #dbeafe !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0d1829 !important;
        border-color: #1e3a5f !important;
    }

    button[data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
    }

    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: #080f1e; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; }

    /* ── Shared components ─────────────────────────────────────── */

    .company-header {
        background-color: #0f1c2e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .ticker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #2563eb;
    }

    .company-name {
        font-size: 1rem;
        font-weight: 500;
        color: #dbeafe;
    }

    .header-sep { color: #1e3a5f; }

    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #ffffff;
        margin: 1.75rem 0 0.75rem;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.4rem;
    }

    .metric-card {
        background-color: #0f1c2e;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: #06b6d4;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .badge-positive { background-color: rgba(16,185,129,0.15);  color: #10b981; }
    .badge-negative { background-color: rgba(239,68,68,0.15);   color: #ef4444; }
    .badge-neutral  { background-color: rgba(245,158,11,0.15);  color: #f59e0b; }
    .badge-high     { background-color: rgba(16,185,129,0.15);  color: #10b981; }
    .badge-medium   { background-color: rgba(245,158,11,0.15);  color: #f59e0b; }
    .badge-low      { background-color: rgba(239,68,68,0.15);   color: #ef4444; }

    .article-card {
        background-color: #0f1c2e;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        flex-wrap: wrap;
    }

    .article-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #ffffff;
        min-width: 5.5rem;
    }

    .callout-card {
        background-color: rgba(37,99,235,0.08);
        border: 1px solid rgba(37,99,235,0.25);
        border-radius: 8px;
        padding: 0.9rem 1.25rem;
        margin-top: 0.75rem;
    }

    .callout-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }

    .callout-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }

    .disclaimer {
        margin-top: 2.5rem;
        padding: 0.75rem 1rem;
        border-top: 1px solid #1e3a5f;
        font-size: 0.75rem;
        color: #475569;
        font-style: italic;
    }

    .muted { color: #ffffff; font-size: 0.875rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Data

@st.cache_data(show_spinner=False)
def load_data() -> tuple:
    
    from pathlib import Path
    DATA_DIR = Path(__file__).parent / "data"
    df = pd.read_csv(DATA_DIR / "sp500_repr_b_posterior.csv", parse_dates=["date"])
    df_text = pd.read_csv(
        DATA_DIR / "sp500_cleaned.csv",
        usecols=["symbol", "date", "headline", "summary"],
        parse_dates=["date"],
    )
    # Integrity check: cumcount-based merge below is only safe if every
    # (symbol, date) group has the same row count across both files AND
    # rows within each group are in the same order. Group-size equality is
    # cheap to verify at load time; order is verified once offline via
    # sentiment-headline coherence (see analysis notebook).
    df["date_str"]      = df["date"].astype(str).str[:10]
    df_text["date_str"] = df_text["date"].astype(str).str[:10]
    g1 = df.groupby(["symbol", "date_str"]).size()
    g2 = df_text.groupby(["symbol", "date_str"]).size()
    shared = g1.index.intersection(g2.index)
    mismatched = (g1.loc[shared] != g2.loc[shared]).sum()
    assert mismatched == 0, (
        f"Merge integrity violated: {mismatched} (symbol, date) groups have "
        f"mismatched row counts between sp500_repr_b_posterior.csv and "
        f"sp500_cleaned.csv. Re-export both files from the same source DataFrame."
    )
    df = df.drop(columns=["date_str"])
    df_text = df_text.drop(columns=["date_str"])
    df["_row_num"]      = df.groupby(["symbol", "date"]).cumcount()
    df_text["_row_num"] = df_text.groupby(["symbol", "date"]).cumcount()
    df["date"]      = df["date"].astype(str)
    df_text["date"] = df_text["date"].astype(str)
    df = df.merge(
        df_text[["symbol", "date", "_row_num", "headline", "summary"]],
        on=["symbol", "date", "_row_num"],
        how="left",
    )
    df = df.drop(columns=["_row_num"])
    df["date"] = pd.to_datetime(df["date"])
    null_rate = df["headline"].isna().mean()
    assert null_rate < 0.01, f"Headline null rate {null_rate:.2%} exceeds 1% threshold."
    return (
        df,
        pd.read_csv(DATA_DIR / "rq1_results.csv"),
        pd.read_csv(DATA_DIR / "rq3_results_with_reliability.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_event_only_1d.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_sentiment_only_1d.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_event_only_7d.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_sentiment_only_7d.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_event_only_30d.csv"),
        pd.read_csv(DATA_DIR / "shap_imp_sentiment_only_30d.csv"),
        df_text,
    )


# Landing page

def main() -> None:
    load_data()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                    color:#475569;line-height:2;padding:0.1rem 0;">
            500 companies / 11 sectors / 50,300 articles<br>
            LR &nbsp;·&nbsp; SVM-RBF &nbsp;·&nbsp; XGBoost<br>
            Horizons: 1d &nbsp;·&nbsp; 7d &nbsp;·&nbsp; 30d
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.html(
        """
        <div style="background:#0f1c2e;border-left:3px solid #2563eb;border-radius:12px;padding:2rem;margin-bottom:2rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:700;color:#2563eb;margin-bottom:0.4rem;">
                NewsSignal
            </div>
            <div style="font-family:'Inter',sans-serif;font-size:1rem;color:#ffffff;margin-bottom:0.85rem;">
                S&amp;P 500 Financial News Signal Explorer
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#ffffff;letter-spacing:0.06em;margin-bottom:1.25rem;">
                500 companies &nbsp;·&nbsp; 11 GICS sectors &nbsp;·&nbsp; 50,300 articles &nbsp;·&nbsp; May - Jul 2025
            </div>
            <p style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#dbeafe;line-height:1.7;margin:0;">
                This tool visualises the outputs of a multi-representation NLP pipeline built for IJC319.
                Use the sidebar to explore sentiment signals, event cluster activations, model performance,
                sector-level reliability, and SHAP-based feature attribution.
            </p>
        </div>
        """
    )

    col_pages, col_method = st.columns(2, gap="large")

    with col_pages:
        st.html(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:600;
                        letter-spacing:0.14em;text-transform:uppercase;color:#06b6d4;
                        border-bottom:1px solid #1e3a5f;padding-bottom:0.4rem;margin-bottom:1rem;">
                Pages
            </div>
            """
        )
        pages = [
            (
                "Signal Explorer",
                "Filter any S&P 500 company by date and inspect article-level FinBERT, "
                "VADER, SentiStrength, and event cluster signals for that company-day.",
            ),
            (
                "Model Performance",
                "Compare ROC-AUC, F1, and calibration across sentiment and event models "
                "at 1-day, 7-day, and 30-day prediction horizons.",
            ),
            (
                "Sector Analysis",
                "Explore signal distributions and posterior event probabilities aggregated "
                "across all 11 GICS sectors.",
            ),
            (
                "SHAP Explainability",
                "Inspect SHAP feature importance rankings for event-only and sentiment-only "
                "models at each prediction horizon.",
            ),
        ]
        for name, desc in pages:
            st.html(
                f"""
                <div style="margin-bottom:1.1rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;
                                font-weight:600;color:#dbeafe;margin-bottom:0.25rem;">
                        {name}
                    </div>
                    <div style="font-family:'Inter',sans-serif;font-size:0.83rem;
                                color:#ffffff;line-height:1.55;">
                        {desc}
                    </div>
                </div>
                """
            )

    with col_method:
        st.html(
            """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;font-weight:600;
                        letter-spacing:0.14em;text-transform:uppercase;color:#06b6d4;
                        border-bottom:1px solid #1e3a5f;padding-bottom:0.4rem;margin-bottom:1rem;">
                Methodology
            </div>
            """
        )
        method_items = [
            ("Sentiment",    "Fine-tuned FinBERT"),
            ("Sentiment",    "VADER"),
            ("Sentiment",    "SentiStrength"),
            ("Events",       "Word2Vec K-means event clustering (12 clusters)"),
            ("Models",       "Logistic Regression / SVM-RBF / XGBoost"),
            ("Horizons",     "1d / 7d / 30d prediction windows"),
        ]
        rows_html = "".join(
            f"""
            <div style="display:flex;align-items:baseline;gap:0.75rem;
                        padding:0.45rem 0;border-bottom:1px solid #0d1829;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
                             font-weight:600;color:#ffffff;min-width:5.5rem;
                             text-transform:uppercase;letter-spacing:0.06em;">
                    {category}
                </span>
                <span style="font-family:'Inter',sans-serif;font-size:0.83rem;color:#dbeafe;">
                    {item}
                </span>
            </div>
            """
            for category, item in method_items
        )
        st.html(
            f'<div style="background:#0f1c2e;border:1px solid #1e3a5f;'
            f'border-radius:10px;padding:1rem 1.25rem;">{rows_html}</div>'
        )


main()
