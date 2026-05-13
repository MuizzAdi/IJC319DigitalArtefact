from __future__ import annotations

import html
import plotly.graph_objects as go
import streamlit as st

from app import load_data
from utils.helpers import (
    CLUSTER_META,
    COLOURS,
    EVENT_COLS,
    finbert_label,
    plotly_defaults,
    section_header,
)

df, _, rq3, *_ = load_data()

_ARTICLE_CAP = 20

# ── Top controls ───────────────────────────────────────────────────────────────

col_sym, col_date = st.columns(2)

symbols_df = (
    df[["symbol", "company"]]
    .drop_duplicates()
    .sort_values("symbol")
)
sym_options = [f"{r.symbol} -- {r.company}" for _, r in symbols_df.iterrows()]
selected_str   = col_sym.selectbox("Company", sym_options)
selected_symbol = selected_str.split(" -- ")[0]

dates_for_sym = sorted(
    df[df["symbol"] == selected_symbol]["date"].dt.strftime("%Y-%m-%d").unique(),
    reverse=True,
)
selected_date = col_date.selectbox("Date", dates_for_sym)

day_df = df[
    (df["symbol"] == selected_symbol)
    & (df["date"].dt.strftime("%Y-%m-%d") == selected_date)
]

if day_df.empty:
    st.warning("No articles found for this selection.")
    st.stop()

first = day_df.iloc[0]
n     = len(day_df)

# ── Company header bar ─────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="company-header">
        <span class="ticker">{first['symbol']}</span>
        <span class="company-name">{first['company']}</span>
        <span class="header-sep">·</span>
        <span class="muted">{first['sector']}</span>
        <span class="header-sep">·</span>
        <span class="muted">{n} article{'s' if n != 1 else ''}</span>
        <span class="header-sep">·</span>
        <span class="muted">{selected_date}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sentiment Signals ──────────────────────────────────────────────────────────

st.markdown(section_header("Sentiment Signals"), unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

avg_pos = day_df["finetuned_finbert_positive"].mean()
avg_neg = day_df["finetuned_finbert_negative"].mean()
avg_neu = day_df["finetuned_finbert_neutral"].mean()
dom_fb  = finbert_label(avg_pos, avg_neg, avg_neu)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">FinBERT (fine-tuned)</div>
            <span class="badge badge-{dom_fb}">{dom_fb.title()}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_fb = go.Figure(go.Bar(
        x=[avg_pos, avg_neg, avg_neu],
        y=["Positive", "Negative", "Neutral"],
        orientation="h",
        marker_color=[COLOURS["positive"], COLOURS["negative"], COLOURS["neutral"]],
        text=[f"{v:.3f}" for v in [avg_pos, avg_neg, avg_neu]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color=COLOURS["text"]),
    ))
    fig_fb.update_layout(
        **plotly_defaults(),
        height=230,
        margin=dict(l=120, r=80, t=50, b=60),
        xaxis=dict(range=[0, 1.15], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig_fb, use_container_width=True)

avg_vader   = day_df["vader_combined_compound"].mean()
vader_sign  = "+" if avg_vader >= 0 else ""
vader_color = (
    COLOURS["positive"] if avg_vader > 0.05
    else COLOURS["negative"] if avg_vader < -0.05
    else COLOURS["neutral"]
)

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">VADER Compound</div>
            <div class="metric-value" style="color:{vader_color}">{vader_sign}{avg_vader:.4f}</div>
            <div class="muted">daily avg &nbsp;·&nbsp; range -1 to +1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

avg_ss_pos = day_df["ss_combined_pos"].mean()
avg_ss_neg = day_df["ss_combined_neg"].mean()

_ss_zero_note = (
    '<div style="font-family:\'Inter\',sans-serif;font-size:0.75rem;'
    'color:#475569;margin-top:0.4rem;line-height:1.5;">'
    'Zero values indicate no SentiStrength signal detected for this company-day.'
    '</div>'
    if avg_ss_pos == 0.0 and avg_ss_neg == 0.0
    else ""
)

with c3:
    st.html(
        f"""
        <div class="metric-card">
            <div class="metric-label">SentiStrength</div>
            <div class="metric-row">
                <span class="muted">Positive</span>
                <span style="color:{COLOURS['positive']}">{avg_ss_pos:.4f}</span>
            </div>
            <div class="metric-row">
                <span class="muted">Negative</span>
                <span style="color:{COLOURS['negative']}">{avg_ss_neg:.4f}</span>
            </div>
            {_ss_zero_note}
        </div>
        """
    )

# ── Signal Interpretation ──────────────────────────────────────────────────────

# Pre-compute event data here; reused by Event Clusters below
event_avgs    = {col: day_df[col].mean() for col in EVENT_COLS}
sorted_events = sorted(event_avgs.items(), key=lambda x: x[1], reverse=True)
dom_event_col   = sorted_events[0][0]
dom_event_label = CLUSTER_META[dom_event_col]["label"]
dom_event_score = event_avgs[dom_event_col]

_CLUSTER_DESC = {
    "event_tech_product_announcements":     "product announcements and technology developments",
    "event_corporate_news_mixed":           "general corporate activity and company news",
    "event_energy_materials_capex":         "capital expenditure and resource sector developments",
    "event_dividend_value_commentary":      "dividend announcements and value-oriented commentary",
    "event_fund_commentary":                "fund performance and portfolio management updates",
    "event_market_sentiment":               "broad market conditions and macroeconomic developments",
    "event_earnings_acquisitions_mixed":    "earnings results and acquisition activity",
    "event_macro_political_mixed":          "macroeconomic and political developments",
    "event_clinical_trial":                 "pharmaceutical developments and clinical trial results",
    "event_dividend_analyst_price_targets": "analyst rating changes and price target updates",
    "event_mna_and_dividend_mixed":         "merger and acquisition activity alongside dividend news",
    "event_dividend_buffett_picks":         "value investing commentary and long-term stock picks",
}

if dom_fb == "positive" and avg_vader > 0.05:
    _sent_line = (
        f"News sentiment for {first['company']} on this date was broadly positive "
        f"across both fine-tuned FinBERT and VADER signals."
    )
elif dom_fb == "negative" and avg_vader < -0.05:
    _sent_line = "News sentiment was broadly negative across both models."
else:
    _sent_line = (
        "Sentiment signals are mixed — FinBERT and VADER disagree on the dominant tone, "
        "suggesting ambiguous or complex news coverage."
    )

_event_line = (
    f"The dominant event cluster was {dom_event_label} "
    f"(cosine similarity {dom_event_score:.3f}), suggesting news coverage was primarily "
    f"focused on {_CLUSTER_DESC.get(dom_event_col, dom_event_label.lower())}."
)

_sector_match = rq3[rq3["sector"] == first["sector"]]
if not _sector_match.empty:
    _sr = _sector_match.iloc[0]
    _rel = str(_sr["reliability"])
    _rel_detail = {
        "High":   "Signal quality is stable and trustworthy.",
        "Medium": "Interpret signals with moderate caution.",
        "Low":    "Substantial distribution shift detected — treat signals as indicative only.",
    }.get(_rel, "")
    _sector_line = (
        f"The {first['sector']} sector has a {_rel} reliability rating for event-based "
        f"signals at the 30-day horizon (AUC = {float(_sr['event_auc']):.3f}). {_rel_detail}"
    )
else:
    _sector_line = ""

st.markdown(section_header("Signal Interpretation"), unsafe_allow_html=True)
st.html(
    f"""<div style="background:#0d1829;border-left:3px solid #2563eb;border-radius:10px;padding:1.2rem;">
        <p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#ffffff;line-height:1.7;margin:0 0 0.75rem 0;">{_sent_line}</p>
        <p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#ffffff;line-height:1.7;margin:0 0 0.75rem 0;">{_event_line}</p>
        <p style="font-family:'Inter',sans-serif;font-size:0.88rem;color:#ffffff;line-height:1.7;margin:0;">{_sector_line}</p>
    </div>"""
)

# ── Event Clusters ─────────────────────────────────────────────────────────────

st.markdown(section_header("Event Clusters"), unsafe_allow_html=True)

y_labels   = [CLUSTER_META[col]["label"] for col, _ in sorted_events]
x_values   = [val for _, val in sorted_events]
bar_colors = [
    COLOURS["accent"] if col == dom_event_col else COLOURS["border"]
    for col, _ in sorted_events
]

fig_ev = go.Figure(go.Bar(
    x=x_values,
    y=y_labels,
    orientation="h",
    marker_color=bar_colors,
    text=[f"{v:.3f}" for v in x_values],
    textposition="outside",
    textfont=dict(family="IBM Plex Mono", size=10, color=COLOURS["text"]),
))
fig_ev.update_layout(
    **plotly_defaults(),
    height=390,
    margin=dict(l=120, r=80, t=50, b=60),
    xaxis=dict(range=[0, 1.12], showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, autorange="reversed"),
)
st.plotly_chart(fig_ev, use_container_width=True)

st.markdown(
    f"""
    <div class="callout-card">
        <div class="callout-label">Dominant Event Cluster</div>
        <div class="callout-value">{dom_event_label}</div>
        <div class="muted">avg posterior probability {dom_event_score:.4f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Per-Article Breakdown ──────────────────────────────────────────────────────

st.markdown(section_header("Per-Article Breakdown"), unsafe_allow_html=True)

display_df = day_df.head(_ARTICLE_CAP)

if n > _ARTICLE_CAP:
    st.markdown(
        f'<div class="muted" style="margin-bottom:0.6rem;">'
        f"Showing {_ARTICLE_CAP} of {n} articles for this date."
        f"</div>",
        unsafe_allow_html=True,
    )

for i, (_, row) in enumerate(display_df.iterrows(), 1):
    art_label     = finbert_label(
        row["finetuned_finbert_positive"],
        row["finetuned_finbert_negative"],
        row["finetuned_finbert_neutral"],
    )
    vader_val     = row["vader_combined_compound"]
    vader_str     = f"{float(vader_val):+.3f}" if vader_val == vader_val else "N/A"
    dom_art_col   = max(EVENT_COLS, key=lambda c: row[c])
    dom_art_event = CLUSTER_META[dom_art_col]["label"]
    dom_art_score = row[dom_art_col]

    raw_headline = row.get("headline", "")
    if isinstance(raw_headline, str) and raw_headline.strip():
        headline_html = (
            f'<div style="font-weight:600;font-size:0.92rem;line-height:1.4;'
            f'color:#ffffff;margin-top:0.2rem;">{html.escape(raw_headline)}</div>'
        )
    else:
        headline_html = (
            '<div style="font-size:0.92rem;color:#94a3b8;font-style:italic;'
            'margin-top:0.2rem;">Headline unavailable</div>'
        )

    raw_summary = row.get("summary", "")
    if isinstance(raw_summary, str) and raw_summary.strip():
        summary_text = raw_summary[:200] + "..." if len(raw_summary) > 200 else raw_summary
        summary_html = (
            f'<div style="font-size:0.8rem;line-height:1.5;color:#dbeafe;'
            f'margin-top:0.3rem;">{html.escape(summary_text)}</div>'
        )
    else:
        summary_html = ""

    st.html(
        f"""
        <div class="article-card" style="flex-direction:column;align-items:flex-start;gap:0.2rem;">
            <div class="article-num">Article {i} of {n}</div>
            {headline_html}
            {summary_html}
            <div class="article-num" style="margin-top:0.3rem;">
                FinBERT: {art_label.title()} &nbsp;·&nbsp; VADER: {vader_str} &nbsp;·&nbsp; Event: {dom_art_event} ({dom_art_score:.3f})
            </div>
        </div>
        """
    )

# ── Sector Context ─────────────────────────────────────────────────────────────

st.markdown(section_header("Sector Context"), unsafe_allow_html=True)

sector_match = rq3[rq3["sector"] == first["sector"]]
if not sector_match.empty:
    s           = sector_match.iloc[0]
    delta       = s["event_auc"] - s["sentiment_auc"]
    delta_sign  = "+" if delta >= 0 else ""
    delta_color = COLOURS["positive"] if delta >= 0 else COLOURS["negative"]
    reliability = str(s["reliability"])
    rel_badge   = {
        "High":   "badge-high",
        "Medium": "badge-medium",
        "Low":    "badge-low",
    }.get(reliability, "badge-neutral")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Event AUC</div>
            <div class="metric-value" style="color:{COLOURS['accent2']}">{s['event_auc']:.4f}</div>
            <div class="muted">{first['sector']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sc2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Sentiment AUC</div>
            <div class="metric-value" style="color:{COLOURS['accent2']}">{s['sentiment_auc']:.4f}</div>
            <div class="muted">{first['sector']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sc3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Delta AUC (Event - Sentiment)</div>
            <div class="metric-value" style="color:{delta_color}">{delta_sign}{delta:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sc4.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Reliability Tier</div>
            <span class="badge {rel_badge}">{reliability}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="muted">No sector data available for this company.</div>',
        unsafe_allow_html=True,
    )

# ── Disclaimer ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="disclaimer">
        Decision-support tool only. ROC-AUC values of 0.51-0.59 are consistent with
        semi-strong market efficiency. This is not financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
