from __future__ import annotations

#  Palette 

COLOURS: dict[str, str] = {
    "bg":       "#080f1e",
    "surface":  "#0f1c2e",
    "surface2": "#0d1829",
    "border":   "#1e3a5f",
    "accent":   "#2563eb",
    "accent2":  "#06b6d4",
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral":  "#f59e0b",
    "text":     "#dbeafe",
    "muted":    "#475569",
    "high":     "#10b981",
    "medium":   "#f59e0b",
    "low":      "#ef4444",
}

#  Event clusters 

CLUSTER_META: dict[str, dict[str, str]] = {
    "event_tech_product_announcements":     {"label": "Tech & Product Launches",     "icon": "[TECH]"},
    "event_corporate_news_mixed":           {"label": "Corporate News Mixed",         "icon": "[CORP]"},
    "event_energy_materials_capex":         {"label": "Energy & Materials CapEx",     "icon": "[ENRG]"},
    "event_dividend_value_commentary":      {"label": "Dividend & Value Commentary",  "icon": "[DIV]"},
    "event_fund_commentary":                {"label": "Fund Commentary",              "icon": "[FUND]"},
    "event_market_sentiment":               {"label": "Market Sentiment / Macro",     "icon": "[MKT]"},
    "event_earnings_acquisitions_mixed":    {"label": "Earnings & Acquisitions",      "icon": "[EARN]"},
    "event_macro_political_mixed":          {"label": "Macro & Political",            "icon": "[MACRO]"},
    "event_clinical_trial":                 {"label": "Clinical Trials & Pharma",     "icon": "[CLIN]"},
    "event_dividend_analyst_price_targets": {"label": "Analyst Ratings & Targets",    "icon": "[ANLST]"},
    "event_mna_and_dividend_mixed":         {"label": "M&A & Dividend Mixed",         "icon": "[M&A]"},
    "event_dividend_buffett_picks":         {"label": "Value / Buffett Picks",        "icon": "[VAL]"},
}

EVENT_COLS: list[str] = list(CLUSTER_META.keys())

#  Sentiment columns 

SENTIMENT_COLS: dict[str, str] = {
    "finetuned_finbert_positive": "FinBERT Positive",
    "finetuned_finbert_negative": "FinBERT Negative",
    "finetuned_finbert_neutral":  "FinBERT Neutral",
    "vader_combined_compound":    "VADER Compound",
    "ss_combined_pos":            "SentiStrength Pos",
    "ss_combined_neg":            "SentiStrength Neg",
}

#  Classification helpers 

def finbert_label(pos: float, neg: float, neu: float) -> str:
    """Return 'positive', 'negative', or 'neutral' based on highest score."""
    if pos >= neg and pos >= neu:
        return "positive"
    if neg >= pos and neg >= neu:
        return "negative"
    return "neutral"


# Pill / badge HTML helpers ─

def _pill(text: str, fg: str, bg_tint: str) -> str:
    """Render a rounded pill span. bg_tint is the colour at ~15% opacity."""
    return (
        f'<span style="'
        f"display:inline-block;"
        f"padding:0.18rem 0.75rem;"
        f"border-radius:999px;"
        f"font-family:'IBM Plex Mono',monospace;"
        f"font-size:0.68rem;"
        f"font-weight:600;"
        f"letter-spacing:0.05em;"
        f"color:{fg};"
        f"background:{bg_tint};"
        f'border:1px solid {fg}26;"'
        f">{text}</span>"
    )


def sentiment_pill(label: str) -> str:
    """Return a coloured pill for a FinBERT sentiment label (case-insensitive)."""
    key = label.strip().lower()
    mapping = {
        "positive": ("Positive", COLOURS["positive"], _hex_tint(COLOURS["positive"])),
        "negative": ("Negative", COLOURS["negative"], _hex_tint(COLOURS["negative"])),
        "neutral":  ("Neutral",  COLOURS["neutral"],  _hex_tint(COLOURS["neutral"])),
    }
    text, fg, bg = mapping.get(key, ("Unknown", COLOURS["muted"], _hex_tint(COLOURS["muted"])))
    return _pill(text, fg, bg)


def reliability_pill(tier: str) -> str:
    """Return a coloured pill for a reliability tier string."""
    key = tier.strip().lower()
    mapping = {
        "high":   ("High Reliability",   COLOURS["high"],   _hex_tint(COLOURS["high"])),
        "medium": ("Medium Reliability", COLOURS["medium"], _hex_tint(COLOURS["medium"])),
        "low":    ("Low Reliability",    COLOURS["low"],    _hex_tint(COLOURS["low"])),
    }
    text, fg, bg = mapping.get(key, (tier, COLOURS["muted"], _hex_tint(COLOURS["muted"])))
    return _pill(text, fg, bg)


def direction_pill(label: float | int) -> str:
    """Return a coloured UP/DOWN/N/A pill for a binary direction label."""
    if label == 1.0:
        return _pill("UP",   COLOURS["positive"], _hex_tint(COLOURS["positive"]))
    if label == 0.0:
        return _pill("DOWN", COLOURS["negative"], _hex_tint(COLOURS["negative"]))
    return _pill("N/A", COLOURS["muted"], _hex_tint(COLOURS["muted"]))


# Chart helpers

def auc_colour(auc: float) -> str:
    """Map an AUC value to a display colour."""
    if auc >= 0.65:
        return COLOURS["positive"]
    if auc >= 0.55:
        return COLOURS["accent2"]
    if auc >= 0.50:
        return COLOURS["neutral"]
    return COLOURS["negative"]


def plotly_defaults() -> dict:
    """Base layout kwargs for all Plotly figures."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font": {
            "family": "IBM Plex Mono",
            "color":  COLOURS["text"],
        },
    }


# HTML component helpers

def section_header(title: str) -> str:
    """Return an HTML section-heading string."""
    return (
        f'<div style="'
        f"font-family:'IBM Plex Mono',monospace;"
        f"font-size:0.68rem;"
        f"font-weight:600;"
        f"letter-spacing:0.14em;"
        f"text-transform:uppercase;"
        f"color:{COLOURS['accent2']};"
        f"border-bottom:1px solid {COLOURS['border']};"
        f"padding-bottom:0.4rem;"
        f'margin-bottom:0.8rem;"'
        f">{title.upper()}</div>"
    )


def metric_card(
    label: str,
    value: str,
    sub: str | None = None,
    value_colour: str | None = None,
) -> str:
    """Return an HTML metric card string."""
    vc = value_colour or COLOURS["text"]
    sub_html = (
        f'<div style="font-size:0.75rem;color:{COLOURS["muted"]};margin-top:0.35rem;">'
        f"{sub}</div>"
        if sub is not None
        else ""
    )
    return (
        f'<div style="'
        f"background:{COLOURS['surface']};"
        f"border:1px solid {COLOURS['border']};"
        f"border-radius:10px;"
        f'padding:1.1rem 1.3rem;">'
        f'<div style="'
        f"font-family:'IBM Plex Mono',monospace;"
        f"font-size:0.68rem;"
        f"font-weight:600;"
        f"letter-spacing:0.1em;"
        f"text-transform:uppercase;"
        f"color:{COLOURS['text']};"
        f'margin-bottom:0.4rem;">{label}</div>'
        f'<div style="'
        f"font-family:'IBM Plex Mono',monospace;"
        f"font-size:1.5rem;"
        f"font-weight:700;"
        f'color:{vc};">{value}</div>'
        f"{sub_html}"
        f"</div>"
    )


# Internal

def _hex_tint(hex_colour: str, alpha: int = 20) -> str:
    """Append a two-digit hex alpha to a #rrggbb colour string."""
    return f"{hex_colour}{alpha:02x}"
