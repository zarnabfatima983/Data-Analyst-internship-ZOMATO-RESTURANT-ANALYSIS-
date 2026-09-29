"""
app.py
------
Streamlit application — Zomato Data Analyst Internship Project
Professional UI with theme system, styled KPI cards, modern sidebar,
animated components, and responsive layout.

ALL backend functions, data calls, ML models, and analysis logic
are preserved exactly as-is. Only the UI/UX layer has been upgraded.

Run with:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# ── Make sure src/ is importable regardless of how/where Streamlit starts ─────
_ROOT = Path(__file__).resolve().parent   # = project root (where app.py lives)
sys.path.insert(0, str(_ROOT))

# ── Matplotlib must use non-interactive Agg backend BEFORE any other import ───
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Analytics Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  THEME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "🌟 Professional": {
        "bg":           "#0f1117",
        "card_bg":      "#1a1d29",
        "card_border":  "#2d3250",
        "sidebar_bg":   "#13151f",
        "sidebar_text": "#c8cde4",
        "text":         "#e8eaf6",
        "subtext":      "#8b92b8",
        "heading":      "#ffffff",
        "accent":       "#6c63ff",
        "accent2":      "#ff6584",
        "accent3":      "#43e97b",
        "metric_val":   "#6c63ff",
        "badge_bg":     "#2d3250",
        "success":      "#43e97b",
        "warning":      "#ffbe0b",
        "info":         "#4cc9f0",
        "chart_bg":     "#1a1d29",
        "chart_text":   "#c8cde4",
        "shadow":       "rgba(108,99,255,0.25)",
        "btn_bg":       "linear-gradient(135deg,#6c63ff,#a78bfa)",
        "btn_text":     "#ffffff",
        "nav_active":   "#6c63ff",
        "input_bg":     "#2d3250",
        "table_head":   "#2d3250",
        "table_row":    "#1a1d29",
        "table_alt":    "#22263a",
        "font":         "Inter",
    },
    "🌊 Modern Blue": {
        "bg":           "#0a0e1a",
        "card_bg":      "#111827",
        "card_border":  "#1e3a5f",
        "sidebar_bg":   "#0d1220",
        "sidebar_text": "#94a3b8",
        "text":         "#e2e8f0",
        "subtext":      "#64748b",
        "heading":      "#f8fafc",
        "accent":       "#3b82f6",
        "accent2":      "#06b6d4",
        "accent3":      "#10b981",
        "metric_val":   "#60a5fa",
        "badge_bg":     "#1e3a5f",
        "success":      "#10b981",
        "warning":      "#f59e0b",
        "info":         "#06b6d4",
        "chart_bg":     "#111827",
        "chart_text":   "#94a3b8",
        "shadow":       "rgba(59,130,246,0.2)",
        "btn_bg":       "linear-gradient(135deg,#3b82f6,#06b6d4)",
        "btn_text":     "#ffffff",
        "nav_active":   "#3b82f6",
        "input_bg":     "#1e3a5f",
        "table_head":   "#1e3a5f",
        "table_row":    "#111827",
        "table_alt":    "#162032",
        "font":         "Poppins",
    },
    "☀️ Light": {
        "bg":           "#f0f4f8",
        "card_bg":      "#ffffff",
        "card_border":  "#e2e8f0",
        "sidebar_bg":   "#ffffff",
        "sidebar_text": "#475569",
        "text":         "#1e293b",
        "subtext":      "#64748b",
        "heading":      "#0f172a",
        "accent":       "#6366f1",
        "accent2":      "#ec4899",
        "accent3":      "#059669",
        "metric_val":   "#6366f1",
        "badge_bg":     "#e0e7ff",
        "success":      "#059669",
        "warning":      "#d97706",
        "info":         "#0284c7",
        "chart_bg":     "#ffffff",
        "chart_text":   "#475569",
        "shadow":       "rgba(99,102,241,0.15)",
        "btn_bg":       "linear-gradient(135deg,#6366f1,#8b5cf6)",
        "btn_text":     "#ffffff",
        "nav_active":   "#6366f1",
        "input_bg":     "#f8fafc",
        "table_head":   "#e0e7ff",
        "table_row":    "#ffffff",
        "table_alt":    "#f8fafc",
        "font":         "Inter",
    },
    "🌿 Dark Green": {
        "bg":           "#0d1b0e",
        "card_bg":      "#132015",
        "card_border":  "#1d4021",
        "sidebar_bg":   "#0d1b0e",
        "sidebar_text": "#86efac",
        "text":         "#dcfce7",
        "subtext":      "#6b7280",
        "heading":      "#f0fdf4",
        "accent":       "#22c55e",
        "accent2":      "#84cc16",
        "accent3":      "#06b6d4",
        "metric_val":   "#4ade80",
        "badge_bg":     "#1d4021",
        "success":      "#22c55e",
        "warning":      "#eab308",
        "info":         "#06b6d4",
        "chart_bg":     "#132015",
        "chart_text":   "#86efac",
        "shadow":       "rgba(34,197,94,0.2)",
        "btn_bg":       "linear-gradient(135deg,#22c55e,#84cc16)",
        "btn_text":     "#0d1b0e",
        "nav_active":   "#22c55e",
        "input_bg":     "#1d4021",
        "table_head":   "#1d4021",
        "table_row":    "#132015",
        "table_alt":    "#0d1b0e",
        "font":         "Roboto",
    },
}

# ── Theme selector (persisted in session state) ───────────────────────────────
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🌟 Professional"

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS INJECTION — applies the selected theme across the entire app
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css(t: dict) -> None:
    font_import = {
        "Inter":   "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
        "Poppins": "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap",
        "Roboto":  "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap",
    }.get(t["font"], "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap")

    st.markdown(f'<link href="{font_import}" rel="stylesheet">', unsafe_allow_html=True)

    css = f"""
    <style>
    /* ── Base & Background ────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        background: {t["bg"]} !important;
        font-family: '{t["font"]}', sans-serif !important;
        color: {t["text"]} !important;
    }}
    [data-testid="stAppViewContainer"] > .main {{
        background: {t["bg"]} !important;
    }}
    [data-testid="block-container"] {{
        padding: 1.5rem 2rem 2rem 2rem !important;
    }}

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {t["sidebar_bg"]} !important;
        border-right: 1px solid {t["card_border"]} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {t["sidebar_text"]} !important;
        font-family: '{t["font"]}', sans-serif !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        padding: 0.55rem 0.9rem !important;
        border-radius: 8px !important;
        margin-bottom: 3px !important;
        transition: all 0.2s ease !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        display: block !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebar"] .stRadio label:hover {{
        background: {t["badge_bg"]} !important;
        color: {t["accent"]} !important;
        transform: translateX(3px) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
        color: {t["subtext"]} !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin-bottom: 0.5rem !important;
    }}

    /* ── Typography — strict hierarchy, no colour collisions ─────────── */
    h1 {{
        font-family: '{t["font"]}', sans-serif !important;
        font-size: 1.95rem !important;
        font-weight: 800 !important;
        color: {t["heading"]} !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.2rem !important;
    }}
    h2 {{
        font-family: '{t["font"]}', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: {t["heading"]} !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.3rem !important;
        line-height: 1.3 !important;
    }}
    h3 {{
        font-family: '{t["font"]}', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: {t["text"]} !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.2rem !important;
    }}
    /* Only target actual prose — NOT every span/label which breaks widgets */
    [data-testid="block-container"] p {{
        font-family: '{t["font"]}', sans-serif !important;
        color: {t["text"]} !important;
        line-height: 1.6 !important;
    }}
    [data-testid="block-container"] li {{
        font-family: '{t["font"]}', sans-serif !important;
        color: {t["text"]} !important;
    }}
    /* Page title subtitle paragraph */
    .page-subtitle {{
        font-family: '{t["font"]}', sans-serif !important;
        color: {t["subtext"]} !important;
        font-size: 0.95rem !important;
        margin-top: 0 !important;
        margin-bottom: 1.25rem !important;
        line-height: 1.5 !important;
    }}

    /* ── Streamlit native metric widget ──────────────────────────────── */
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: {t["metric_val"]} !important;
        font-family: '{t["font"]}', sans-serif !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        color: {t["subtext"]} !important;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.75rem !important;
        color: {t["success"]} !important;
    }}
    [data-testid="metric-container"] {{
        background: {t["card_bg"]} !important;
        border: 1px solid {t["card_border"]} !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.4rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 2px 12px {t["shadow"]} !important;
    }}
    [data-testid="metric-container"]:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px {t["shadow"]} !important;
    }}

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button {{
        background: {t["btn_bg"]} !important;
        color: {t["btn_text"]} !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.6rem !important;
        font-family: '{t["font"]}', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px {t["shadow"]} !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {t["shadow"]} !important;
        filter: brightness(1.08) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0px) !important;
        filter: brightness(0.95) !important;
    }}
    [data-testid="stFormSubmitButton"] > button {{
        background: {t["btn_bg"]} !important;
        color: {t["btn_text"]} !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-family: '{t["font"]}', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px {t["shadow"]} !important;
        width: 100% !important;
    }}
    [data-testid="stFormSubmitButton"] > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {t["shadow"]} !important;
        filter: brightness(1.1) !important;
    }}

    /* ── Input widgets ────────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {{
        background: {t["input_bg"]} !important;
        color: {t["text"]} !important;
        border: 1px solid {t["card_border"]} !important;
        border-radius: 8px !important;
        font-family: '{t["font"]}', sans-serif !important;
        font-size: 0.9rem !important;
    }}
    .stSelectbox label, .stMultiSelect label,
    .stNumberInput label, .stSlider label,
    .stCheckbox label {{
        color: {t["subtext"]} !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        font-family: '{t["font"]}', sans-serif !important;
    }}
    .stSlider > div > div > div > div {{
        background: {t["accent"]} !important;
    }}
    [data-baseweb="slider"] [role="slider"] {{
        background: {t["accent"]} !important;
        border: 2px solid {t["bg"]} !important;
    }}

    /* ── DataFrames / Tables ──────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid {t["card_border"]} !important;
    }}
    .dataframe-container {{
        font-family: '{t["font"]}', sans-serif !important;
    }}

    /* ── Expanders ────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background: {t["card_bg"]} !important;
        border: 1px solid {t["card_border"]} !important;
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        overflow: hidden !important;
    }}
    [data-testid="stExpander"] summary {{
        font-family: '{t["font"]}', sans-serif !important;
        font-weight: 600 !important;
        color: {t["text"]} !important;
        padding: 0.75rem 1rem !important;
        background: {t["card_bg"]} !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        background: {t["badge_bg"]} !important;
        color: {t["accent"]} !important;
    }}

    /* ── Info / Success / Warning boxes ──────────────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 10px !important;
        border-left-width: 4px !important;
        font-family: '{t["font"]}', sans-serif !important;
        background: {t["card_bg"]} !important;
    }}
    .stSuccess {{
        border-left-color: {t["success"]} !important;
        color: {t["success"]} !important;
    }}
    .stWarning {{
        border-left-color: {t["warning"]} !important;
    }}
    .stInfo {{
        border-left-color: {t["info"]} !important;
    }}

    /* ── Dividers ─────────────────────────────────────────────────────── */
    hr {{
        border-color: {t["card_border"]} !important;
        opacity: 0.5 !important;
    }}

    /* ── Spinners ─────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] {{
        color: {t["accent"]} !important;
    }}

    /* ── Top toolbar — complete override ─────────────────────────────── */
    /* Bar background */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stApp > header,
    section[data-testid="stSidebar"] ~ div > header {{
        background: {t["card_bg"]} !important;
        border-bottom: 1px solid {t["card_border"]} !important;
        backdrop-filter: none !important;
    }}
    /* EVERY svg inside the header — fill + colour */
    header[data-testid="stHeader"] svg,
    [data-testid="stToolbar"] svg {{
        fill:   {t["text"]} !important;
        color:  {t["text"]} !important;
        stroke: {t["text"]} !important;
        opacity: 1 !important;
    }}
    /* EVERY button/anchor inside the header */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] a {{
        color:   {t["text"]} !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}
    /* Deploy button — pill style */
    [data-testid="stToolbar"] button[kind="header"],
    [data-testid="baseButton-header"],
    button[data-testid="stBaseButton-header"] {{
        background:    {t["badge_bg"]}    !important;
        color:         {t["accent"]}      !important;
        border:        1px solid {t["accent"]} !important;
        border-radius: 8px !important;
        font-family:   '{t["font"]}', sans-serif !important;
        font-weight:   600 !important;
        font-size:     0.8rem !important;
        padding:       0.25rem 0.9rem !important;
        transition:    all 0.2s ease !important;
    }}
    [data-testid="baseButton-header"]:hover,
    button[data-testid="stBaseButton-header"]:hover {{
        background: {t["accent"]} !important;
        color:      {t["btn_text"]} !important;
    }}
    /* Fullscreen / wide-mode double-arrow button */
    [data-testid="StyledFullScreenButton"],
    button[title*="fullscreen" i],
    button[title*="wide" i],
    button[aria-label*="wide" i],
    button[aria-label*="fullscreen" i] {{
        background:    {t["badge_bg"]}  !important;
        border:        1px solid {t["card_border"]} !important;
        border-radius: 6px !important;
        color:         {t["text"]} !important;
        opacity:       1 !important;
        visibility:    visible !important;
    }}
    [data-testid="StyledFullScreenButton"] svg,
    button[title*="fullscreen" i] svg,
    button[title*="wide" i] svg {{
        fill:   {t["text"]} !important;
        color:  {t["text"]} !important;
        stroke: none !important;
    }}
    /* Hamburger / kebab menu button */
    [data-testid="stMainMenuButton"],
    button[aria-label="Main menu"],
    button[title="Main menu"] {{
        color:      {t["text"]} !important;
        opacity:    1 !important;
        visibility: visible !important;
    }}
    [data-testid="stMainMenuButton"] svg,
    button[aria-label="Main menu"] svg,
    button[title="Main menu"] svg {{
        fill:   {t["text"]} !important;
        color:  {t["text"]} !important;
        stroke: {t["text"]} !important;
    }}
    /* Recording / status dot (red circle when running) */
    [data-testid="stStatusWidget"],
    [data-testid="stStatusWidget"] *,
    [data-testid="stDecoration"],
    .stStatusWidget {{
        color:      {t["text"]} !important;
        opacity:    1 !important;
        visibility: visible !important;
    }}
    [data-testid="stStatusWidget"] svg,
    [data-testid="stDecoration"] svg {{
        fill:   {t["text"]} !important;
        stroke: {t["text"]} !important;
    }}
    /* Hover states for all toolbar buttons */
    header[data-testid="stHeader"] button:hover svg,
    [data-testid="stToolbar"] button:hover svg {{
        fill:   {t["accent"]} !important;
        color:  {t["accent"]} !important;
        stroke: {t["accent"]} !important;
    }}
    /* Hamburger dropdown panel */
    [data-testid="stMainMenu"],
    ul[data-testid="stMainMenu"] {{
        background:    {t["card_bg"]}    !important;
        border:        1px solid {t["card_border"]} !important;
        border-radius: 12px !important;
        box-shadow:    0 8px 32px {t["shadow"]} !important;
        padding:       0.4rem !important;
    }}
    [data-testid="stMainMenu"] li,
    [data-testid="stMainMenu"] a,
    [data-testid="stMainMenu"] span,
    [data-testid="stMainMenu"] button {{
        color:         {t["text"]}  !important;
        font-family:   '{t["font"]}', sans-serif !important;
        font-size:     0.88rem !important;
        border-radius: 8px !important;
    }}
    [data-testid="stMainMenu"] li:hover,
    [data-testid="stMainMenu"] button:hover {{
        background: {t["badge_bg"]} !important;
        color:      {t["accent"]}   !important;
    }}
    [data-testid="stMainMenu"] svg {{
        fill:  {t["text"]} !important;
        color: {t["text"]} !important;
    }}

    /* ── Scrollbar ────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {t["bg"]}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb {{ background: {t["card_border"]}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {t["accent"]}; }}

    /* ── Matplotlib chart containers ──────────────────────────────────── */
    [data-testid="stImage"] img {{
        border-radius: 12px !important;
        border: 1px solid {t["card_border"]} !important;
    }}

    /* ── Page fade-in animation ──────────────────────────────────────── */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to   {{ opacity: 1; transform: translateY(0);    }}
    }}
    [data-testid="block-container"] > div:first-child {{
        animation: fadeInUp 0.35s ease forwards !important;
    }}

    /* ── Column cards (wrap columns in a card feel) ────────────────────── */
    [data-testid="stHorizontalBlock"] > div {{
        transition: transform 0.15s ease !important;
    }}

    /* ── Checkbox ──────────────────────────────────────────────────────── */
    .stCheckbox span {{
        color: {t["text"]} !important;
        font-family: '{t["font"]}', sans-serif !important;
    }}

    /* ── Tab-like section separators ─────────────────────────────────── */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.8rem 0 0.6rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid {t["card_border"]};
        clear: both;
    }}
    .section-header span {{
        font-size: 0.95rem;
        font-weight: 700;
        color: {t["heading"]};
        font-family: '{t["font"]}', sans-serif;
        letter-spacing: 0.2px;
    }}
    .section-dot {{
        width: 8px; height: 8px;
        background: {t["accent"]};
        border-radius: 50%;
        flex-shrink: 0;
    }}
    /* First section-header on a page — no top margin */
    .page-badge + h1 + p + .section-header,
    .section-header:first-of-type {{
        margin-top: 0.5rem;
    }}

    /* ── KPI custom card ─────────────────────────────────────────────── */
    .kpi-card {{
        background: {t["card_bg"]};
        border: 1px solid {t["card_border"]};
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 16px {t["shadow"]};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 30px {t["shadow"]};
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: {t["btn_bg"]};
    }}
    .kpi-icon {{
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
        display: block;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {t["metric_val"]};
        font-family: '{t["font"]}', sans-serif;
        line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {t["subtext"]};
        font-family: '{t["font"]}', sans-serif;
        margin-top: 0.3rem;
    }}
    .kpi-desc {{
        font-size: 0.78rem;
        color: {t["subtext"]};
        font-family: '{t["font"]}', sans-serif;
        margin-top: 0.2rem;
    }}

    /* ── Page title badge ─────────────────────────────────────────────── */
    .page-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: {t["badge_bg"]};
        border: 1px solid {t["card_border"]};
        border-radius: 100px;
        padding: 0.25rem 0.85rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: {t["accent"]};
        letter-spacing: 0.5px;
        font-family: '{t["font"]}', sans-serif;
        margin-bottom: 0.5rem;
    }}

    /* ── Result score bar ─────────────────────────────────────────────── */
    .score-bar-container {{
        background: {t["card_bg"]};
        border: 1px solid {t["card_border"]};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.4rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .score-bar-container:hover {{
        transform: translateX(4px);
        box-shadow: 0 4px 16px {t["shadow"]};
    }}
    .score-bar-fill {{
        height: 6px;
        border-radius: 3px;
        background: {t["btn_bg"]};
        transition: width 0.6s ease;
    }}

    /* ── Insight card ─────────────────────────────────────────────────── */
    .insight-card {{
        background: {t["card_bg"]};
        border: 1px solid {t["card_border"]};
        border-left: 4px solid {t["accent"]};
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        font-family: '{t["font"]}', sans-serif;
    }}
    .insight-card p {{
        color: {t["text"]};
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }}

    /* ── Sidebar logo area ────────────────────────────────────────────── */
    .sidebar-logo {{
        text-align: center;
        padding: 1rem 0 1.25rem 0;
        border-bottom: 1px solid {t["card_border"]};
        margin-bottom: 1rem;
    }}
    .sidebar-logo-title {{
        font-size: 1.1rem;
        font-weight: 800;
        color: {t["heading"]};
        font-family: '{t["font"]}', sans-serif;
        margin-top: 0.4rem;
    }}
    .sidebar-logo-sub {{
        font-size: 0.72rem;
        color: {t["subtext"]};
        font-family: '{t["font"]}', sans-serif;
        letter-spacing: 0.5px;
    }}

    /* ── Recommendation result card ──────────────────────────────────── */
    .rec-card {{
        background: {t["card_bg"]};
        border: 1px solid {t["card_border"]};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 0.6rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 2px 12px {t["shadow"]};
    }}
    .rec-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 28px {t["shadow"]};
    }}
    .rec-card-name {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {t["heading"]};
        font-family: '{t["font"]}', sans-serif;
    }}
    .rec-card-meta {{
        font-size: 0.82rem;
        color: {t["subtext"]};
        font-family: '{t["font"]}', sans-serif;
        margin-top: 0.2rem;
    }}
    .rec-tag {{
        display: inline-block;
        background: {t["badge_bg"]};
        color: {t["accent"]};
        border-radius: 100px;
        padding: 0.15rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0.15rem 0.1rem;
        font-family: '{t["font"]}', sans-serif;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Helper: styled section header ─────────────────────────────────────────────
def section(title: str, icon: str = "") -> None:
    label = f"{icon} {title}" if icon else title
    st.markdown(
        f'<div class="section-header"><div class="section-dot"></div>'
        f'<span>{label}</span></div>',
        unsafe_allow_html=True,
    )


# ── Helper: KPI card HTML ─────────────────────────────────────────────────────
def kpi_card(icon: str, value: str, label: str, desc: str = "") -> str:
    desc_html = f'<div class="kpi-desc">{desc}</div>' if desc else ""
    return (
        f'<div class="kpi-card">'
        f'<span class="kpi-icon">{icon}</span>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'{desc_html}</div>'
    )


# ── Helper: apply theme to matplotlib figures ─────────────────────────────────
def style_fig(fig: plt.Figure, t: dict) -> None:
    """Apply theme background and text colours to a matplotlib figure."""
    fig.patch.set_facecolor(t["chart_bg"])
    for ax in fig.axes:
        ax.set_facecolor(t["chart_bg"])
        ax.tick_params(colors=t["chart_text"], labelsize=9)
        ax.xaxis.label.set_color(t["chart_text"])
        ax.yaxis.label.set_color(t["chart_text"])
        ax.title.set_color(t["heading"])
        for spine in ax.spines.values():
            spine.set_edgecolor(t["card_border"])


# ── Cached data & model loaders (unchanged) ───────────────────────────────────
@st.cache_data(show_spinner="Loading and cleaning dataset …")
def get_data():
    from src.data_cleaning import load_clean
    try:
        return load_clean()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource(show_spinner="Building recommendation engine …")
def get_recommender(df):
    from src.recommendation import RestaurantRecommender
    rec = RestaurantRecommender()
    rec.fit(df)
    return rec


@st.cache_resource(show_spinner="Training ML models (this takes ~30 s) …")
def get_ml_results(df):
    from src.rating_prediction import train_and_evaluate
    return train_and_evaluate(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

# Theme selector (must be before inject_css)
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '<span style="font-size:2.2rem">🍽️</span>'
        '<div class="sidebar-logo-title">Zomato Analytics</div>'
        '<div class="sidebar-logo-sub">Data Analyst Internship Project</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1.5px;opacity:0.6;margin-bottom:0.2rem">APPEARANCE</p>',
        unsafe_allow_html=True,
    )
    chosen_theme = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme_name),
        label_visibility="collapsed",
    )
    st.session_state.theme_name = chosen_theme
    T = THEMES[chosen_theme]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1.5px;opacity:0.6;margin-bottom:0.2rem">NAVIGATION</p>',
        unsafe_allow_html=True,
    )

    PAGES = [
        "🏠  Home",
        "🏙️  City Analysis",
        "🛵  Online Delivery",
        "💰  Price Range",
        "🤖  Rating Prediction",
        "🔍  Recommender",
    ]
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")

    st.markdown(
        f"""
        <div style='margin-top:2rem;padding:1rem;background:{T["card_bg"]};
             border:1px solid {T["card_border"]};border-radius:12px;
             font-size:0.78rem;color:{T["subtext"]};line-height:1.7'>
        <b style='color:{T["text"]}'>Dataset</b><br>
        Zomato Restaurants<br>
        <b style='color:{T["text"]}'>Source</b><br>
        Kaggle / Shruti Mehta<br>
        <b style='color:{T["text"]}'>Records</b><br>
        9,551 raw → 7,403 cleaned
        </div>
        """,
        unsafe_allow_html=True,
    )

# Apply CSS after T is defined
inject_css(T)

# Set seaborn/matplotlib to match theme
sns.set_theme(style="dark" if T["bg"] < "#888888" else "whitegrid",
              palette="muted", font_scale=1.0)
plt.rcParams.update({
    "figure.facecolor":  T["chart_bg"],
    "axes.facecolor":    T["chart_bg"],
    "axes.edgecolor":    T["card_border"],
    "axes.labelcolor":   T["chart_text"],
    "xtick.color":       T["chart_text"],
    "ytick.color":       T["chart_text"],
    "text.color":        T["chart_text"],
    "axes.titlecolor":   T["heading"],
    "grid.color":        T["card_border"],
    "grid.alpha":        0.4,
})


# ── Helper: render and close a figure ─────────────────────────────────────────
def render(fig: plt.Figure) -> None:
    style_fig(fig, T)
    st.pyplot(fig)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown('<span class="page-badge">📊 Overview</span>', unsafe_allow_html=True)
    st.title("🍽️ Zomato Data Analyst Project")
    st.markdown(
        '<p class="page-subtitle">'
        "Complete analysis pipeline on 7,403 real Zomato restaurant records across 141 cities."
        "</p>",
        unsafe_allow_html=True,
    )

    df = get_data()
    from src.analysis import dataset_overview
    ov = dataset_overview(df)

    # ── KPI grid ─────────────────────────────────────────────────────────────
    section("Key Metrics", "📈")
    kpis = [
        ("🏪", f"{ov['total_restaurants']:,}", "Total Restaurants", "After cleaning"),
        ("🌆", f"{ov['unique_cities']:,}",       "Unique Cities",     "Across 15 countries"),
        ("⭐", f"{ov['avg_rating']}",             "Avg Rating",        "Out of 5.0"),
        ("🛵", f"{ov['pct_online_delivery']}%",  "Online Delivery",   "Penetration rate"),
        ("🍜", f"{ov['unique_cuisines']:,}",      "Unique Cuisines",   "Primary types"),
        ("👍", f"{ov['avg_votes']:,}",            "Avg Votes",         "Per restaurant"),
        ("💰", f"₹{ov['avg_cost']:,.0f}",         "Avg Cost for Two",  "In local currency"),
        ("📅", f"{ov['pct_table_booking']}%",    "Table Booking",     "Offer reservations"),
    ]
    cols = st.columns(4)
    for i, (icon, val, lbl, desc) in enumerate(kpis):
        with cols[i % 4]:
            st.markdown(kpi_card(icon, val, lbl, desc), unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Task overview — proper clickable nav cards ────────────────────────────
    section("Project Tasks", "🗂️")

    # Add CSS for the task nav cards
    st.markdown(f"""
    <style>
    .task-nav-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin-bottom: 1rem;
    }}
    .task-nav-card {{
        background: {T["card_bg"]};
        border: 1px solid {T["card_border"]};
        border-radius: 14px;
        padding: 1.1rem 1rem 1rem 1rem;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        cursor: default;
    }}
    .task-nav-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }}
    .task-nav-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 28px {T["shadow"]};
        border-color: {T["accent"]};
    }}
    .task-nav-card:hover::before {{
        background: {T["btn_bg"]};
    }}
    .tnc-num {{
        font-size: 0.62rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {T["accent"]};
        font-family: '{T["font"]}', sans-serif;
        margin-bottom: 0.45rem;
    }}
    .tnc-icon {{
        font-size: 1.5rem;
        display: block;
        margin-bottom: 0.4rem;
        line-height: 1;
    }}
    .tnc-title {{
        font-size: 0.88rem;
        font-weight: 700;
        color: {T["heading"]};
        font-family: '{T["font"]}', sans-serif;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }}
    .tnc-desc {{
        font-size: 0.73rem;
        color: {T["subtext"]};
        font-family: '{T["font"]}', sans-serif;
        line-height: 1.45;
    }}
    .tnc-badge {{
        display: inline-block;
        background: {T["badge_bg"]};
        color: {T["accent"]};
        border-radius: 100px;
        padding: 0.1rem 0.55rem;
        font-size: 0.65rem;
        font-weight: 700;
        margin-top: 0.5rem;
        font-family: '{T["font"]}', sans-serif;
        letter-spacing: 0.5px;
    }}
    @media (max-width: 900px) {{
        .task-nav-grid {{ grid-template-columns: repeat(3, 1fr); }}
    }}
    @media (max-width: 600px) {{
        .task-nav-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    </style>
    <div class="task-nav-grid">
      <div class="task-nav-card">
        <div class="tnc-num">Task 1</div>
        <span class="tnc-icon">🏙️</span>
        <div class="tnc-title">City Analysis</div>
        <div class="tnc-desc">Restaurant distribution &amp; ratings across 141 cities</div>
        <span class="tnc-badge">EDA · Charts</span>
      </div>
      <div class="task-nav-card">
        <div class="tnc-num">Task 2</div>
        <span class="tnc-icon">🛵</span>
        <div class="tnc-title">Online Delivery</div>
        <div class="tnc-desc">Delivery penetration &amp; rating comparison analysis</div>
        <span class="tnc-badge">Analysis · Stats</span>
      </div>
      <div class="task-nav-card">
        <div class="tnc-num">Task 3</div>
        <span class="tnc-icon">💰</span>
        <div class="tnc-title">Price Range</div>
        <div class="tnc-desc">Price tier analysis &amp; customer value insights</div>
        <span class="tnc-badge">Distribution</span>
      </div>
      <div class="task-nav-card">
        <div class="tnc-num">Task 4</div>
        <span class="tnc-icon">🤖</span>
        <div class="tnc-title">Rating Prediction</div>
        <div class="tnc-desc">Gradient Boosting — R²&nbsp;=&nbsp;0.60, MAE&nbsp;=&nbsp;0.25</div>
        <span class="tnc-badge">ML · Scikit-learn</span>
      </div>
      <div class="task-nav-card">
        <div class="tnc-num">Task 5</div>
        <span class="tnc-icon">🔍</span>
        <div class="tnc-title">Recommender</div>
        <div class="tnc-desc">TF-IDF + Cosine Similarity content-based filtering</div>
        <span class="tnc-badge">NLP · Streamlit</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample data ───────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    section("Sample Data", "🔎")
    st.dataframe(
        df[["Restaurant Name", "City", "Cuisines", "Aggregate rating",
            "Price range", "Has Online delivery", "Has Table booking",
            "Votes", "Average Cost for two"]].head(10),
        use_container_width=True,
        height=320,
    )

    # ── Rating histogram ──────────────────────────────────────────────────────
    section("Rating Distribution", "📊")
    from src.visualization import plot_rating_distribution_hist
    fig = plot_rating_distribution_hist(df)
    render(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CITY ANALYSIS (TASK 1)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏙️  City Analysis":
    st.markdown('<span class="page-badge">📍 Task 1</span>', unsafe_allow_html=True)
    st.title("🏙️ City-wise Restaurant Analysis")
    st.markdown(
        '<p class="page-subtitle">'
        "Restaurant distribution, counts, and average ratings across 141 cities."
        "</p>",
        unsafe_allow_html=True,
    )

    df = get_data()
    from src.analysis      import (city_restaurant_counts, city_avg_rating,
                                    top_n_cities, city_summary, top_cuisines)
    from src.visualization import (plot_top_cities_bar, plot_city_avg_rating,
                                    plot_city_dual_axis, plot_top_cuisines,
                                    plot_correlation_heatmap)

    # ── Filter ────────────────────────────────────────────────────────────────
    with st.container():
        st.markdown(
            f'<div style="background:{T["card_bg"]};border:1px solid {T["card_border"]};'
            f'border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem">',
            unsafe_allow_html=True,
        )
        top_n = st.slider("Number of cities to display", 5, 30, 10,
                          help="Adjust how many top cities appear in the charts")
        st.markdown("</div>", unsafe_allow_html=True)

    city_counts = city_restaurant_counts(df)

    # ── Quick stats ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    top1 = city_counts.iloc[0]
    with c1:
        st.markdown(kpi_card("🥇", top1["City"], "Top City", f"{top1['Restaurant Count']:,} restaurants"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("🌆", str(city_counts.shape[0]), "Total Cities", "In dataset"),
                    unsafe_allow_html=True)
    with c3:
        avg_r = city_avg_rating(df)
        best_city = avg_r[avg_r["Restaurant Count"] >= 20].iloc[0]
        st.markdown(kpi_card("⭐", str(best_city["Avg Rating"]),
                              "Highest Rated City", best_city["City"]),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("🍜", str(top_cuisines(df, 1).iloc[0]["Cuisine"]),
                              "Top Cuisine", "Most common in dataset"),
                    unsafe_allow_html=True)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Charts in pairs ───────────────────────────────────────────────────────
    section(f"Top {top_n} Cities by Restaurant Count", "📊")
    fig = plot_top_cities_bar(city_counts, top_n=top_n)
    render(fig)

    section(f"Top {top_n} Cities by Average Rating", "⭐")
    ratings = city_avg_rating(df)
    ratings_filtered = ratings[ratings["Restaurant Count"] >= 20]
    fig = plot_city_avg_rating(ratings_filtered, top_n=top_n)
    render(fig)

    section("Restaurant Count vs Average Rating (Dual Axis)", "📈")
    top_cities_df = top_n_cities(df, n=top_n)
    fig = plot_city_dual_axis(top_cities_df)
    render(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        section("Top 15 Cuisines", "🍜")
        cuisine_counts = top_cuisines(df, n=15)
        fig = plot_top_cuisines(cuisine_counts)
        render(fig)
    with col_b:
        section("Feature Correlation Heatmap", "🔥")
        fig = plot_correlation_heatmap(df)
        render(fig)

    section("Full City Summary Table", "📋")
    summary = city_summary(df)
    st.dataframe(summary, use_container_width=True, height=400)

    with st.expander("📌 Business Insights"):
        st.markdown(
            f'<div class="insight-card"><p>'
            "• <b>New Delhi</b> dominates with ~55% of all listed restaurants — Zomato started in India.<br>"
            "• Smaller international cities often have <b>higher average ratings</b> because only standout restaurants list on Zomato there.<br>"
            "• Positive but moderate correlation between <b>votes and rating</b> (~0.30) — popular restaurants tend to be rated higher.<br>"
            "• <b>North Indian</b> cuisine is the most common, reflecting the Delhi-heavy dataset."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ONLINE DELIVERY (TASK 2)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛵  Online Delivery":
    st.markdown('<span class="page-badge">🛵 Task 2</span>', unsafe_allow_html=True)
    st.title("🛵 Online Delivery Analysis")
    st.markdown(
        '<p class="page-subtitle">'
        "Delivery penetration, rating comparison, and city-level adoption analysis."
        "</p>",
        unsafe_allow_html=True,
    )

    df = get_data()
    from src.analysis      import (delivery_counts, delivery_rating_comparison,
                                    delivery_by_city)
    from src.visualization import (plot_delivery_pie, plot_delivery_rating_bar,
                                    plot_delivery_by_city)

    d_counts = delivery_counts(df)
    with_del = d_counts.loc[d_counts["Delivery Flag"]==1, "Count"].values[0]
    pct      = d_counts.loc[d_counts["Delivery Flag"]==1, "Percentage"].values[0]
    without  = df.shape[0] - with_del

    # ── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("🛵", f"{with_del:,}", "With Delivery", "Offer online ordering"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("📊", f"{pct:.1f}%", "Delivery Penetration", "Of total restaurants"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("🏪", f"{without:,}", "Without Delivery", "Dine-in / pickup only"), unsafe_allow_html=True)
    with c4:
        comp = delivery_rating_comparison(df)
        diff = round(
            comp.loc[comp["Has Online Delivery"]=="No","Avg Rating"].values[0]
            - comp.loc[comp["Has Online Delivery"]=="Yes","Avg Rating"].values[0], 2)
        st.markdown(kpi_card("⭐", f"+{diff}", "Rating Gap", "No-delivery restaurants rate higher"), unsafe_allow_html=True)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1.6])
    with col_a:
        section("Delivery Availability", "🥧")
        fig = plot_delivery_pie(d_counts)
        render(fig)
    with col_b:
        section("Rating / Votes / Cost Comparison", "📊")
        fig = plot_delivery_rating_bar(comp)
        render(fig)

    section("Delivery vs No-Delivery — Comparison Table", "📋")
    st.dataframe(comp, use_container_width=True, height=120)

    section("Rating Distribution — Box Plot", "📦")
    df_plot = df.copy()
    df_plot["Delivery"] = df_plot["Has Online delivery"].map({1:"Yes", 0:"No"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_plot, x="Delivery", y="Aggregate rating",
                palette=["#2ecc71","#e74c3c"], width=0.45, ax=ax)
    ax.set_title("Rating Distribution — Online Delivery vs None", fontweight="bold")
    ax.set_xlabel("Has Online Delivery")
    ax.set_ylabel("Aggregate Rating")
    plt.tight_layout()
    render(fig)

    top_n_del = st.slider("Cities to show in delivery chart", 5, 30, 15,
                          help="Number of top cities to display in delivery penetration chart")
    section(f"Online Delivery Penetration — Top {top_n_del} Cities", "🌆")
    city_del = delivery_by_city(df, top_n=top_n_del)
    fig = plot_delivery_by_city(city_del)
    render(fig)

    with st.expander("📌 Business Insights"):
        st.markdown(
            f'<div class="insight-card"><p>'
            "• Only <b>~32%</b> of restaurants offer online delivery — a significant growth opportunity.<br>"
            "• Restaurants <b>without</b> delivery have a marginally higher avg rating (3.47 vs 3.38).<br>"
            "• Delivery restaurants receive <b>more votes</b> on average — higher visibility & engagement.<br>"
            "• <b>Budget (Price 1)</b> restaurants lead delivery adoption; Luxury (Price 4) rarely deliver.<br>"
            "• Recommendation: target <b>mid-range (Price 2-3)</b> restaurants for delivery expansion."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PRICE RANGE (TASK 3)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰  Price Range":
    st.markdown('<span class="page-badge">💰 Task 3</span>', unsafe_allow_html=True)
    st.title("💰 Price Range Analysis")
    st.markdown(
        '<p class="page-subtitle">'
        "Distribution, ratings, votes, and cost analysis across 4 price tiers."
        "</p>",
        unsafe_allow_html=True,
    )

    df = get_data()
    from src.analysis      import (price_range_distribution, price_range_avg_rating,
                                    best_price_range, price_delivery_crosstab, PRICE_LABELS)
    from src.visualization import (plot_price_range_bar, plot_price_range_pie,
                                    plot_price_avg_rating, plot_price_cost_rating)

    price_dist  = price_range_distribution(df)
    price_stats = price_range_avg_rating(df)
    best        = best_price_range(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    icons = ["🟢","🟡","🟠","🔴"]
    cols = st.columns(4)
    for col, row, icon in zip(cols, price_stats.itertuples(), icons):
        with col:
            st.markdown(
                kpi_card(icon, f"{row.Count:,}", row.Label,
                         f"Avg Rating: {row._4} ⭐"),
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="insight-card" style="border-left-color:{T["accent3"]}">'
        f'<p>★ Best value price range: <b>{best["Label"]}</b> — '
        f'Avg Rating <b>{best["Avg Rating"]}</b> | '
        f'{best["Count"]:,} restaurants</p></div>',
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        section("Restaurant Count by Price Range", "📊")
        fig = plot_price_range_bar(price_dist)
        render(fig)
    with col_b:
        section("Price Range Distribution (Pie)", "🥧")
        fig = plot_price_range_pie(price_dist)
        render(fig)

    section("Average Rating per Price Range", "⭐")
    fig = plot_price_avg_rating(price_stats)
    render(fig)

    col_c, col_d = st.columns(2)
    with col_c:
        section("Avg Cost vs Avg Rating (Bubble Chart)", "🔵")
        fig = plot_price_cost_rating(price_stats)
        render(fig)
    with col_d:
        section("Rating Distribution — Box Plot", "📦")
        df_plot = df.copy()
        df_plot["Price Label"] = df_plot["Price range"].map(PRICE_LABELS)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_plot, x="Price Label", y="Aggregate rating",
                    palette="Set2", width=0.45,
                    order=list(PRICE_LABELS.values()), ax=ax)
        ax.set_title("Rating Distribution per Price Range", fontweight="bold")
        ax.set_xlabel("Price Range")
        ax.set_ylabel("Aggregate Rating")
        plt.tight_layout()
        render(fig)

    section("Full Price Range Statistics", "📋")
    st.dataframe(price_stats, use_container_width=True, height=200)

    with st.expander("📌 Business Insights"):
        st.markdown(
            f'<div class="insight-card"><p>'
            "• <b>Budget (1) + Mid-Range (2)</b> = ~74% of all restaurants — mass-market dominance.<br>"
            "• Clear pattern: <b>higher price → higher average rating</b>.<br>"
            "• <b>Premium (3)</b> restaurants receive the most votes (~455 avg) — the engagement sweet spot.<br>"
            "• Average cost spans <b>₹285 (Budget) to ₹1,475 (Luxury)</b> — tiers are well-separated.<br>"
            "• For best dining value, <b>Premium (3)</b> offers the highest rating with strong engagement."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: RATING PREDICTION (TASK 4)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Rating Prediction":
    st.markdown('<span class="page-badge">🤖 Task 4</span>', unsafe_allow_html=True)
    st.title("🤖 Restaurant Rating Prediction")
    st.markdown(
        '<p class="page-subtitle">'
        "Four ML regression models trained and compared to predict Aggregate Rating."
        "</p>",
        unsafe_allow_html=True,
    )

    df = get_data()
    from src.visualization     import (plot_actual_vs_predicted, plot_feature_importance,
                                        plot_model_comparison, plot_residuals)
    from src.rating_prediction import predict_single, ALL_FEATURES

    with st.spinner("Training models … (first load takes ~30 seconds)"):
        results = get_ml_results(df)

    best     = results["best_name"]
    best_row = results["results"][results["results"]["Model"]==best].iloc[0]

    # ── Model KPIs ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("🏆", best.split()[0], "Best Model", best), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("📐", str(best_row["R2 Score"]), "R² Score", "Variance explained"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("📏", str(best_row["MAE"]), "MAE", "Mean Absolute Error"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("📉", str(best_row["RMSE"]), "RMSE", "Root Mean Squared Error"), unsafe_allow_html=True)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Model comparison table ────────────────────────────────────────────────
    section("Model Comparison Table", "📋")
    st.dataframe(results["results"], use_container_width=True, height=200)

    # ── Metric selector ───────────────────────────────────────────────────────
    section("Model Comparison Chart", "📊")
    metric_choice = st.selectbox("Select metric to compare",
                                 ["R2 Score", "MAE", "RMSE"],
                                 help="Switch between R², MAE and RMSE to compare models")
    fig = plot_model_comparison(results["results"], metric=metric_choice)
    render(fig)

    # ── Actual vs Predicted + Residuals ───────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        section("Actual vs Predicted Ratings", "🎯")
        fig = plot_actual_vs_predicted(
            results["y_test"], results["y_pred_best"], model_name=best)
        render(fig)
    with col_b:
        section("Residual Plot", "📉")
        fig = plot_residuals(results["y_test"], results["y_pred_best"], model_name=best)
        render(fig)

    residuals    = np.array(results["y_test"]) - np.array(results["y_pred_best"])
    pct_within   = (np.abs(residuals) <= 0.5).mean() * 100
    st.markdown(
        f'<div class="insight-card" style="border-left-color:{T["accent3"]}">'
        f'<p>✅ <b>{pct_within:.1f}%</b> of predictions are within '
        f'<b>±0.5 rating points</b> of the actual rating.</p></div>',
        unsafe_allow_html=True,
    )

    # ── Feature importance ────────────────────────────────────────────────────
    if results["feature_importance"] is not None:
        section("Top 15 Feature Importances", "🔑")
        col_c, col_d = st.columns([1.5, 1])
        with col_c:
            fig = plot_feature_importance(results["feature_importance"], top_n=15)
            render(fig)
        with col_d:
            st.dataframe(results["feature_importance"].head(15),
                         use_container_width=True, height=420)

    # ── Custom prediction form ────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    section("🔮 Predict Rating for a Custom Restaurant", "✨")
    st.markdown(
        '<p class="page-subtitle">'
        "Fill in the restaurant details below and click Predict to get a model-generated rating estimate."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            votes     = st.number_input("Votes",                0, 10000, 200)
            avg_cost  = st.number_input("Average Cost for Two", 0, 10000, 600)
            has_del   = st.selectbox("Has Online Delivery",  ["No","Yes"])
            has_book  = st.selectbox("Has Table Booking",    ["No","Yes"])
        with col2:
            price_rng = st.selectbox(
                "Price Range", [1,2,3,4],
                format_func=lambda x: {1:"🟢 Budget",2:"🟡 Mid-Range",
                                        3:"🟠 Premium",4:"🔴 Luxury"}[x])
            city      = st.selectbox("City", sorted(df["City"].unique().tolist()))
            cuisine   = st.selectbox("Primary Cuisine",
                                     sorted(df["Primary Cuisine"].dropna().unique().tolist()))
        submitted = st.form_submit_button("🔮 Predict Rating")

    if submitted:
        row = {
            "Votes":                int(votes),
            "Average Cost for two": int(avg_cost),
            "Has Online delivery":  1 if has_del=="Yes" else 0,
            "Has Table booking":    1 if has_book=="Yes" else 0,
            "Price range":          price_rng,
            "Primary Cuisine":      cuisine,
            "City":                 city,
        }
        pred = predict_single(results["best_model"], row)
        stars = "⭐" * round(pred)
        st.markdown(
            f'<div style="background:{T["card_bg"]};border:1px solid {T["accent"]};'
            f'border-radius:14px;padding:1.5rem 2rem;text-align:center;margin-top:1rem;'
            f'box-shadow:0 4px 20px {T["shadow"]}">'
            f'<div style="font-size:2.8rem;font-weight:900;color:{T["metric_val"]}">{pred:.2f}</div>'
            f'<div style="font-size:1.2rem;margin:0.2rem 0">{stars}</div>'
            f'<div style="color:{T["subtext"]};font-size:0.85rem">Predicted rating out of 5.00 '
            f'· Model: {best}</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander("📌 Key Findings"):
        st.markdown(
            f'<div class="insight-card"><p>'
            "• <b>Gradient Boosting</b> is the best model with R²≈0.60, MAE≈0.25.<br>"
            "• <b>Votes</b> is the single most important predictor — engagement signals quality.<br>"
            "• <b>Price Range</b> and <b>City</b> are the next strongest factors.<br>"
            "• ~60% of rating variation is explained by available features; the rest is subjective.<br>"
            "• The model is accurate within ±0.5 stars for most restaurants."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: RECOMMENDER (TASK 5)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Recommender":
    st.markdown('<span class="page-badge">🔍 Task 5</span>', unsafe_allow_html=True)
    st.title("🔍 Restaurant Recommendation System")
    st.markdown(
        '<p class="page-subtitle">'
        "Content-Based Filtering using TF-IDF + Cosine Similarity across 7,403 restaurants."
        "</p>",
        unsafe_allow_html=True,
    )

    df  = get_data()
    rec = get_recommender(df)
    from src.analysis import PRICE_LABELS

    # ── Filter form ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{T["card_bg"]};border:1px solid {T["card_border"]};'
        f'border-radius:14px;padding:1.5rem;margin-bottom:1rem">',
        unsafe_allow_html=True,
    )
    with st.form("recommend_form"):
        section("Your Preferences", "🎯")
        col1, col2, col3 = st.columns(3)
        with col1:
            cuisine_options = ["Any"] + sorted(df["Primary Cuisine"].dropna().unique().tolist())
            cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
        with col2:
            city_options = ["Any"] + sorted(df["City"].dropna().unique().tolist())
            city = st.selectbox("Preferred City", city_options)
        with col3:
            price_options = {0:"Any", 1:"🟢 Budget (1)", 2:"🟡 Mid-Range (2)",
                             3:"🟠 Premium (3)", 4:"🔴 Luxury (4)"}
            price_range = st.selectbox("Price Range",
                                       list(price_options.keys()),
                                       format_func=lambda x: price_options[x])

        col4, col5, col6 = st.columns(3)
        with col4:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1,
                                   help="Only show restaurants at or above this rating")
        with col5:
            top_n = st.slider("Number of Results", 3, 20, 8,
                              help="How many recommendations to return")
        with col6:
            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
            need_delivery = st.checkbox("🛵 Must have Online Delivery")
            need_booking  = st.checkbox("📅 Must have Table Booking")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Find Restaurants")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    if submitted:
        cuisine_query = "" if cuisine == "Any" else cuisine
        city_query    = "" if city    == "Any" else city

        with st.spinner("Finding the best matches …"):
            results = rec.recommend(
                cuisine=cuisine_query,
                city=city_query,
                price_range=price_range,
                min_rating=min_rating,
                top_n=top_n,
                require_delivery=need_delivery,
                require_booking=need_booking,
            )

        if results.empty:
            st.markdown(
                f'<div class="insight-card" style="border-left-color:{T["warning"]}">'
                "<p>⚠️ No restaurants matched your filters. "
                "Try relaxing one or more criteria (lower min rating, any price range).</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="insight-card" style="border-left-color:{T["success"]}">'
                f"<p>✅ Found <b>{len(results)}</b> matching restaurant(s) for your preferences!</p></div>",
                unsafe_allow_html=True,
            )

            # ── Results table ─────────────────────────────────────────────────
            section("Results Table", "📋")
            display_cols = ["Restaurant Name","City","Cuisines","Aggregate rating",
                            "Price Label","Online Delivery","Table Booking",
                            "Average Cost for two","Votes","Composite Score"]
            show_cols = [c for c in display_cols if c in results.columns]
            st.dataframe(results[show_cols], use_container_width=True, height=300)

            # ── Score bars ────────────────────────────────────────────────────
            section("Recommendation Scores", "📊")
            fig, ax = plt.subplots(figsize=(10, max(4, len(results)*0.55)))
            colors = sns.color_palette("viridis", len(results))
            ax.barh(results["Restaurant Name"][::-1],
                    results["Composite Score"][::-1], color=colors)
            ax.set_xlabel("Composite Score", fontsize=11)
            ax.set_title("Restaurants Ranked by Composite Score\n"
                         "(0.5 × Cosine Similarity + 0.5 × Normalised Rating)",
                         fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1)
            for i, val in enumerate(results["Composite Score"][::-1]):
                ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)
            plt.tight_layout()
            render(fig)

            # ── Detail cards ──────────────────────────────────────────────────
            section("Restaurant Details", "🏪")
            for i, row in results.iterrows():
                rating_bar_pct = int(row["Aggregate rating"] / 5.0 * 100)
                score_pct      = int(row["Composite Score"] * 100)

                reasons = []
                if cuisine_query and cuisine_query.lower() in row["Cuisines"].lower():
                    reasons.append(f"✅ Serves {cuisine_query}")
                if city_query and city_query.lower() in row["City"].lower():
                    reasons.append(f"✅ Located in {city_query}")
                if price_range > 0:
                    pl = price_options[price_range].split("(")[0].strip()
                    if pl.lower() in row.get("Price Label","").lower():
                        reasons.append(f"✅ {row.get('Price Label','')}")
                if row["Aggregate rating"] >= min_rating:
                    reasons.append(f"✅ Rating {row['Aggregate rating']} ≥ {min_rating}")

                tags_html = "".join(f'<span class="rec-tag">{r}</span>' for r in reasons)

                delivery_badge = (
                    f'<span class="rec-tag" style="color:{T["success"]}">🛵 Delivery</span>'
                    if row.get("Online Delivery") == "Yes" else ""
                )
                booking_badge = (
                    f'<span class="rec-tag" style="color:{T["info"]}">📅 Booking</span>'
                    if row.get("Table Booking") == "Yes" else ""
                )

                st.markdown(
                    f'<div class="rec-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                    f'<div>'
                    f'<div class="rec-card-name">#{i+1} &nbsp; {row["Restaurant Name"]}</div>'
                    f'<div class="rec-card-meta">📍 {row["City"]} &nbsp;·&nbsp; '
                    f'🍽️ {row["Cuisines"][:50]}{"…" if len(row["Cuisines"])>50 else ""}</div>'
                    f'</div>'
                    f'<div style="text-align:right;flex-shrink:0;margin-left:1rem">'
                    f'<div style="font-size:1.6rem;font-weight:900;color:{T["metric_val"]}">'
                    f'{row["Aggregate rating"]}</div>'
                    f'<div style="font-size:0.7rem;color:{T["subtext"]}">/ 5.0</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin:0.6rem 0 0.3rem 0">'
                    f'<div style="font-size:0.72rem;color:{T["subtext"]};margin-bottom:3px">'
                    f'Rating: {row["Aggregate rating"]}/5</div>'
                    f'<div style="background:{T["card_border"]};border-radius:4px;height:5px">'
                    f'<div class="score-bar-fill" style="width:{rating_bar_pct}%"></div></div>'
                    f'</div>'
                    f'<div style="margin-bottom:0.5rem">'
                    f'<div style="font-size:0.72rem;color:{T["subtext"]};margin-bottom:3px">'
                    f'Match Score: {row["Composite Score"]:.3f}</div>'
                    f'<div style="background:{T["card_border"]};border-radius:4px;height:5px">'
                    f'<div class="score-bar-fill" style="width:{score_pct}%;'
                    f'background:linear-gradient(90deg,{T["accent2"]},{T["accent"]})">'
                    f'</div></div>'
                    f'</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:0.2rem;margin-top:0.5rem">'
                    f'<span class="rec-tag">{row.get("Price Label","N/A")}</span>'
                    f'<span class="rec-tag">💰 ₹{row["Average Cost for two"]:,.0f}</span>'
                    f'<span class="rec-tag">👍 {row["Votes"]:,} votes</span>'
                    f'{delivery_badge}{booking_badge}'
                    f'</div>'
                    f'{("<div style=\"margin-top:0.5rem\">" + tags_html + "</div>") if tags_html else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with st.expander("ℹ️ How the Recommender Works"):
        st.markdown(
            f'<div class="insight-card"><p>'
            "<b>Algorithm:</b> Content-Based Filtering · TF-IDF + Cosine Similarity<br><br>"
            "1. Each restaurant becomes a TF-IDF text document (cuisine · city · price · rating · services)<br>"
            "2. Your preferences become a query document with the same vocabulary<br>"
            "3. <b>Cosine Similarity</b> ranks all 7,403 restaurants by closeness to your query<br>"
            "4. <b>Composite Score</b> = 0.5 × Similarity + 0.5 × Normalised Rating<br>"
            "5. Hard filters (city, cuisine, price, min rating, delivery/booking) narrow candidates first"
            "</p></div>",
            unsafe_allow_html=True,
        )
