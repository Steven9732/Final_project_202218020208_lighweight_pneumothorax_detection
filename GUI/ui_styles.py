from __future__ import annotations

import streamlit as st

APP_CSS = r"""
        <style>
        .perf-wrap {
            background: linear-gradient(180deg, #121A23 0%, #111821 100%);
            border: 1px solid #22303E;
            border-radius: 20px;
            padding: 22px 22px 18px 22px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.24);
            margin-bottom: 18px;
        }

        .perf-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            flex-wrap: wrap;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid #22303E;
        }

        .perf-head-left {
            min-width: 280px;
        }

        .perf-title {
            font-size: 1.12rem;
            font-weight: 750;
            color: #EAF2F7;
            margin-bottom: 0.35rem;
        }

        .perf-subtitle {
            color: #9CB0BF;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        .perf-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            justify-content: flex-start;
        }

        .perf-chip {
            display: inline-block;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            border: 1px solid #304252;
            background: #17212B;
            color: #C6D7E3;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .perf-row-gap {
            height: 14px;
        }

        .metric-card-feature {
            background: linear-gradient(180deg, #172331 0%, #14202B 100%);
            border: 1px solid #35516A;
            border-radius: 18px;
            padding: 22px 22px;
            min-height: 142px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        }

        .metric-card-feature .metric-label {
            color: #9CB6C9;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .metric-card-feature .metric-value {
            color: #F5FAFD;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 10px;
        }

        .metric-card-feature .metric-note {
            color: #9CB0BF;
            font-size: 0.90rem;
            line-height: 1.55;
        }

        .metric-card-compact {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 18px 18px;
            min-height: 122px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        }

        .metric-card-compact .metric-label {
            color: #8EA1B1;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .metric-card-compact .metric-value {
            color: #F2F7FB;
            font-size: 1.45rem;
            font-weight: 780;
            line-height: 1.15;
            margin-bottom: 8px;
        }

        .metric-card-compact .metric-note {
            color: #93A6B6;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .perf-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #233243 20%, #233243 80%, transparent 100%);
            margin: 14px 0 6px 0;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(77,163,217,0.10), transparent 22%),
                radial-gradient(circle at left top, rgba(79,182,168,0.08), transparent 18%),
                linear-gradient(180deg, #0B1117 0%, #0D141C 100%);
            color: #EAF2F7;
        }

        [data-testid="stHeader"] {
            background: rgba(11, 17, 23, 0.88);
            border-bottom: 1px solid #1E2A36;
            backdrop-filter: blur(8px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E151D 0%, #101922 100%);
            border-right: 1px solid #1E2A36;
        }

        .top-hero {
            background: linear-gradient(180deg, rgba(18,26,35,0.96) 0%, rgba(16,24,32,0.96) 100%);
            border: 1px solid #233243;
            border-radius: 22px;
            padding: 24px 26px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.28);
            margin-bottom: 18px;
        }

        .title-xl {
            font-size: 2rem;
            font-weight: 800;
            color: #EAF2F7;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9FB0BF;
            font-size: 1rem;
            line-height: 1.65;
        }

        .section-card {
            background: linear-gradient(180deg, #121A23 0%, #111821 100%);
            border: 1px solid #22303E;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
            margin-bottom: 16px;
        }

        .section-title {
            font-size: 1.06rem;
            font-weight: 700;
            color: #DCE8F2;
            margin-bottom: 0.75rem;
        }

        .metric-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border-radius: 16px;
            padding: 16px 18px;
            border: 1px solid #243241;
            min-height: 112px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        }

        .metric-label {
            color: #8EA1B1;
            font-size: 0.92rem;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .metric-value {
            color: #F2F7FB;
            font-size: 1.42rem;
            font-weight: 750;
            line-height: 1.2;
        }

        .muted {
            color: #93A6B6;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .decision-banner {
            background: linear-gradient(180deg, rgba(20,33,45,0.98) 0%, rgba(15,24,33,0.98) 100%);
            border: 1px solid #274258;
            border-left: 6px solid #4FB6A8;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.22);
        }

        .decision-title {
            font-size: 0.88rem;
            color: #89A0B2;
            font-weight: 700;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .decision-value {
            font-size: 1.58rem;
            font-weight: 800;
            color: #F3F8FC;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            padding: 0.30rem 0.72rem;
            border-radius: 999px;
            border: 1px solid #304252;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            background: #17212B;
            color: #BCD0DE;
            font-size: 0.82rem;
        }

        .report-heading {
            font-size: 0.96rem;
            font-weight: 700;
            color: #D8E7F2;
            margin-top: 1rem;
            margin-bottom: 0.35rem;
        }

        .report-body {
            color: #B7C6D2;
            line-height: 1.72;
            font-size: 0.97rem;
        }

        .case-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        }

        .case-title {
            font-weight: 700;
            color: #E4EEF6;
            margin-bottom: 0.45rem;
        }

        .case-meta {
            color: #98AABA;
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .chunk-card {
            background: linear-gradient(180deg, #141D27 0%, #121A23 100%);
            border: 1px solid #243241;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.18);
        }

        .disclaimer {
            background: rgba(214,162,74,0.10);
            border: 1px solid rgba(214,162,74,0.28);
            color: #E4C987;
            border-radius: 14px;
            padding: 14px 16px;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 10px;
        }

        .stButton > button {
            border-radius: 12px;
            height: 44px;
            border: 1px solid #33506A;
            background: linear-gradient(180deg, #215172 0%, #1B4663 100%);
            color: #F4F8FB;
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(0,0,0,0.24);
        }

        .stButton > button:hover {
            border: 1px solid #41759B;
            background: linear-gradient(180deg, #276286 0%, #205675 100%);
            color: #FFFFFF;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            height: 42px;
            border: 1px solid #304252;
            background: #17212B;
            color: #EAF2F7;
            font-weight: 600;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            background: #131B24;
            border: 1px solid #22303E;
            border-radius: 12px;
            color: #9EB0BF;
            padding: 10px 16px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: #1A2733;
            border: 1px solid #3A5368;
            color: #F2F7FB;
        }

        .stTextInput input {
            background: #111923 !important;
            color: #EAF2F7 !important;
            border: 1px solid #2A3947 !important;
            border-radius: 10px !important;
        }

        .stAlert {
            border-radius: 14px;
        }
        </style>
        
"""


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)