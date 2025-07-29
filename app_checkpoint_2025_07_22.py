# --- Checkpoint updated: 2025-07-22 ---
# This is a direct copy of the current working app.py as a checkpoint for rollback or reference.

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from train_model import get_symptom_map, map_symptom, emergency_check, predict, CLUSTERS
import io
from fpdf import FPDF
import re

# Set Streamlit to wide layout for full-bleed background
st.set_page_config(layout="wide")

# --- Chat UI CSS (single block, after imports) ---
st.markdown("""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700&display=swap');
body, .main, .block-container {
    background: linear-gradient(135deg, #e3f6fc 0%, #f6fff8 100%);
}
.chat-container {
    max-width: 600px;
    margin: 0 auto;
    padding: 16px 0 80px 0;
}
.bubble {
    display: inline-block;
    padding: 12px 18px;
    border-radius: 20px;
    margin-bottom: 6px;
    max-width: 80%;
    word-break: break-word;
    font-size: 1.05em;
    animation: fadeIn 0.5s;
}
.bubble.user {
    background: #1976d2;
    color: #fff;
    border-radius: 20px 20px 4px 20px;
    box-shadow: 0 2px 8px #1976d233;
    }
.bubble.bot {
        background: #e3f6fc;
        color: #1976d2;
    border: 1.5px solid #b2dfdb;
    border-radius: 20px 20px 20px 4px;
    box-shadow: 0 2px 8px #b2dfdb33;
    }
.chip-group {
        display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
    margin-bottom: 2px;
    }
.chip, .option-chip {
    background: #1976d2;
    color: #fff;
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 1.05em;
    border: none;
    margin-bottom: 2px;
    margin-right: 4px;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer;
}
.chip:hover, .option-chip:hover, .chip:focus, .option-chip:focus {
    background: #43e97b;
        color: #fff;
    box-shadow: 0 2px 8px #43e97b33;
    }
.input-container {
    position: fixed;
    bottom: 0;
    left: 0; right: 0;
    width: 100vw;
    background: #f4fafd;
    padding: 16px 0 12px 0;
    box-shadow: 0 -2px 12px rgba(30,60,90,0.04);
    z-index: 100;
    }
.stTextInput>div>div>input {
    border-radius: 20px !important;
    border: 2px solid #1976d2 !important;
    background: #fff !important;
    color: #222 !important;
    font-size: 1.08em !important;
    padding: 10px 16px !important;
    }
.header-brand {
        display: flex;
        align-items: center;
    justify-content: center;
    margin-top: 2em;
    margin-bottom: 1.5em;
    gap: 1em;
    }
.header-logo {
    font-size: 3em;
    background: #fff;
    border-radius: 50%;
    box-shadow: 0 2px 12px #1976d233;
    padding: 0.2em 0.4em;
    border: 2px solid #b2dfdb;
    }
.header-title {
    font-family: 'Nunito', Arial, sans-serif;
    font-size: 2.5em;
        font-weight: 700;
        color: #1976d2;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px #e3f6fc;
    }
.accessibility-bar {
    display: flex; gap: 1em; align-items: center; margin-bottom: 1.5em;
    }
.accessibility-btn {
        background: #1976d2;
    color: #fff;
    border: none;
    border-radius: 16px;
    padding: 8px 22px;
    font-size: 1.08em;
    cursor: pointer;
    font-weight: 600;
    box-shadow: 0 2px 8px #1976d233;
    transition: background 0.2s, color 0.2s;
    }
.accessibility-btn.selected {
    background: #43e97b;
    color: #fff;
}
.accessibility-btn:focus {
    outline: 2px solid #1976d2;
    }
.bot-msg {
    background: #fff;
        color: #1976d2;
    border: 1.5px solid #1976d2;
    border-radius: 20px;
    padding: 12px 18px;
    margin-bottom: 8px;
    font-size: 1.08em;
        font-weight: 500;
    }
.option-chip, .chip {
        background: #1976d2;
        color: #fff;
    border-radius: 16px;
    padding: 4px 14px;
    font-size: 1.05em;
    border: none;
    margin-bottom: 2px;
    margin-right: 4px;
    }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: none; }
}
.stButton>button:hover, .stButton>button:focus {
    background: #43e97b !important;
    color: #fff !important;
    box-shadow: 0 2px 8px #43e97b33 !important;
}
.disease-card {
    background: #f6fff8;
    border-radius: 1.2em;
    box-shadow: 0 2px 16px #b2dfdb44;
    padding: 1.2em 1.5em;
    margin-bottom: 1.2em;
    border-left: 8px solid #1976d2;
    color: #1976d2;
    position: relative;
}
.confidence-badge {
    display: inline-block;
        font-size: 1em;
    font-weight: 700;
    padding: 0.3em 1em;
    border-radius: 1em;
    margin-right: 0.7em;
    color: #fff;
}
.confidence-high { background: #43e97b; }
.confidence-med { background: #ffd600; color: #222; }
.confidence-low { background: #ff9800; }
.download-btn {
    background: #1976d2;
    color: #fff;
    font-size: 1.1em;
    font-weight: 700;
    border: none;
    border-radius: 1.5em;
    padding: 0.7em 2.2em;
    margin: 1em 0.5em 0 0;
    cursor: pointer;
    box-shadow: 0 2px 8px #1976d233;
    transition: background 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 0.7em;
}
.download-btn:hover { background: #43e97b; color: #fff; }
.restart-btn {
    background: #43e97b;
    color: #fff;
    font-size: 1.1em;
    font-weight: 700;
    border: none;
    border-radius: 1.5em;
    padding: 0.7em 2.2em;
    margin: 1em 0.5em 0 0;
    cursor: pointer;
    box-shadow: 0 2px 8px #43e97b33;
    transition: background 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 0.7em;
}
.restart-btn:hover { background: #1976d2; color: #fff; }
    </style>
""", unsafe_allow_html=True)

# ...existing code from your current app.py...
