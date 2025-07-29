import joblib
import numpy as np
import pandas as pd
import time
from train_model import get_symptom_map, map_symptom, emergency_check, predict, CLUSTERS
import io
from fpdf import FPDF
import re
import difflib
import streamlit as st

# Set Streamlit to wide layout for full-bleed background
st.set_page_config(layout="wide")

# --- Inject viewport meta tag for mobile responsiveness ---
st.markdown("""
    <meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'>
    <style>
    @media (max-width: 700px) {
        .chat-container {
            max-width: 98vw !important;
            padding: 8px 0 80px 0 !important;
        }
        .bubble, .chat-bubble {
            font-size: 1em !important;
            max-width: 98vw !important;
            padding: 10px 10px !important;
        }
        .header-brand {
            flex-direction: column !important;
            gap: 0.5em !important;
        }
        .header-logo {
            font-size: 2em !important;
            padding: 0.1em 0.2em !important;
        }
        .header-title {
            font-size: 1.5em !important;
        }
        .modern-progress-bar {
            width: 98vw !important;
            max-width: 100vw !important;
            min-width: 0 !important;
        }
        .modern-progress-step {
            width: 36px !important;
        }
        .modern-progress-circle {
            width: 24px !important;
            height: 24px !important;
            font-size: 0.9em !important;
        }
        .modern-progress-label {
            font-size: 0.8em !important;
        }
        .disease-card {
            padding: 0.7em 0.5em !important;
            font-size: 1em !important;
        }
        .mp-btn-row {
            flex-direction: column !important;
            gap: 1em !important;
        }
        .mp-btn-row .stDownloadButton, .mp-btn-row .stButton {
            min-width: 120px !important;
            max-width: 98vw !important;
        }
        .severity-chip-row {
            gap: 0.5em !important;
            margin-left: 0.5em !important;
        }
        .chat-avatar {
            width: 28px !important;
            height: 28px !important;
            font-size: 1.1em !important;
            margin: 0 0.3em !important;
        }
        .stTextInput>div>div>input {
            font-size: 1em !important;
            padding: 8px 8px !important;
        }
    }
    html, body, .main, .block-container {
        overflow-x: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

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
.severity-chip-btn button {
            background: linear-gradient(135deg, #1976d2 80%, #43e97b 100%) !important;
            color: #fff !important;
            border-radius: 10px !important;
            padding: 0.5em 1.2em !important;
            font-size: 1.2em !important;
            font-weight: 700 !important;
            border: none !important;
            cursor: pointer !important;
            transition: background 0.2s;
            margin-bottom: 0 !important;
        }
        .severity-chip-btn button:hover, .severity-chip-btn button:focus {
            background: linear-gradient(135deg, #43e97b 80%, #1976d2 100%) !important;
            color: #fff !important;
        }
    .severity-chip-row .stButton>button, .severity-chip-row button {
            background: linear-gradient(135deg, #1976d2 80%, #43e97b 100%) !important;
            background-color: #1976d2 !important;
            color: #fff !important;
            border-radius: 10px !important;
            padding: 0.5em 1.2em !important;
            font-size: 1.2em !important;
            font-weight: 700 !important;
            border: none !important;
            cursor: pointer !important;
            transition: background 0.2s;
            margin-bottom: 0 !important;
        }
        .severity-chip-row .stButton>button:hover, .severity-chip-row .stButton>button:focus,
        .severity-chip-row button:hover, .severity-chip-row button:focus {
            background: linear-gradient(135deg, #43e97b 80%, #1976d2 100%) !important;
            background-color: #43e97b !important;
            color: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper: Show typing indicator ---
def show_typing_indicator():
    st.markdown('<div class="bubble bot" style="opacity:0.7;"><span style="font-size:1.2em;">🤖</span> <i>Bot is typing...</i></div>', unsafe_allow_html=True)

# --- Helper: Format chat and results as text ---
def format_symptom_severity_summary_text(symptom_severity):
    lines = ["Symptom severity summary:"]
    for s, v in symptom_severity.items():
        s_disp = s.replace('_', ' ').capitalize()
        # Removed emoji from summary
        lines.append(f"- {s_disp}: {v}")
    return '\n'.join(lines)

def format_chat_and_results_text():
    lines = []
    lines.append("MediPredict - Chat Transcript\n")
    for msg in st.session_state.get('chat', []):
        role = 'You' if msg['role'] == 'user' else 'Bot'
        content = msg['content']
        # Remove HTML tags from bot messages
        content = re.sub(r'<[^>]+>', '', content)
        lines.append(f"{role}: {content}")
    # Add symptom severity summary
    if st.session_state.get('symptom_severity'):
        lines.append("")
        lines.append(format_symptom_severity_summary_text(st.session_state['symptom_severity']))
    lines.append("\n--- Results ---\n")
    for disease, score, user_syms, precaution, doctor in st.session_state.get('top3_results', []):
        lines.append(f"Disease: {disease}  (Confidence: {int(score*100)}%)")
        lines.append(f"Your symptoms: {', '.join([s.replace('_', ' ') for s in user_syms])}")
        # Precautions as bulleted list
        precautions_list = [p.strip().capitalize() for p in precaution.replace(';', ',').split(',') if p.strip()]
        if precautions_list:
            lines.append("Precautions:")
            for item in precautions_list:
                lines.append(f"- {item}")
        # Doctor(s) as bulleted list or fallback
        doctor_list = [d.strip() for d in doctor.replace(';', ',').split(',') if d.strip() and d.strip().lower() not in ['n/a', 'na']]
        if doctor_list:
            lines.append("Doctor to consult:")
            for d in doctor_list:
                lines.append(f"- {d}")
        else:
            lines.append("Doctor to consult:")
            lines.append("- No specific doctor recommendation. Please consult a general physician or healthcare provider.")
        lines.append("")
    lines.append("Note: This tool is for informational purposes only. Please consult a healthcare professional.")
    return '\n'.join(lines)

# --- Helper: Remove emoji/non-ASCII from text ---
def remove_emoji(text):
    # Remove all emoji and non-ASCII characters
    return re.sub(r'[^\x00-\x7F]+', '', text)

# --- Helper: Format chat and results as PDF ---
def format_chat_and_results_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 12, remove_emoji("MediPredict - Chat Transcript"), ln=1)
    pdf.set_font("Arial", '', 12)
    for msg in st.session_state.get('chat', []):
        role = 'You' if msg['role'] == 'user' else 'Bot'
        content = msg['content']
        # Remove HTML tags and emoji from bot messages
        content = re.sub(r'<[^>]+>', '', content)
        content = remove_emoji(content)
        pdf.multi_cell(0, 8, f"{role}: {content}")
    pdf.ln(4)
    # Add symptom severity summary (no emoji)
    if st.session_state.get('symptom_severity'):
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(0, 8, "Symptom severity summary:", ln=1)
        pdf.set_font("Arial", '', 12)
        for s, v in st.session_state['symptom_severity'].items():
            s_disp = remove_emoji(s.replace('_', ' ').capitalize())
            pdf.cell(0, 8, f"- {s_disp}: {v}", ln=1)
        pdf.ln(2)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, remove_emoji("Results"), ln=1)
    pdf.set_font("Arial", '', 12)
    for disease, score, user_syms, precaution, doctor in st.session_state.get('top3_results', []):
        pdf.cell(0, 8, f"Disease: {remove_emoji(disease)}  (Confidence: {int(score*100)}%)", ln=1)
        pdf.cell(0, 8, f"Your symptoms: {', '.join([remove_emoji(s.replace('_', ' ')) for s in user_syms])}", ln=1)
        precautions_list = [p.strip().capitalize() for p in precaution.replace(';', ',').split(',') if p.strip()]
        if precautions_list:
            pdf.cell(0, 8, "Precautions:", ln=1)
            for item in precautions_list:
                pdf.cell(0, 8, f"- {remove_emoji(item)}", ln=1)
        doctor_list = [d.strip() for d in doctor.replace(';', ',').split(',') if d.strip() and d.strip().lower() not in ['n/a', 'na']]
        if doctor_list:
            pdf.cell(0, 8, "Doctor to consult:", ln=1)
            for d in doctor_list:
                pdf.cell(0, 8, f"- {remove_emoji(d)}", ln=1)
        else:
            pdf.cell(0, 8, "Doctor to consult:", ln=1)
            pdf.cell(0, 8, "- No specific doctor recommendation. Please consult a general physician or healthcare provider.", ln=1)
        pdf.ln(2)
    pdf.set_text_color(180, 28, 28)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, remove_emoji("Note: This tool is for informational purposes only. Please consult a healthcare professional."))
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

# --- Chat messaging app ---
# --- Messaging app input handler (must be defined before use) ---
def handle_send():
    user_input = st.session_state.get('chat_input', '').strip()
    if not user_input:
        return
    st.session_state.chat.append({'role': 'user', 'content': user_input})
    st.session_state['typing'] = True
    st.session_state['pending_user_input'] = user_input
    st.session_state['pending_step'] = st.session_state.get('step', 'input_symptoms')
    st.session_state['chat_input'] = ''

def process_bot_response():
    user_input = st.session_state.pop('pending_user_input', None)
    step = st.session_state.pop('pending_step', None)
    if not user_input or not step:
        st.session_state['typing'] = False
        return
    if step == 'input_symptoms':
        import re
        import string
        from collections import Counter
        user_text = user_input.lower().replace('-', ' ')
        user_text = user_text.translate(str.maketrans('', '', string.punctuation))
        found = set()
        all_phrases = set(features)
        all_phrases.update(synonym_map.keys())
        all_phrases.update(synonym_map.values())
        def canonicalize_symptom(s):
            s = s.strip().lower().replace(' ', '_')
            if s in features:
                return s
            if s in synonym_map:
                mapped = synonym_map[s]
                mapped = mapped.strip().lower().replace(' ', '_')
                if mapped in features:
                    return mapped
            import difflib
            match = difflib.get_close_matches(s, features, n=1, cutoff=0.7)
            if match:
                return match[0]
            return s  # fallback to original
        for phrase in all_phrases:
            norm_phrase = phrase.replace('_', ' ').lower()
            if ' ' in norm_phrase:
                if norm_phrase in user_text:
                    mapped = canonicalize_symptom(phrase)
                    if mapped in features or (mapped == 'fever' and 'high_fever' in features):
                        found.add('high_fever' if mapped in ['fever', 'high fever'] and 'high_fever' in features else mapped)
            else:
                pattern = r'\b' + re.escape(norm_phrase) + r'\b'
                if re.search(pattern, user_text):
                    mapped = canonicalize_symptom(phrase)
                    if mapped in features or (mapped == 'fever' and 'high_fever' in features):
                        found.add('high_fever' if mapped in ['fever', 'high fever'] and 'high_fever' in features else mapped)
        fever_match = re.search(r'(fever)[^\d]*(\d{2,3}\.\d*)', user_text)
        if fever_match:
            found.add('high_fever' if 'high_fever' in features else 'fever')
        if 'jaundice' in user_text:
            for c in CLUSTERS:
                if c['disease'].lower() == 'jaundice':
                    for s in c['cluster']:
                        if s in features:
                            found.add(s)
        if not found:
            st.session_state.chat.append({'role': 'bot', 'content': 'Sorry, I could not recognize any symptoms. Please try again.'})
        else:
            mapped_set = set(found)
            cluster_matches = []
            for c in CLUSTERS:
                present = list(set([s for s in c['cluster'] if s in mapped_set]))
                missing = list(set([s for s in c['cluster'] if s not in mapped_set]))
                if present:
                    cluster_matches.append((len(present), c, present, missing))
            cluster_matches.sort(reverse=True, key=lambda x: x[0])
            # --- Filter out too-common symptoms from missing suggestions ---
            all_cluster_symptoms = [s for c in CLUSTERS for s in c['cluster']]
            symptom_counts = Counter(all_cluster_symptoms)
            COMMON_SYMPTOM_THRESHOLD = 3  # Only suggest symptoms that appear in <3 clusters
            # Always suggest missing symptoms from the most relevant cluster if at least one symptom is matched
            if cluster_matches and cluster_matches[0][2]:
                c = cluster_matches[0][1]
                present_disp = [s.replace('_', ' ') for s in cluster_matches[0][2]]
                # Filter missing symptoms to only those that are not too common
                missing = cluster_matches[0][3]
                filtered_missing = [s for s in missing if symptom_counts[s] < COMMON_SYMPTOM_THRESHOLD]
                missing_disp = [s.replace('_', ' ') for s in filtered_missing]
                if missing_disp:
                    # Deduplicate missing before display
                    filtered_missing = list(dict.fromkeys(filtered_missing))  # preserves order, removes duplicates
                    missing_disp = [s.replace('_', ' ') for s in filtered_missing]
                    msg = f'You mentioned symptoms related to <b>{c["disease"]}</b>: {", ".join(present_disp)}.<br>Are you also experiencing: <b>{", ".join(missing_disp)}</b>? (Type yes/no or list symptoms)'
                    st.session_state.chat.append({'role': 'bot', 'content': msg})
                    st.session_state.symptom_severity = {s: None for s in found}
                    st.session_state.step = 'suggest_clusters'
                    st.session_state.cluster_index = 0
                    st.session_state.suggested_clusters = []
                    st.session_state['pending_cluster'] = c['disease']
                else:
                    st.session_state.symptom_severity = {s: None for s in found}
                    st.session_state.step = 'input_severity'
                    st.session_state.cluster_index = 0
                    st.session_state.suggested_clusters = []
            else:
                st.session_state.symptom_severity = {s: None for s in found}
                st.session_state.step = 'input_severity'
                st.session_state.cluster_index = 0
                st.session_state.suggested_clusters = []
    elif step == 'suggest_clusters':
        # Robust fix: If skip_suggest_clusters is set, immediately proceed to severity and clear the flag
        if st.session_state.get('skip_suggest_clusters', False):
            st.session_state['skip_suggest_clusters'] = False
            st.session_state.step = 'input_severity'
            st.session_state.cluster_index = 0
            st.session_state.suggested_clusters = []
            st.session_state.pop('pending_cluster', None)
            st.rerun()
            return
        # Use the stored pending_cluster for correct symptom addition
        pending_cluster_name = st.session_state.get('pending_cluster', None)
        pending_cluster = None
        if pending_cluster_name:
            for c in CLUSTERS:
                if c['disease'] == pending_cluster_name:
                    pending_cluster = c
                    break
        mapped_set = set(st.session_state.symptom_severity.keys())
        idx = st.session_state.cluster_index
        max_sug = st.session_state.max_suggestions
        # Only consider clusters not already suggested
        clusters_to_suggest = [c for c in CLUSTERS if c['disease'] not in st.session_state.suggested_clusters]
        if pending_cluster is not None:
            present = [s for s in pending_cluster['cluster'] if s in mapped_set]
            missing = [s for s in pending_cluster['cluster'] if s not in mapped_set]
            response = user_input.lower()
            actually_added = []
            if response in ['yes', 'y']:
                for s in missing:
                    if s not in st.session_state.symptom_severity:
                        st.session_state.symptom_severity[s] = None
                        actually_added.append(s)
                if actually_added:
                    st.session_state.chat.append({'role': 'bot', 'content': f"Thank you! Adding symptoms: {', '.join(actually_added)}."})
                # Instead of setting skip_suggest_clusters, immediately advance to severity input
                st.session_state.step = 'input_severity'
                st.session_state.cluster_index = 0
                st.session_state.suggested_clusters = []
                st.session_state.pop('pending_cluster', None)
                st.rerun()
                return
            elif response not in ['no', 'n'] and response.strip():
                for s in missing:
                    if s.replace('_', ' ') in response and s not in st.session_state.symptom_severity:
                        st.session_state.symptom_severity[s] = None
                        actually_added.append(s)
                if actually_added:
                    for s in actually_added:
                        st.session_state.chat.append({'role': 'bot', 'content': f"Adding symptom: {s}"})
                st.session_state['skip_suggest_clusters'] = True
                st.session_state.pop('pending_cluster', None)
                st.rerun()
                return
            else:  # user said 'no'
                st.session_state.suggested_clusters.append(pending_cluster['disease'])
                st.session_state.cluster_index += 1
                st.session_state.pop('pending_cluster', None)
                # Find the next most relevant cluster (not already suggested)
                mapped_set = set(st.session_state.symptom_severity.keys())
                # Re-rank clusters by number of present symptoms, excluding already suggested
                cluster_matches = []
                for c in CLUSTERS:
                    if c['disease'] in st.session_state.suggested_clusters:
                        continue
                    present = [s for s in c['cluster'] if s in mapped_set]
                    missing = [s for s in c['cluster'] if s not in mapped_set]
                    if present:
                        cluster_matches.append((len(present), c, present, missing))
                cluster_matches.sort(reverse=True, key=lambda x: x[0])
                if st.session_state.cluster_index < max_sug and cluster_matches:
                    # Suggest the next most relevant cluster
                    next_c = cluster_matches[0][1]
                    present_disp = [s.replace('_', ' ') for s in cluster_matches[0][2]]
                    missing_disp = [s.replace('_', ' ') for s in cluster_matches[0][3]]
                    msg = f'You mentioned symptoms related to <b>{next_c["disease"]}</b>: {", ".join(present_disp)}.<br>Are you also experiencing: <b>{", ".join(missing_disp)}</b>? (Type yes/no or list symptoms)'
                    st.session_state.chat.append({'role': 'bot', 'content': msg})
                    st.session_state['pending_cluster'] = next_c['disease']
                    st.rerun()
                    return
                else:
                    # After max_suggestions or last cluster, proceed to severity for current symptoms or inform user
                    if len(st.session_state.symptom_severity) == 0:
                        st.session_state.chat.append({'role': 'bot', 'content': 'Sorry, I could not recognize enough symptoms for a prediction. Please enter more symptoms.'})
                        st.session_state.step = 'input_symptoms'
                        st.session_state.cluster_index = 0
                        st.session_state.suggested_clusters = []
                        st.session_state.pop('pending_cluster', None)
                        st.rerun()
                        return
                    else:
                        st.session_state.step = 'input_severity'
                        st.session_state.cluster_index = 0
                        st.session_state.suggested_clusters = []
                        st.session_state.pop('pending_cluster', None)
                        st.rerun()
                        return
    elif step == 'input_severity':
        pending = [s for s, v in st.session_state.symptom_severity.items() if v is None]
        if pending:
            try:
                sev = int(user_input)
                if 1 <= sev <= 5:
                    st.session_state.symptom_severity[pending[0]] = sev
                else:
                    st.session_state.chat.append({'role': 'bot', 'content': 'Please enter a number from 1 to 5 for severity.'})
            except ValueError:
                st.session_state.chat.append({'role': 'bot', 'content': 'Please enter a valid integer for severity.'})
        if all(v is not None for v in st.session_state.symptom_severity.values()):
            st.session_state.step = 'predict'
    elif step == 'predict':
        emerg, msg = emergency_check(st.session_state.symptom_severity)
        if emerg:
            st.session_state.chat.append({'role': 'bot', 'content': msg})
            st.session_state.chat.append({'role': 'bot', 'content': "Note: This tool is for informational purposes only. Please consult a healthcare professional for medical advice."})
            st.session_state.step = 'done'
        else:
            # Debug: Show the actual symptoms and severities being used for prediction
            st.write('DEBUG: Symptoms and severities for prediction:', st.session_state.symptom_severity)
            matched_disease = None
            for c in CLUSTERS:
                if all(s in st.session_state.symptom_severity for s in c['cluster']):
                    matched_disease = c['disease']
                    break
            if matched_disease:
                top3 = [(matched_disease, 0.99)]
                model_top3 = predict(st.session_state.symptom_severity, features, model, le)
                for d, s in model_top3:
                    if d != matched_disease and len(top3) < 3:
                        top3.append((d, s))
            else:
                top3 = predict(st.session_state.symptom_severity, features, model, le)
            msg = '<b>Top 3 probable diseases:</b><ol>'
            for i, (disease, score) in enumerate(top3, 1):
                msg += f"<li><b>{disease}</b> <span style='color:#666'>(Confidence: {score:.2f})</span>"
                info = disease_info.get(disease.strip().lower(), {})
                precaution = info.get('precaution', 'N/A')
                doctor = info.get('doctor_type', 'N/A')
                if precaution != 'N/A':
                    msg += f"<br><b>Precautions:</b> <ul>" + ''.join(f'<li>{p.strip()}</li>' for p in precaution.split(';')) + "</ul>"
                if doctor != 'N/A':
                    msg += f"<b>Recommended Doctor:</b> <ul>" + ''.join(f'<li>{d.strip()}</li>' for d in doctor.split(';')) + "</ul>"
                msg += "</li>"
            msg += "</ol><br><span style='color:#b91c1c'><b>Note:</b> This tool is for informational purposes only. Please consult a healthcare professional for medical advice.</span>"
            st.session_state.chat.append({'role': 'bot', 'content': msg})
            st.session_state.step = 'done'
        # Clear top3_results to avoid stale results on next run
        st.session_state.pop('top3_results', None)
    elif step == 'done':
        if user_input.lower() == 'start over':
            st.session_state.chat = []
            st.session_state.symptom_severity = {}
            st.session_state.step = 'input_symptoms'
            st.session_state.suggested_clusters = []
            st.session_state.cluster_index = 0
    st.session_state['typing'] = False
    st.rerun()

# --- Handler for severity chip click ---
def handle_severity_chip(severity):
    pending = [s for s, v in st.session_state.symptom_severity.items() if v is None]
    if pending:
        st.session_state.symptom_severity[pending[0]] = severity
        st.session_state.chat.append({'role': 'user', 'content': str(severity)})
        # If all severities are filled, add summary bot message and advance to 'predict' step
        if all(v is not None for v in st.session_state.symptom_severity.values()):
            # --- Add summary bot message for all symptoms and their severities ---
            summary = '<b>Symptom severity summary:</b><br>'
            summary += '<div style="display:flex; flex-wrap:wrap; gap:0.7em 1.2em; margin-top:0.7em;">'
            for s, v in st.session_state.symptom_severity.items():
                s_disp = s.replace('_', ' ').capitalize()
                summary += f'<span style="display:inline-flex;align-items:center;background:#1976d2;color:#fff;border-radius:1.2em;padding:0.4em 1.2em;font-weight:600;font-size:1.08em;box-shadow:0 2px 8px #1976d233;">{s_disp}: <span style="margin:0 0.5em 0 0.8em;font-size:1.1em;">{v}</span></span>'
            summary += '</div>'
            st.session_state.chat.append({'role': 'bot', 'content': summary})
            st.session_state.step = 'predict'
        # No st.rerun() needed; Streamlit will rerun after button click

# --- Ensure session state is initialized before any function uses it ---
if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'symptom_severity' not in st.session_state:
    st.session_state.symptom_severity = {}
if 'step' not in st.session_state:
    st.session_state.step = 'input_symptoms'
if 'suggested_clusters' not in st.session_state:
    st.session_state.suggested_clusters = []
if 'cluster_index' not in st.session_state:
    st.session_state.cluster_index = 0
if 'max_suggestions' not in st.session_state:
    st.session_state.cluster_index = 0
if 'max_suggestions' not in st.session_state:
    st.session_state.max_suggestions = 4

# --- Load model and data ---
@st.cache_resource
def load_model():
    model = joblib.load('models/model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    with open('features/rf_features.txt') as f:
        features = [line.strip() for line in f if line.strip()]
    try:
        kb = pd.read_csv('data/disease_centric_knowledgebase_with_doctor.csv')
        disease_info = {k.strip().lower(): v for k, v in kb.groupby('disease').agg({'precaution': 'first', 'doctor_type': 'first'}).to_dict('index').items()}
    except Exception:
        disease_info = {}
    return model, le, features, disease_info

model, le, features, disease_info = load_model()
synonym_map = get_symptom_map(features)

# --- Fever/High Fever mapping utility ---
def fever_high_fever_canonical():
    """Return canonical symptom for fever/high fever, preferring 'high_fever' if present."""
    if 'high_fever' in features:
        return 'high_fever'
    elif 'fever' in features:
        return 'fever'
    return None

# --- Update normalize_symptom to always map 'fever' and 'high fever' to canonical ---
def normalize_symptom(s):
    s = s.strip().lower().replace(' ', '_')
    # Map both 'fever' and 'high_fever' to canonical
    if s in ['fever', 'high_fever', 'high fever']:
        canonical = fever_high_fever_canonical()
        if canonical:
            return canonical
    if s in features:
        return s
    if s in synonym_map:
        mapped = synonym_map[s]
        mapped = mapped.strip().lower().replace(' ', '_')
        # If synonym is fever/high fever, map to canonical
        if mapped in ['fever', 'high_fever', 'high fever']:
            canonical = fever_high_fever_canonical()
            if canonical:
                return canonical
        if mapped in features:
            return mapped
    match = difflib.get_close_matches(s, features, n=1, cutoff=0.7)
    if match:
        # If fuzzy match is fever/high fever, map to canonical
        if match[0] in ['fever', 'high_fever', 'high fever']:
            canonical = fever_high_fever_canonical()
            if canonical:
                return canonical
        return match[0]
    return None

# --- Replace all previous normalize_symptom definitions with this one ---
# --- Also update cluster normalization at load time ---
for c in CLUSTERS:
    c['cluster'] = [normalize_symptom(s) for s in c['cluster'] if normalize_symptom(s)]

# --- Onboarding/Welcome Screen ---
if 'started' not in st.session_state or not st.session_state['started']:
    st.markdown("""
    <div style="text-align:center; margin-top: 3em;">
        <div style="font-size:3em;">🩺</div>
        <div style="font-size:2.2em; font-weight:700; color:#1976d2; margin-bottom:0.3em;">MediPredict</div>
        <div style="font-size:1.2em; color:#1976d2; margin-bottom:1.5em;">Hi! I'm MediPredict, your AI health companion.<br>How can I help you today?</div>
    """, unsafe_allow_html=True)
    if st.button('Start', key='start_btn', help='Begin chatting with MediPredict', use_container_width=True):
        st.session_state['started'] = True
    st.stop()

# --- Header ---
st.markdown(
    '<div class="header-brand">'
    '<span class="header-logo">🩺</span>'
    '<span class="header-title">MediPredict</span>'
    '</div>',
    unsafe_allow_html=True
)

# --- Progress Bar at the Top (Active pointer is always blue, completed steps green, upcoming gray) ---
if 'started' in st.session_state and st.session_state['started']:
    steps = [
        ("Describe", "input_symptoms"),
        ("Severity", "input_severity"),
        ("Output", "predict"),
    ]
    step_map = {s: i for i, (_, s) in enumerate(steps)}
    current_step = st.session_state.get('step', 'input_symptoms')
    if current_step == 'done':
        step_idx = 2
    else:
        step_idx = step_map.get(current_step, 0)
    st.markdown('''
    <style>
    .modern-progress-bar {
        position: relative;
        width: 80%;
        max-width: 520px;
        margin: 1.2em auto 2.2em auto;
        height: 54px;
    }
    .modern-progress-bar-line {
        position: absolute;
        top: 26px;
        left: 0;
        right: 0;
        height: 6px;
        background: #e0f2f1;
        border-radius: 3px;
        z-index: 1;
    }
    .modern-progress-bar-line-filled {
        position: absolute;
        top: 26px;
        left: 0;
        height: 6px;
        background: #43e97b;
        border-radius: 3px;
        z-index: 2;
        transition: width 0.3s;
    }
    .modern-progress-step {
        position: absolute;
        top: 0;
        width: 54px;
        text-align: center;
        z-index: 3;
    }
    .modern-progress-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #e0f2f1;
        color: #b0b0b0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2em;
        margin: 0 auto 2px auto;
        border: 3px solid #e0f2f1;
        box-shadow: 0 2px 8px #1976d233;
        transition: background 0.2s, color 0.2s, border 0.2s;
    }
    .modern-progress-circle.completed {
        background: #43e97b;
        color: #fff;
        border: 3px solid #43e97b;
    }
    .modern-progress-circle.active {
        background: #1976d2;
        color: #fff;
        border: 3px solid #1976d2;
    }
    .modern-progress-label {
        font-size: 0.98em;
        color: #1976d2;
        font-weight: 600;
        margin-top: 0.1em;
        letter-spacing: 0.5px;
    }
    </style>
    <div class="modern-progress-bar">
        <div class="modern-progress-bar-line"></div>
        <div class="modern-progress-bar-line-filled" style="width: {(step_idx/(len(steps)-1))*100 if step_idx>0 and len(steps)>1 else 0}%"></div>
    ''', unsafe_allow_html=True)
    for i, (label, s) in enumerate(steps):
        left = f"calc({i/(len(steps)-1)*100 if len(steps)>1 else 0}% - 27px)"  # 27px centers the 54px step
        circle_class = "modern-progress-circle"
        if i < step_idx:
            circle_class += " completed"
        elif i == step_idx:
            circle_class += " active"
        st.markdown(f'<div class="modern-progress-step" style="left:{left};"><div class="{circle_class}">{i+1}</div><div class="modern-progress-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Chatbot Conversation UI ---
if 'started' in st.session_state and st.session_state['started']:
    font_size = '1.05em'
    bot_color = '#1976d2'
    chip_bg = '#1976d2'
    chip_color = '#fff'

    chat_has_messages = bool(st.session_state.chat)

    # Show welcome illustration ONLY if there are NO chat messages
    if not chat_has_messages:
        st.markdown('''
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:320px;">
            <div style="font-size:5em; margin-bottom:0.2em;">🤖</div>
            <div style="font-size:1.5em; color:#1976d2; font-weight:700; margin-bottom:0.3em;">Welcome to your AI Health Chat</div>
            <div style="font-size:1.1em; color:#1976d2; max-width:400px; text-align:center;">Describe your symptoms or ask a health question to get started!</div>
        </div>
        ''', unsafe_allow_html=True)
    # Show chat area ONLY if there ARE chat messages (no white card)
    elif chat_has_messages:
        st.markdown('''
        <style>
        .chat-bubble-row {
            display: flex;
            align-items: flex-end;
            margin-bottom: 1.2em;
        }
        .chat-bubble-row.user { justify-content: flex-end; }
        .chat-bubble-row.bot { justify-content: flex-start; }
        .chat-avatar {
            width: 38px; height: 38px;
            border-radius: 50%;
            background: #e3f6fc;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.7em;
            margin: 0 0.7em;
            box-shadow: 0 2px 8px #1976d233;
        }
        .chat-bubble {
            padding: 14px 20px;
            border-radius: 1.5em 1.5em 1.5em 0.4em;
            font-size: 1.08em;
            max-width: 70vw;
            min-width: 60px;
            word-break: break-word;
            box-shadow: 0 2px 12px #1976d222;
            margin: 0;
            position: relative;
            animation: fadeIn 0.5s;
        }
        .chat-bubble.user {
            background: linear-gradient(135deg, #1976d2 80%, #43e97b 100%);
            color: #fff;
            border-radius: 1.5em 1.5em 0.4em 1.5em;
            margin-left: 2em;
        }
        .chat-bubble.bot {
            background: #f4fafd;
            color: #1976d2;
            border: 1.5px solid #b2dfdb;
            margin-right: 2em;
        }
        .severity-chip-row {
            display: flex;
            gap: 1.2em;
            margin: 0.7em 0 0.2em 3.7em;
        }
        </style>
        ''', unsafe_allow_html=True)
        # Render chat bubbles directly, no card
        for msg in st.session_state.chat:
            align = "user" if msg["role"] == "user" else "bot"
            avatar = "<div class='chat-avatar'>🧑</div>" if align == "user" else "<div class='chat-avatar'>🤖</div>"
            bubble = f"<div class='chat-bubble {align}'>{msg['content']}</div>"
            row = f"<div class='chat-bubble-row {align}'>"
            if align == "bot":
                row += avatar + bubble
            else:
                row += bubble + avatar
            row += "</div>"
            st.markdown(row, unsafe_allow_html=True)
        # If severity is needed, show as a bot bubble at the end (ONLY here, nowhere else)
        if st.session_state.get('step') == 'input_severity':
            pending = [s for s, v in st.session_state.symptom_severity.items() if v is None]
            if pending:
                symptom = pending[0].replace('_', ' ')
                # Show the severity prompt as a chat bubble with chips visually attached below
                st.markdown(f"""
                <div class='chat-bubble-row bot'>
                    <div class='chat-avatar'>🤖</div>
                    <div class='chat-bubble bot'>
                        <span style='font-weight:600;'>On a scale of 1 (mild) to 5 (severe), how severe is your <b>{symptom}</b>?</span>
                        <div class='severity-chip-row'>
                """, unsafe_allow_html=True)
                chip_cols = st.columns(5, gap="small")
                for i, col in enumerate(chip_cols, 1):
                    with col:
                        st.button(str(i), key=f'severity_chip_{symptom}_{i}', on_click=handle_severity_chip, args=(i,), help=f'Severity {i}', use_container_width=True)
                st.markdown("""
                        </div>
                    </div>
                </div>
                <style>
                .severity-chip-row {
                    display: flex;
                    gap: 1.2em;
                    margin: 0.7em 0 0.2em 3.7em;
                }
                .severity-chip-row .stButton>button {
                    background: linear-gradient(135deg, #1976d2 80%, #43e97b 100%) !important;
                    color: #fff !important;
                    border-radius: 10px !important;
                    padding: 0.5em 1.2em !important;
                    font-size: 1.2em !important;
                    font-weight: 700 !important;
                    border: none !important;
                    cursor: pointer !important;
                    transition: background 0.2s;
                    margin-bottom: 0 !important;
                    box-shadow: 0 2px 8px #1976d233 !important;
                }
                .severity-chip-row .stButton>button:hover, .severity-chip-row .stButton>button:focus {
                    background: linear-gradient(135deg, #43e97b 80%, #1976d2 100%) !important;
                    color: #fff !important;
                }
                </style>
                """, unsafe_allow_html=True)
                # Handle custom button click via query param
                query_params = st.query_params
                if 'severity_chip_custom' in query_params:
                    try:
                        val = int(query_params['severity_chip_custom'][0])
                        handle_severity_chip(val)
                        # Remove param to avoid double submit
                        st.query_params.clear()
                        st.experimental_rerun()
                    except Exception:
                        pass
    # --- Emergency alert banner ---
    emerg, emerg_msg = emergency_check(st.session_state.get('symptom_severity', {}))
    if emerg:
        st.markdown(f'<div style="background:#ffd6d6; color:#d32f2f; border:2px solid #d32f2f; border-radius:12px; padding:1em; margin-bottom:1em; font-size:1.1em; text-align:center;">⚠️ <b>Emergency:</b> {emerg_msg}</div>', unsafe_allow_html=True)

    # --- Chat input box at the bottom (always visible, with blinking cursor and Send button) ---
    st.markdown('''
    <style>
    .stTextInput input:focus { border-color: #43e97b !important; box-shadow: 0 0 0 2px #43e97b33 !important; }
    .stTextInput input { caret-color: #1976d2 !important; }
    .mp-chat-input-row { display: flex; align-items: center; gap: 0.7em; margin-top: 1.2em; }
    .mp-send-btn button {
        background: linear-gradient(135deg, #1976d2 70%, #43e97b 100%) !important;
        color: #fff !important;
        font-size: 1.13em !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 1.5em !important;
        padding: 0.7em 2.2em !important;
        margin: 0 !important;
        box-shadow: 0 2px 8px #1976d233 !important;
        transition: background 0.2s, color 0.2s, box-shadow 0.2s;
        width: 100%;
        outline: none !important;
    }
    .mp-send-btn button:hover, .mp-send-btn button:focus {
        background: linear-gradient(135deg, #43e97b 70%, #1976d2 100%) !important;
        color: #fff !important;
        box-shadow: 0 4px 16px #43e97b33 !important;
    }
    </style>
    <div class="mp-chat-input-row">
    ''', unsafe_allow_html=True)
    chat_input = st.text_input(
        "Chat input",  # Non-empty label for accessibility
        placeholder="Type your symptoms, reply, or severity (1-5)...",
        key="chat_input",
        label_visibility="hidden",  # Hide label visually
        on_change=handle_send
    )
    send_clicked = st.button("Send", key="send_btn", help="Send message", type="primary")
    if send_clicked:
        handle_send()
    st.markdown('</div>', unsafe_allow_html=True)
    # --- Apply modern style to Send button ---
    st.markdown('''
    <style>
    /* Target the Send button specifically in the chat input row */
    .mp-chat-input-row .stButton>button {
        background: linear-gradient(135deg, #1976d2 70%, #43e97b 100%) !important;
        color: #fff !important;
        font-size: 1.13em !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 1.5em !important;
        padding: 0.7em 2.2em !important;
        margin: 0 !important;
        box-shadow: 0 2px 8px #1976d233 !important;
        transition: background 0.2s, color 0.2s, box-shadow 0.2s;
        width: 100%;
        outline: none !important;
    }
    .mp-chat-input-row .stButton>button:hover, .mp-chat-input-row .stButton>button:focus {
        background: linear-gradient(135deg, #43e97b 70%, #1976d2 100%) !important;
        color: #fff !important;
        box-shadow: 0 4px 16px #43e97b33 !important;
    }
    </style>
    ''', unsafe_allow_html=True)
    # Debug: If nothing happens, show a message
    if not st.session_state.get('typing', False) and st.session_state.get('step') == 'input_symptoms' and not st.session_state.get('chat_input', '').strip():
        st.markdown('<div class="bot-msg" style="color:#b91c1c;">Please enter your symptoms and press Enter or click Send.</div>', unsafe_allow_html=True)
    # --- Force process_bot_response if in input_symptoms or suggest_clusters and pending_user_input exists ---
    if st.session_state.get('step') in ['input_symptoms', 'suggest_clusters'] and st.session_state.get('pending_user_input'):
        process_bot_response()
    # --- If step is 'predict', store results and advance to 'done' ---
    if st.session_state.get('step') == 'predict':
        if 'top3_results' not in st.session_state:
            def store_results():
                matched_disease = None
                best_cluster = None
                best_match_count = 0
                user_syms = list(st.session_state.symptom_severity.keys())
                for c in CLUSTERS:
                    present = [s for s in c['cluster'] if s in user_syms]
                    if len(present) >= 2:
                        if len(present) > best_match_count:
                            best_match_count = len(present)
                            best_cluster = c
                if best_cluster:
                    matched_disease = best_cluster['disease']
                else:
                    for c in CLUSTERS:
                        if all(s in user_syms for s in c['cluster']):
                            matched_disease = c['disease']
                            break
                if matched_disease:
                    top3 = [(matched_disease, 0.99)]
                    model_top3 = predict(st.session_state.symptom_severity, features, model, le)
                    for d, s in model_top3:
                        if d != matched_disease and len(top3) < 3:
                            top3.append((d, s))
                else:
                    top3 = predict(st.session_state.symptom_severity, features, model, le)
                st.session_state['top3_results'] = []
                for disease, score in top3:
                    info = disease_info.get(disease.strip().lower(), {})
                    precaution = info.get('precaution', 'N/A')
                    doctor = info.get('doctor_type', 'N/A')
                    st.session_state['top3_results'].append((disease, score, user_syms, precaution, doctor))
            store_results()
        st.session_state.step = 'done'
        st.rerun()

def get_precaution_message(precaution):
    """
    Returns a user-friendly precaution message.
    If the input is missing, empty, or 'N/A'/'na', returns a default message.
    """
    if not precaution or str(precaution).strip().lower() in ['n/a', 'na']:
        return "No specific precautions available. Please consult a healthcare professional."
    return precaution

# --- Results as card-style containers ---
if st.session_state.step in ['predict', 'done'] and 'top3_results' in st.session_state:
    st.markdown('<div style="padding: 12px 20px; margin-bottom: 10px; background: #f8fbff; border-radius: 8px; color:#1976d2; font-size:1.2em; font-weight:700;"><b>Results:</b></div>', unsafe_allow_html=True)
    for disease, score, user_syms, precaution, doctor in st.session_state['top3_results']:
        chips = ''.join([f'<span class="chip">{s.replace("_", " ")}</span>' for s in user_syms])
        precaution_msg = get_precaution_message(precaution)
        precautions_list = [p.strip().capitalize() for p in precaution_msg.replace(';', ',').split(',') if p.strip()]
        precautions_html = ''.join(f'<li>{item}</li>' for item in precautions_list)
        # --- Confidence badge color class ---
        if score >= 0.85:
            conf_class = 'confidence-high'
        elif score >= 0.6:
            conf_class = 'confidence-med'
        else:
            conf_class = 'confidence-low'
        # --- Doctor display with fallback ---
        doctor_list = [d.strip() for d in str(doctor).replace(';', ',').split(',') if d.strip() and d.strip().lower() not in ['n/a', 'na']]
        if doctor_list:
            doctor_html = ', '.join(doctor_list)
        else:
            doctor_html = 'No specific doctor recommendation. Please consult a general physician or healthcare provider.'
        st.markdown(f'''
        <div class="disease-card">
            <span class="confidence-badge {conf_class}">{int(score*100)}%</span>
            <span class="disease-title" style="color:#1976d2; font-size:1.2em; font-weight:700;">🦠 {disease.capitalize()}</span>
            <div style="margin-bottom:0.3em; color:#1976d2;"><b>Your symptoms:</b> {chips}</div>
            <div class="precaution-list" style="color:#222;"><b>Precautions:</b><ul>{precautions_html}</ul></div>
            <div style="color:#1976d2;"><b>Doctor to consult:</b> {doctor_html}</div>
        </div>
        ''', unsafe_allow_html=True)
    # --- Improved Download and Restart Buttons UI ---
    st.markdown('''
    <style>
    .mp-btn-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2.5em;
        margin-top: 2em;
        margin-bottom: 1.5em;
        width: 100%;
    }
    .mp-btn-row .stDownloadButton, .mp-btn-row .stButton {
        flex: 0 0 260px;
        max-width: 260px;
        min-width: 180px;
        display: flex;
        justify-content: center;
    }
    .stDownloadButton>button, .mp-btn-row .stButton>button {
        background: linear-gradient(135deg, #1976d2 70%, #43e97b 100%) !important;
        color: #fff !important;
        font-size: 1.13em !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 1.5em !important;
        padding: 0.8em 0 !important;
        margin: 0 !important;
        box-shadow: 0 2px 8px #1976d233 !important;
        transition: background 0.2s, color 0.2s;
        width: 100%;
        outline: none !important;
        letter-spacing: 0.01em;
    }
    .stDownloadButton>button:hover, .mp-btn-row .stButton>button:hover {
        background: linear-gradient(135deg, #43e97b 70%, #1976d2 100%) !important;
        color: #fff !important;
    }
    </style>
    <div class="mp-btn-row">
    ''', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1], gap="large")
    with col1:
        pdf_bytes = format_chat_and_results_pdf()
        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_bytes,
            file_name="symptom_checker_results.pdf",
            mime="application/pdf",
            key="download_pdf_btn",
            help="Download your chat and results as a PDF",
        )
    with col2:
        text_str = format_chat_and_results_text()
        st.download_button(
            label="⬇️ Download as Text",
            data=text_str,
            file_name="symptom_checker_results.txt",
            mime="text/plain",
            key="download_txt_btn",
            help="Download your chat and results as a text file",
        )
    with col3:
        if st.button("🔄 Restart", key="restart_btn", help="Restart the chat and symptom checker"):
            st.session_state.chat = []
            st.session_state.symptom_severity = {}
            st.session_state.step = 'input_symptoms'
            st.session_state.suggested_clusters = []
            st.session_state.cluster_index = 0
            st.session_state.started = False
            if 'top3_results' in st.session_state:
                del st.session_state['top3_results']
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Modern Stylish Button CSS for all main actions ---
    st.markdown('''
    <style>
    .stButton>button, .stDownloadButton>button, .mp-send-btn button {
        background: linear-gradient(135deg, #1976d2 70%, #43e97b 100%) !important;
        color: #fff !important;
        font-size: 1.15em !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 2em !important;
        padding: 0.85em 2.3em !important;
        margin: 0.2em 0.2em 0.2em 0 !important;
        box-shadow: 0 4px 16px #1976d244 !important;
        transition: background 0.2s, color 0.2s, box-shadow 0.2s;
        outline: none !important;
        letter-spacing: 0.02em;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, .mp-send-btn button:hover,
    .stButton>button:focus, .stDownloadButton>button:focus, .mp-send-btn button:focus {
        background: linear-gradient(135deg, #43e97b 70%, #1976d2 100%) !important;
        color: #fff !important;
        box-shadow: 0 6px 24px #43e97b33 !important;
    }
    .mp-btn-row { display: flex; gap: 1.2em; margin-top: 1.2em; margin-bottom: 0.5em; }
    </style>
    ''', unsafe_allow_html=True)

# --- Enhanced Animated Gradient Background with SVG Overlay (Final Fix) ---
st.markdown("""
    <style>
    /* Remove all backgrounds from containers except .stApp */
    html, body, .main, .block-container, footer, [data-testid="stFooter"], .reportview-container, .css-1outpf7, .css-18ni7ap {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        min-height: 100vh !important;
        height: 100vh !important;
    }
    html, body {
        background-color: #f6fff8 !important; /* fallback color */
    }
    /* Ensure .stApp covers the full viewport */
    .stApp {
        position: relative;
        min-height: 100vh !important;
        height: 100vh !important;
        width: 100vw !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        overflow-x: hidden !important;
    }
    /* Animated gradient background on .stApp for full viewport coverage */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -2;
        background: linear-gradient(120deg, #f6fff8 0%, #e3f6fc 40%, #fafdff 80%, #b2dfdb 100%);
        background-size: 200% 200%;
        animation: gradientMove 16s ease-in-out infinite alternate;
        opacity: 0.92;
        pointer-events: none;
    }
    /* SVG overlay with medical icons (very subtle, low opacity) */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        pointer-events: none;
        opacity: 0.10;
        background: url('data:image/svg+xml;utf8,<svg width=\"100%\" height=\"100%\" xmlns=\"http://www.w3.org/2000/svg\"><g fill=\"%231976d2\" fill-opacity=\"0.12\"><rect x=\"10%\" y=\"10%\" width=\"60\" height=\"60\" rx=\"16\"/><circle cx=\"80%\" cy=\"20%\" r=\"32\"/><rect x=\"70%\" y=\"70%\" width=\"48\" height=\"48\" rx=\"12\"/><path d=\"M180 120 h40 v10 h-40z\"/><path d=\"M200 100 v40 h-10 v-40z\"/><ellipse cx=\"30%\" cy=\"80%\" rx=\"28\" ry=\"16\"/><rect x=\"60%\" y=\"50%\" width=\"36\" height=\"36\" rx=\"8\"/></g></svg>');
        background-repeat: no-repeat;
        background-size: 400px 400px, 300px 300px, 200px 200px;
        background-position: 10% 10%, 80% 20%, 70% 70%, 30% 80%, 60% 50%;
    }
    /* Hide Streamlit's default footer text if present */
    footer, [data-testid="stFooter"] { display: none !important; }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
""", unsafe_allow_html=True)