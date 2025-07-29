import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import difflib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import streamlit as st

# --- CONFIG ---
DATA_PATH = 'outputs/p.csv'  # Patient-centric dataset
MODEL_PATH = 'models/model.pkl'
FEATURES_PATH = 'features/rf_features.txt'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'

# --- Load Data ---
def load_data():
    df = pd.read_csv(DATA_PATH)
    with open(FEATURES_PATH) as f:
        features = [line.strip() for line in f if line.strip()]
    return df, features

# --- Symptom Mapping and Fuzzy Matching ---
def get_symptom_map(features):
    # Map synonyms and fuzzy matches to feature names
    synonym_map = {
        'tiredness': 'fatigue', 'sore throat': 'throat_irritation', 'throat pain': 'throat_irritation',
        'head ache': 'headache', 'head pain': 'headache', 'runny nose': 'runny_nose',
        'stomach pain': 'abdominal_pain', 'belly pain': 'abdominal_pain', 'loose motion': 'diarrhoea',
        'vomitting': 'vomiting', 'throwing up': 'vomiting', 'diarrhea': 'diarrhoea',
        'weight loss': 'weight_loss', 'weight gain': 'weight_gain', 'loss of appetite': 'loss_of_appetite',
        'loss of smell': 'loss_of_smell', 'loss of taste': 'loss_of_taste',
        'difficulty breathing': 'breathlessness',
        'shortness of breath': 'shortness_of_breath',
        'palpitations': 'palpitations', 'joint pain': 'joint_pain', 'joint ache': 'joint_pain',
        'muscle ache': 'muscle_pain', 'muscle pain': 'muscle_pain', 'chest pain': 'chest_pain',
        'rash': 'skin_rash', 'itching': 'itching', 'itchy': 'itching',
        'high temperature': 'high_fever', 'fever': 'high_fever', 'cold': 'runny_nose',
        'sneezing': 'continuous_sneezing', 'cramps': 'cramps', 'back pain': 'back_pain',
        'yellow skin': 'yellowish_skin', 'yellowing skin': 'yellowish_skin', 'yellow eyes': 'yellowish_eyes',
        'yellowing eye': 'yellowish_eyes', 'yellowish eyes': 'yellowish_eyes', 'dark urine': 'dark_urine',
        'pale stools': 'pale_stools', 'fatigue': 'fatigue', 'chills': 'chills', 'sweating': 'sweating',
        'nausea': 'nausea', 'dizziness': 'dizziness', 'migraine': 'headache',
        # Expanded mappings for all cluster symptoms with underscores
        'abdominal pain': 'abdominal_pain',
        'chest tightness': 'chest_tightness',
        'body ache': 'body_ache',
        'skin rash': 'skin_rash',
        'high fever': 'high_fever',
        'mild fever': 'mild_fever',
        'chronic cough': 'chronic_cough',
        'lower abdominal pain': 'lower_abdominal_pain',
        'pain in left arm': 'pain_in_left_arm',
        'swelling of hands and feet': 'swelling_of_hands_and_feet',
        'hearing loss': 'hearing_loss',
        'eye pain': 'eye_pain',
        'blurred vision': 'blurred_and_distorted_vision',
        'neck stiffness': 'neck_stiffness',
        'facial droop': 'facial_droop',
        'speech difficulty': 'speech_difficulty',
        'loss of balance': 'loss_of_balance',
        'sleep disturbance': 'sleep_disturbance',
        'appetite change': 'appetite_change',
        'painful periods': 'painful_periods',
        'pain during intercourse': 'pain_during_intercourse',
        'urine changes': 'urine_changes',
        'chest discomfort': 'chest_discomfort',
        'morning stiffness': 'morning_stiffness',
        'muscle weakness': 'muscle_weakness',
        'muscle rigidity': 'muscle_rigidity',
        'pain crises': 'pain_crises',
        'scaly patches': 'scaly_patches',
        'staring spells': 'staring_spells',
        'tearing': 'tearing',
        'gritty sensation': 'gritty_sensation',
        'cloudy urine': 'cloudy_urine',
        'burning urination': 'burning_urination',
        'pelvic pain': 'pelvic_pain',
        'pain around ear': 'pain_around_ear',
        'rectal bleeding': 'rectal_bleeding',
        'leg pain': 'leg_pain',
        'lower back pain': 'lower_back_pain',
        'throat irritation': 'throat_irritation',
        'swelling': 'swelling',
        'swelling joints': 'swelling_joints',
        'swelling of stomach': 'swelling_of_stomach',
        'swollen blood vessels': 'swollen_blood_vessels',
        'swollen extremeties': 'swollen_extremeties',
        'swollen legs': 'swollen_legs',
        'joint swelling': 'swelling_joints',
        'joint stiffness': 'stiffness',
        'joint redness': 'redness',
        'joint heat': 'heat',
        'joint morning stiffness': 'morning_stiffness',
        'joint reduced range of motion': 'reduced_range_of_motion',
        'photosensitivity': 'photosensitivity',
        'weight gain': 'weight_gain',
        'weight loss': 'weight_loss',
        'irregular periods': 'irregular_periods',
        'excess hair growth': 'excess_hair_growth',
        'memory loss': 'memory_loss',
        'difficulty communicating': 'difficulty_communicating',
        'mood changes': 'mood_changes',
        'disorientation': 'disorientation',
        'numbness': 'numbness',
        'tingling': 'tingling',
        'vision problems': 'vision_problems',
        'rapid pulse': 'rapid_pulse',
        'drooping': 'drooping',
        'loss of taste': 'loss_of_taste',
        'tearing': 'tearing',
        'pain around ear': 'pain_around_ear',
        'rectal bleeding': 'rectal_bleeding',
        'leg pain': 'leg_pain',
        'lower back pain': 'lower_back_pain',
        # Covid symptom variants
        'fever': 'high_fever',
        'cough': 'cough',
        'breathlessness': 'breathlessness',
        'shortness of breath': 'breathlessness',
        'difficulty breathing': 'breathlessness',
    }
    return synonym_map

def map_symptom(user_symptom, features, synonym_map):
    # Try direct, synonym, and fuzzy match
    s = user_symptom.strip().lower().replace(' ', '_')
    if s in features:
        return s
    if user_symptom in synonym_map:
        mapped = synonym_map[user_symptom]
        if mapped in features:
            return mapped
    # Fuzzy match
    match = difflib.get_close_matches(s, features, n=1, cutoff=0.8)
    if match:
        return match[0]
    return None

# --- Emergency Rule-Based Checks ---
def emergency_check(symptom_severity):
    # Ensure None values are treated as 0
    chest_pain = symptom_severity.get('chest_pain', 0)
    breathlessness = symptom_severity.get('breathlessness', 0)
    try:
        chest_pain = int(chest_pain) if chest_pain is not None else 0
    except Exception:
        chest_pain = 0
    try:
        breathlessness = int(breathlessness) if breathlessness is not None else 0
    except Exception:
        breathlessness = 0
    if chest_pain >= 3 and breathlessness >= 3:
        return True, 'Emergency alert: Chest pain and breathlessness detected. Please seek immediate medical attention!'
    return False, ''

# --- Input Handling ---
def get_user_input(features, synonym_map):
    print('Enter your symptoms (comma separated):')
    user_input = input('> ').strip().lower()
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    symptom_severity = {}
    for s in user_symptoms:
        mapped = map_symptom(s, features, synonym_map)
        if not mapped:
            # Suggest possible matches
            suggestions = difflib.get_close_matches(s.replace(' ', '_'), features, n=3, cutoff=0.6)
            print(f"Unrecognized symptom: '{s}'. Did you mean: {', '.join(suggestions) if suggestions else 'No suggestions'}?")
            continue
        # Ask for severity
        while True:
            try:
                sev = int(input(f"On a scale of 1 (mild) to 5 (severe), how severe is your '{s}'? "))
                if 1 <= sev <= 5:
                    break
                else:
                    print('Please enter a number from 1 to 5.')
            except ValueError:
                print('Please enter a valid integer.')
        symptom_severity[mapped] = sev
    return symptom_severity

def get_user_input_conversational(features, synonym_map):
    print("Hello! I'm your AI medical assistant. Please describe your symptoms in a sentence or list (e.g., 'I have fever, cough, and sore throat').")
    user_input = input('You: ').strip().lower()
    # Build a set of all possible symptom phrases (features and synonyms)
    all_phrases = set(features)
    all_phrases.update(synonym_map.keys())
    all_phrases.update(synonym_map.values())
    found = set()
    print("\n--- Symptom Mapping Debug ---")
    for phrase in all_phrases:
        if phrase.replace('_', ' ') in user_input:
            mapped = synonym_map.get(phrase, phrase)
            if mapped in features:
                print(f"Input phrase '{phrase.replace('_', ' ')}' mapped to feature '{mapped}' (recognized)")
                found.add((phrase, mapped))
            else:
                print(f"Input phrase '{phrase.replace('_', ' ')}' mapped to '{mapped}' but NOT in features (NOT recognized)")
    if not found:
        # --- Suggest possible symptoms using fuzzy matching ---
        print('Sorry, I could not recognize any symptoms directly.')
        input_phrases = [p.strip() for p in user_input.replace(',', ' ').split() if p.strip()]
        suggestions = set()
        for phrase in input_phrases:
            matches = difflib.get_close_matches(phrase.replace(' ', '_'), features, n=3, cutoff=0.6)
            matches += difflib.get_close_matches(phrase, synonym_map.keys(), n=2, cutoff=0.6)
            matches += difflib.get_close_matches(phrase, synonym_map.values(), n=2, cutoff=0.6)
            for m in matches:
                mapped = synonym_map.get(m, m)
                if mapped in features:
                    suggestions.add(mapped)
        if suggestions:
            print(f"Did you mean any of these symptoms? {', '.join(sorted(suggestions))}")
            print("Please enter the symptoms you are experiencing from the above list (comma separated):")
            selected = input('You: ').strip().lower()
            selected_symptoms = [s.strip().replace(' ', '_') for s in selected.split(',') if s.strip()]
            found = set()
            for s in selected_symptoms:
                if s in features:
                    found.add((s, s))
                elif s in synonym_map and synonym_map[s] in features:
                    found.add((s, synonym_map[s]))
            if not found:
                print('No valid symptoms selected. Suggesting possible disease clusters...')
        else:
            print('No close matches found. Suggesting possible disease clusters...')
        # --- Suggest up to 3 clusters ---
        suggested_clusters = 0
        print('\nHere are some common disease clusters and their main symptoms:')
        for c in CLUSTERS[:3]:
            print(f"- {c['disease'].title()}: {', '.join(c['cluster'])}")
        print("\nPlease enter any symptoms you are experiencing from the above clusters (comma separated):")
        selected = input('You: ').strip().lower()
        selected_symptoms = [s.strip().replace(' ', '_') for s in selected.split(',') if s.strip()]
        found = set()
        for s in selected_symptoms:
            if s in features:
                found.add((s, s))
            elif s in synonym_map and synonym_map[s] in features:
                found.add((s, synonym_map[s]))
        if not found:
            print('No valid symptoms selected. Please try again.')
            return {}
    print("\nI understood these symptoms:")
    for orig, mapped in found:
        print(f"- {orig} (mapped to: {mapped})")
    # --- Suggest missing symptoms from up to 3 most relevant clusters ---
    mapped_set = set(m for _, m in found)
    # --- Prioritize clusters with highly specific symptoms (e.g., jaundice/hepatitis) ---
    jaundice_specific = {'yellowish_skin', 'yellowish_eyes', 'dark_urine', 'pale_stools', 'abdominal_pain'}
    n_jaundice = len(jaundice_specific & mapped_set)
    cluster_matches = []
    for c in CLUSTERS:
        present = [s for s in c['cluster'] if s in mapped_set]
        missing = [s for s in c['cluster'] if s not in mapped_set]
        # Boost score for jaundice/hepatitis clusters if at least 2 specific symptoms present
        boost = 0
        if c['disease'] in ['jaundice', 'hepatitis'] and n_jaundice >= 2:
            boost = 100  # Large boost to ensure top ranking
        cluster_matches.append((len(present) + boost, c, present, missing))
    # Sort clusters by most matches (with boost)
    cluster_matches.sort(reverse=True, key=lambda x: x[0])
    suggested = 0
    for _, c, present, missing in cluster_matches[:3]:
        if missing:
            print(f"\nYou mentioned symptoms related to {c['disease']}: {', '.join(present)}.")
            print(f"Are you also experiencing: {', '.join(missing)}? (yes/no or list symptoms)")
            followup = input('You: ').strip().lower()
            if followup in ['yes', 'y']:
                for s in missing:
                    mapped_set.add(s)
                print(f"Thank you! Adding symptoms: {', '.join(missing)}.")
                suggested += 1
                if suggested >= 3:
                    break
            elif followup not in ['no', 'n'] and followup:
                # Add only the symptoms mentioned by the user
                added = []
                for s in missing:
                    # Accept comma/space separated, robust match
                    if s.replace('_', ' ') in followup or s in followup.replace(' ', '_').split(',') or s in followup.split():
                        mapped_set.add(s)
                        added.append(s)
                        print(f"Adding symptom: {s}")
                if added:
                    print(f"Thank you! Adding symptoms: {', '.join(added)}.")
                suggested += 1
                if suggested >= 3:
                    break
            else:
                suggested += 1
                if suggested >= 3:
                    break
    # Ask for severity for each unique mapped symptom
    symptom_severity = {}
    for mapped in sorted(mapped_set):
        while True:
            try:
                sev = int(input(f"On a scale of 1 (mild) to 5 (severe), how severe is your '{mapped.replace('_', ' ')}'? "))
                if 1 <= sev <= 5:
                    break
                else:
                    print('Please enter a number from 1 to 5.')
            except ValueError:
                print('Please enter a valid integer.')
        symptom_severity[mapped] = sev
    print(f"\nFinal recognized symptoms for prediction: {list(symptom_severity.keys())}")
    return symptom_severity

# --- Prediction ---
def predict(symptom_severity, features, model, le):
    # Debug: Show the mapping from symptom_severity to features and the input vector
    st.write('DEBUG: predict() called with symptom_severity:', symptom_severity)
    X = np.zeros((1, len(features)))
    for i, feat in enumerate(features):
        X[0, i] = symptom_severity.get(feat, 0)
    st.write('DEBUG: Model input vector X:', X)
    probs = model.predict_proba(X)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)
    top3_scores = probs[top3_idx]
    st.write('DEBUG: Model top3 diseases:', list(zip(top3_diseases, top3_scores)))
    return list(zip(top3_diseases, top3_scores))

# --- Output ---
def show_output(top3, disease_info):
    print('\nTop 3 probable diseases:')
    for i, (disease, score) in enumerate(top3, 1):
        print(f"{i}. {disease} (Confidence: {score:.2f})")
        info = disease_info.get(disease.strip().lower(), {})
        precaution = info.get('precaution', '').strip()
        doctor = info.get('doctor_type', 'N/A')
        if not precaution:
            precaution = 'No specific precautions available. Please consult a healthcare professional.'
        print(f"   Precautions: {precaution}")
        if doctor != 'N/A':
            print(f"   Recommended Doctor: {doctor}")
    print("\nNote: This tool is for informational purposes only. Please consult a healthcare professional for medical advice.\n")

# --- Disease Clusters (shared for suggestions and hybrid override) ---
CLUSTERS = [
    # Common cold
    {'cluster': ['cough', 'runny_nose', 'sore_throat', 'throat_irritation', 'sneezing'], 'disease': 'common cold'},
    # Flu
    {'cluster': ['fever', 'body_ache', 'fatigue', 'chills', 'headache', 'cough'], 'disease': 'flu'},
    # Covid
    {'cluster': ['fever', 'cough', 'loss_of_taste', 'loss_of_smell', 'breathlessness'], 'disease': 'covid'},
    # Diabetes
    {'cluster': ['excessive_thirst', 'frequent_urination', 'fatigue', 'blurred_and_distorted_vision'], 'disease': 'diabetes'},
    # Hypertension
    {'cluster': ['headache', 'shortness_of_breath', 'nosebleeds', 'flushing'], 'disease': 'hypertension'},
    # Asthma
    {'cluster': ['cough', 'wheezing', 'shortness_of_breath', 'chest_tightness'], 'disease': 'asthma'},
    # Allergy
    {'cluster': ['sneezing', 'itching', 'runny_nose', 'cough', 'wheezing', 'skin_rash'], 'disease': 'allergy'},
    # Arthritis
    {'cluster': ['joint_pain', 'stiffness', 'swelling', 'reduced_range_of_motion'], 'disease': 'arthritis'},
    # Migraine
    {'cluster': ['headache', 'nausea', 'sensitivity_to_light', 'aura', 'throbbing_pain'], 'disease': 'migraine'},
    # Pneumonia
    {'cluster': ['cough', 'fever', 'shortness_of_breath', 'chest_pain', 'fatigue'], 'disease': 'pneumonia'},
    # Tuberculosis
    {'cluster': ['cough', 'fever', 'night_sweats', 'weight_loss', 'fatigue'], 'disease': 'tuberculosis'},
    # Whooping cough
    {'cluster': ['cough', 'runny_nose', 'fever', 'vomiting', 'exhaustion'], 'disease': 'whooping cough'},
    # Jaundice
    {'cluster': ['yellowish_skin', 'yellowish_eyes', 'dark_urine', 'pale_stools', 'fatigue', 'yellowish_eyes'], 'disease': 'jaundice'},
    # Dengue
    {'cluster': ['fever', 'headache', 'muscle_pain', 'joint_pain', 'skin_rash', 'nausea'], 'disease': 'dengue'},
    # Malaria
    {'cluster': ['fever', 'chills', 'sweating', 'headache', 'nausea', 'vomiting'], 'disease': 'malaria'},
    # Typhoid
    {'cluster': ['fever', 'abdominal_pain', 'headache', 'diarrhoea', 'fatigue'], 'disease': 'typhoid'},
    # Emergency
    {'cluster': ['chest_pain', 'breathlessness'], 'disease': 'emergency'},
    # Chickenpox
    {'cluster': ['fever', 'skin_rash', 'itching', 'fatigue', 'loss_of_appetite'], 'disease': 'chickenpox'},
    # Measles
    {'cluster': ['fever', 'skin_rash', 'cough', 'runny_nose', 'conjunctivitis'], 'disease': 'measles'},
    # Hepatitis
    {'cluster': ['fatigue', 'yellowish_skin', 'yellowish_eyes', 'dark_urine', 'abdominal_pain'], 'disease': 'hepatitis'},
    # Heart attack
    {'cluster': ['chest_pain', 'breathlessness', 'sweating', 'nausea', 'pain_in_left_arm'], 'disease': 'heart attack'},
    # Stroke
    {'cluster': ['sudden_weakness', 'facial_droop', 'speech_difficulty', 'confusion', 'loss_of_balance'], 'disease': 'stroke'},
    # Gastroenteritis
    {'cluster': ['diarrhoea', 'vomiting', 'abdominal_pain', 'fever', 'nausea'], 'disease': 'gastroenteritis'},
    # Sinusitis
    {'cluster': ['headache', 'facial_pain', 'runny_nose', 'nasal_congestion', 'fever'], 'disease': 'sinusitis'},
    # Bronchitis
    {'cluster': ['cough', 'chest_discomfort', 'fatigue', 'shortness_of_breath', 'mild_fever'], 'disease': 'bronchitis'},
    # Depression
    {'cluster': ['persistent_sadness', 'loss_of_interest', 'fatigue', 'sleep_disturbance', 'appetite_change'], 'disease': 'depression'},
    # Anxiety
    {'cluster': ['excessive_worry', 'restlessness', 'fatigue', 'irritability', 'sleep_disturbance'], 'disease': 'anxiety'},
    # Anemia
    {'cluster': ['fatigue', 'pallor', 'shortness_of_breath', 'dizziness', 'palpitations'], 'disease': 'anemia'},
    # Hypothyroidism
    {'cluster': ['fatigue', 'weight_gain', 'cold_intolerance', 'dry_skin', 'constipation'], 'disease': 'hypothyroidism'},
    # Hyperthyroidism
    {'cluster': ['weight_loss', 'palpitations', 'heat_intolerance', 'irritability', 'tremor'], 'disease': 'hyperthyroidism'},
    # Urinary tract infection (UTI)
    {'cluster': ['burning_urination', 'frequent_urination', 'lower_abdominal_pain', 'cloudy_urine', 'fever'], 'disease': 'urinary tract infection'},
    # Appendicitis
    {'cluster': ['abdominal_pain', 'loss_of_appetite', 'nausea', 'vomiting', 'fever'], 'disease': 'appendicitis'},
    # Meningitis
    {'cluster': ['fever', 'headache', 'neck_stiffness', 'photophobia', 'nausea'], 'disease': 'meningitis'},
    # Otitis media (ear infection)
    {'cluster': ['ear_pain', 'fever', 'hearing_loss', 'irritability', 'fluid_discharge'], 'disease': 'otitis media'},
    # Conjunctivitis
    {'cluster': ['eye_redness', 'itching', 'tearing', 'discharge', 'gritty_sensation'], 'disease': 'conjunctivitis'},
    # GERD (acid reflux)
    {'cluster': ['heartburn', 'regurgitation', 'chest_pain', 'sore_throat', 'chronic_cough'], 'disease': 'GERD'},
    # Peptic ulcer
    {'cluster': ['abdominal_pain', 'nausea', 'vomiting', 'bloating', 'loss_of_appetite'], 'disease': 'peptic ulcer'},
    # Polycystic Ovary Syndrome (PCOS)
    {'cluster': ['irregular_periods', 'weight_gain', 'acne', 'excess_hair_growth', 'infertility'], 'disease': 'PCOS'},
    # Chronic kidney disease
    {'cluster': ['fatigue', 'swelling', 'loss_of_appetite', 'nausea', 'urine_changes'], 'disease': 'chronic kidney disease'},
    # Chronic obstructive pulmonary disease (COPD)
    {'cluster': ['chronic_cough', 'shortness_of_breath', 'wheezing', 'chest_tightness', 'fatigue'], 'disease': 'COPD'},
    # Epilepsy
    {'cluster': ['seizures', 'confusion', 'loss_of_consciousness', 'staring_spells', 'muscle_rigidity'], 'disease': 'epilepsy'},
    # Gout
    {'cluster': ['joint_pain', 'swelling', 'redness', 'heat', 'stiffness'], 'disease': 'gout'},
    # Psoriasis
    {'cluster': ['skin_rash', 'itching', 'scaly_patches', 'joint_pain', 'redness'], 'disease': 'psoriasis'},
    # Celiac disease
    {'cluster': ['abdominal_pain', 'diarrhoea', 'bloating', 'weight_loss', 'fatigue'], 'disease': 'celiac disease'},
    # Sickle cell anemia
    {'cluster': ['fatigue', 'pain_crises', 'anemia', 'swelling_of_hands_and_feet', 'jaundice'], 'disease': 'sickle cell anemia'},
    # Lupus
    {'cluster': ['joint_pain', 'skin_rash', 'fatigue', 'fever', 'photosensitivity'], 'disease': 'lupus'},
    # Rheumatoid arthritis
    {'cluster': ['joint_pain', 'swelling', 'morning_stiffness', 'fatigue', 'weight_loss'], 'disease': 'rheumatoid arthritis'},
    # Glaucoma
    {'cluster': ['eye_pain', 'blurred_vision', 'halos_around_lights', 'nausea', 'vomiting'], 'disease': 'glaucoma'},
    # Parkinson's disease
    {'cluster': ['tremor', 'slowed_movement', 'rigid_muscles', 'impaired_posture', 'speech_changes'], 'disease': "parkinson's disease"},
    # Alzheimer's disease
    {'cluster': ['memory_loss', 'confusion', 'difficulty_communicating', 'mood_changes', 'disorientation'], 'disease': ",alzheimer's disease"},
    # Multiple sclerosis
    {'cluster': ['numbness', 'tingling', 'vision_problems', 'muscle_weakness', 'fatigue'], 'disease': 'multiple sclerosis'},
    # Pancreatitis
    {'cluster': ['abdominal_pain', 'nausea', 'vomiting', 'fever', 'rapid_pulse'], 'disease': 'pancreatitis'},
    # Endometriosis
    {'cluster': ['pelvic_pain', 'painful_periods', 'pain_during_intercourse', 'infertility', 'fatigue'], 'disease': 'endometriosis'},
    # Varicella (shingles)
    {'cluster': ['skin_rash', 'pain', 'burning', 'tingling', 'blisters'], 'disease': 'shingles'},
    # Bell's palsy
    {'cluster': ['facial_weakness', 'drooping', 'loss_of_taste', 'tearing', 'pain_around_ear'], 'disease': "bell's palsy"},
    # Hemorrhoids
    {'cluster': ['rectal_bleeding', 'pain', 'itching', 'swelling', 'discomfort'], 'disease': 'hemorrhoids'},
    # Sciatica
    {'cluster': ['lower_back_pain', 'leg_pain', 'numbness', 'tingling', 'weakness'], 'disease': 'sciatica'},
    # Add more as needed
]

# --- Main Training Script ---
def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print("Columns in dataset:", list(df.columns))
    # Use all symptom columns as features (exclude non-symptom columns)
    non_symptom_cols = {'patient_id', 'age', 'age_group', 'gender', 'disease', 'precaution', 'doctor_type', 'total_symptoms'}
    symptom_cols = [col for col in df.columns if col not in non_symptom_cols]
    X_bin = df[symptom_cols].copy()
    y = df['disease']
    features = list(X_bin.columns)
    # Save features to rf_features.txt
    with open(FEATURES_PATH, 'w') as f:
        for feat in features:
            f.write(f'{feat}\n')
    print("Features from file:", features)
    # Model training as before, but use X_bin.values as X
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X = X_bin.values
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []
    all_y_true = []
    all_y_pred = []
    print('--- Cross-validation results (5-fold) ---')
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f'Fold {fold}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}')
        print(classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))))
        print('Confusion matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('-'*40)
    print(f'Avg Acc={np.mean(accs):.3f} Prec={np.mean(precs):.3f} Rec={np.mean(recs):.3f} F1={np.mean(f1s):.3f}')

    # Per-class metrics (macro)
    print('\n--- Per-class metrics (all folds combined) ---')
    report = classification_report(all_y_true, all_y_pred, target_names=le.inverse_transform(np.unique(all_y_true)), output_dict=True, zero_division=0)
    print(f"{'Disease':30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Support':>7s}")
    for disease, stats in report.items():
        if disease in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{disease:30s} {stats['precision']:7.3f} {stats['recall']:7.3f} {stats['f1-score']:7.3f} {int(stats['support']):7d}")
    print(f"{'Macro avg':30s} {report['macro avg']['precision']:7.3f} {report['macro avg']['recall']:7.3f} {report['macro avg']['f1-score']:7.3f}")
    print(f"{'Weighted avg':30s} {report['weighted avg']['precision']:7.3f} {report['weighted avg']['recall']:7.3f} {report['weighted avg']['f1-score']:7.3f}")

    # Train final model on all data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y_enc)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f'Final model saved to {MODEL_PATH}')
    print(f'Label encoder saved to {LABEL_ENCODER_PATH}')

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print('\nTop 15 Feature Importances:')
    for feat, imp in feat_imp[:15]:
        print(f'{feat}: {imp:.4f}')
    # Save to file
    with open('outputs/feature_importances.txt', 'w') as f:
        for feat, imp in feat_imp:
            f.write(f'{feat}\t{imp}\n')
    print('Feature importances saved to outputs/feature_importances.txt')

    # Load disease info for output
    try:
        kb = pd.read_csv('data/disease_centric_knowledgebase_with_doctor.csv')
        disease_info = {k.strip().lower(): v for k, v in kb.groupby('disease').agg({'precaution': 'first', 'doctor_type': 'first'}).to_dict('index').items()}
        # Add/override precaution for covid and parkinson's disease
        if 'covid' in disease_info:
            disease_info['covid']['precaution'] = 'Isolate, wear a mask, monitor oxygen, consult doctor if breathless'
        if "parkinson's disease" in disease_info:
            disease_info["parkinson's disease"]['precaution'] = 'Take medications as prescribed, regular exercise, fall prevention, consult neurologist'
    except Exception:
        disease_info = {}

    synonym_map = get_symptom_map(features)
    print("\nWelcome to the conversational AI Symptom Checker!")
    while True:
        symptom_severity = get_user_input_conversational(features, synonym_map)
        if not symptom_severity:
            print('No valid symptoms entered. Please try again.')
            continue
        emerg, msg = emergency_check(symptom_severity)
        if emerg:
            print('\n' + msg)
            print("\nNote: This tool is for informational purposes only. Please consult a healthcare professional for medical advice.\n")
            continue
        # Hybrid logic: if all symptoms in a cluster are present, override prediction
        matched_disease = None
        for c in CLUSTERS:
            if all(s in symptom_severity for s in c['cluster']):
                matched_disease = c['disease']
                break
        if matched_disease:
            # Show this disease as top result with high confidence
            top3 = [(matched_disease, 0.99)]
            # Fill up with model predictions for the rest
            model_top3 = predict(symptom_severity, features, model, le)
            for d, s in model_top3:
                if d != matched_disease and len(top3) < 3:
                    top3.append((d, s))
        else:
            top3 = predict(symptom_severity, features, model, le)
        show_output(top3, disease_info)
        again = input('Would you like to check another case? (y/n): ').strip().lower()
        if again != 'y':
            print('Thank you for using the AI Symptom Checker. Stay healthy!')
            break

if __name__ == '__main__':
    main()
