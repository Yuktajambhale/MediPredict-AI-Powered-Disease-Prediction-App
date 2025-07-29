import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches

# Load disease knowledgebase for precautions, doctor type, and symptom clusters
kb = pd.read_csv('data/disease_centric_knowledgebase_with_doctor.csv')
disease_info = {k.strip().lower(): v for k, v in kb.groupby('disease').agg({'precaution': 'first', 'doctor_type': 'first', 'symptom_list': 'first'}).to_dict('index').items()}

# Build dynamic disease clusters: each disease maps to a list of symptoms (single cluster)
disease_clusters = {}
for _, row in kb.iterrows():
    disease = row['disease'].strip().lower()
    symptoms = [s.strip().lower() for s in row['symptom_list'].split(',') if s.strip()]
    if symptoms:
        disease_clusters[disease] = [symptoms]  # single cluster per disease

# Build a mapping from disease to set of symptoms for rule-based
kb['symptom_set'] = kb['symptom_list'].apply(lambda x: set(s.strip().lower() for s in x.split(',')))
disease_symptom_map = dict(zip(kb['disease'].str.strip().str.lower(), kb['symptom_set']))

# Load the best Random Forest model and label encoder
model = joblib.load('models/final_symptom_checker_model_rf.pkl')
le = joblib.load('models/label_encoder_rf.pkl')

# Load the exact feature list used for model training
with open('features/rf_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f if line.strip()]

# Synonym map for common symptom variants
synonym_map = {
    'running nose': 'runny nose', 'runny nose': 'runny nose', 'high temperature': 'fever', 'temperature': 'fever',
    'tiredness': 'fatigue', 'fatigue': 'fatigue', 'sore throat': 'sore throat', 'throat pain': 'sore throat',
    'throat irritation': 'sore throat', 'head ache': 'headache', 'head pain': 'headache', 'migraine': 'headache',
    'body pain': 'body pain', 'body ache': 'body pain', 'muscle ache': 'muscle pain', 'muscle aches': 'muscle pain',
    'muscle pains': 'muscle pain', 'muscle cramp': 'muscle pain', 'muscle cramps': 'muscle pain',
    'chest discomfort': 'chest pain', 'chest tightness': 'chest pain', 'stomach pain': 'abdominal pain',
    'belly pain': 'abdominal pain', 'abdominal ache': 'abdominal pain', 'loose motion': 'diarrhoea',
    'loose motions': 'diarrhoea', 'loose stools': 'diarrhoea', 'vomitting': 'vomiting', 'throwing up': 'vomiting',
    'nausea': 'nausea', 'feeling sick': 'nausea', 'high fever': 'high fever', 'fever': 'fever', 'coughing': 'cough',
    'cough': 'cough', 'cold': 'cold', 'sneezing': 'continuous sneezing', 'continuous sneezing': 'continuous sneezing',
    'joint pain': 'joint pain', 'joint ache': 'joint pain', 'swelling': 'swelling', 'swollen': 'swelling',
    'rash': 'skin rash', 'skin rash': 'skin rash', 'itching': 'itching', 'itchy': 'itching', 'weight loss': 'weight loss',
    'weight gain': 'weight gain', 'loss of appetite': 'loss of appetite', 'appetite loss': 'loss of appetite',
    'loss of smell': 'loss of smell', 'loss of taste': 'loss of taste', 'difficulty breathing': 'breathlessness',
    'shortness of breath': 'breathlessness', 'breathlessness': 'breathlessness', 'palpitations': 'palpitations',
    'irregular heartbeat': 'palpitations', 'dizziness': 'dizziness', 'vertigo': '(vertigo) paroymsal  positional vertigo',
    'cramps': 'cramps', 'menstrual cramps': 'cramps', 'period pain': 'cramps', 'periods': 'cramps',
    'back pain': 'back pain', 'lower back pain': 'back pain', 'yellow skin': 'yellowish skin', 'yellowish skin': 'yellowish skin',
    'yellowing skin': 'yellowish skin', 'yellowing eye': 'yellowish eyes', 'yellow eyes': 'yellowish eyes', 'yellowish eyes': 'yellowish eyes',
    'dark urine': 'dark urine', 'pale stools': 'pale stools', 'excessive hunger': 'excessive hunger', 'hunger': 'excessive hunger',
    'diarrhea': 'diarrhoea', 'loose stool': 'diarrhoea', 'chills': 'chills', 'sweating': 'sweating', 'vomiting': 'vomiting', 'palpitation': 'palpitations',
}

# --- Normalize cluster symptoms at load time ---ness': 'dizziness', 'vertigo': '(vertigo) paroymsal  positional vertigo',
def normalize_symptom(s):
    s = s.strip().lower()
    s = synonym_map.get(s, s)
    # Map to feature name if possible
    for feat in selected_features:
        if s == feat.replace('_', ' '):
            return feat.replace('_', ' ')
    return s

# Normalize cluster symptoms at load time
for disease, clusters in disease_clusters.items():
    norm_clusters = []
    for cluster in clusters:
        norm_cluster = [normalize_symptom(s) for s in cluster]
        norm_clusters.append(norm_cluster)
    disease_clusters[disease] = norm_clusters

for disease in disease_symptom_map:
    disease_symptom_map[disease] = set([normalize_symptom(s) for s in disease_symptom_map[disease]])

print("\nWelcome to the AI-Powered Symptom Checker Chatbot!")
print("Choose mode: [1] ML-based  [2] Rule-based  [3] Hybrid (both)")
mode = input("Enter 1, 2, or 3: ").strip()
if mode not in {'1', '2', '3'}:
    print("Invalid mode. Defaulting to ML-based.")
    mode = '1'
print("\nType your symptoms in a sentence (e.g., 'I have fever, headache, and cough'). Type 'exit' to quit.\n")

while True:
    user_input = input("\nDescribe your symptoms (comma separated, or sentence): ").strip()
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    # Normalize and map user symptoms
    user_symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    found = set()
    debug_lines = []
    for s in user_symptoms:
        mapped = normalize_symptom(s)
        debug_lines.append(f"Input: '{s}' -> Mapped: '{mapped}'")
        if mapped:
            found.add(mapped)
    print("\nSymptom Mapping Debug:")
    for line in debug_lines:
        print(line)
    mapped_set = set(found)
    # --- Suggest missing symptoms from up to 3 relevant clusters, even if 2+ symptoms matched ---
    cluster_matches = []
    for disease, clusters in disease_clusters.items():
        for cluster in clusters:
            present = [s for s in cluster if s in mapped_set]
            missing = [s for s in cluster if s not in mapped_set]
            if present:
                cluster_matches.append((len(present), disease, present, missing))
    cluster_matches.sort(reverse=True, key=lambda x: x[0])
    # --- Always suggest missing symptoms from up to 3 relevant clusters, one at a time ---
    cluster_idx = 0
    added_any = False
    while cluster_idx < min(3, len(cluster_matches)):
        _, disease, present, missing = cluster_matches[cluster_idx]
        if missing:
            print(f"You mentioned symptoms related to {disease.title()}: {', '.join(present)}.")
            print(f"Are you also experiencing: {', '.join(missing)}? (Type yes/no or list symptoms)")
            response = input(f"Add missing symptoms for {disease.title()} (comma separated): ").strip().lower()
            if response in ['yes', 'y']:
                for s in missing:
                    mapped = normalize_symptom(s)
                    if mapped:
                        found.add(mapped)
                print(f"Thank you! Adding symptoms: {', '.join(missing)}.")
                added_any = True
                break  # Immediately proceed to severity after adding symptoms for a cluster
            elif response not in ['no', 'n'] and response:
                entered_syms = [r.strip() for r in response.split(',') if r.strip()]
                for r in entered_syms:
                    mapped = normalize_symptom(r)
                    if mapped:
                        found.add(mapped)
                        print(f"Adding symptom: {r}")
        cluster_idx += 1
    # After all suggestions, or after a 'yes', always proceed to severity input for all recognized symptoms
    if found:
        symptom_severity = {}
        for s in found:
            while True:
                sev = input(f"On a scale of 1 (mild) to 5 (severe), how severe is your {s.replace('_', ' ')}? ").strip()
                try:
                    sev = int(sev)
                    if 1 <= sev <= 5:
                        symptom_severity[s] = sev
                        break
                    else:
                        print("Please enter a number from 1 to 5.")
                except ValueError:
                    print("Please enter a valid integer.")
        print("\nSymptom severity summary:")
        for s, v in symptom_severity.items():
            print(f"- {s.replace('_', ' ').capitalize()}: {v}")
        # ...existing prediction and output logic...
    else:
        print("Sorry, I could not recognize any symptoms. Please try again.")

# Add 'yellowish eyes' to jaundice cluster if not present
for disease, clusters in disease_clusters.items():
    if disease == 'jaundice':
        for cluster in clusters:
            if 'yellowish eyes' not in cluster:
                cluster.append('yellowish eyes')

# Add/override precaution for covid and parkinson's disease
if 'covid' in disease_info:
    disease_info['covid']['precaution'] = 'Isolate, wear a mask, monitor oxygen, consult doctor if breathless'
if "parkinson's disease" in disease_info:
    disease_info["parkinson's disease"]['precaution'] = 'Take medications as prescribed, regular exercise, fall prevention, consult neurologist'

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

# Display disease information with precautions and doctor type
print("\nDisease Information:")
for disease, info in disease_info.items():
    print(f"\nDisease: {disease.title()}")
    precaution = info.get('precaution', '').strip()
    doctor = info.get('doctor_type', 'N/A')
    if not precaution:
        precaution = 'No specific precautions available. Please consult a healthcare professional.'
    print(f"   Precautions: {precaution}")
    print(f"   Recommended Doctor Type: {doctor}")