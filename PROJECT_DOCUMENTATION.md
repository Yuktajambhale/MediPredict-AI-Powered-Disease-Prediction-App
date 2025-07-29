# MediPredict: AI-Powered Disease Prediction and Symptom Checker

## Introduction

MediPredict is an AI-powered platform designed to assist users in identifying probable diseases based on their symptoms. By leveraging machine learning and a curated medical knowledge base, MediPredict bridges the gap between symptom onset and professional medical advice, empowering users to make informed decisions about their health.

## Problem Statement

Early and accurate disease prediction is crucial for timely medical intervention and improved patient outcomes. However, many individuals lack access to immediate medical advice or struggle to interpret their symptoms. MediPredict aims to bridge this gap by providing an AI-powered, user-friendly platform that predicts probable diseases based on user-reported symptoms, suggests relevant symptom clusters, and offers precautionary advice and doctor recommendations.

## Tech Stack

- **Python 3:**  
  Chosen for its simplicity, readability, and rich ecosystem of data science and machine learning libraries.

- **Streamlit:**  
  Enables rapid development of interactive web applications with minimal code, making it easy for users to interact with the model.

- **scikit-learn (Random Forest Classifier):**  
  Provides robust, well-tested machine learning algorithms. Random Forest is used for its accuracy, interpretability, and ability to handle high-dimensional data.

- **pandas & numpy:**  
  Essential for efficient data manipulation, cleaning, and numerical operations.

- **joblib:**  
  Used for model persistence (saving/loading trained models) due to its efficiency with large numpy arrays.

- **fpdf:**  
  Allows exporting results as PDF, enhancing usability for users who want to save or share their predictions.

- **difflib (fuzzy matching) & re (regex):**  
  Used for robust symptom normalization, handling typos, synonyms, and variations in user input.

These technologies were selected to ensure the system is user-friendly, reliable, and easy to maintain or extend.

## Architecture

```
User (Web/Console)
   |
   v
[Streamlit App / Console Chatbot]
   |
   v
[Symptom Normalization & Fuzzy Matching]
   |
   v
[Symptom Cluster Suggestion Logic]
   |
   v
[Severity Input]
   |
   v
[Random Forest Model Prediction]
   |
   v
[Disease Info Lookup (Precautions, Doctor)]
   |
   v
[Results Display & Export]
```

- **src/app.py:** Streamlit web app (main user interface)
- **src/chatbot.py:** Console-based chatbot
- **src/train_model.py:** Model training, normalization, and core logic
- **models/:** Trained model and label encoder
- **features/:** List of model features (symptoms)
- **data/:** Datasets and disease knowledge base

## Workflow

1. **User Input:**
   - User enters symptoms (web or console)
   - Symptoms are normalized using synonym mapping and fuzzy matching
2. **Cluster Suggestion:**
   - The system suggests missing symptoms from relevant disease clusters
   - User can confirm or add more symptoms
3. **Severity Input:**
   - User rates severity (1-5) for each recognized symptom
4. **Prediction:**
   - Symptoms and severities are fed to a Random Forest model
   - Top 3 probable diseases are predicted
5. **Results & Advice:**
   - Precautions and doctor recommendations are displayed
   - Results can be exported as PDF or text (web app)

## Workflow (Detailed)

1. **User Input Collection**
   - User enters symptoms via the Streamlit web app or console chatbot.
   - Input is normalized: spelling corrections, synonym mapping, and fuzzy matching are applied to match known symptoms.

2. **Symptom Cluster Suggestion**
   - The system identifies relevant disease clusters based on initial symptoms.
   - Up to 3 clusters are suggested, prioritizing those with highly specific symptoms.
   - The user is interactively asked about missing symptoms from these clusters to improve diagnostic accuracy.

3. **Severity Input**
   - For each recognized symptom, the user rates its severity on a scale of 1 to 5.
   - This step ensures nuanced input for the prediction model.

4. **Prediction**
   - The normalized symptoms and their severities are encoded as model features.
   - The Random Forest classifier predicts the top 3 probable diseases.

5. **Results & Recommendations**
   - The app displays the predicted diseases, associated precautions, and recommended doctor types.
   - If no specific precaution is available, a user-friendly default message is shown.

6. **Export & Follow-up**
   - In the web app, users can export results as PDF or text.
   - The system encourages users to consult a healthcare professional for confirmation and further advice.

## Web App and Chatbot: How They Work

### Web App (Streamlit)
1. **Launch:** User runs `streamlit run src/app.py` to open the web interface.
2. **Symptom Input:** User enters symptoms in a text box (e.g., `fever, cough, fatigue`).
3. **Symptom Normalization:** The app standardizes input, correcting typos and mapping synonyms.
4. **Cluster Suggestions:** The app suggests up to 3 relevant symptom clusters, asking about additional symptoms (e.g., “Do you also have sore throat?”).
5. **User Confirmation:** User confirms or adds symptoms based on suggestions.
6. **Severity Input:** User rates severity (1-5) for each symptom using sliders or input fields.
7. **Prediction:** The app encodes symptoms/severities and predicts the top 3 probable diseases using the Random Forest model.
8. **Results Display:** The app shows predicted diseases, precautions, and doctor recommendations. If no specific precaution is available, a default message is shown.
9. **Export:** User can export results as PDF or text.

### Console Chatbot
1. **Launch:** User runs `python src/chatbot.py` in the terminal.
2. **Symptom Input:** Chatbot prompts user to enter symptoms (e.g., `headache, nausea`).
3. **Symptom Normalization:** Chatbot standardizes input, handling typos and synonyms.
4. **Cluster Suggestions:** Chatbot interactively asks about missing symptoms from relevant clusters (e.g., “Do you also have vomiting?”).
5. **User Confirmation:** User responds “yes” or “no” to each suggestion.
6. **Severity Input:** Chatbot asks user to rate severity (1-5) for each symptom.
7. **Prediction:** Chatbot encodes input and predicts the top 3 diseases using the model.
8. **Results Display:** Chatbot prints predicted diseases, precautions, and doctor recommendations. If no specific precaution is available, a default message is shown.

Both interfaces guide the user step-by-step, from symptom entry to personalized prediction and advice. The web app offers a graphical, interactive experience with export options, while the console chatbot provides a conversational, text-based flow.

## Dataset

MediPredict utilizes several curated datasets to enable robust disease prediction and symptom analysis:

- **Patient-centric dataset (`outputs/p.csv`):**
  - Contains rows of patient records with binary indicators for the presence or absence of each symptom.
  - Each record is labeled with the diagnosed disease.
  - Used for training and evaluating the machine learning model.

- **Feature list (`features/rf_features.txt`):**
  - Lists all symptoms (features) considered by the model.
  - Ensures consistent feature ordering and mapping between data and model.

- **Disease knowledge base (`data/disease_centric_knowledgebase_with_doctor.csv`):**
  - Maps diseases to their associated symptoms, recommended precautions, and doctor types.
  - Used for cluster-based symptom suggestion, precaution display, and doctor recommendations.

- **Symptom clusters:**
  - Defined in code, grouping symptoms that commonly co-occur for specific diseases.
  - Used to suggest additional relevant symptoms to users, improving diagnostic accuracy.

**Note:** All datasets are anonymized and structured for research and educational purposes. The knowledge base is curated with input from medical sources and domain experts.

## Key Features

- Robust symptom normalization, synonym mapping, and fuzzy matching
- Intelligent symptom cluster suggestions to improve diagnostic accuracy
- Interactive, stateful user experience (web and console)
- Severity input for nuanced prediction
- Top-3 disease prediction with confidence scores
- Precaution and doctor recommendations for each disease
- User-friendly default messages for missing data
- Export results as PDF or text (web app)
- Modular, extensible codebase for future enhancements

## Usage Guide

### 1. Streamlit Web App

- Run the app: `streamlit run src/app.py`
- Enter your symptoms in the input box (e.g., "fever, cough, headache")
- Confirm or add symptoms suggested by the system
- Rate the severity (1-5) for each symptom
- View predicted diseases, precautions, and doctor recommendations
- Export results as PDF or text if desired

### 2. Console Chatbot

- Run the chatbot: `python src/chatbot.py`
- Enter symptoms as prompted
- Respond to cluster suggestions interactively
- Enter severity ratings when asked
- View predictions and advice in the terminal

## Project Structure

- `src/app.py` — Streamlit web application
- `src/chatbot.py` — Console-based chatbot
- `src/train_model.py` — Model training and core logic
- `models/` — Trained model and label encoder files
- `features/` — List of model features (symptoms)
- `data/` — Datasets and disease knowledge base
- `outputs/` — Model outputs, feature importances, and predictions
- `notebooks/` — Jupyter notebooks for data exploration and modeling
- `README.md` — Quick start and setup instructions
- `PROJECT_DOCUMENTATION.md` — Comprehensive documentation (this file)

## Example Usage

### Web App Example

- **Input:** `fever, cough, fatigue`
- **System:** Suggests missing symptoms from relevant clusters (e.g., "Do you also have sore throat?")
- **User:** Confirms or adds symptoms
- **Severity:** User rates each symptom
- **Output:**
  - Top 3 probable diseases (e.g., Flu, COVID-19, Common Cold)
  - Precautions (e.g., "Stay hydrated, rest, consult a doctor if symptoms worsen.")
  - Doctor recommendation (e.g., "General Physician")

### Console Example

- **Input:** `headache, nausea`
- **System:** Suggests cluster symptoms (e.g., "Do you also have vomiting?")
- **User:** Responds interactively
- **Severity:** User inputs severity
- **Output:**
  - Disease predictions and advice in terminal

## Limitations & Known Issues

- Not a substitute for professional medical diagnosis
- Limited to diseases and symptoms present in the dataset
- May not recognize rare or highly specific symptoms
- English language input only (multi-language planned for future)
- Model accuracy depends on data quality and coverage

## Contributors

- [Your Name] (Project Lead)
- [Other Contributors]

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For setup, usage, and contribution guidelines, see the full README.md.

## Exploratory Data Analysis (EDA)

Before building the prediction model, comprehensive EDA was performed to understand the dataset and extract meaningful insights:

- **Data Overview:** Inspected the distribution of diseases and symptoms, checked for missing values, and validated data consistency.
- **Symptom Frequency:** Analyzed the frequency of each symptom across the dataset to identify common and rare symptoms.
- **Disease Distribution:** Visualized the number of cases per disease to detect class imbalance and ensure adequate representation.
- **Symptom Correlation:** Explored relationships between symptoms and diseases, and identified symptom clusters that frequently co-occur.
- **Severity Patterns:** Examined severity ratings (if available) to understand their distribution and impact on disease prediction.
- **Visualization:** Used bar charts, heatmaps, and pair plots to visualize data patterns and correlations.
- **Data Cleaning:** Addressed missing or inconsistent entries, standardized symptom names, and removed duplicates.

EDA helped guide feature selection, cluster definition, and model design, ensuring the robustness and reliability of the MediPredict system.

## Data Processing

Robust data processing is essential for accurate disease prediction. The following steps are performed:

- **Data Cleaning:**  
  - Remove duplicates and handle missing values.
  - Standardize symptom names (e.g., lowercasing, removing extra spaces, correcting typos).

- **Symptom Normalization:**  
  - Map synonyms and variations to a standard symptom vocabulary.
  - Use fuzzy matching to handle user typos and non-standard input.

- **Feature Engineering:**  
  - Convert symptom presence/absence into binary features for model input.
  - Encode severity ratings as numerical features.

- **Label Encoding:**  
  - Encode disease names as numerical labels for model training and prediction.

- **Data Splitting:**  
  - Split the dataset into training and testing sets to evaluate model performance.

- **Cluster Definition:**  
  - Group symptoms into clusters based on co-occurrence and medical relevance for improved suggestions.

These steps ensure the data is clean, consistent, and suitable for machine learning, directly impacting the reliability of predictions.
