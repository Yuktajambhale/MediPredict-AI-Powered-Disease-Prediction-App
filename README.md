# 🩺 MediPredict: AI-Powered Medical Disease Prediction

[🔗 Live Demo](https://medipredict-bot-2025.azurewebsites.net/)  

MediPredict is an AI-based healthcare web app that predicts the top 3 possible diseases based on user-reported symptoms and severity. It also provides mapped precautions, doctor suggestions, and a medical chatbot for basic health queries.  

Designed with a modern Streamlit UI and deployed on Azure for real-time access, MediPredict simplifies early-stage symptom analysis and disease awareness.

---

## 🚀 Features

- 🔍 **Disease Prediction** using Random Forest, XGBoost, and other ML models  
- 🧾 **Symptom Input** with autocomplete, cluster suggestions, and severity scale  
- 📊 **Top 3 Predictions** with confidence scores, precautions, and doctor recommendations  
- 🤖 **Medical Chatbot** for answering health-related queries  
- 📄 **PDF Report Download** of prediction results  
- ⚙️ **Fallback Advice** to visit general physician when no match is found  
- 🌐 **Deployed on Azure** using Docker for scalable and secure access  

---

## 🧠 Tech Stack

- **Frontend/UI**: Streamlit (Glassmorphism UI)
- **Backend/Logic**: Python 3.8+
- **Machine Learning**: scikit-learn, XGBoost, Random Forest
- **Data Processing**: Pandas, NumPy
- **PDF Report Generation**: FPDF
- **EDA & Modeling**: Jupyter Notebooks
- **Deployment**: Docker, Azure App Service

---

 🗂️ Project Structure

bash
├── data/            # Medical datasets (raw and processed CSVs)
├── notebooks/       # EDA, model training, tuning (GridSearchCV, etc.)
├── src/             # Source code for Streamlit app and chatbot
├── models/          # Trained models and encoders
├── features/        # Feature files used in ML input
├── outputs/         # Generated reports and visualizations
tion and model training.
- Use the scripts in `src/` for running the chatbot, training models, and data processing.

How to Use MediPredict
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run notebooks for EDA or model training

bash
Copy
Edit
Open Jupyter and explore the notebooks/ directory
Launch the web app

bash
Copy
Edit
streamlit run src/app.py
Use the chatbot

bash
Copy
Edit
Run chatbot script in src/ for medical Q&A
🧪 Model Training & Optimization
Models: KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, Logistic Regression

Tuning: Cross-validation and Grid/RandomizedSearchCV

EDA: Used to identify missing values, outliers, feature impact

Cleaned data stored in outputs/, models in models/

☁️ Deployment (Azure)
Dockerized App: For portability and consistency

Azure App Service: Secure cloud deployment

CI/CD: GitHub Actions & Azure Pipelines

Environment Variables: Managed via Azure

Live Access: https://medipredict-bot-2025.azurewebsites.net/

