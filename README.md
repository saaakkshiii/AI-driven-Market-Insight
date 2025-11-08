AI Financial Market Prediction Dashboard
📊 Overview

The AI Financial Market Prediction Dashboard is a full-stack Flask web application that predicts financial market performance and company stock impact using Machine Learning and Quantum-inspired models.
It provides interactive visualizations, real-time analytics, user authentication, and model comparison — all in one intelligent dashboard.

⚙️ Features
🔍 Machine Learning & Analytics

Multiple ML Models: Linear Regression, Support Vector Regressor (SVR), Gradient Boosting Regressor (GBR), and Quantum-inspired models (VQC).

Automatic Model Selection: Compares R² and MSE values to pick the best-performing model dynamically.

Feature Importance & Model Comparison: Interactive bar and line charts using Chart.js.

Historical Trend View: Visualizes company growth and stock impact trends over time.

Residual Error Analysis: Evaluates bias and prediction variance across models.

🧩 Web Application (Flask)

Secure User Authentication: Register/login system built using Flask-Login and SQLite.

Prediction Dashboard: Input company data and instantly get multi-model predictions.

Personalized History Page: Stores user predictions with timestamps for tracking.

CSV Export: Download your prediction history for further analysis.

Dark Mode Toggle: Modern UI with black background and neon highlights.

Auto Browser Launch: Automatically opens at http://127.0.0.1:5000 when started.

🧰 Tech Stack
Category	Technologies Used
Frontend	HTML5, CSS3, Bootstrap 5, Chart.js
Backend	Flask, Python 3
Machine Learning	scikit-learn, XGBoost, NumPy, Pandas
Quantum ML	Qiskit (optional for quantum circuit learning)
Database	SQLite
Visualization	Chart.js, Matplotlib
Authentication	Flask-Login
🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/AI_Market_Dashboard.git
cd AI_Market_Dashboard

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate   # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


(If you don’t have requirements.txt, use:)

pip install flask flask-login pandas numpy scikit-learn matplotlib chart-studio qiskit

4️⃣ Run the Model Training (first-time setup)
python train_models_classical.py

5️⃣ Start the Web App
python app.py
