🫀 Heart Disease Prediction using Logistic Regression 📌 Overview

This project predicts the likelihood of heart disease in a person based on clinical parameters using Logistic Regression. A Flask-based web application has been developed to make predictions interactively, and the project is hosted online using Render.

🚀 Features

✅ Machine Learning model: Logistic Regression

✅ User-friendly Flask Web App

✅ Interactive UI for inputting patient data

✅ Deployed and accessible online (Render Hosting)

✅ Explainability using SHAP and LIME visualizations

🏗️ Project Structure Heart-Disease-Project │── app.py # Flask main application
│── heart.csv # Dataset (Cleveland Heart Disease Dataset)
│── /static # CSS, JS, and visualization files
│ ├── css/
│ ├── js/
│ ├── shap/
│ └── lime/
│── /templates
│ └── index.html # Frontend UI
│── requirements.txt # Dependencies
│── README.md # Documentation

⚙️ Tech Stack

Programming Language: Python 🐍

Framework: Flask

Machine Learning: Scikit-learn (Logistic Regression)

Frontend: HTML, CSS, JavaScript

Visualization: SHAP, LIME

Deployment: Render

📊 Dataset

The project uses the Heart Disease Dataset, which contains medical attributes such as:

Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, Slope, Major Vessels, Thalassemia.

Target: Presence (1) or absence (0) of heart disease.

🔮 Model Details

Algorithm: Logistic Regression

Why Logistic Regression?

Handles binary classification problems effectively.

Provides interpretable coefficients for medical insights.

Works well with this dataset with good accuracy.

🌐 Live Demo

👉 Click here to try the app on Render - [https://heart-disease-project-6xdv.onrender.com](https://heart-disease-prediction-3-ftle.onrender.com/)

⚡ Installation & Usage (Local Setup)

Clone the repository

git clone https://github.com/your-username/Heart-Disease-Project.git cd Heart-Disease-Project

Create a virtual environment & activate

python -m venv venv source venv/bin/activate # Linux/Mac venv\Scripts\activate # Windows

Install dependencies

pip install -r requirements.txt

Run the app

python app.py

Open your browser at http://127.0.0.1:5000/

👨‍💻 Author

SOMA KAVYA SREE


📧 Email: kavyasreesoma1@gmail.com


⭐ Contribute

Feel free to fork this repo, raise issues, or submit pull requests to improve the project!
