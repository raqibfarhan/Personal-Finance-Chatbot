Personel Finance chatbot using LLM 💰🤖

ersonal Finance Chatbot with Hybrid ML and LLM Approach Overview This project is a Personal Finance Chatbot designed to assist users with budgeting, expense tracking, savings prediction, and investment risk profiling. It integrates a hybrid approach combining traditional machine learning models  for core analytics and Large Language Models LLMs powered by OpenAI for natural language interaction, delivering an intuitive and powerful financial assistant.

Features Core Machine Learning Components  Budget Categorization Model: Classifies user transactions into categories such as food, bills, entertainment, etc., using classifiers like Naive Bayes and Random Forest.

Expense Anomaly Detection: Detects unusual or fraudulent spending behavior through models like Isolation Forest or One-Class SVM.

Savings Prediction Model: Predicts monthly savings based on past spending patterns with regression models such as XGBoost or Linear Regression.

Investment Risk Profiling: Clusters users into risk profiles (e.g., conservative, moderate, aggressive) using unsupervised learning (K-Means clustering).

Large Language Model Component  Provides a conversational interface where users can ask finance-related questions naturally.

This project is a Streamlit-based interactive chatbot that allows users to upload and analyze personal finance data (CSV files). It uses FAISS for vector search, Groq LLM for natural language understanding, and custom calculation functions to provide insights, summaries, and visualizations.

🚀 Features

📂 Upload CSV (finance transactions or any dataset)

🔎 Search & Query with LLM (ask natural language questions)

📊 Charts & Visualizations using Plotly

🧮 Custom Finance Calculations (e.g., total spend, savings, category breakdown)

⚡ FAISS Indexing for semantic search in uploaded data

🔑 Environment variables support with .env

🛠️ Tech Stack

Python 3.10+

Streamlit – UI

Pandas – Data processing

Plotly – Charts

FAISS – Vector indexing

Groq API – LLM queries

Dotenv – Environment variables



⚙️ Setup Instructions
1️⃣ Clone Repo
https://github.com/raqibfarhan/Personal-Finance-Chatbot.git
cd finance-csv-chatbot

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate 

3️⃣ Set Environment Variables

Create a .env file in the project root:

GROQ_API_KEY="your_api_key_here"
DATA_PATH=".csv file"
MODEL_NAME="llama-3.1-70b-versatile"

4️⃣ Run the App

streamlit run app.py
