# VeriFy - AI-Powered Fake News Detector 🛡️

**Internship Project | Advanced Fake News Detection System**

## 🚨 Problem Statement
In the digital age, misinformation spreads rapidly across social media and web platforms. Manual verification of news is time-consuming and often biased. **VeriFy** is an automated AI system designed to help students and users identify fake news articles with high accuracy using Machine Learning.

## 🌟 Features
- **Real-time Detection**: Classifies news as **Real** or **Fake** instantly.
- **Confidence Score**: Provides a probability percentage for the prediction.
- **Explainability**: Highlights key terms influencing the decision.
- **Clean UI**: Professional, easy-to-use web interface built with Streamlit.
- **Multiple Models**: Trained on Logistic Regression, Naive Bayes, and Passive Aggressive Classifier.

## 🗂️ Dataset
We use the **ISOT Fake News Dataset** from Kaggle, a widely accepted benchmark for fake news detection research.
- **Fake.csv**: Contains articles flagged as misinformation.
- **True.csv**: Contains verified real news articles.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Data Analysis**: Pandas, NumPy
- **NLP**: NLTK, Scikit-learn (TF-IDF Vectorizer)
- **ML Models**: 
  - Logistic Regression (Best for binary classification)
  - Naive Bayes (Standard for text classification)
  - Passive Aggressive Classifier (Great for dynamic data streams)
- **Frontend**: Streamlit

## 🚀 How to Run the Project

### 1. Setup Environment
Ensure you have Python installed. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Running the project requires the dataset. 
- Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
- Place them in the fake-news-detector root folder or let the script handle them.
- Run the preprocessing script:
```bash
python src/preprocess.py
```

### 3. Train Models
Train the machine learning models and save the best performing one:
```bash
python src/train.py
```
*Artifacts will be saved in the `models/` directory.*

### 4. Launch Application
Start the Streamlit web app:
```bash
streamlit run app.py
```

## 📊 Model Evaluation
We evaluate models based on:
- **Accuracy**: Overall correctness.
- **Precision**: How many predicted "Fake" are actually Fake.
- **Recall**: How many actual Fake items were correctly detected. *Crucial for staying safe from misinformation.*
- **F1-Score**: Balance between Precision and Recall.

*Confusion Matrices and detailed reports are generated during the training phase.*

## 🔮 Future Scope
- **Browser Extension**: For real-time checking while browsing.
- **URL Analysis**: Scrape and verify content directly from links.
- **Multilingual Support**: Detect fake news in Hindi/Spanish using Transformers (BERT/RoBERTa).
- **Social Media Integration**: Bot to verify tweets/posts automatically.

---
*Developed by [Lalit]*
