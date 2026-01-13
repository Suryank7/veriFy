import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def load_data(filepath='data/processed_data.csv'):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}. Please run preprocess.py first.")
        return None
    return pd.read_csv(filepath)

def train_models():
    # 1. Load Data
    print("Loading processed data...")
    df = load_data()
    if df is None: return

    # Handle missing values if any created during processing
    df = df.dropna(subset=['cleaned_text'])

    x = df['cleaned_text']
    y = df['class']

    # 2. Split Data
    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 3. Vectorization
    print("Vectorizing text...")
    vectorization = TfidfVectorizer(max_features=10000, ngram_range=(1,2)) # Increased features for better accuracy
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Dictionary to store model performance
    model_performance = {}

    # 4. Model Training & Evaluation
    
    # --- Logistic Regression ---
    print("\nTRAINING LOGISTIC REGRESSION...")
    lr = LogisticRegression()
    lr.fit(xv_train, y_train)
    pred_lr = lr.predict(xv_test)
    acc_lr = accuracy_score(y_test, pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(classification_report(y_test, pred_lr))
    model_performance['LR'] = {'model': lr, 'accuracy': acc_lr}

    # --- Naive Bayes ---
    print("\nTRAINING NAIVE BAYES...")
    nb = MultinomialNB()
    nb.fit(xv_train, y_train)
    pred_nb = nb.predict(xv_test)
    acc_nb = accuracy_score(y_test, pred_nb)
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
    print(classification_report(y_test, pred_nb))
    model_performance['NB'] = {'model': nb, 'accuracy': acc_nb}

    # --- Passive Aggressive Classifier ---
    print("\nTRAINING PASSIVE AGGRESSIVE CLASSIFIER...")
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(xv_train, y_train)
    pred_pac = pac.predict(xv_test)
    acc_pac = accuracy_score(y_test, pred_pac)
    print(f"Passive Aggressive Classifier Accuracy: {acc_pac:.4f}")
    print(classification_report(y_test, pred_pac))
    model_performance['PAC'] = {'model': pac, 'accuracy': acc_pac}

    # 5. Select Best Model
    best_model_name = max(model_performance, key=lambda k: model_performance[k]['accuracy'])
    best_model = model_performance[best_model_name]['model']
    print(f"\nBest Model: {best_model_name} with Accuracy: {model_performance[best_model_name]['accuracy']:.4f}")

    # Plot Confusion Matrix for Best Model
    print("Generating confusion matrix plot...")
    plt.figure(figsize=(8, 6))
    if best_model_name == 'LR':
        y_pred = pred_lr
    elif best_model_name == 'NB':
        y_pred = pred_nb
    else:
        y_pred = pred_pac
        
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")

    # 6. Save Artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("Saving model and vectorizer...")
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorization, 'models/vectorizer.pkl')
    print("Model saved to models/best_model.pkl")
    print("Vectorizer saved to models/vectorizer.pkl")

    # Save logic for "Recall" explanation in documentation
    # Just printing here for the log
    print("\nNOTE: Recall is critical in fake news detection because False Negatives (Fake news classified as Real) can be very damaging. A high recall score for the 'Fake' class ensures we catch most misinformation.")

if __name__ == "__main__":
    train_models()
