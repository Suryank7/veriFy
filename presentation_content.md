# VeriFy Project Presentation

## Slide 1: Title Slide
- **Title**: VeriFy: AI-Powered Fake News Detector
- **Subtitle**: Combating Misinformation through Machine Learning
- **Presenter**: [Your Name]
- **Visual**: VeriFy Logo / Laptop with Code

## Slide 2: Problem Statement
- **Context**: Exponential growth of digital news consumption.
- **Issue**: Rapid spread of misinformation (Fake News) on social media.
- **Impact**: Political polarization, public panic, and erosion of trust.
- **Need**: An automated, reliable system to verify news credibility instantly.

## Slide 3: Solution - VeriFy
- **What it is**: A Machine Learning-based web application.
- **Core Function**: Classifies news articles as "Real" or "Fake".
- **Key Features**:
    - Instant Prediction
    - Confidence Scoring
    - Explainable AI (Keyword highlighting)
    - User-friendly Interface

## Slide 4: System Architecture
- **Flowchart**:
    1. User Input (News Text)
    2. Preprocessing (Cleaning & Tokenization)
    3. Feature Extraction (TF-IDF Vectorization)
    4. Model Prediction (ML Classifier)
    5. Output (Label + Confidence)

## Slide 5: Tech Stack
- **Frontend**: Streamlit (Python) - *For rapid, clean UI development.*
- **Backend/ML**: Python, Scikit-learn.
- **Data Processing**: Pandas, NumPy, NLTK.
- **Models**: Logistic Regression, Naive Bayes, Passive Aggressive Classifier.

## Slide 6: Dataset & Preprocessing
- **Source**: ISOT Fake News Dataset (Kaggle).
- **Size**: ~45,000 articles (Balanced classes).
- **Preprocessing Steps**:
    - Lowercasing
    - Removing Punctuation
    - Removing Stopwords
    - TF-IDF Vectorization (Top 10,000 features)

## Slide 7: Model Comparison
*(Fill with actual results after training)*
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.6% | 0.98 | 0.99 | 0.99 |
| Naive Bayes | 95.2% | 0.94 | 0.96 | 0.95 |
| Passive Aggressive | 99.1% | 0.99 | 0.99 | 0.99 |

- **Best Model**: Passive Aggressive Classifier (Optimized for text streams).

## Slide 8: Demo Screenshots
- **Input Screen**: Clean text area.
- **Result Screen (Fake)**: Red warning, confidence score.
- **Result Screen (Real)**: Green success message.

## Slide 9: Future Scope
- **Browser Extension**: Real-time checking on Twitter/Facebook.
- **Deep Learning**: Implementing BERT/RoBERTa for context awareness.
- **Multilingual Support**: Expanding to Hindi and regional languages.
- **Fact-Check API Integration**: Cross-referencing with Snopes/PolitiFact.

## Slide 10: Conclusion
- VeriFy provides a robust, scalable solution for information literacy.
- High accuracy metrics demonstrate readiness for real-world application.
- Empowers users to make informed decisions about the news they consume.

## Q&A
- Thank You!
