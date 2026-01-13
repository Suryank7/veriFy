import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

def load_and_preprocess_data(fake_path='data/Fake.csv', true_path='data/True.csv'):
    """
    Loads Fake and True datasets, adds labels, combines them, 
    and applies text cleaning.
    
    Returns:
        pd.DataFrame: A dataframe with 'text', 'title', 'subject', 'date', 'class', and 'cleaned_text'.
    """
    print("Loading datasets...")
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
    except FileNotFoundError:
        print(f"Error: Datasets not found at {fake_path} or {true_path}. Please ensure files exist.")
        raise

    # Add labels: Fake = 0, Real = 1
    df_fake['class'] = 0
    df_true['class'] = 1

    # Combine datasets
    df_manual_testing = pd.concat([df_fake.tail(10), df_true.tail(10)], axis=0) # Save some for manual testing if needed, though split is better
    
    # Remove last 10 rows for manual testing from the main training set
    df_fake = df_fake.iloc[:-10]
    df_true = df_true.iloc[:-10]
    
    df = pd.concat([df_fake, df_true], axis=0)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Cleaning text (this might take a moment)...")
    # Combine title and text for better context? The user requested "Accept news text or article content".
    # Often title + text is better. Let's create a 'content' column to be safe.
    df['content'] = df['title'] + " " + df['text']
    
    df['cleaned_text'] = df['content'].apply(clean_text)
    
    print("Data preprocessing complete.")
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    print(f"Total samples: {len(df)}")
    print(df.head())
    # Save processed data for faster loading next time
    df.to_csv('data/processed_data.csv', index=False)
    print("Saved to data/processed_data.csv")
