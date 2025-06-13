import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle
from pathlib import Path

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """
    Clean and preprocess text data
    Args:
        text: Input text string
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_data():
    """
    Main preprocessing function
    Returns:
        tuple: Processed training and testing data
    """
    # Load data
    df = pd.read_csv('../news_dataset.csv')
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def process_text(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    
    print("Removing stopwords and lemmatizing...")
    df['processed_text'] = df['cleaned_text'].apply(process_text)
    
    # Split data
    X = df['processed_text']
    y = df['category']  # Using category as target variable
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    print("Performing TF-IDF vectorization...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save processed data
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)
    
    print("Saving processed data...")
    with open(results_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train_tfidf': X_train_tfidf,
            'X_test_tfidf': X_test_tfidf,
            'y_train': y_train,
            'y_test': y_test,
            'tfidf_vectorizer': tfidf
        }, f)
    
    print("Preprocessing completed!")
    return X_train_tfidf, X_test_tfidf, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
