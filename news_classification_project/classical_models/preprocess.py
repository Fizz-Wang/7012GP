# =================================================================================================
#
#                                       GENERAL INSTRUCTIONS
#
# This script prepares the final dataset for model training. It now performs these key tasks:
# 1. Simplified Category Extraction: It reads the 'category' column (e.g., "Science and
#    Technology_positive_2024...") and extracts only the part before the first underscore
#    as the final, single category label.
# 2. Text Preprocessing: It cleans the text, removes stopwords, performs lemmatization,
#    and then vectorizes the text data using TF-IDF.
#
# The final output is 'processed_data.pkl', ready to be used by the training scripts.
#
# =================================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle
from pathlib import Path

# --- Download required NLTK data (if not already present) ---
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    """
    Cleans and preprocesses a single text string.
    Args:
        text (str): Input text string.
    Returns:
        str: Cleaned text.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def preprocess_and_transform_data():
    """
    Main preprocessing function that extracts a single category,
    cleans text, and saves the final processed data.
    """
    # --- Step 1: Load the initial filtered dataset ---
    # This assumes the '01_load_data.py' script has been run.
    # The path is relative to the location of this script in 'classical_models/'
    input_csv_path = '../news_dataset_english_filtered.csv'
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: '{input_csv_path}' not found. Please run '01_load_data.py' first.")
        return

    # --- Step 2: Extract a Single Category from the 'category' column ---
    print("Extracting a single category from the 'category' column...")
    # The 'category' column has a format like 'Science and Technology_positive_2024...'.
    # We split by the first underscore and take the first part.
    # .str.get(0) retrieves the first element from the split list.
    df['single_category'] = df['category'].str.split('_', n=1).str.get(0)
    print("Category extraction complete.")
    print("\nCategory distribution after extraction:")
    print(df['single_category'].value_counts())

    # --- Step 3: Text Cleaning and Processing ---
    print("\nCleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Initialize lemmatizer and stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def process_text(text):
        words = text.split()
        # Lemmatize and remove stopwords
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    print("Removing stopwords and lemmatizing...")
    df['processed_text'] = df['cleaned_text'].apply(process_text)

    # --- Step 4: Split data into Training and Testing sets ---
    print("Splitting data into training and testing sets...")
    X = df['processed_text']
    y = df['single_category']  # Use the new single_category column as the target

    # Use stratify=y to ensure the category distribution is similar in train and test sets
    # This might fail if some categories have only one sample. We'll add a check.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("Stratify failed, likely due to categories with only one sample. Splitting without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # --- Step 5: TF-IDF Vectorization ---
    print("Performing TF-IDF vectorization...")
    tfidf = TfidfVectorizer(
        max_features=50000,  # Limit the number of features to the top 50,000
        min_df=5,  # Ignore terms that appear in less than 5 documents
        max_df=0.9,  # Ignore terms that appear in more than 90% of documents
        ngram_range=(1, 2)  # Include both unigrams and bigrams
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # --- Step 6: Save the Processed Data ---
    # The path is relative to this script's location
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True)

    output_pickle_path = results_dir / 'processed_data.pkl'
    print(f"Saving processed data to '{output_pickle_path}'...")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump({
            'X_train_tfidf': X_train_tfidf,
            'X_test_tfidf': X_test_tfidf,
            'y_train': y_train,
            'y_test': y_test,
            'tfidf_vectorizer': tfidf
        }, f)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    preprocess_and_transform_data()
