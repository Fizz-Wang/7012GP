# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from pathlib import Path
import ftfy  # Import the ftfy library to fix potential text encoding issues
import re  # Import the regular expression library
from langdetect import detect, LangDetectException  # Import language detection library


def detect_language(text):
    """
    Detects the language of a given text using langdetect.
    Args:
        text (str): The input text.
    Returns:
        str: The detected language code (e.g., 'en', 'es') or 'unknown' if detection fails.
    """
    try:
        # We need a minimum amount of text to make an accurate prediction.
        if len(text) < 50:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        # This exception occurs for texts that are too short or ambiguous (e.g., only numbers/symbols)
        return 'unknown'


def extract_article_info(json_path):
    """
    Extracts required information from a single JSON file.
    The category is derived from the name of the parent directory.

    Args:
        json_path (Path): A Path object pointing to the JSON file.

    Returns:
        dict or None: A dictionary containing article information, or None if processing fails.
    """
    try:
        # Safely read and parse the JSON file using 'utf-8' encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Proactively use ftfy.fix_text() on text fields to fix potential encoding errors
        # that might exist in the source file.
        title = ftfy.fix_text(data.get('title', ''))
        text = ftfy.fix_text(data.get('text', ''))

        # The parent directory's name is used as the category for this article.
        category = json_path.parent.name

        # --- Extract the required fields directly from the JSON content ---
        return {
            'category': category,  # Category derived from folder name
            'source_file': json_path.name,  # Add traceability by recording the source filename
            'url': data.get('url', ''),
            'title': title,
            'text': text,
            'language': data.get('language', 'unknown'),
            'categories_from_json': data.get('categories', [])  # Keep original categories from JSON
        }
    except json.JSONDecodeError:
        print(f"Error: File {json_path.name} in folder {json_path.parent.name} is not a valid JSON file. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing file {json_path.name}: {e}. Skipping.")
        return None


def process_all_articles(base_path):
    """
    Processes all JSON files found in all subdirectories of the base path.

    Args:
        base_path (str): Path to the main directory (e.g., 'docs') containing category subfolders.

    Returns:
        list: A list of dictionaries, each containing information for one article.
    """
    articles = []
    base_path = Path(base_path)

    # Check if the base path exists and is a directory
    if not base_path.is_dir():
        print(f"Error: Base path '{base_path}' does not exist or is not a directory.")
        return articles

    print(f"Scanning all subdirectories in: '{base_path}'...")

    # Walk through all subdirectories in the base path
    for sub_dir in base_path.iterdir():
        if sub_dir.is_dir():
            print(f"  - Processing folder: {sub_dir.name}")
            # Process all JSON files in the current subdirectory
            for json_file in sub_dir.glob('*.json'):
                article_info = extract_article_info(json_file)
                if article_info:
                    articles.append(article_info)

    return articles


def main():
    """
    Main execution function.
    """
    # ========================================================================
    # This path should point to the main directory (e.g., 'docs') which contains
    # all the category subfolders. The script will traverse all of them.
    # ========================================================================
    ARTICLES_BASE_PATH = '../free-news-datasets-master/docs'

    # Process all articles from all subfolders
    articles_data = process_all_articles(ARTICLES_BASE_PATH)

    if not articles_data:
        print(
            "No data was processed. Please check if the path is correct and if subfolders with JSON files exist. Exiting.")
        return

    # Convert the list of results into a Pandas DataFrame
    print("\nConverting data to DataFrame...")
    df = pd.DataFrame(articles_data)

    # ========================================================================
    # Step 1: Filtering for articles initially marked as English
    # ========================================================================
    print("\nFiltering for articles marked as English in metadata...")
    english_df = df[df['language'].str.lower().str.strip() == 'english'].copy()
    print(f"Found {len(df)} total articles, {len(english_df)} of which are marked as English.")

    if english_df.empty:
        print("No articles marked as English found. Exiting.")
        return

    # ========================================================================
    # Step 2: Content-based language detection to verify English
    # ========================================================================
    print(f"\nPerforming content-based language detection on {len(english_df)} articles...")
    # Apply the language detection function to the 'text' column. This may take a moment.
    english_df['detected_language'] = english_df['text'].apply(detect_language)

    initial_count_step2 = len(english_df)
    verified_english_df = english_df[english_df['detected_language'] == 'en'].copy()
    removed_count_step2 = initial_count_step2 - len(verified_english_df)
    print(
        f"Removed {removed_count_step2} mislabeled non-English articles. Verified English articles: {len(verified_english_df)}.")

    if verified_english_df.empty:
        print("No articles verified as English by content. Exiting.")
        return

    # ========================================================================
    # Step 3: Filter out articles with very short text content
    # ========================================================================
    print(f"\nFiltering out articles with less than 256 characters in text...")
    initial_count_step3 = len(verified_english_df)
    final_df = verified_english_df[verified_english_df['text'].str.len() >= 256].copy()
    removed_count_step3 = initial_count_step3 - len(final_df)
    print(f"Removed {removed_count_step3} short articles. Final article count: {len(final_df)}.")

    if final_df.empty:
        print("No articles remaining after length filtering. Exiting.")
        return

    # Define the output CSV filename for the final filtered data
    output_path = 'news_dataset_english_filtered.csv'

    # Save the final filtered DataFrame to a CSV file, ensuring UTF-8 encoding
    print(f"\nSaving final filtered articles to '{output_path}'...")
    final_df.to_csv(output_path, index=False, encoding='utf-8')

    print("\n================== PROCESSING COMPLETE ==================")
    print(f"Successfully processed and saved {len(final_df)} final articles.")
    print(f"CSV file saved to: {output_path}")
    print("\nFinal Dataset Info:")
    final_df.info()
    print("\nCategory distribution in final dataset:")
    print(final_df['category'].value_counts())
    print("=======================================================")


if __name__ == "__main__":
    main()
