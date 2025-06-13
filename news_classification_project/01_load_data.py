import os
import json
import pandas as pd
from pathlib import Path

def extract_article_info(json_path):
    """
    Extract relevant information from a single JSON file
    Args:
        json_path: Path to the JSON file
    Returns:
        dict: Dictionary containing article information
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract category and sentiment from folder name
        folder_name = os.path.basename(os.path.dirname(json_path))
        category, sentiment, _ = folder_name.split('_', 2)
        
        return {
            'category': category,
            'sentiment': sentiment,
            'title': data.get('title', ''),
            'text': data.get('text', ''),
            'url': data.get('url', ''),
            'published_date': data.get('published_date', ''),
            'source': data.get('source', '')
        }
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None

def process_all_articles(base_path):
    """
    Process all JSON files in the dataset
    Args:
        base_path: Path to the docs directory
    Returns:
        list: List of dictionaries containing article information
    """
    articles = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories
    for category_dir in base_path.iterdir():
        if category_dir.is_dir():
            for json_file in category_dir.glob('*.json'):
                article_info = extract_article_info(json_file)
                if article_info:
                    articles.append(article_info)
    
    return articles

def main():
    # Path to the docs directory
    docs_path = '../free-news-datasets-master/docs'
    
    # Process all articles
    print("Processing articles...")
    articles = process_all_articles(docs_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(articles)
    
    # Save to CSV
    output_path = 'news_dataset.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset saved to {output_path}")
    print(f"Total articles processed: {len(df)}")
    print("\nDataset statistics:")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

if __name__ == "__main__":
    main()
