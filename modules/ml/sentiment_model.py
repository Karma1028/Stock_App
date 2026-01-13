"""
Sentiment Analysis Model for Stock News
Trains a classifier on news headlines to predict sentiment scores.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

class SentimentModel:
    """Sentiment analysis model for stock news."""
    
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or (Config.MODELS_DIR / "sentiment_model.pkl")
        self.vectorizer = None
        self.classifier = None
        
    def _analyze_with_textblob(self, text: str) -> float:
        """Get TextBlob sentiment as baseline."""
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _categorize_sentiment(self, polarity: float) -> str:
        """Convert polarity to category."""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def prepare_training_data(self, news_csv_path: Path):
        """
        Prepare training data from news CSV.
        Uses TextBlob for initial labels, then trains a more sophisticated model.
        """
        print(f"Loading news data from {news_csv_path}")
        df = pd.read_csv(news_csv_path)
        
        print(f"Columns in CSV: {df.columns.tolist()}")
        
        # Detect title column (could be 'title', 'Title', 'headline', etc.)
        title_col = None
        for col in ['title', 'Title', 'headline', 'Headline', 'text', 'Text']:
            if col in df.columns:
                title_col = col
                break
        
        if title_col is None:
            raise ValueError(f"No title column found. Available columns: {df.columns.tolist()}")
        
        print(f"Using column '{title_col}' for news text")
        
        # Check if sentiment column exists, if not generate using TextBlob
        if 'sentiment' not in df.columns and title_col in df.columns:
            print("Generating sentiment labels using TextBlob...")
            df['sentiment_score'] = df[title_col].apply(self._analyze_with_textblob)
            df['sentiment'] = df['sentiment_score'].apply(self._categorize_sentiment)
        elif 'sentiment' in df.columns:
            # If numeric, categorize
            if df['sentiment'].dtype in [np.float64, np.int64]:
                df['sentiment'] = df['sentiment'].apply(self._categorize_sentiment)
        
        # Filter out rows without titles
        df = df[df[title_col].notna()].copy()
        
        print(f"Prepared {len(df)} training samples")
        
        return df[title_col].values, df['sentiment'].values
    
    def train(self, news_csv_path: Path, test_size: float = 0.2):
        """Train the sentiment classifier."""
        X, y = self.prepare_training_data(news_csv_path)
        
        print(f"Training on {len(X)} samples...")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Logistic Regression
        print("Training Logistic Regression classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=1.0
        )
        
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Training Complete!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.save()
        
        return accuracy
    
    def predict(self, texts):
        """Predict sentiment for new texts."""
        if self.vectorizer is None or self.classifier is None:
            if not self.load():
                raise ValueError("Model not trained and cannot be loaded")
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Vectorize
        X_tfidf = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.classifier.predict(X_tfidf)
        probabilities = self.classifier.predict_proba(X_tfidf)
        
        # Convert to scores (-1 to 1)
        label_to_score = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}
        scores = [label_to_score[pred] for pred in predictions]
        
        # Adjust scores based on confidence
        # If probability is high for positive/negative, adjust score
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            max_proba = np.max(proba)
            if pred == 'positive':
                scores[i] = min(1.0, max_proba * 1.5 - 0.5)
            elif pred == 'negative':
                scores[i] = max(-1.0, -(max_proba * 1.5 - 0.5))
        
        return scores
    
    def save(self):
        """Save model to disk."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'version': '1.0'
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {self.model_path}")
    
    def load(self):
        """Load model from disk."""
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            
            print(f"✓ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Train the sentiment model."""
    # Path to news data
    news_data_path = Config.DATA_DIR / "all_stocks_news_consolidated.csv"
    
    if not news_data_path.exists():
        print(f"News data not found at {news_data_path}")
        print("Please ensure the news data is available.")
        return
    
    # Initialize and train model
    model = SentimentModel()
    accuracy = model.train(news_data_path, test_size=0.2)
    
    # Test predictions
    print("\n--- Testing Predictions ---")
    test_texts = [
        "Stock surges to new highs on strong earnings",
        "Company reports massive losses, shares plummet",
        "Market opens flat amid mixed economic data"
    ]
    
    predictions = model.predict(test_texts)
    for text, score in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Sentiment Score: {score:.3f}\n")

if __name__ == "__main__":
    main()
