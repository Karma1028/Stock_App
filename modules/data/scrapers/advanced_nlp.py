"""
Advanced NLP module for news analysis using embeddings and topic modeling.
"""
import pandas as pd
import numpy as np
from datetime import datetime
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️ sentence-transformers not installed. Using basic features only.")

try:
    from bertopic import BERTopic
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    print("⚠️ BERTopic not installed. Topic modeling disabled.")

class AdvancedNLPAnalyzer:
    def __init__(self):
        self.embedding_model = None
        self.topic_model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            # Load a lightweight model for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of texts.
        Returns a numpy array of shape (n_texts, embedding_dim).
        """
        if not HAS_SENTENCE_TRANSFORMERS or not texts:
            return np.array([])
        
        return self.embedding_model.encode(texts, show_progress_bar=False)
    
    def extract_topics(self, df_news):
        """
        Extracts topics from news headlines using BERTopic.
        """
        if not HAS_BERTOPIC or df_news.empty:
            return df_news, None
        
        headlines = df_news['title'].tolist()
        
        # Generate topic model
        self.topic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=False
        )
        
        topics, probs = self.topic_model.fit_transform(headlines)
        
        # Add topics to dataframe
        df_news['topic'] = topics
        df_news['topic_prob'] = probs.max(axis=1)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        return df_news, topic_info
    
    def detect_events(self, df_news):
        """
        Detects specific event types (earnings, M&A, guidance) using keyword matching.
        """
        if df_news.empty:
            return df_news
        
        event_keywords = {
            'earnings': ['earnings', 'quarterly', 'results', 'profit', 'loss', 'revenue'],
            'ma': ['merger', 'acquisition', 'acquires', 'deal', 'takeover', 'buyout'],
            'guidance': ['guidance', 'outlook', 'forecast', 'target', 'estimate'],
            'dividend': ['dividend', 'payout', 'distribution'],
            'regulatory': ['regulation', 'compliance', 'lawsuit', 'investigation', 'fine']
        }
        
        def categorize_event(title):
            title_lower = title.lower()
            events = []
            for event_type, keywords in event_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    events.append(event_type)
            return events if events else ['general']
        
        df_news['event_types'] = df_news['title'].apply(categorize_event)
        df_news['has_earnings'] = df_news['event_types'].apply(lambda x: 'earnings' in x)
        df_news['has_ma'] = df_news['event_types'].apply(lambda x: 'ma' in x)
        
        return df_news
    
    def compute_news_surprise(self, df_news, baseline_days=7):
        """
        Computes a "news surprise" metric by comparing recent sentiment to baseline.
        """
        if df_news.empty or 'sentiment' not in df_news.columns:
            return 0
        
        # Sort by date
        df_sorted = df_news.sort_values('date')
        
        # Baseline sentiment (older news)
        if len(df_sorted) > baseline_days:
            baseline_sentiment = df_sorted.iloc[:-baseline_days]['sentiment'].mean()
            recent_sentiment = df_sorted.iloc[-baseline_days:]['sentiment'].mean()
            
            surprise = recent_sentiment - baseline_sentiment
            return surprise
        
        return 0
    
    def enhance_news_dataframe(self, df_news):
        """
        Enhances news dataframe with advanced NLP features.
        """
        if df_news.empty:
            return df_news
        
        # 1. Add embeddings (as average for aggregation)
        if HAS_SENTENCE_TRANSFORMERS:
            embeddings = self.generate_embeddings(df_news['title'].tolist())
            if len(embeddings) > 0:
                # Store average embedding per news item
                for i in range(min(5, embeddings.shape[1])):  # First 5 dimensions
                    df_news[f'emb_{i}'] = embeddings[:, i]
        
        # 2. Detect events
        df_news = self.detect_events(df_news)
        
        # 3. Extract topics
        df_news, topic_info = self.extract_topics(df_news)
        
        return df_news
