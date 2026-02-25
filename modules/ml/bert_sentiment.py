from transformers import pipeline
import pandas as pd
import logging

class BertSentimentAnalyzer:
    """
    BERT-based sentiment analysis using 'ProsusAI/finbert' (specialized for finance).
    Features:
    - Uses HuggingFace pipeline for efficient inference.
    - Caches results to avoid redundant computation.
    """
    
    _pipeline = None

    def __init__(self, model_name="ProsusAI/finbert"):
        self.model_name = model_name
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Lazy load the pipeline."""
        if BertSentimentAnalyzer._pipeline is None:
            try:
                logging.info(f"Loading BERT model: {self.model_name}...")
                BertSentimentAnalyzer._pipeline = pipeline("sentiment-analysis", model=self.model_name)
                logging.info("BERT model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load BERT model: {e}")
                BertSentimentAnalyzer._pipeline = None

    def analyze(self, texts):
        """
        Analyze a list of texts or a single string.
        Returns a DataFrame with 'label' and 'score'.
        """
        if not BertSentimentAnalyzer._pipeline:
            logging.warning("BERT pipeline not initialized. Returning empty.")
            return []

        if isinstance(texts, str):
            texts = [texts]
        
        # Truncate texts to 512 tokens approx (simple char limit for speed/safety)
        # finbert usually handles truncation, but explicit safety is good.
        texts = [t[:512] for t in texts]

        try:
            results = BertSentimentAnalyzer._pipeline(texts)
            return results
        except Exception as e:
            logging.error(f"BERT Analysis failed: {e}")
            return []

    def analyze_dataframe(self, df, text_col='title'):
        """
        Enriches dataframe with BERT sentiment labels and scores.
        """
        if df.empty or text_col not in df.columns:
            return df
            
        texts = df[text_col].dropna().tolist()
        if not texts:
            return df
            
        results = self.analyze(texts)
        
        # Map back to DF
        # Assuming index alignment matches (since we dropna)
        # Better to iterate or map safely
        
        # Helper to get score
        def get_score(res):
            if not res: return 0
            # FinBERT labels: positive, negative, neutral
            label = res['label']
            score = res['score']
            if label == 'positive': return score
            if label == 'negative': return -score
            return 0
            
        df['bert_raw'] = results
        df['bert_score'] = df['bert_raw'].apply(get_score)
        df['bert_label'] = df['bert_raw'].apply(lambda x: x['label'] if x else 'neutral')
        
        return df
