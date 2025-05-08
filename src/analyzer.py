import pandas as pd
import re
import nltk
from textblob import TextBlob
from collections import defaultdict
import logging

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ReviewAnalyzer:
    """A class for analyzing product reviews from Flipkart"""
    
    def __init__(self):
        """Initialize the analyzer with product feature categories"""
        self.feature_categories = {
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'powerful'],
            'battery': ['battery', 'backup', 'charge', 'life'],
            'display': ['display', 'screen', 'resolution', 'color'],
            'design': ['design', 'look', 'build', 'quality', 'weight'],
            'price': ['price', 'value', 'worth', 'expensive', 'cheap', 'cost'],
            'camera': ['camera', 'photo', 'picture', 'image', 'video'],
            'sound': ['sound', 'audio', 'speaker', 'bass', 'noise']
        }
    
    def clean_text(self, text):
        """Clean and preprocess review text"""
        if not isinstance(text, str):
            return ""
        
        # Remove special characters, URLs, etc.
        text = re.sub('@[A-Za-z0-9_]+', '', text)
        text = re.sub('#','',text)
        text = re.sub('https?:\/\/\S+', '', text)
        text = re.sub('\n',' ',text)
        text = re.sub(r'www\S+', " ", text)
        text = re.sub(r'\.|/|:|-', " ", text)
        text = re.sub(r'[^\w\s]','',text)
        
        return text.lower().strip()
    
    def get_sentiment(self, text):
        """Get sentiment scores using TextBlob"""
        if not text:
            return {'polarity': 0, 'subjectivity': 0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def get_sentiment_label(self, polarity):
        """Convert polarity score to sentiment label"""
        if polarity >= 0.1:
            return 'Positive'
        elif polarity <= -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_sentiment_from_rating(self, rating):
        """Convert rating to sentiment label"""
        if rating >= 4:
            return 'Positive'
        elif rating <= 2:
            return 'Negative'
        else:
            return 'Neutral'
    
    def extract_features(self, text):
        """Extract product features from text"""
        if not text:
            return {}
        
        features_found = {}
        for category, terms in self.feature_categories.items():
            for term in terms:
                if term in text:
                    if category not in features_found:
                        features_found[category] = []
                    features_found[category].append(term)
        
        return features_found
    
    def analyze_review(self, review_text, rating=None):
        """Analyze a single review"""
        # Clean text
        cleaned_text = self.clean_text(review_text)
        
        # Get sentiment
        sentiment = self.get_sentiment(cleaned_text)
        sentiment_label = self.get_sentiment_label(sentiment['polarity'])
        
        # Get rating-based sentiment if rating is provided
        rating_sentiment = self.get_sentiment_from_rating(rating) if rating else None
        
        # Extract features
        features = self.extract_features(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'sentiment_label': sentiment_label,
            'rating_sentiment': rating_sentiment,
            'features': features
        }
    
    def process_dataframe(self, df, review_column='Review', rating_column='Rating'):
        """Process all reviews in a DataFrame"""
        results = []
        
        for idx, row in df.iterrows():
            review_text = row[review_column]
            rating = row[rating_column] if rating_column in df.columns else None
            
            analysis = self.analyze_review(review_text, rating)
            analysis['index'] = idx
            
            # Add other columns from the original DataFrame
            for col in df.columns:
                if col not in analysis:
                    analysis[col] = row[col]
            
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def aggregate_product_features(self, df, product_column='Product_name'):
        """Aggregate feature mentions and sentiments by product"""
        product_features = {}
        
        for product in df[product_column].unique():
            product_df = df[df[product_column] == product]
            
            feature_mentions = defaultdict(int)
            feature_sentiments = defaultdict(list)
            
            for _, row in product_df.iterrows():
                for feature_category in row['features']:
                    feature_mentions[feature_category] += 1
                    feature_sentiments[feature_category].append(row['polarity'])
            
            # Calculate average sentiment for each feature
            feature_avg_sentiments = {}
            for feature, sentiments in feature_sentiments.items():
                feature_avg_sentiments[feature] = sum(sentiments) / len(sentiments)
            
            product_features[product] = {
                'mentions': dict(feature_mentions),
                'sentiments': feature_avg_sentiments
            }
        
        return product_features
    
    def get_product_strengths_weaknesses(self, product_features):
        """Get strengths and weaknesses for each product"""
        results = {}
        
        for product, features in product_features.items():
            sentiments = features['sentiments']
            
            # Sort features by sentiment
            sorted_features = sorted(sentiments.items(), key=lambda x: x[1], reverse=True)
            
            # Get strengths (positive sentiment)
            strengths = [feature for feature, sentiment in sorted_features if sentiment > 0.1]
            
            # Get weaknesses (negative sentiment)
            weaknesses = [feature for feature, sentiment in sorted_features if sentiment < -0.1]
            
            results[product] = {
                'strengths': strengths[:3],  # Top 3 strengths
                'weaknesses': weaknesses[:3]  # Top 3 weaknesses
            }
        
        return results