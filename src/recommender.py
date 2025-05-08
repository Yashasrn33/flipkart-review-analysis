import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SimpleRecommender:
    """A simple recommendation system for Flipkart products based on reviews"""
    
    def __init__(self):
        """Initialize the recommender"""
        self.product_profiles = None
        self.tfidf_vectorizer = None
        self.product_vectors = None
        self.similarity_matrix = None
    
    def fit(self, df, text_column='cleaned_text', product_column='Product_name', rating_column='Rating'):
        """Build product profiles based on review text"""
        # Aggregate reviews by product
        self.product_profiles = df.groupby(product_column).agg({
            text_column: ' '.join,
            rating_column: 'mean',
            'polarity': 'mean' if 'polarity' in df.columns else lambda x: 0
        }).reset_index()
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.product_vectors = self.tfidf_vectorizer.fit_transform(
            self.product_profiles[text_column]
        )
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.product_vectors)
        
        return self
    
    def get_recommendations(self, product_name, top_n=5):
        """Get recommendations for a product"""
        if self.product_profiles is None:
            return "Recommender not fitted yet. Call fit() first."
        
        # Find product index
        try:
            idx = self.product_profiles[self.product_profiles['Product_name'] == product_name].index[0]
        except (IndexError, KeyError):
            return f"Product '{product_name}' not found."
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity (excluding the product itself)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:top_n+1]
        
        # Get recommended products
        product_indices = [i[0] for i in similarity_scores]
        similarity_values = [i[1] for i in similarity_scores]
        
        # Create recommendations DataFrame
        recommendations = self.product_profiles.iloc[product_indices][['Product_name', 'Rating']]
        recommendations['similarity'] = similarity_values
        
        return recommendations
    
    def get_recommendations_by_features(self, target_features, top_n=5):
        """Get recommendations based on target features"""
        if self.product_profiles is None:
            return "Recommender not fitted yet. Call fit() first."
        
        # Create a pseudo-document with target features
        target_text = ' '.join(target_features)
        
        # Transform using the fitted vectorizer
        target_vector = self.tfidf_vectorizer.transform([target_text])
        
        # Calculate similarity with all products
        similarity_scores = cosine_similarity(target_vector, self.product_vectors).flatten()
        
        # Sort by similarity
        product_indices = similarity_scores.argsort()[::-1][:top_n]
        
        # Create recommendations DataFrame
        recommendations = self.product_profiles.iloc[product_indices][['Product_name', 'Rating']]
        recommendations['similarity'] = similarity_scores[product_indices]
        
        return recommendations
