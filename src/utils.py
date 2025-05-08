import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def create_wordcloud(text_list):
    """Create word cloud from text list"""
    if not text_list:
        return None
        
    text = ' '.join([str(text) for text in text_list if isinstance(text, str)])
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    return wordcloud

def plot_rating_distribution(df, rating_column='Rating'):
    """Plot rating distribution"""
    plt.figure(figsize=(10, 6))
    
    # Create rating distribution
    rating_counts = df[rating_column].value_counts().sort_index()
    
    # Plot bar chart
    ax = sns.barplot(x=rating_counts.index, y=rating_counts.values)
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(rating_counts.values):
        percentage = count / total * 100
        ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    return plt

def plot_sentiment_distribution(df, sentiment_column='sentiment_label'):
    """Plot sentiment distribution"""
    plt.figure(figsize=(10, 6))
    
    # Create sentiment distribution
    sentiment_counts = df[sentiment_column].value_counts()
    
    # Define colors
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    
    # Plot bar chart
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(sentiment_counts.values):
        percentage = count / total * 100
        ax.text(i, count + 5, f"{percentage:.1f}%", ha='center')
    
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    return plt

def plot_feature_distribution(product_features, product_name):
    """Plot feature distribution for a product"""
    if product_name not in product_features:
        return None
    
    features = product_features[product_name]
    mentions = features['mentions']
    sentiments = features['sentiments']
    
    # Sort features by mentions
    sorted_features = sorted(mentions.items(), key=lambda x: x[1], reverse=True)
    
    # Get top features
    top_features = [f[0] for f in sorted_features[:7]]  # Top 7 features
    
    # Prepare data for plotting
    plot_data = []
    for feature in top_features:
        sentiment = sentiments.get(feature, 0)
        plot_data.append({
            'Feature': feature.capitalize(),
            'Mentions': mentions[feature],
            'Sentiment': sentiment
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(121)
    bars = sns.barplot(x='Feature', y='Mentions', data=plot_df, ax=ax1)
    plt.title(f'Feature Mentions for {product_name}')
    plt.xticks(rotation=45, ha='right')
    
    ax2 = plt.subplot(122)
    sentiment_bars = sns.barplot(x='Feature', y='Sentiment', data=plot_df, ax=ax2)
    
    # Color bars based on sentiment
    for i, bar in enumerate(sentiment_bars.patches):
        if plot_df.iloc[i]['Sentiment'] > 0.1:
            bar.set_color('green')
        elif plot_df.iloc[i]['Sentiment'] < -0.1:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    plt.title(f'Feature Sentiment for {product_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt
