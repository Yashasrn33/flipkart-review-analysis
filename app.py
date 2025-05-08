import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import os
import sys

from src.analyzer import ReviewAnalyzer
from src.recommender import SimpleRecommender
from src.utils import create_wordcloud, plot_rating_distribution, plot_sentiment_distribution, plot_feature_distribution

# Set page configuration
st.set_page_config(
    page_title="Flipkart Review Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load and process data
@st.cache_data
def load_and_process_data(file_path):
    """Load and process data"""
    # Read data
    df = pd.read_csv(file_path)
    
    # Initialize analyzer
    analyzer = ReviewAnalyzer()
    
    # Process the data
    processed_df = analyzer.process_dataframe(df)
    
    # Aggregate product features
    product_features = analyzer.aggregate_product_features(processed_df)
    
    # Get product strengths and weaknesses
    strengths_weaknesses = analyzer.get_product_strengths_weaknesses(product_features)
    
    # Initialize recommender
    recommender = SimpleRecommender()
    recommender.fit(processed_df)
    
    return df, processed_df, product_features, strengths_weaknesses, recommender, analyzer

# Sidebar
st.sidebar.title("Flipkart Review Analysis")
st.sidebar.markdown("Analyze product reviews and sentiments")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Flipkart reviews CSV", type="csv")

if uploaded_file is not None:
    # Load and process data
    try:
        with st.spinner('Loading and processing data...'):
            file_path = "temp_uploaded.csv"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            df, processed_df, product_features, strengths_weaknesses, recommender, analyzer = load_and_process_data(file_path)
            
            st.session_state.df = df
            st.session_state.processed_df = processed_df
            st.session_state.product_features = product_features
            st.session_state.strengths_weaknesses = strengths_weaknesses
            st.session_state.recommender = recommender
            st.session_state.analyzer = analyzer
            st.session_state.data_loaded = True
        
        st.sidebar.success("Data loaded and processed!")
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
else:
    # Try to use default data path
    try:
        if not st.session_state.data_loaded:
            default_path = "data/flipkart.csv"
            if os.path.exists(default_path):
                with st.spinner('Loading default data...'):
                    df, processed_df, product_features, strengths_weaknesses, recommender, analyzer = load_and_process_data(default_path)
                    
                    st.session_state.df = df
                    st.session_state.processed_df = processed_df
                    st.session_state.product_features = product_features
                    st.session_state.strengths_weaknesses = strengths_weaknesses
                    st.session_state.recommender = recommender
                    st.session_state.analyzer = analyzer
                    st.session_state.data_loaded = True
                
                st.sidebar.success("Default data loaded!")
    except:
        pass

# Main content
if st.session_state.data_loaded:
    df = st.session_state.df
    processed_df = st.session_state.processed_df
    product_features = st.session_state.product_features
    strengths_weaknesses = st.session_state.strengths_weaknesses
    recommender = st.session_state.recommender
    analyzer = st.session_state.analyzer
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Filter by product
    all_products = df['Product_name'].unique().tolist()
    selected_product = st.sidebar.selectbox("Select Product", options=all_products)
    
    # Filter data by selected product
    filtered_df = df[df['Product_name'] == selected_product]
    filtered_processed = processed_df[processed_df['Product_name'] == selected_product]
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sentiment Analysis", "Feature Analysis", "Recommendations"])
    
    with tab1:
        # Overview tab
        st.header(f"Review Analysis for {selected_product}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Number of Reviews", f"{len(filtered_df)}")
        
        with col2:
            avg_rating = filtered_df['Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}/5")
        
        with col3:
            positive_count = len(filtered_processed[filtered_processed['sentiment_label'] == 'Positive'])
            positive_percent = (positive_count / len(filtered_processed)) * 100 if len(filtered_processed) > 0 else 0
            st.metric("Positive Sentiment", f"{positive_percent:.1f}%")
        
        with col4:
            negative_count = len(filtered_processed[filtered_processed['sentiment_label'] == 'Negative'])
            negative_percent = (negative_count / len(filtered_processed)) * 100 if len(filtered_processed) > 0 else 0
            st.metric("Negative Sentiment", f"{negative_percent:.1f}%")
        
        # Rating distribution
        st.subheader("Rating Distribution")
        
        rating_counts = filtered_df['Rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title='Rating Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample reviews
        st.subheader("Sample Reviews")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Reviews**")
            positive_reviews = filtered_processed[filtered_processed['sentiment_label'] == 'Positive'].head(3)
            for i, row in positive_reviews.iterrows():
                st.markdown(f"*Rating: {row['Rating']}/5*")
                st.markdown(f"> {row['Review']}")
                st.markdown("---")
        
        with col2:
            st.markdown("**Negative Reviews**")
            negative_reviews = filtered_processed[filtered_processed['sentiment_label'] == 'Negative'].head(3)
            for i, row in negative_reviews.iterrows():
                st.markdown(f"*Rating: {row['Rating']}/5*")
                st.markdown(f"> {row['Review']}")
                st.markdown("---")
    
    with tab2:
        # Sentiment Analysis tab
        st.header(f"Sentiment Analysis for {selected_product}")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        
        sentiment_counts = filtered_processed['sentiment_label'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
            title="Sentiment Distribution"
        )
        fig.update_layout(legend_title="Sentiment")
        st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds
        st.subheader("Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Reviews Word Cloud**")
            positive_texts = filtered_processed[filtered_processed['sentiment_label'] == 'Positive']['cleaned_text'].tolist()
            wordcloud = create_wordcloud(positive_texts)
            
            if wordcloud:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("No positive reviews to display")
        
        with col2:
            st.markdown("**Negative Reviews Word Cloud**")
            negative_texts = filtered_processed[filtered_processed['sentiment_label'] == 'Negative']['cleaned_text'].tolist()
            wordcloud = create_wordcloud(negative_texts)
            
            if wordcloud:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("No negative reviews to display")
        
        # Rating vs Sentiment
        st.subheader("Rating vs Sentiment")
        
        rating_sentiment = filtered_processed.groupby(['Rating', 'sentiment_label']).size().reset_index(name='count')
        fig = px.bar(
            rating_sentiment, 
            x='Rating', 
            y='count', 
            color='sentiment_label',
            barmode='group',
            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
            title="Rating vs Sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Feature Analysis tab
        st.header(f"Feature Analysis for {selected_product}")
        
        if selected_product in product_features:
            features = product_features[selected_product]
            
            # Feature mentions
            st.subheader("Feature Mentions")
            
            mentions = features['mentions']
            sentiments = features['sentiments']
            
            # Sort features by mentions
            sorted_features = sorted(mentions.items(), key=lambda x: x[1], reverse=True)
            
            # Create DataFrame
            feature_df = pd.DataFrame([
                {'Feature': feature.capitalize(), 'Mentions': count, 'Sentiment': sentiments.get(feature, 0)}
                for feature, count in sorted_features
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot feature mentions
                fig = px.bar(
                    feature_df.head(7),  # Top 7 features
                    x='Feature',
                    y='Mentions',
                    title="Feature Mentions"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot feature sentiments
                fig = px.bar(
                    feature_df.head(7),  # Top 7 features
                    x='Feature',
                    y='Sentiment',
                    color='Sentiment',
                    color_continuous_scale=['red', 'gray', 'green'],
                    range_color=[-0.5, 0.5],
                    title="Feature Sentiment"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Strengths and weaknesses
            st.subheader("Strengths and Weaknesses")
            
            col1, col2 = st.columns(2)
            
            if selected_product in strengths_weaknesses:
                product_sw = strengths_weaknesses[selected_product]
                
                with col1:
                    st.markdown("**Strengths**")
                    if product_sw['strengths']:
                        for strength in product_sw['strengths']:
                            sentiment_score = sentiments.get(strength, 0)
                            st.markdown(f"- **{strength.capitalize()}** (Sentiment: {sentiment_score:.2f})")
                    else:
                        st.write("No clear strengths identified")
                
                with col2:
                    st.markdown("**Weaknesses**")
                    if product_sw['weaknesses']:
                        for weakness in product_sw['weaknesses']:
                            sentiment_score = sentiments.get(weakness, 0)
                            st.markdown(f"- **{weakness.capitalize()}** (Sentiment: {sentiment_score:.2f})")
                    else:
                        st.write("No clear weaknesses identified")
            
            # Feature-specific reviews
            st.subheader("Feature-Specific Reviews")
            
            # Select feature
            top_features = [feature.capitalize() for feature, _ in sorted_features[:7]]  # Top 7 features
            selected_feature = st.selectbox("Select Feature", options=top_features)
            selected_feature_lower = selected_feature.lower()
            
            # Find reviews mentioning the feature
            feature_reviews = []
            for _, row in filtered_processed.iterrows():
                features_mentioned = row['features']
                if selected_feature_lower in features_mentioned:
                    feature_reviews.append(row)
            
            if feature_reviews:
                # Convert to DataFrame
                feature_reviews_df = pd.DataFrame(feature_reviews)
                
                # Display positive and negative reviews
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Positive {selected_feature} Reviews**")
                    positive_feat_reviews = feature_reviews_df[feature_reviews_df['sentiment_label'] == 'Positive'].head(2)
                    for i, row in positive_feat_reviews.iterrows():
                        st.markdown(f"*Rating: {row['Rating']}/5*")
                        st.markdown(f"> {row['Review']}")
                        st.markdown("---")
                
                with col2:
                    st.markdown(f"**Negative {selected_feature} Reviews**")
                    negative_feat_reviews = feature_reviews_df[feature_reviews_df['sentiment_label'] == 'Negative'].head(2)
                    for i, row in negative_feat_reviews.iterrows():
                        st.markdown(f"*Rating: {row['Rating']}/5*")
                        st.markdown(f"> {row['Review']}")
                        st.markdown("---")
            else:
                st.write(f"No reviews specifically mentioning {selected_feature}")
        else:
            st.write("No feature analysis available for this product")
    
    with tab4:
        # Recommendations tab
        st.header(f"Similar Products to {selected_product}")
        
        # Get recommendations
        recommendations = recommender.get_recommendations(selected_product)
        
        if isinstance(recommendations, pd.DataFrame):
            # Display recommendations
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"### {i}. {row['Product_name']}")
                    st.markdown(f"**Rating:** {row['Rating']:.2f}/5")
                    st.markdown(f"**Similarity:** {row['similarity']:.2f}")
                
                with col2:
                    # Get product strengths
                    if row['Product_name'] in strengths_weaknesses:
                        product_sw = strengths_weaknesses[row['Product_name']]
                        
                        strengths_text = ", ".join([s.capitalize() for s in product_sw['strengths']]) if product_sw['strengths'] else "None identified"
                        weaknesses_text = ", ".join([w.capitalize() for w in product_sw['weaknesses']]) if product_sw['weaknesses'] else "None identified"
                        
                        st.markdown(f"**Strengths:** {strengths_text}")
                        st.markdown(f"**Weaknesses:** {weaknesses_text}")
                    
                    # Get sample review
                    product_reviews = df[df['Product_name'] == row['Product_name']]
                    if not product_reviews.empty:
                        sample_review = product_reviews.iloc[0]['Review']
                        st.markdown(f"**Sample Review:**")
                        st.markdown(f"> {sample_review}")
                
                st.markdown("---")
        else:
            st.write(recommendations)  # Display error message
        
        # Feature-based recommendations
        st.subheader("Feature-Based Recommendations")
        st.markdown("Find products based on specific features you're interested in")
        
        # Select features
        all_features = list(analyzer.feature_categories.keys())
        selected_features = st.multiselect(
            "Select Features You're Interested In",
            options=[f.capitalize() for f in all_features],
            default=[]
        )
        
        if selected_features:
            # Convert back to lowercase
            selected_features_lower = [f.lower() for f in selected_features]
            
            # Get recommendations
            feature_recommendations = recommender.get_recommendations_by_features(selected_features_lower)
            
            if isinstance(feature_recommendations, pd.DataFrame):
                # Display recommendations
                for i, (_, row) in enumerate(feature_recommendations.iterrows(), 1):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"### {i}. {row['Product_name']}")
                        st.markdown(f"**Rating:** {row['Rating']:.2f}/5")
                        st.markdown(f"**Match Score:** {row['similarity']:.2f}")
                    
                    with col2:
                        # Get product strengths
                        if row['Product_name'] in strengths_weaknesses:
                            product_sw = strengths_weaknesses[row['Product_name']]
                            
                            # Highlight selected features in strengths
                            strengths = []
                            for s in product_sw['strengths']:
                                if s in selected_features_lower:
                                    strengths.append(f"**{s.capitalize()}**")
                                else:
                                    strengths.append(s.capitalize())
                            
                            strengths_text = ", ".join(strengths) if strengths else "None identified"
                            weaknesses_text = ", ".join([w.capitalize() for w in product_sw['weaknesses']]) if product_sw['weaknesses'] else "None identified"
                            
                            st.markdown(f"**Strengths:** {strengths_text}")
                            st.markdown(f"**Weaknesses:** {weaknesses_text}")
                        
                        # Check if product has the selected features
                        if row['Product_name'] in product_features:
                            product_feats = product_features[row['Product_name']]
                            mentions = product_feats['mentions']
                            
                            feature_matches = []
                            for feature in selected_features_lower:
                                if feature in mentions:
                                    feature_matches.append(feature.capitalize())
                            
                            if feature_matches:
                                st.markdown(f"**Matching Features:** {', '.join(feature_matches)}")
                            else:
                                st.markdown("**Matching Features:** None directly mentioned")
                    
                    st.markdown("---")
            else:
                st.write(feature_recommendations)  # Display error message
else:
    # Show instructions to upload data
    st.title("Flipkart Review Analysis")
    st.markdown("Please upload a Flipkart reviews CSV file using the sidebar to start the analysis.")
    
    st.markdown("""
    ### Expected CSV Format:
    The CSV file should contain the following columns:
    - **Product_name**: Name of the product
    - **Review**: Text of the review
    - **Rating**: Numerical rating (1-5)
    
    ### What You Can Do:
    Once data is loaded, you can:
    - Analyze sentiment distribution
    - Extract key product features
    - Identify strengths and weaknesses
    - Get product recommendations
    - Compare different products
    """)
