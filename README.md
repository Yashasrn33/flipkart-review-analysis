# Flipkart Review Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.23.1-red)
![TextBlob](https://img.shields.io/badge/TextBlob-0.17.1-green)

A streamlined application for analyzing customer reviews from Flipkart to extract insights, sentiment, and product features.

## 📋 Overview

This application performs sentiment analysis and feature extraction on Flipkart product reviews to help identify customer preferences, product strengths and weaknesses, and generate recommendations.

![Dashboard Screenshot](https://raw.githubusercontent.com/USERNAME/flipkart-review-analysis/main/assets/dashboard.png)

## ✨ Features

- **Review Sentiment Analysis**: Analyze customer sentiments using TextBlob and rating-based classification
- **Product Feature Extraction**: Identify specific product features mentioned in reviews
- **Feature-Specific Sentiment**: Determine customer satisfaction with specific product aspects
- **Strengths & Weaknesses Identification**: Automatically identify product strengths and weaknesses
- **Product Recommendations**: Get similar products based on review content
- **Feature-Based Recommendations**: Find products based on specific features
- **Interactive Dashboard**: Explore all insights through an intuitive Streamlit interface

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Yashasrn33/flipkart-review-analysis.git
cd flipkart-review-analysis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Dashboard

```bash
# Run the Streamlit app
streamlit run app.py
```

The application will open in your default web browser. Upload a Flipkart reviews CSV file using the sidebar.

### Expected CSV Format

The CSV file should contain the following columns:

- `Product_name`: Name of the product
- `Review`: Text of the review
- `Rating`: Numerical rating (1-5)

Example:

```
Product_name,Review,Rating
Lenovo Ideapad Gaming 3,Great performance and battery life,5
Dell Inspiron 15,Average build quality but good specs,3
```

## 🧠 How It Works

1. **Data Processing**:

   - Reviews are cleaned to remove special characters and irrelevant text
   - Sentiment is analyzed using TextBlob to determine polarity (positive/negative)
   - Product features are extracted using keyword matching

2. **Sentiment Analysis**:

   - Each review is classified as Positive, Neutral, or Negative
   - Sentiment is derived from both review text and numerical ratings
   - Word clouds visualize common terms in positive and negative reviews

3. **Feature Analysis**:

   - Identifies features like performance, battery, display, design
   - Calculates sentiment for each specific feature
   - Determines strengths (positively mentioned features) and weaknesses (negatively mentioned features)

4. **Recommendations**:
   - Calculates product similarity based on review content (TF-IDF vectors)
   - Recommends similar products based on review content
   - Offers feature-based recommendations for finding products with specific characteristics

## 📊 Dashboard Sections

### Overview

- Basic product statistics
- Rating distribution
- Sample positive and negative reviews

### Sentiment Analysis

- Sentiment distribution
- Word clouds for positive and negative reviews
- Relationship between ratings and sentiment

### Feature Analysis

- Feature mentions and sentiment
- Product strengths and weaknesses
- Feature-specific reviews

### Recommendations

- Similar products based on review content
- Feature-based product recommendations
- Product highlights and sample reviews

## 🚧 Future Improvements

- Add support for multi-language reviews
- Implement aspect-based sentiment analysis for more nuanced feature analysis
- Add time-series analysis to track sentiment trends over time
- Improve feature extraction with machine learning models
- Add user-based collaborative filtering for personalized recommendations
- Implement a chatbot interface for natural language queries

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
