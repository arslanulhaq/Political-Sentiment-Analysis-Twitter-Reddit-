# 🗳️ Political Sentiment Analysis (Twitter & Reddit)

This project performs sentiment analysis on social media data (Twitter & Reddit) to understand public opinion on political discussions.  
It combines traditional NLP techniques, lexicon-based sentiment analyzers (TextBlob, VADER), and machine learning models to classify text into **Positive, Negative, or Neutral sentiments.

---

## 📌 Project Overview
- Datasets: Twitter & Reddit political text datasets  
- Goal: Detect and analyze public sentiment in political conversations  
- Approach:
  - Data Cleaning & Preprocessing (text normalization, stopword removal, etc.)
  - Lexicon-based analysis using TextBlob and VADER
  - Feature extraction using **TF-IDF**
  - ML models: Logistic Regression, Naive Bayes, SVM, Random Forest
  - Data visualization (EDA, sentiment distribution, word clouds, platform comparisons)

---

## 📂 Dataset
- Source: [Twitter & Reddit Sentiment Analysis Dataset](https://www.kaggle.com/datasets)  
- Size:
  - ~162k tweets
  - ~20k Reddit posts
- Labels: `Positive`, `Negative`, `Neutral`

---

## ⚙️ Installation
Run the following in your environment (Kaggle/Colab recommended):

```bash
pip install textblob vaderSentiment wordcloud plotly streamlit scikit-learn seaborn
python -m textblob.corpora.download_lite
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

---

## 🚀 Usage
1. Clone the repo and open the Jupyter notebook:
   ```bash
   git clone https://github.com/your-username/political-sentiment-analysis.git
   cd political-sentiment-analysis
   jupyter notebook sentiment-analysis.ipynb
   ```
2. Run all cells to:
   - Load datasets
   - Preprocess text
   - Apply sentiment analysis (TextBlob + VADER)
   - Train ML models
   - Visualize results

---

## 📊 Results
- Sentiment distribution across platforms (Twitter vs Reddit)  
- Lexicon-based vs ML-based predictions 
- Model performance metrics: Accuracy, Precision, Recall, F1-score  
- Visual insights:
  - Word Clouds
  - Sentiment trends
  - Confusion matrices
  - Platform-specific sentiment breakdowns  

---

## 🛠️ Technologies
- Languages: Python  
- Libraries: Pandas, NumPy, NLTK, TextBlob, VADER, Scikit-learn, Seaborn, Matplotlib, Plotly, WordCloud  
- Visualization: Plotly, Seaborn, Matplotlib   

---

## 📌 Future Work
- Deploy interactive dashboard using **Streamlit**  
- Extend dataset with real-time Twitter API data  
- Experiment with **deep learning models (LSTM, BERT)** for improved accuracy  

---
