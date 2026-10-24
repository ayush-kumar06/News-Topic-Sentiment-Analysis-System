# 📰 News Topic & Sentiment Analysis Dashboard

An AI-powered dashboard that automatically groups news articles into topics and detects the sentiment of each article. It combines classic machine learning (TF-IDF + K-Means) with VADER sentiment analysis and presents the results in an interactive Streamlit app.

---

## ✨ Features

- **Automatic topic discovery**: news articles are clustered into topics using TF-IDF and K-Means (no predefined categories).
- **Sentiment analysis**: every article is classified as Positive, Neutral or Negative using VADER.
- **Interactive dashboard** built with Streamlit and Altair:
  - KPI cards for total news and the Positive / Negative / Neutral counts
  - Sentiment distribution donut chart
  - Sentiment-by-topic stacked bar chart
  - Topic cards showing the top keywords of each cluster
  - Articles-per-topic bar chart
- **Sidebar filters**: filter by sentiment, topic, or search headlines by keyword.
- **Searchable articles table** with a **CSV download** of the filtered results.
- **Data preprocessing**: validation, duplicate removal, and text cleaning (URLs, HTML tags, punctuation, extra spaces).

---

## 🧰 Tech Stack

| Purpose | Tool |
|---|---|
| Dashboard | Streamlit |
| Charts | Altair |
| Data handling | Pandas |
| Text features & clustering | scikit-learn (TF-IDF, K-Means, CountVectorizer) |
| Sentiment analysis | VADER (vaderSentiment) |
| Language | Python 3 |

---

## 📁 Project Structure

```
news-intelligence-dashboard/
│
├── app.py                # Streamlit dashboard
├── train_model.py        # Topic clustering + sentiment analysis pipeline
├── preprocessing.py      # Data validation and text cleaning
├── data/
│   └── news.csv          # Input dataset (headline, article)
├── news_analysis.csv     # Generated results used by the dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. **Load and preprocess**: `news.csv` is validated, empty rows and duplicates are removed, and the headline and article are combined and cleaned into a `clean_text` column.
2. **Vectorize**: the cleaned text is converted into numerical features with TF-IDF (up to 1000 features, unigrams and bigrams, English stop words removed).
3. **Cluster into topics**: K-Means groups similar articles into up to 4 topics. The top keywords of each topic are printed in the terminal.
4. **Analyze sentiment**: VADER computes a compound score for each article:
   - `>= 0.05` → **Positive**
   - `<= -0.05` → **Negative**
   - otherwise → **Neutral**
5. **Save results**: the output is written to `news_analysis.csv`.
6. **Visualize**: `app.py` reads `news_analysis.csv` and renders the interactive dashboard.

---

## 📋 Requirements

- Python 3.9 or higher
- pip (Python package manager)
- A CSV dataset at `data/news.csv` with `headline` and `article` columns
- Python packages (listed in `requirements.txt`):

| Package | Used for |
|---|---|
| `streamlit` | Dashboard interface |
| `pandas` | Data loading and processing |
| `altair` | Interactive charts |
| `scikit-learn` | TF-IDF, K-Means, keyword extraction |
| `vaderSentiment` | Sentiment scoring |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd news-intelligence-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas altair scikit-learn vaderSentiment
```

### 3. Prepare your data

Place your dataset at `data/news.csv`. It must contain these columns:

| Column | Description |
|---|---|
| `headline` | The news headline |
| `article` | The full article text |

### 4. Run the analysis

```bash
python train_model.py
```

This creates `news_analysis.csv`.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

---

## 📊 Output Format

`news_analysis.csv` contains:

| Column | Description |
|---|---|
| `headline` | Original headline |
| `article` | Original article text |
| `clean_text` | Cleaned, combined text used by the models |
| `topic_id` | Cluster assigned by K-Means |
| `sentiment` | Positive, Neutral or Negative |

---

## 🔧 Customization

- **Number of topics**: change `number_of_topics` in `train_model.py` (default: 4).
- **Sentiment thresholds**: adjust the `0.05` / `-0.05` cutoffs in `analyze_sentiment()`.
- **Keywords per topic**: change `top_n` in `topic_keywords()` in `app.py`.
- **Theme colors**: edit `SENTIMENT_COLORS` and `ACCENT` at the top of `app.py`.

---

## ⚠️ Limitations

- Topic IDs are unsupervised clusters, so they are numbered but not named automatically.
- VADER is rule-based and works best on short, English text; results on long articles can lean strongly positive or negative.
- Cleaning keeps only English letters, so non-English text is not supported.
- With very small datasets, clusters may be uneven.

---

## 🔮 Future Improvements

- Automatic topic naming
- Choosing the number of topics with silhouette score
- Transformer-based sentiment models
- Live news ingestion via an API or RSS feeds
- Time-based trend charts

---

## 👤 Author

**Ayush Kumar**
- GitHub: [ayush-kumar06](https://github.com/ayush-kumar06)
- LinkedIn: [Ayush Kumar](https://www.linkedin.com/in/ayush-kumar-161380327?utm_source=share_via&utm_content=profile&utm_medium=member_android)
- Email: ayush.aiml6@gmail.com

---

## 📄 License

This project is for educational and personal use. Add your preferred license here.
