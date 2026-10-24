
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing import preprocess_news


# 1. Load and preprocess the dataset
df = pd.read_csv("data/news.csv")
df = preprocess_news(df)

if len(df) < 2:
    raise ValueError("At least 2 valid news articles are required.")

print("Dataset loaded:", len(df), "articles")


# 2. Convert news text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1, 2)
)    

X = vectorizer.fit_transform(df["clean_text"])

print("TF-IDF conversion completed!")


# 3. Group similar news using K-Means
number_of_topics = min(4, len(df))

kmeans = KMeans(
    n_clusters=number_of_topics,
    random_state=42,
    n_init=10
)

df["topic_id"] = kmeans.fit_predict(X)

# Display the most important words in each topic
terms = vectorizer.get_feature_names_out()
print("\n========== DISCOVERED TOPICS ==========")

for topic in range(number_of_topics):
    center = kmeans.cluster_centers_[topic]
    top_indices = center.argsort()[-5:][::-1]
    keywords = [terms[i] for i in top_indices]

    print(f"Topic {topic + 1}: {', '.join(keywords)}")


# 4. Analyze sentiment using VADER
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["clean_text"].apply(analyze_sentiment)

# 5. Save analyzed news
df.to_csv("news_analysis.csv", index=False)

print("\n========== SENTIMENT SUMMARY ==========")
print(df["sentiment"].value_counts())

print("\n========== SAMPLE RESULTS ==========")
print(df[["headline", "topic_id", "sentiment"]].to_string(index=False))

print("\nAnalysis completed successfully!")
print("Results saved to news_analysis.csv")
