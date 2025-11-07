"""
Classical NLP Agent
Goal: Implement tokenization, POS tagging, lemmatization, NER, keyword extraction,
and sentiment analysis for the corpus.

Expected output → results/classical_output.json
"""

# TODO: import spacy, nltk, scikit-learn
# TODO: load dataset (data/news_corpus.csv)
# TODO: implement classical NLP pipeline
#   - Tokenize & POS-tag texts
#   - Extract entities (ORG, PERSON, DATE)
#   - Compute TF-IDF keywords
#   - Add sentiment polarity score
# TODO: save results as JSON file


#!/usr/bin/env python3
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import json
import os

# CSV path
csv_path = "/Users/darshan/Desktop/AI_Journalist/MiniHackathon/data/news_corpus.csv"

# Check if file exists
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

# Load dataset
df = pd.read_csv(csv_path)

# Use 'title' column
text_col = "title"
if text_col not in df.columns:
    raise ValueError(f"Column '{text_col}' not found in CSV.")

# Get first 10 non-empty titles
titles = df[text_col].dropna().head(10).tolist()
ids = df['ID'].dropna().head(10).tolist()  # Get corresponding IDs

if len(titles) == 0:
    raise ValueError(f"No non-empty titles found in column '{text_col}'")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# TF-IDF setup
vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(titles)
keywords = vectorizer.get_feature_names_out()

# Create results folder
os.makedirs("results", exist_ok=True)

# Process each title
output = []
for i, (id_val, title) in enumerate(zip(ids, titles)):
    doc = nlp(title)

    # Tokenization
    tokens = [token.text for token in doc]

    # POS tagging
    pos_tags = [(token.text, token.pos_) for token in doc]

    # NER
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Sentiment
    sentiment = TextBlob(title).sentiment.polarity

    # TF-IDF keywords (based on global TF-IDF)
    text_vector = vectorizer.transform([title])
    top_indices = text_vector.toarray()[0].argsort()[-5:][::-1]
    top_keywords = [keywords[idx]
                    for idx in top_indices if text_vector[0, idx] > 0]

    output.append({
        "id": int(id_val),
        "title": title,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entities": entities,
        "tfidf_keywords": top_keywords,
        "sentiment_score": sentiment
    })

# Save results
output_path = os.path.join("results", "classical_output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"✅ Task complete! Output saved to {output_path}")
