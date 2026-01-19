import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")

TMDB_PATH = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
BOLLY_PATH = os.path.join(DATA_DIR, "bollywood_movies.csv")

# ================= LOAD TMDB =================
print("Loading TMDB dataset...")
tmdb = pd.read_csv(TMDB_PATH)

tmdb = tmdb[['title', 'overview']]
tmdb['overview'] = tmdb['overview'].fillna("")

# 🔥 LIMIT SIZE (Render free tier safe)
tmdb = tmdb.head(1500)

# ================= LOAD BOLLYWOOD =================
print("Loading Bollywood dataset...")
bolly = pd.read_csv(BOLLY_PATH)

# 🔥 LIMIT SIZE
bolly = bolly.head(800)

# ================= CREATE TEXT FOR BOLLYWOOD =================
print("Creating text features for Bollywood movies...")

text_cols = [
    col for col in bolly.columns
    if col.lower() in ['genre', 'genres', 'actors', 'cast', 'director', 'keywords', 'story']
]

if text_cols:
    bolly['overview'] = bolly[text_cols].astype(str).agg(" ".join, axis=1)
else:
    bolly['overview'] = bolly['title'].astype(str)

bolly = bolly[['title', 'overview']]
bolly['overview'] = bolly['overview'].fillna("")

# ================= MERGE DATA =================
movies = pd.concat([tmdb, bolly], ignore_index=True)
print(f"Total movies used for training: {len(movies)}")

# ================= VECTORIZE =================
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000  # 🔥 MEMORY CONTROL
)

vectors = vectorizer.fit_transform(movies['overview'])

# ================= SIMILARITY =================
print("Computing similarity matrix...")
similarity = cosine_similarity(vectors)

# ================= SAVE MODELS =================
print("Saving model files...")
joblib.dump(movies, os.path.join(BASE_DIR, "movies.pkl"))
joblib.dump(similarity, os.path.join(BASE_DIR, "similarity.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("✅ Hollywood + Bollywood model trained successfully")
