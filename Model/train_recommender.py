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

# ================= LOAD BOLLYWOOD =================
print("Loading Bollywood dataset...")
bolly = pd.read_csv(BOLLY_PATH)

# 🔥 FORCE TEXT CREATION FOR BOLLYWOOD
print("Creating text features for Bollywood movies...")

text_parts = []

for col in bolly.columns:
    if col.lower() in ['genre', 'genres', 'actors', 'cast', 'director', 'keywords', 'story']:
        text_parts.append(bolly[col].astype(str))

if not text_parts:
    # LAST RESORT: use title itself
    bolly['overview'] = bolly['title'].astype(str)
else:
    bolly['overview'] = " ".join(text_parts)

bolly = bolly[['title', 'overview']]

# ================= MERGE =================
movies = pd.concat([tmdb, bolly], ignore_index=True)
print(f"Total movies loaded: {len(movies)}")

# ================= VECTORIZE =================
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies['overview'])

# ================= SIMILARITY =================
similarity = cosine_similarity(vectors)

# ================= SAVE =================
joblib.dump(movies, os.path.join(BASE_DIR, "movies.pkl"))
joblib.dump(similarity, os.path.join(BASE_DIR, "similarity.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("✅ Hollywood + Bollywood model trained successfully")
