import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Load saved objects using absolute paths
movies = joblib.load(os.path.join(BASE_DIR, "Model", "movies.pkl"))
similarity = joblib.load(os.path.join(BASE_DIR, "Model", "similarity.pkl"))

app = FastAPI(title="Movie Recommendation API")

# Input schema
class MovieInput(BaseModel):
    movie_name: str

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.post("/recommend")
def recommend_movies(data: MovieInput):
    movie_name = data.movie_name

    if movie_name not in movies['title'].values:
        return {"error": "Movie not found in database"}

    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = [movies.iloc[i[0]].title for i in movie_list]

    return {
        "input_movie": movie_name,
        "recommended_movies": recommendations
    }
