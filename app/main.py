from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load saved objects
movies = joblib.load("model/movies.pkl")
similarity = joblib.load("model/similarity.pkl")


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

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return {
        "input_movie": movie_name,
        "recommended_movies": recommendations
    }
