import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite movie.")

# Backend API URL
API_URL = "https://movie-recommendation-y323.onrender.com/recommend"

# Input
movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name")
    else:
        payload = {
            "movie_name": movie_name
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()

                if "error" in data:
                    st.error(data["error"])
                else:
                    st.success(f"Recommendations for **{data['input_movie']}**:")
                    for movie in data["recommended_movies"]:
                        st.write("👉", movie)
            else:
                st.error("Backend API error")

        except Exception as e:
            st.error("Could not connect to backend API")
