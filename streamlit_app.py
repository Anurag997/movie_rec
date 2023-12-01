# Import necessary libraries
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd
import streamlit as st

# Load the modified ratings.csv file
ratings_df = pd.read_csv('ratings_modified.csv')

# Load the movies.csv file
movies_df = pd.read_csv('movies.csv')

# Merge ratings and movies DataFrames
ratings_df = pd.merge(ratings_df, movies_df[['movieId', 'title', 'genres']], on='movieId')

# Define the Reader object
reader = Reader(rating_scale=(1, 5))

# Load the data into the Surprise Dataset
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create a collaborative filtering model using k-NN
model = KNNBasic(sim_options={'user_based': True})  # User-based collaborative filtering

# Train the model on the training set
model.fit(trainset)

# Function to get top N movie recommendations for a user based on genres
def get_genre_recommendations(model, user_id, genres, num_recommendations=10):
    # Get a list of movieIds for movies with the specified genres
    movies_with_genres = ratings_df[ratings_df['genres'].str.contains('|'.join(genres))]['movieId'].unique()

    # Get the average rating for movies in the dataset
    average_rating = ratings_df['rating'].mean()

    # Get user ratings for these movies with a more conservative default rating
    user_ratings = [(user_id, movie_id, average_rating) for movie_id in movies_with_genres]

    # Exclude movies the user has already rated
    movies_to_exclude = trainset.ur[user_id]

    # Make predictions for the selected movies
    predictions = model.test(user_ratings)

    # Filter out movies the user has already rated
    predictions = [pred for pred in predictions if pred.iid not in [item[1] for item in movies_to_exclude]]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Return the top N recommended movies
    return predictions[:num_recommendations]

# Streamlit app
def main():
    st.title("Movie Recommendation App")

    # Get user input for genres
    user_input_genres = st.text_input("Enter movie genres (comma-separated):")
    
    if user_input_genres:
        user_input_genres = user_input_genres.split(',')
        
        # Example user ID (modify as needed)
        user_id_to_recommend = 1

        # Get genre-based recommendations
        genre_recommendations = get_genre_recommendations(model, user_id_to_recommend, user_input_genres)

        # Display the genre-based recommendations
        st.subheader(f"Top 10 Recommendations for Genres {user_input_genres}:")
        for prediction in genre_recommendations:
            movie_id = prediction.iid
            score = prediction.est
            movie_title = ratings_df[ratings_df['movieId'] == movie_id]['title'].values[0]
            st.write(f"Movie: {movie_title} (MovieId: {movie_id}), Predicted Rating: {score:.2f}")

if __name__ == '__main__':
    main()
