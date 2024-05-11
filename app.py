import streamlit as st
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing

class RecommendationSystemModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,
    ):
        super(RecommendationSystemModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )

        # Hidden layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, users, movies):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        # Pass through hidden layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }



# Load the necessary data
df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")
num_users = df['userId'].nunique()
num_movies = df['movieId'].nunique()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recommendation_model = RecommendationSystemModel(
    num_users=num_users,
    num_movies=num_movies,
    embedding_size=128,
    hidden_dim=256,
    dropout_rate=0.1,
).to(device)
recommendation_model.load_state_dict(torch.load("recommendation_model.pt",map_location=torch.device('cpu')))
recommendation_model.eval()




# Streamlit app
st.title("Neural Network Collaborative Filtering Movie Recommender")
# st.markdown("##### Recommending movies using a Neural Network Collaborative Filtering model.")

# user_id = st.selectbox("Select a user ID", df["userId"].unique())
user_id = st.sidebar.selectbox("Select a user ID", df["userId"].unique())
user_id_shown = user_id

le_user = preprocessing.LabelEncoder()
le_movie = preprocessing.LabelEncoder()
df.userId = le_user.fit_transform(df.userId.values)
df.movieId = le_movie.fit_transform(df.movieId.values)

user_id = le_user.transform([user_id])[0]  # Convert to model's expected format

all_movie_ids = set(df.movieId.unique())
    
seen_movie_ids = set(df[df.userId == user_id].movieId.unique())

original_seen_movie_ids = le_movie.inverse_transform(list(seen_movie_ids))

seen_movies = movies_df[movies_df['movieId'].isin(original_seen_movie_ids)]

# st.write("Movies seen by the selected user:")
st.markdown(f"#### Movies seen by the User {user_id_shown}:")
# st.write(seen_movies)
st.dataframe(seen_movies)

# Add a dropdown menu to select the number of recommendations
# num_recommendations = st.sidebar.selectbox("Select the number of recommendations", [3, 5, 10, 15, 20])
num_recommendations = st.sidebar.slider("Select the number of recommendations", 1, 100, 10, step=1)

unseen_movie_ids = all_movie_ids - seen_movie_ids
unseen_movie_ids = list(unseen_movie_ids)

users = [user_id] * len(unseen_movie_ids)
movies = unseen_movie_ids
ratings = [0] * len(unseen_movie_ids)  # dummy ratings

BATCH_SIZE = 32
dataset = MovieLensDataset(users, movies, ratings)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

recommendation_model.eval()
predicted_ratings = []
with torch.no_grad():
    for data in data_loader:
        output = recommendation_model(data["users"].to(device), data["movies"].to(device))
        predicted_ratings.extend(output.squeeze().tolist())

movie_ratings = list(zip(unseen_movie_ids, predicted_ratings))

movie_ratings.sort(key=lambda x: x[1], reverse=True)

top_n_movie_ids = []
top_n_movie_names = []
top_n_movie_genres = []
for movie_id, _ in movie_ratings:
    if movie_id in movies_df.movieId.values:
        top_n_movie_ids.append(movie_id)
        movie_index = movies_df[movies_df.movieId == movie_id].index[0]
        top_n_movie_names.append(movies_df.title.values[movie_index])
        top_n_movie_genres.append(movies_df.genres.values[movie_index])
        if len(top_n_movie_ids) == num_recommendations:
            break

# st.write("Top 10 movie recommendations:")
st.markdown(f"#### Top {num_recommendations} movie recommendations for User {user_id_shown}:")
for i, movie_name in enumerate(top_n_movie_names, start=0):
    st.write(f"{i+1}. {movie_name} (Genres: {top_n_movie_genres[i]})")
    # st.write(f"{i+1}. {movie_name} (ID: {top_n_movie_ids[i]}) | (Genres: {top_n_movie_genres[i]})")