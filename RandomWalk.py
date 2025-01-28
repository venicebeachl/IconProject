import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import matplotlib.pyplot as plt

# Funzione per caricare i dati di MovieLens 100K
def load_movielens_data():
    user_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.user",
        sep="|", names=["UserID", "Age", "Gender", "Occupation", "Zip Code"]
    ).set_index("UserID")

    item_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        sep="|", encoding="ISO-8859-1",
        names=["ItemID", "Title", "Release Date", "Video Release Date", "IMDb URL",
               "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    ).set_index("ItemID").drop(columns=["Video Release Date", "IMDb URL", "unknown"])

    feedback_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"]
    ).drop(columns=["Timestamp"])

    return user_df, item_df, feedback_df

# Funzione per calcolare la similarità tra film
def compute_similarity(item_df):
    # Selezioniamo solo i generi
    item_features_str = item_df.columns[5:]  # I generi partono dalla colonna 5
    item_df['genres'] = item_df[item_features_str].apply(lambda row: " ".join([col for col in item_features_str if row[col] == 1]), axis=1)

    # Usando TfidfVectorizer per creare un vettore di caratteristiche
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(item_df['genres'])

    # Calcoliamo la similarità tra film
    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)
    return similarity_matrix

# Funzione per eseguire un Random Walk su CSP
def csp_random_walk(user_id, feedback_df, similarity_matrix, item_df, walk_length=50, top_n=15):
    # Trova i film che l'utente ha valutato positivamente
    user_ratings = feedback_df[feedback_df['UserID'] == user_id]
    liked_movies = user_ratings[user_ratings['Rating'] == 5]['ItemID'].tolist()
    if len(liked_movies) < 3:
        liked_movies = user_ratings[user_ratings['Rating'] >= 4]['ItemID'].tolist()

    # Definiamo i vincoli (ad esempio, preferenze per genere)
    preferred_genres = ["Action", "Comedy", "Drama"]  # Esempio di preferenza dell'utente

    # Iniziamo il Random Walk da più film che l'utente ha apprezzato
    walk_path = []
    current_movies = random.sample(liked_movies, min(3, len(liked_movies)))  # Selezioniamo fino a 3 film apprezzati
    walk_path.extend(current_movies)

    # Funzione per verificare se un film soddisfa i vincoli
    def satisfies_constraints(movie_id):
        genres = item_df.loc[movie_id, item_df.columns[5:]]
        movie_genres = [col for col, value in genres.items() if value == 1]
        genre_weight = sum(1 for genre in movie_genres if genre in preferred_genres)
        return genre_weight > 0  # Solo film con almeno un genere preferito

    # Cammina nel grafo per un numero di passi
    for _ in range(walk_length):
        next_movies = []
        for current_movie in current_movies:
            similar_movies = list(enumerate(similarity_matrix[current_movie - 1]))  # Trova i film più simili
            similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)  # Ordina per similarità decrescente

            # Filtra i film con similarità inferiore a una soglia
            similarity_threshold = 0.2
            weighted_movies = [(movie[0] + 1, movie[1] + (1.0 if (movie[0] + 1) in liked_movies else 0.0))
                              for movie in similar_movies if movie[1] >= similarity_threshold]

            # Seleziona solo i film che soddisfano i vincoli
            valid_movies = [movie[0] for movie in weighted_movies if satisfies_constraints(movie[0])]
            if valid_movies:
                next_movie = max(valid_movies, key=lambda x: similarity_matrix[current_movie - 1][x - 1])
                next_movies.append(next_movie)

        # Rinnova la lista dei film attuali (filtrando per i migliori)
        current_movies = next_movies[:top_n]  # Limitiamo a top_n film

        # Aggiungi il film alla sequenza di cammino
        walk_path.extend(current_movies)

    # Conta la frequenza dei film visitati
    movie_counts = defaultdict(int)
    for movie in walk_path:
        movie_counts[movie] += 1

    # Ordina per frequenza e restituisci i top N film
    recommended_movie_ids = sorted(movie_counts, key=movie_counts.get, reverse=True)[:top_n]
    recommended_movies = [(item_df.loc[movie_id, 'Title'], movie_id) for movie_id in recommended_movie_ids]

    return recommended_movies, liked_movies

# Funzione per calcolare precisione e richiamo
def calculate_metrics(recommended_items, relevant_items):
    recommended_set = {item_id for _, item_id in recommended_items}
    relevant_set = set(relevant_items)
    relevant_recommendations = recommended_set & relevant_set

    precision = len(relevant_recommendations) / len(recommended_set) if recommended_set else 0
    recall = len(relevant_recommendations) / len(relevant_set) if relevant_set else 0

    return precision, recall

# Funzione per visualizzare metriche
def plot_metrics(precision, recall):
    metrics = [precision, recall]
    labels = ['Precision', 'Recall']

    plt.bar(labels, metrics, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.ylabel('Value')
    plt.show()

# Funzione principale
if __name__ == "__main__":
    user_df, item_df, feedback_df = load_movielens_data()

    # Calcoliamo la similarità tra i film
    similarity_matrix = compute_similarity(item_df)

    # Esempio: Raccomandazioni per un utente
    user_id = 1
    recommendations, relevant_items = csp_random_walk(user_id, feedback_df, similarity_matrix, item_df, walk_length=50, top_n=15)

    # Calcola precisione e richiamo
    precision, recall = calculate_metrics(recommendations, relevant_items)

    # Mostra i risultati
    print(f"Recommended Movies (CSP Random Walk):")
    for title, movie_id in recommendations:
        print(f"- {title} (ID: {movie_id})")

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Visualizza il grafico con precisione e richiamo
    plot_metrics(precision, recall)