import pandas as pd
from sklearn.metrics import precision_score, recall_score
from collections import Counter
import random

# Funzione per costruire i generi dei film
def build_genres(item_df):
    genre_columns = item_df.columns[2:]  # I generi partono dalla colonna 2 (altrimenti avremmo 'Title' come colonna 5)
    item_df['genres'] = item_df[genre_columns].apply(
        lambda row: " ".join([col for col in genre_columns if row[col] == 1]), axis=1
    )
    # Gestione di film senza generi
    item_df['genres'] = item_df['genres'].replace("", "unknown")

# Funzione per calcolare precisione e richiamo
def calculate_precision_recall(recommended, liked):
    true_positive = len(set(recommended) & set(liked))
    precision = true_positive / len(recommended) if recommended else 0
    recall = true_positive / len(liked) if liked else 0
    return precision, recall

# Funzione per raccomandare film basati sui generi
def recommend_movies_by_genres(liked_movies, item_df, feedback_df, top_n=10):
    liked_genres = Counter()
    for movie_id in liked_movies:
        genres = item_df.loc[movie_id, 'genres'] if movie_id in item_df.index else ""
        liked_genres.update(genres.split())

    if not liked_genres or "unknown" in liked_genres:  # Se i generi sono "unknown", non fare raccomandazioni basate sui generi
        return fallback_recommendations(feedback_df, item_df, top_n)

    print("Generi apprezzati:", liked_genres)  # Debug generi

    top_genres = [genre for genre, count in liked_genres.items() if count > 1]  # Generi con più di 1 occorrenza
    if not top_genres:  # Se nessun genere prevale, usa i generi più comuni
        top_genres = [genre for genre, _ in liked_genres.most_common(3)]

    # Raccomandazioni con almeno due generi condivisi e considerazione della valutazione media
    recommended = item_df[item_df['genres'].apply(lambda x: sum(1 for genre in top_genres if genre in x) >= 2)]
    recommended = recommended[~recommended.index.isin(liked_movies)]  # Escludi i film già apprezzati

    # Ordina i film per valutazione media (opzionale, solo se disponibile)
    recommended['avg_rating'] = recommended.index.map(lambda movie_id: feedback_df[feedback_df['ItemID'] == movie_id]['Rating'].mean())
    recommended = recommended.sort_values(by='avg_rating', ascending=False)

    recommended = recommended.head(top_n)
    
    return [(row['Title'], idx) for idx, row in recommended.iterrows()]

# Funzione di fallback per raccomandazioni casuali o con alte valutazioni
def fallback_recommendations(feedback_df, item_df, top_n=10):
    top_rated = (
        feedback_df.groupby('ItemID')['Rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    return [(item_df.loc[movie_id, 'Title'], movie_id) for movie_id in top_rated if movie_id in item_df.index]

# Dati di esempio
item_data = {
    'ItemID': [1, 2, 3, 4, 5],
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre1': [1, 0, 1, 0, 1],
    'Genre2': [0, 1, 0, 1, 0],
    'Genre3': [0, 0, 1, 1, 0],
}
feedback_data = {
    'UserID': [1, 1, 1, 2, 2],
    'ItemID': [1, 2, 3, 4, 5],
    'Rating': [5, 4, 5, 3, 4],
}

liked_movies = [1, 3]  # Film apprezzati dall'utente
item_df = pd.DataFrame(item_data).set_index('ItemID')
feedback_df = pd.DataFrame(feedback_data)

# Costruzione dei generi
build_genres(item_df)

# Debug dei dati
def debug_dataframes():
    print("\nDataframe Film:")
    print(item_df[['Title', 'genres']])
    print("\nDataframe Feedback:")
    print(feedback_df)

debug_dataframes()

# Raccomandazioni
recommended_movies = recommend_movies_by_genres(liked_movies, item_df, feedback_df, top_n=10)

# Precisione e Richiamo
recommended_ids = [movie_id for _, movie_id in recommended_movies]
precision, recall = calculate_precision_recall(recommended_ids, liked_movies)

# Output
print(f"Liked Movies: {liked_movies}")
print(f"Recommended Movies:")
for title, movie_id in recommended_movies:
    print(f"- {title} (ID: {movie_id})")
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
