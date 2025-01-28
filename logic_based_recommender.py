import pandas as pd

def load_movielens_data():
    # Carica i dati degli utenti
    user_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.user",
        sep="|", names=["UserID", "Age", "Gender", "Occupation", "Zip Code"]
    ).set_index("UserID")

    # Carica i dati dei film
    item_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        sep="|", encoding="ISO-8859-1",
        names=["ItemID", "Title", "Release Date", "Video Release Date", "IMDb URL",
               "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    ).set_index("ItemID").drop(columns=["Video Release Date", "IMDb URL", "unknown"])

    # Carica i dati delle valutazioni degli utenti sui film
    feedback_df = pd.read_csv(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"]
    ).drop(columns=["Timestamp"])

    return user_df, item_df, feedback_df

# Carica i dati
user_df, item_df, feedback_df = load_movielens_data()

def build_knowledge_base(user_df, item_df, feedback_df):
    kb = {}

    # Iteriamo attraverso ogni utente
    for user_id, user in user_df.iterrows():
        liked_movies = feedback_df[feedback_df["UserID"] == user_id]
        liked_movies = liked_movies[liked_movies["Rating"] > 3]
        liked_movie_ids = liked_movies["ItemID"].tolist()

        # Log per vedere quali film l'utente ha apprezzato
        print(f"User {user_id} liked movies: {liked_movie_ids}")

        # Raccomandare film simili ai generi che l'utente ha apprezzato
        recommended_movies = set()  # Usa un set per evitare duplicati
        for movie_id in liked_movie_ids:
            movie_info = item_df.loc[movie_id]
            genres = movie_info.index[movie_info == 1].tolist()

            # Log per vedere i generi di ogni film
            print(f"Movie {movie_id} has genres: {genres}")

            for genre in genres:
                if genre != "Title" and genre != "Release Date":  # Evita colonne inutili
                    # Log per vedere quali film vengono aggiunti come raccomandazioni
                    print(f"Adding movies from genre {genre}")
                    recommended_movies.update(item_df[item_df[genre] == 1].index)  # Aggiungi film del genere

        # Verifica se sono stati aggiunti film raccomandati
        print(f"Recommended movies for User {user_id}: {recommended_movies}")
        kb[user_id] = list(recommended_movies)

    return kb


def calculate_precision_recall(kb, feedback_df, item_df, top_n=10):
    total_precision = 0
    total_recall = 0
    total_users = len(kb)

    for user_id, recommended_movies in kb.items():
        # Otteniamo i film che l'utente ha visto e valutato positivamente
        liked_movies = feedback_df[feedback_df["UserID"] == user_id]
        liked_movies = liked_movies[liked_movies["Rating"] > 3]
        liked_movie_ids = liked_movies["ItemID"].tolist()

        # Calcoliamo la precisione
        true_positives = len(set(recommended_movies[:top_n]) & set(liked_movie_ids))
        precision = true_positives / top_n if top_n > 0 else 0

        # Calcoliamo il recall
        recall = true_positives / len(liked_movie_ids) if len(liked_movie_ids) > 0 else 0

        # Aggiungiamo alla somma
        total_precision += precision
        total_recall += recall

    # Calcoliamo la precisione e il recall medi
    avg_precision = total_precision / total_users if total_users > 0 else 0
    avg_recall = total_recall / total_users if total_users > 0 else 0

    return avg_precision, avg_recall

# Passo 1: costruire la knowledge base
kb = build_knowledge_base(user_df, item_df, feedback_df)

# Passo 2: calcolare la precisione e il recall
avg_precision, avg_recall = calculate_precision_recall(kb, feedback_df, item_df)

# Risultati finali
print(f"Precisione media: {avg_precision}")
print(f"Recall medio: {avg_recall}")
