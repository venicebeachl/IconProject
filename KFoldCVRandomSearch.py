import os
import sys
import numpy as np
import pandas as pd
import cornac
from cornac.utils import cache
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import BPR
from sklearn.model_selection import ParameterSampler, KFold
from scipy.stats import uniform, randint
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Ignora il warning specifico
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Seed per riproducibilità
SEED = 42
VERBOSE = False

# Funzione per caricare il dataset MovieLens 100K
def load_movielens_data():
    user_df = pd.read_csv(
        cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.user"),
        sep="|", names=["UserID", "Age", "Gender", "Occupation", "Zip Code"]
    ).set_index("UserID")

    item_df = pd.read_csv(
        cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.item"),
        sep="|", encoding="ISO-8859-1",
        names=["ItemID", "Title", "Release Date", "Video Release Date", "IMDb URL",
               "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    ).set_index("ItemID").drop(columns=["Video Release Date", "IMDb URL", "unknown"])

    feedback_df = pd.read_csv(
        cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.data"),
        sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"]
    ).drop(columns=["Timestamp"])

    return user_df, item_df, feedback_df

# Funzione per mappare gli ID in un intervallo continuo
def map_ids(feedback_df):
    user_ids = feedback_df['UserID'].unique()
    item_ids = feedback_df['ItemID'].unique()

    user_id_map = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(item_ids)}

    feedback_df['UserID'] = feedback_df['UserID'].map(user_id_map)
    feedback_df['ItemID'] = feedback_df['ItemID'].map(item_id_map)

    return feedback_df, user_id_map, item_id_map

# Funzione per valutare un singolo modello
def evaluate_model(params, train_set, test_feedback, threshold):
    model = BPR(
        k=params['k'],
        max_iter=params['max_iter'],
        learning_rate=params['learning_rate'],
        lambda_reg=params['lambda_reg'],
        verbose=VERBOSE,
        seed=SEED
    )
    model.fit(train_set)

    y_true, scores = [], []
    for row in test_feedback:
        user, item, rating = row[0], row[1], row[2]
        try:
            pred_score = model.score(user, item)
            if not np.isnan(pred_score):  # Ignora i valori NaN
                y_true.append(1 if rating >= threshold else 0)
                scores.append(pred_score)
        except cornac.exception.ScoreException:
            continue

    if y_true:
        # Calcola la soglia ottimale
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Classifica le previsioni usando la soglia ottimale
        y_pred = [1 if score >= optimal_threshold else 0 for score in scores]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
        return roc_auc, precision, recall, model, optimal_threshold, y_true, scores
    return -1, 0, 0, None, None, None, None

# Funzione per valutare un set di parametri in parallelo
def evaluate_params(params, train_set, test_feedback, threshold):
    roc_auc, precision, recall, model, optimal_threshold, y_true, scores = evaluate_model(params, train_set, test_feedback, threshold)
    return roc_auc, precision, recall, model, optimal_threshold, y_true, scores, params  # Restituisci anche i parametri

# Funzione per generare il grafico a barre di Precision e Recall
def plot_precision_recall_bars(precision, recall, threshold):
    labels = ['Precision', 'Recall']
    values = [precision, recall]
    colors = ['#1f77b4', '#ff7f0e']  # Colori per Precision e Recall

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.xlabel('Metriche')
    plt.ylabel('Valore')
    plt.title(f'Precision e Recall (Threshold: {threshold})')
    plt.ylim(0, 1.1)  # Imposta il limite dell'asse y tra 0 e 1.1

    # Aggiungi i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom')

    plt.show()

# Funzione per generare la curva ROC
def plot_roc_curve(y_true, scores, threshold):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Threshold: {threshold})')
    plt.legend(loc="lower right")
    plt.show()

# Definizione della griglia di iperparametri per BPR
param_grid = {
    'k': randint(50, 300),  # Fattori latenti (range più ampio)
    'max_iter': randint(200, 1000),  # Iterazioni (range più ampio)
    'learning_rate': uniform(0.001, 0.2),  # Tasso di apprendimento (range più ampio)
    'lambda_reg': uniform(0.001, 0.2)  # Regolarizzazione (range più ampio)
}

# Main function
if __name__ == "__main__":
    # Carica il dataset MovieLens 100K
    user_df, item_df, feedback_df = load_movielens_data()

    # Filtra utenti e item con meno di N valutazioni
    min_ratings = 5
    user_counts = feedback_df['UserID'].value_counts()
    item_counts = feedback_df['ItemID'].value_counts()
    feedback_df = feedback_df[feedback_df['UserID'].isin(user_counts[user_counts >= min_ratings].index)]
    feedback_df = feedback_df[feedback_df['ItemID'].isin(item_counts[item_counts >= min_ratings].index)]

    # Mappa gli ID in un intervallo continuo
    feedback_df, user_id_map, item_id_map = map_ids(feedback_df)

    # Prepara i feedback come una lista di tuple (user, item, rating)
    feedback = feedback_df.to_records(index=False)

    # Esegui la K-Fold Cross-Validation con Random Search e Parallelizzazione
    print("Eseguendo K-Fold Cross Validation con Random Search e Parallelizzazione...")
    threshold = 3.0  # Soglia più alta
    n_folds = 5  # Numero di fold per la cross-validation
    n_iter = 20  # Numero di combinazioni casuali da testare

    print(f"\nThreshold: {threshold}")
    best_roc_auc = -1
    best_model = None
    best_precision = 0
    best_recall = 0
    best_params = None
    best_threshold = None
    best_y_true = None
    best_scores = None

    # Lista per memorizzare le metriche di ogni fold
    roc_auc_list = []
    precision_list = []
    recall_list = []

    # K-Fold Cross-Validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(feedback)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        train_feedback = [feedback[i] for i in train_idx]
        test_feedback = [feedback[i] for i in test_idx]
        train_set = cornac.data.Dataset.from_uir(train_feedback)

        # Random Search con Parallelizzazione
        param_samples = ParameterSampler(param_grid, n_iter=n_iter, random_state=SEED)
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_params)(params, train_set, test_feedback, threshold)
            for params in param_samples
        )

        # Trova il miglior risultato per questo fold
        best_fold_result = max(results, key=lambda x: x[0])  # Trova il risultato con il miglior ROC AUC
        roc_auc, precision, recall, model, optimal_threshold, y_true, scores, params = best_fold_result  # Estrai i valori

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_precision = precision
            best_recall = recall
            best_params = params  # Memorizza i parametri migliori
            best_threshold = optimal_threshold
            best_y_true = y_true
            best_scores = scores

        # Memorizza le metriche di ogni fold
        roc_auc_list.append(roc_auc)
        precision_list.append(precision)
        recall_list.append(recall)

    print(f"\nMigliori parametri trovati: {best_params}")
    print(f"Soglia ottimale: {best_threshold}")
    print(f"Miglior ROC AUC: {best_roc_auc:.3f}")
    print(f"Precision: {best_precision:.3f}")
    print(f"Recall: {best_recall:.3f}")

    # Calcola la deviazione standard delle metriche
    roc_auc_std = np.std(roc_auc_list)
    precision_std = np.std(precision_list)
    recall_std = np.std(recall_list)

    print(f"\nDeviazione standard ROC AUC: {roc_auc_std:.3f}")
    print(f"Deviazione standard Precision: {precision_std:.3f}")
    print(f"Deviazione standard Recall: {recall_std:.3f}")

    # Genera il grafico a barre per Precision e Recall
    plot_precision_recall_bars(best_precision, best_recall, best_threshold)

    # Genera la curva ROC
    if best_y_true is not None and best_scores is not None:
        plot_roc_curve(best_y_true, best_scores, best_threshold)

