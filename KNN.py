import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN

SEED = 42
VERBOSE = False

# 1. CARICAMENTO DATI DAL DATASET MOVIELENS
user_df = pd.read_csv(
    cornac.utils.cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.user"),
    sep="|", names=["UserID", "Age", "Gender", "Occupation", "Zip Code"]
).set_index("UserID")

item_df = pd.read_csv(
    cornac.utils.cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.item"),
    sep="|", encoding="ISO-8859-1",
    names=["ItemID", "Title", "Release Date", "Video Release Date", "IMDb URL",
           "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
           "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
           "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
).set_index("ItemID").drop(columns=["Video Release Date", "IMDb URL", "unknown"])

feedback_df = pd.read_csv(
    cornac.utils.cache("http://files.grouplens.org/datasets/movielens/ml-100k/u.data"),
    sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"]
).drop(columns=["Timestamp"])

# 2. DIVISIONE DEL DATASET (Train-Test)
feedback = movielens.load_feedback(variant="100K")
ratio_split = RatioSplit(feedback, test_size=0.2, seed=SEED, verbose=VERBOSE)

# 3. CREAZIONE MODELLO UserKNN
K = 50  # numero di vicini
user_knn = UserKNN(k=K, similarity="pearson", name="UserKNN", verbose=VERBOSE)
user_knn.fit(ratio_split.train_set)

# 4. FUNZIONE PER GENERARE PREDIZIONI BINARIE
def predict_knn_binary(model, test_set, threshold=4.0):
    """
    Predice i rating come punteggi binari per un modello KNN.
    - model: il modello UserKNN o ItemKNN allenato
    - test_set: il set di test (Cornac Dataset)
    - threshold: valore per definire rating positivi/negativi
    """
    y_true = []
    y_scores = []

    # Ottieni i dati di test
    test_data = test_set.item_data

    # Itera sugli indici degli utenti, degli oggetti e sui rating
    for i in range(test_data.shape[0]):  # Aggiustato per iterare correttamente sugli indici
        user_idx, item_idx, rating = test_data[i]
        pred = model.score(user_idx, item_idx)  # Predizione del modello
        y_true.append(1 if rating >= threshold else 0)  # Rating binario
        y_scores.append(pred)

    return np.array(y_true), np.array(y_scores)

# Generazione delle predizioni
y_true, y_scores = predict_knn_binary(user_knn, ratio_split.test_set)

# Calcolo delle curve ROC e Precision-Recall
fpr, tpr, _ = roc_curve(y_true, y_scores)
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Calcolo AUC
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

# 6. VISUALIZZAZIONE DELLE CURVE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax1.plot(fpr, tpr, color="b", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend(loc="lower right")

# Precision-Recall Curve
ax2.plot(recall, precision, color="g", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.legend(loc="lower left")

plt.tight_layout()
plt.show()

# 7. STAMPA DEI RISULTATI
print("== RISULTATI ==")
print(f"Area Under ROC Curve (AUC): {roc_auc:.2f}")
print(f"Area Under Precision-Recall Curve (AUC): {pr_auc:.2f}")
