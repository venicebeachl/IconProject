import os
import sys
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cornac
from cornac.utils import cache
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN, ItemKNN
from sklearn.metrics import precision_score, recall_score, roc_curve, auc

SEED = 42
VERBOSE = False

# Download some information of MovieLens 100K dataset
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

# Convert to a list of (user, item, rating) tuples
dataset = cornac.data.Dataset.from_uir(feedback_df.itertuples(index=False, name=None))

# UserKNN models
uknn_cosine = UserKNN(k=2, similarity="cosine", verbose=VERBOSE).fit(dataset)
print(f"Cosine(1,3) = {uknn_cosine.sim_mat[0, 2]:.3f}")

uknn_pearson = UserKNN(k=2, similarity="pearson", verbose=VERBOSE).fit(dataset)
print(f"Pearson(1,3) = {uknn_pearson.sim_mat[0, 2]:.3f}")

# UserKNN methods
K = 50  # number of nearest neighbors
uknn_base = UserKNN(
  k=K, similarity="pearson", name="UserKNN-Base", verbose=VERBOSE
)
uknn_amp1 = UserKNN(
  k=K, similarity="pearson", amplify=0.5, name="UserKNN-Amp0.5", verbose=VERBOSE
)
uknn_amp2 = UserKNN(
  k=K, similarity="pearson", amplify=3.0, name="UserKNN-Amp3.0", verbose=VERBOSE
)
uknn_idf = UserKNN(
  k=K, similarity="pearson", weighting="idf", name="UserKNN-IDF", verbose=VERBOSE
)
uknn_bm25 = UserKNN(
  k=K, similarity="pearson", weighting="bm25", name="UserKNN-BM25", verbose=VERBOSE
)

feedback = movielens.load_feedback(variant="100K")
ratio_split = RatioSplit(feedback, test_size=0.1, seed=SEED, verbose=VERBOSE)
cornac.Experiment(
  eval_method=ratio_split,
  models=[uknn_base, uknn_amp1, uknn_amp2, uknn_idf, uknn_bm25],
  metrics=[cornac.metrics.RMSE()],
).run()

# ItemKNN model
iknn_adjusted = ItemKNN(k=50, similarity="cosine", name="ItemKNN-Adjusted", verbose=VERBOSE)
iknn_adjusted.fit(ratio_split.train_set)

# Extract rating matrix and mappings
rating_mat = iknn_adjusted.train_set.matrix
user_id2idx = iknn_adjusted.train_set.uid_map
user_idx2id = list(iknn_adjusted.train_set.user_ids)
item_id2idx = iknn_adjusted.train_set.iid_map
item_idx2id = list(iknn_adjusted.train_set.item_ids)

# User-specific analysis
TOPK = 5
UID = 1
UIDX = user_id2idx[str(UID)]

print(f"UserID = {UID}")
print("-" * 25)
print(user_df.loc[UID])

# Top-rated items by user
rating_arr = rating_mat[UIDX].A.ravel()
top_rated_items = np.argsort(rating_arr)[-TOPK:]
print(f"\nTOP {TOPK} RATED ITEMS BY USER {UID}:")
print("Ratings:", rating_arr[top_rated_items])
print(item_df.loc[[int(item_idx2id[i]) for i in top_rated_items]])

# Recommendations
recommendations, scores = iknn_adjusted.rank(UIDX)
print(f"\nTOP {TOPK} RECOMMENDATIONS FOR USER {UID}:")
print("Scores:", scores[recommendations[:TOPK]])
print(item_df.loc[[int(item_idx2id[i]) for i in recommendations[:TOPK]]])

# Analyze nearest neighbors and contributions
df = defaultdict(list)
score_arr = iknn_adjusted.ui_mat[UIDX].A.ravel()
rated_items = np.nonzero(rating_mat[UIDX])[1]

for rec in recommendations[:TOPK]:
    sim_arr = iknn_adjusted.sim_mat[rec].A.ravel()
    nearest_neighbor = rated_items[np.argsort(sim_arr[rated_items])[-1]]
    sim = sim_arr[nearest_neighbor]
    score = score_arr[nearest_neighbor]
    df["Recommendation"].append(item_df.loc[[int(item_idx2id[rec])]]["Title"].values[0])
    df["Item NN"].append(nearest_neighbor)
    df["Similarity"].append(sim)
    df["Score of the NN"].append(score)
    df["Contribution"].append((score * sim) / np.abs(sim))

rec_df = pd.DataFrame.from_dict(df)
print(rec_df)

# Number of nearest neighbors rated
n_nearest_neighbors = []
for rec in recommendations[:TOPK]:
    nearest_neighbors = np.argsort(iknn_adjusted.sim_mat[rec].A.ravel())[-K:]
    n_nearest_neighbors.append(len(np.intersect1d(nearest_neighbors, rated_items)))

rec_df["Number of rated NN"] = n_nearest_neighbors

# Visualization
fig, ax = plt.subplots(figsize=(14, 5))
sns.barplot(x="Recommendation", y="Number of rated NN", data=rec_df, palette="ch:.25", ax=ax)
ax.set_xticklabels([textwrap.fill(x.get_text(), 25) for x in ax.get_xticklabels()])
plt.show()

# Precision, Recall, ROC, and AUC Calculation

# Set a threshold for recommendations
threshold = 3.5  # Rating threshold for considering a recommendation as positive

# True positives (if user rated positively)
y_true = (rating_mat[UIDX] >= threshold).A.ravel()

# Predicted scores (1 if score >= threshold, else 0)
y_pred = (scores >= threshold).astype(int)

# Calculate Precision and Recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC: {roc_auc:.3f}")

# Visualize the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
