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
from cornac.models import MF
from sklearn.metrics import precision_score, recall_score, roc_curve, auc

SEED = 42
VERBOSE = False

# Download and preprocess MovieLens 100K dataset
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

# Function to perform K-Fold Cross Validation
def cross_validate_model(feedback, k=10, threshold=3.0):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    precisions, recalls, roc_aucs, all_fpr, all_tpr, all_thresholds = [], [], [], [], [], []

    feedback = np.array(feedback)

    for train_idx, test_idx in kf.split(feedback):
        # Split data
        train_feedback = feedback[train_idx]
        test_feedback = feedback[test_idx]

        # Convert to Cornac Dataset
        train_set = cornac.data.Dataset.from_uir(train_feedback)
        test_set = cornac.data.Dataset.from_uir(test_feedback)

        # Train the model
        model = MF(k=50, max_iter=50, learning_rate=0.01, lambda_reg=0.1, verbose=VERBOSE)
        model.fit(train_set)

        # Predict on the test set
        y_true, y_pred, scores = [], [], []
        for user, item, rating in test_feedback:
            try:
                pred_score = model.score(user, item)
                y_true.append(1 if rating >= threshold else 0)
                y_pred.append(1 if pred_score >= threshold else 0)
                scores.append(pred_score)
            except cornac.exception.ScoreException:
                # Ignore items that cannot be scored
                continue

        # Calculate metrics if there are valid predictions
        if y_true and scores:
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_aucs.append(auc(fpr, tpr))
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_thresholds.append(thresholds)

    # Return mean metrics and ROC curve values
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0
    mean_roc_auc = np.mean(roc_aucs) if roc_aucs else 0

    # Plot Precision and Recall
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(k), precisions, label="Precision")
    plt.plot(range(k), recalls, label="Recall")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Precision and Recall per Fold")
    plt.legend()

    # Plot ROC AUC Curve
    plt.subplot(1, 2, 2)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    plt.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return mean_precision, mean_recall, mean_roc_auc

# Main function
if __name__ == "__main__":
    user_df, item_df, feedback_df = load_movielens_data()

    # Prepare feedback as a list of tuples (user, item, rating)
    feedback = feedback_df.to_records(index=False)

    # Perform K-Fold Cross Validation
    print("Eseguendo K-Fold Cross Validation...")
    precision, recall, roc_auc = cross_validate_model(feedback, k=10, threshold=3.0)

    # Output results
    print("\nRisultati:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
