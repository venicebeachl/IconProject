import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dei dati MovieLens
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
).set_index("ItemID")

feedback_df = pd.read_csv(
    "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
    sep="\t", names=["UserID", "ItemID", "Rating", "Timestamp"]
).drop(columns=["Timestamp"])

# Merge dei dati per creare un unico dataset
merged_df = feedback_df.merge(user_df, left_on="UserID", right_index=True)
merged_df = merged_df.merge(item_df, left_on="ItemID", right_index=True)

# Preprocessing del dataset
merged_df["Gender"] = merged_df["Gender"].map({"M": 1, "F": 0})
merged_df["Rating"] = merged_df["Rating"].apply(lambda x: "Positive" if x >= 4 else "Negative")
features = ["Age", "Gender", "Action", "Comedy", "Drama"]
data = merged_df[features + ["Rating"]]

# Dividere i dati in train e test
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Separare le caratteristiche (X) dalla variabile target (y)
X_train = train_data[features]
y_train = train_data["Rating"]
X_test = test_data[features]
y_test = test_data["Rating"]

# Gestione dello squilibrio delle classi con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Creazione di un modello Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
rf_model.fit(X_train_res, y_train_res)

# Predizione sul test set
y_pred = rf_model.predict(X_test)

# Calcolo delle metriche
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Grafico dell'accuratezza
performance_metrics = {
    "Accuracy": accuracy
}
plt.bar(performance_metrics.keys(), performance_metrics.values(), color='skyblue')
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()
