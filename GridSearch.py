import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caricamento dei dati
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

# Filtraggio delle feature
selected_genres = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi"]  # Generi principali
merged_df["Gender"] = merged_df["Gender"].map({"M": 1, "F": 0})
merged_df["Rating"] = merged_df["Rating"].apply(lambda x: "Positive" if x >= 4 else "Negative")
feature_columns = ["Age", "Gender"] + selected_genres
data = merged_df[feature_columns + ["Rating"]]

# Dividere i dati in train e test
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Separare le caratteristiche (X) dalla variabile target (y)
X_train = train_data[feature_columns]
y_train = train_data["Rating"]
X_test = test_data[feature_columns]
y_test = test_data["Rating"]

# Gestione dello squilibrio delle classi con SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Creazione del modello Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Parametri da testare nella GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Ottimizzazione del modello tramite GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)

# Best parameters found by GridSearch
print(f"Best Parameters: {grid_search.best_params_}")

# Predizioni con il miglior modello
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Report delle performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy:.2f}")
print("\nOptimized Classification Report:")
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
