import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def process_dataset(filename):
    # Load and prepare the data
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initial model training
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(
        f'./results/feature_importances_{os.path.basename(filename).split(".")[0]}.png'
    )
    plt.close()

    # Prune features with importance less than threshold
    threshold = 0.01
    pruned_features = [
        index for index, importance in enumerate(importances) if importance >= threshold
    ]
    X_train_pruned = X_train.iloc[:, pruned_features]
    X_test_pruned = X_test.iloc[:, pruned_features]

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        "n_estimators": [50, 75, 100, 125, 150, 200],
        "learning_rate": [0.01, 0.015, 0.035, 0.055, 0.075, 0.09, 0.1, 0.15, 0.2],
        "max_depth": [3, 4, 5],
    }
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train_pruned, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print(f"Best parameters found for {filename}: ", best_params)

    # Use the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test_pruned)
    print(
        f"Classification Report for {filename}:\n",
        classification_report(y_test, y_pred),
    )
    print(f"Accuracy for {filename}: ", accuracy_score(y_test, y_pred))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(
        f"./results/confusion_matrix_{os.path.basename(filename).split('.')[0]}_n{best_params['n_estimators']}_lr{best_params['learning_rate']}_md{best_params['max_depth']}.png"
    )
    plt.close()

    # Plot learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        best_model,
        X_train_pruned,
        y_train,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=42,
    )
    train_errors = 1 - np.mean(train_scores, axis=1)
    test_errors = 1 - np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, label="Training error")
    plt.plot(train_sizes, test_errors, label="Testing error")
    plt.xlabel("Training Size")
    plt.ylabel("Error Rate")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"./results/learning_curve_{os.path.basename(filename).split('.')[0]}_n{best_params['n_estimators']}_lr{best_params['learning_rate']}_md{best_params['max_depth']}.png"
    )
    plt.close()


os.makedirs("./results", exist_ok=True)

# Toggle datasets
process_dataset("../../data/credit_card_default_tw.csv")
# process_dataset("../../data/diabetes_balanced.csv")
