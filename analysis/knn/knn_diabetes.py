import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Ensure the results directory exists
if not os.path.exists("./results"):
    os.makedirs("./results")

# Load the dataset
data = pd.read_csv("../../data/diabetes_balanced.csv")

# Split data into features and target
X = data.iloc[:, 1:]  # All columns except the first
y = data.iloc[:, 0]  # The first column (Diabetes_binary)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def knn_classification(
    X_train, X_test, y_train, y_test, n_neighbors=5, grid_search=False
):
    if grid_search:
        # Define the parameter grid
        param_grid = {
            "n_neighbors": np.arange(1, 31),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }

        # Perform grid search
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        knn = grid_search.best_estimator_
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        best_params = {
            "n_neighbors": n_neighbors,
            "weights": "uniform",
            "metric": "minkowski",
        }

    # Predict on training and testing data
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return train_accuracy, test_accuracy, best_params


def plot_results(param_values, train_errors, test_errors, param_name, filename):
    plt.figure()
    plt.plot(param_values, train_errors, label="Training Error")
    plt.plot(param_values, test_errors, label="Testing Error")
    plt.xlabel(param_name)
    plt.ylabel("Error Rate")
    plt.title(f"Training and Testing Error Rates for different {param_name}")
    plt.legend()
    plt.savefig(f"./results/{filename}")
    plt.close()


# Example of using individual hyperparameter
n_neighbors_values = np.arange(1, 31)
train_errors = []
test_errors = []

for n in n_neighbors_values:
    train_acc, test_acc, _ = knn_classification(
        X_train, X_test, y_train, y_test, n_neighbors=n
    )
    train_errors.append(1 - train_acc)
    test_errors.append(1 - test_acc)

plot_results(
    n_neighbors_values,
    train_errors,
    test_errors,
    "Number of Neighbors",
    "knn_neighbors_diabetes.png",
)

# Example of using grid search
_, _, best_params = knn_classification(
    X_train, X_test, y_train, y_test, grid_search=True
)
print(f"Best hyperparameters from grid search: {best_params}")

# Calculate error rates for grid search best parameters
train_acc, test_acc, _ = knn_classification(
    X_train, X_test, y_train, y_test, n_neighbors=best_params["n_neighbors"]
)
train_error = 1 - train_acc
test_error = 1 - test_acc

print(f"Train Error: {train_error}")
print(f"Test Error: {test_error}")
