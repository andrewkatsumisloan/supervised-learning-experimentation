import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(file_path):
    data = pd.read_csv(file_path)
    base_name = os.path.basename(file_path).split(".")[0]
    return data, base_name


def preprocess_data(data):
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    return X_train, X_test, Y_train, Y_test


def hyperparameter_tuning(X_train, Y_train):
    param_grid = {
        "n_neighbors": [1, 5, 10, 15, 20, 25, 30, 35],
        # "n_neighbors": [30, 35, 40],
        "weights": ["uniform"],
        "metric": ["euclidean", "manhattan"],
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, Y_train)
    return grid_search


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    Y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)

    Y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)

    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    class_report = classification_report(Y_test, Y_test_pred)

    return train_accuracy, test_accuracy, conf_matrix, class_report


def plot_validation_curve(results, base_name):
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=results,
        x="param_n_neighbors",
        y="mean_train_score",
        marker="o",
        label="Train Accuracy",
        color="orange",
    )
    sns.lineplot(
        data=results,
        x="param_n_neighbors",
        y="mean_test_score",
        marker="o",
        label="Test Accuracy",
        color="g",
    )
    plt.title("Accuracy vs. Number of Neighbors")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")

    plt.tight_layout()

    output_path = f"./results/knn_validation_curve_{base_name}.png"
    plt.savefig(output_path)
    plt.show()


def plot_learning_curve(model, X, Y, base_name):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=Y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
    )

    train_error = 1 - np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_error = 1 - np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot training error
    sns.lineplot(
        x=train_sizes, y=train_error, marker="o", label="Training error", color="r"
    )
    plt.fill_between(
        train_sizes,
        train_error - train_std,
        train_error + train_std,
        alpha=0.1,
        color="orange",
    )

    # Plot cross-validation error
    sns.lineplot(
        x=train_sizes,
        y=test_error,
        marker="o",
        label="Cross-validation error",
        color="g",
    )
    plt.fill_between(
        train_sizes, test_error - test_std, test_error + test_std, alpha=0.1, color="g"
    )

    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Error Rate")
    plt.legend(loc="best")

    learning_curve_output_path = f"./results/knn_learning_curve_{base_name}.png"
    plt.savefig(learning_curve_output_path)
    plt.show()


def main():
    # Load and preprocess data
    # file_path = "../../data/diabetes_balanced.csv"
    file_path = "../../data/credit_card_default_tw.csv"
    data, base_name = load_data(file_path)
    X_train, X_test, Y_train, Y_test = preprocess_data(data)

    # Hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, Y_train)

    # Evaluate the best model
    best_model = grid_search.best_estimator_
    train_accuracy, test_accuracy, conf_matrix, class_report = evaluate_model(
        best_model, X_train, Y_train, X_test, Y_test
    )

    # Print evaluation results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Plot validation curve
    plot_validation_curve(pd.DataFrame(grid_search.cv_results_), base_name)

    # Plot learning curve
    plot_learning_curve(best_model, X_train, Y_train, base_name)


if __name__ == "__main__":
    main()
