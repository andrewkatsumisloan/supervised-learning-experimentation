import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


def train_and_evaluate_model(C, kernel, gamma, X_train, Y_train, X_test, Y_test):
    svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    svm.fit(X_train, Y_train)

    Y_train_pred = svm.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)

    Y_test_pred = svm.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)

    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    class_report = classification_report(Y_test, Y_test_pred)

    return svm, train_accuracy, test_accuracy, conf_matrix, class_report


def plot_learning_curve(model, X, Y, base_name, C, gamma):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=Y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
    )

    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid()

    learning_curve_output_path = (
        f"./results/svm_learning_curve_{base_name}_C{C}_gamma{gamma}.png"
    )
    plt.savefig(learning_curve_output_path)
    plt.show()


def main():
    file_path = "../../data/credit_card_default_tw.csv"
    file_path = "../../data/diabetes_balanced.csv"

    data, base_name = load_data(file_path)
    X_train, X_test, Y_train, Y_test = preprocess_data(data)

    # Manually set hyperparameters
    C = 1
    kernel = "rbf"
    gamma = 0.1

    # Train and evaluate the model
    model, train_accuracy, test_accuracy, conf_matrix, class_report = (
        train_and_evaluate_model(C, kernel, gamma, X_train, Y_train, X_test, Y_test)
    )

    print(f"Parameters: C={C}, gamma={gamma}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    plot_learning_curve(model, X_train, Y_train, base_name, C, gamma)


if __name__ == "__main__":
    main()
