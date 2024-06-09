import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Reading and Preprocessing Data
print("Step 1: Reading and Preprocessing Data")

# Load the data, choose which dataset by uncommenting
file_path = "../../data/diabetes_balanced.csv"
# file_path = "../../data/credit_card_default_tw.csv"

data = pd.read_csv(file_path)

# Determine the base name of the file for naming outputs
base_name = os.path.basename(file_path).split(".")[0]

# Separate features and target
X = data.iloc[:, :-1].values  # Features
Y = data.iloc[:, -1].values  # Target (last column)

# Standardization of the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 2: Use Grid Search to tune Hyperparameters
print("Step 2: Hyperparameter Tuning with Grid Search")

# Define the parameter grid with reduced complexity
param_grid = {
    "C": [0.1, 1, 10],  # Fewer, broader options for the regularization parameter
    "kernel": ["rbf"],
    "gamma": [1, 0.1, 0.01],  # Reduced number of options for gamma
}

# Create the SVM model
svm = SVC(random_state=42)

# Create the GridSearchCV object with adjusted settings
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
    return_train_score=True,
)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters found by the grid search
print(f"Best Parameters: {grid_search.best_params_}")

# Step 3: Evaluating the Best Model
print("Step 3: Evaluating the Best Model")
best_model = grid_search.best_estimator_

# Predict on the training set
Y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Predict on the test set
Y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(Y_test, Y_test_pred)
print("Classification Report:")
print(class_report)

# Step 4: Visualizing Training and Testing Errors
results = pd.DataFrame(grid_search.cv_results_)

# Plotting training and testing error rates as a function of hyperparameters
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot for parameter 'C'
sns.lineplot(
    data=results, x="param_C", y="mean_train_score", label="Train Accuracy", ax=ax[0]
)
sns.lineplot(
    data=results, x="param_C", y="mean_test_score", label="Test Accuracy", ax=ax[0]
)
ax[0].set_title("Accuracy vs. C")
ax[0].set_xlabel("C")
ax[0].set_ylabel("Accuracy")

# Plot for parameter 'gamma'
sns.lineplot(
    data=results,
    x="param_gamma",
    y="mean_train_score",
    label="Train Accuracy",
    ax=ax[1],
)
sns.lineplot(
    data=results, x="param_gamma", y="mean_test_score", label="Test Accuracy", ax=ax[1]
)
ax[1].set_title("Accuracy vs. Gamma")
ax[1].set_xlabel("Gamma")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()

# Save the plots
output_path = f"./results/svm_validation_curve_{base_name}.png"
plt.savefig(output_path)

# Show the plots
plt.show()

# Step 5: Plotting the Learning Curve
print("Step 5: Plotting the Learning Curve")

train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model,
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
    train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r"
)
plt.fill_between(
    train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
)

plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid()

# Save the learning curve plot
learning_curve_output_path = f"./results/svm_learning_curve_{base_name}.png"
plt.savefig(learning_curve_output_path)

# Show the learning curve plot
plt.show()
