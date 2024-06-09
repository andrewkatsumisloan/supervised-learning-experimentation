import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Reading and Preprocessing Data
print("Step 1: Reading and Preprocessing Data")
# Load the data
data = pd.read_csv("../../data/credit_card_default_tw.csv")

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
    "C": [1, 10],  # Fewer, broader options for the regularization parameter
    "kernel": [
        "rbf"
    ],  # Focus on the RBF kernel as it is generally more effective for complex datasets
    "gamma": [0.1, 0.01],  # Reduced number of options for gamma
}

# Create the SVM model
svm = SVC(random_state=42)

# Create the GridSearchCV object with adjusted settings
grid_search = GridSearchCV(
    estimator=svm, param_grid=param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1
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
