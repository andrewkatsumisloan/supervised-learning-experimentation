import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    X = data.iloc[:, :-1].values  # Features
    Y = data.iloc[:, -1].values  # Target (last column)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, Y_train, Y_test


class ANN(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, layer3_size, layer4_size):
        super(ANN, self).__init__()
        self.layer_1 = nn.Linear(input_size, layer1_size)
        self.layer_2 = nn.Linear(layer1_size, layer2_size)
        self.layer_3 = nn.Linear(layer2_size, layer3_size)
        self.layer_4 = nn.Linear(layer3_size, layer4_size)
        self.layer_5 = nn.Linear(layer4_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.sigmoid(self.layer_5(x))
        return x


def train_and_evaluate_model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    lr,
    num_epochs,
    layer1_size,
    layer2_size,
    layer3_size,
    layer4_size,
):
    model = ANN(X_train.shape[1], layer1_size, layer2_size, layer3_size, layer4_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_error_rates = []
    test_error_rates = []

    for epoch in range(num_epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_outputs = (outputs > 0.5).float()
        train_error_rate = (
            1 - (train_outputs == Y_train).sum().item() / Y_train.shape[0]
        )
        train_error_rates.append(train_error_rate)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)

            test_outputs = (test_outputs > 0.5).float()
            test_error_rate = (
                1 - (test_outputs == Y_test).sum().item() / Y_test.shape[0]
            )
            test_error_rates.append(test_error_rate)

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Error Rate: {train_error_rate:.4f}, Test Error Rate: {test_error_rate:.4f}"
            )

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = (test_outputs > 0.5).float()
        accuracy = (test_outputs == Y_test).sum().item() / Y_test.shape[0]
        Y_test_np = Y_test.numpy()
        test_outputs_np = test_outputs.numpy()
        class_report = classification_report(
            Y_test_np,
            test_outputs_np,
            target_names=["Class 0", "Class 1"],
            output_dict=True,
        )
        return accuracy, class_report, model, train_error_rates, test_error_rates


def plot_error_rates(train_error_rates, test_error_rates, num_epochs, filename):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_error_rates, label="Training Error Rate")
    plt.plot(epochs, test_error_rates, label="Test Error Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Error Rate")
    plt.title("Training and Test Error Rate vs Epochs")
    plt.legend()
    plt.savefig(filename)
    plt.show()


def main():
    # file_path = "../../data/credit_card_default_tw.csv"
    file_path = "../../data/diabetes_balanced.csv"

    filename = os.path.splitext(os.path.basename(file_path))[0] + "_error_rate_plot.png"
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(file_path)

    # So far optimal hps for credit card data
    lr = 0.01
    num_epochs = 75
    layer1_size = 1028
    layer2_size = 512
    layer3_size = 256
    layer4_size = 64

    # So far optimal hps for diabetes
    # lr = 0.01
    # num_epochs = 80
    # layer1_size = 512
    # layer2_size = 128
    # layer3_size = 128
    # layer4_size = 64

    accuracy, class_report, model, train_error_rates, test_error_rates = (
        train_and_evaluate_model(
            X_train,
            Y_train,
            X_test,
            Y_test,
            lr,
            num_epochs,
            layer1_size,
            layer2_size,
            layer3_size,
            layer4_size,
        )
    )

    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(
        classification_report(
            Y_test.numpy(),
            (model(X_test) > 0.5).float().numpy(),
            target_names=["Class 0", "Class 1"],
        )
    )

    plot_error_rates(train_error_rates, test_error_rates, num_epochs, filename)


if __name__ == "__main__":
    main()
