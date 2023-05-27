import torch
import torch.nn as nn
import torch.optim as optim
import logging
from nn_class import NeuralNetwork
from util_functions import train, test
from compare_functions import train_loader, test_loader
import matplotlib.pyplot as plt

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename="testing.txt",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 10
learning_rate = 0.001

activation_functions = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}

results = {}

# train/test with different activation functions
for name, activation_function in activation_functions.items():
    logging.info(f"Training with {name} activation function")
    model = NeuralNetwork(input_size, hidden_size, num_classes, activation_function).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
        )

    results[name] = {
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
        "test_accuracy_history": test_accuracy_history,
    }

# plot training loss
plt.figure()
for name, data in results.items():
    plt.plot(data["train_loss_history"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

# plot test loss
plt.figure()
for name, data in results.items():
    plt.plot(data["test_loss_history"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.legend()
plt.show()

# plot test accuracy
plt.figure()
for name, data in results.items():
    plt.plot(data["test_accuracy_history"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.show()
