"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    temp_array = np.sum(encodings, 1, keepdims=True)
    temp = 0
    probabilities = np.array([])
    counts = np.array([])

    for i in range(temp_array.size):
        temp += temp_array[i] / encodings.shape[1]
        print(
            i,
            temp_array[i],
            temp_array[i] / encodings.shape[1],
            np.log(temp_array[i] / encodings.shape[1]),
        )
        probabilities = np.append(probabilities, temp_array[i] / encodings.shape[1])
        counts = np.append(counts, temp_array[i])
    log_probabilities = np.log(probabilities)
    known_min_probability = -counts.T @ log_probabilities

    print("Log Probabilities: ", log_probabilities)
    print("Known Minimum Probability: ", known_min_probability)
    # set number of iterations and learning rate
    num_iterations = 1000  # SET THIS
    learning_rate = 0.01  # SET THIS

    # initialize lists to store loss and iteration values
    losses = []
    iterations = []
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        # append loss and iteration values to lists
        losses.append(loss.item())
        iterations.append(_)

    # plotting loss as a function of iterations, with known mimimum loss value
    plt.plot(iterations, losses, label="Loss")
    plt.axhline(
        y=known_min_probability,
        color="r",
        linestyle="--",
        label="Known Minimum Loss Value",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    # plt.savefig("./Resources/Iterative_Losses.png")

    gd_assigned_probabilities = normalize(torch.sigmoid(model.s))
    assigned_probabilities = np.array([])
    actual_probabilities = np.array([])
    difference = np.array([])
    characters = vocabulary.copy()
    characters[-1] = "None"
    for i in range(len(gd_assigned_probabilities)):
        print(
            i,  # Index
            characters[i],  # Character
            gd_assigned_probabilities[
                i
            ].item(),  # Predicted Probability based on Gradient Descent
            probabilities[i],  # True Probability based on the training data
            probabilities[i]
            - gd_assigned_probabilities[
                i
            ].item(),  # Difference between Actual and Predicted
        )
        assigned_probabilities = np.append(
            assigned_probabilities, gd_assigned_probabilities[i].item()
        )  # Predicted Probabilities based on Gradient Descent
        actual_probabilities = np.append(
            actual_probabilities, probabilities[i]
        )  # True Probabilities based on the training data
        difference = np.append(
            difference, probabilities[i] - gd_assigned_probabilities[i].item()
        )  # Difference between Actual and Predicted

    plt.title("Known Minimum Probabilities vs Model Optimal Probabilities")

    X = characters
    Y = actual_probabilities
    Z = assigned_probabilities

    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, Y, 0.4, label="True Probabilities")
    plt.bar(X_axis + 0.2, Z, 0.4, label="Assigned Probabilities")
    plt.xlabel("Vocabulary Element")
    plt.ylabel("Probabilities")
    plt.xticks(np.arange(len(characters)), characters)
    plt.legend()
    plt.show()
    # plt.savefig("..\Resources/Probability_Comparisons.png")


if __name__ == "__main__":
    gradient_descent_example()
