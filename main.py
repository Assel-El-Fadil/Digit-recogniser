import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data).T

labels = data[0]
pixels = data[1:]

# normalise pixel values
pixels = pixels / 255.0

# one hot encoding the labels :
encoded_labels = np.zeros((labels.size, labels.max() + 1))
encoded_labels[np.arange(labels.size), labels] = 1
encoded_labels = encoded_labels.T

testdata = pd.read_csv('test.csv')
testdata = np.array(testdata).T
testdata = testdata / 255.0


def reLu(X):
    return np.maximum(0, X)


def ReLU_deriv(X):
    return X > 0


def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


# HE Initialisation
def init_params():
    weights1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    biases1 = np.zeros((10, 1))
    weights2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    biases2 = np.zeros((10, 1))
    return weights1, biases1, weights2, biases2


def propagation(w1, b1, w2, b2, inputs):
    weightedresults1 = w1.dot(inputs) + b1
    hiddenlayer = reLu(weightedresults1)
    weightedresults2 = w2.dot(hiddenlayer) + b2
    output = softmax(weightedresults2)
    return weightedresults1, hiddenlayer, weightedresults2, output


def back_prop(weightedresults1, hiddenlayer, output, weights2, pixelData, Y_batch):
    dZ2 = output - Y_batch
    m = Y_batch.shape[1]
    dW2 = 1 / m * dZ2.dot(hiddenlayer.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = weights2.T.dot(dZ2) * ReLU_deriv(weightedresults1)
    dW1 = 1 / m * dZ1.dot(pixelData.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(weights1, biases1, weights2, biases2, dW1, db1, dW2, db2, alpha):
    W1 = weights1 - alpha * dW1
    b1 = biases1 - alpha * db1
    W2 = weights2 - alpha * dW2
    b2 = biases2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# SGD
def stochastic_gradient_descent(X, Y, alpha, iterations, batch_size=64):
    W1, b1, W2, b2 = init_params()
    m = X.shape[1]

    for i in range(iterations + 1):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[:, j:j + batch_size]

            Z1, A1, Z2, A2 = propagation(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_prop(
                Z1, A1, A2, W2, X_batch, Y_batch
            )
            W1, b1, W2, b2 = update_params(
                W1, b1, W2, b2, dW1, db1, dW2, db2, alpha
            )

        if i % 50 == 0:
            print("Epoch:", i)
            _, _, _, A2_full = propagation(W1, b1, W2, b2, X)
            predictions = get_predictions(A2_full)
            print(get_accuracy(predictions, labels))

    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = testdata[:, index, None]
    prediction = make_predictions(testdata[:, index, None], W1, b1, W2, b2)
    print("Prediction: ", prediction)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


W1, b1, W2, b2 = stochastic_gradient_descent(pixels, encoded_labels, 0.01, 400)


print("====================transition to test data======================")
_, _, _, A2_full = propagation(W1, b1, W2, b2, testdata)

test = 1000

while test != 'a':
    test = int(test)
    test_prediction(test, W1, b1, W2, b2)
    test = input("Which data point to test")
