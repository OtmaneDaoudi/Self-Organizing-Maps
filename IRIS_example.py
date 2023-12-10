from sklearn import datasets
import matplotlib.pyplot as plt
from SOM import *

iris = datasets.load_iris()
data = iris["data"]
labels = iris["target"]
label_names = iris["target_names"]
feature_names = iris["feature_names"]

# Data normalization
petal_lengths = data[:, 2]
petal_lengths = petal_lengths / petal_lengths.max()

petal_wdiths = data[:, 3]
petal_wdiths = petal_wdiths / petal_wdiths.max()

# SOM
som = SelfOrganazingMap(2, 2, 2, init_learning_rate=.2, iterations=3000)


# plot data
fig, ax = plt.subplots(1, 3)
colors = {
    0: "r",
    1: "b",
    2: "g",
    3: "orange"
}
# initial map
ax[0].set_title("Initial SOM")
ax[0].set_xlabel("Petal length")
ax[0].set_ylabel("Petal width")
for petal_length, petal_width, label in zip(petal_lengths, petal_wdiths, labels):
    ax[0].scatter(petal_length, petal_width, color=colors[label])
legend_elements = [
    plt.Line2D([0], [0], color='r', lw=2, label='Sesota'),
    plt.Line2D([0], [0], color='b', lw=2, label='Versicolor'),
    plt.Line2D([0], [0], color='g', lw=2, label='Virginica'),
    plt.Line2D([0], [0], color='black', lw=2, label='Map nodes'),
]
for row in som.weights:
    for neuron in row:
        ax[0].scatter(neuron[0], neuron[1], color="black")
ax[0].legend(handles=legend_elements)

# train SOM
som.train(np.column_stack((petal_lengths, petal_wdiths)))

# plot resulting map
ax[1].set_title("Trained SOM")
ax[1].set_xlabel("Petal length")
ax[1].set_ylabel("Petal width")
for petal_length, petal_width, label in zip(petal_lengths, petal_wdiths, labels):
    ax[1].scatter(petal_length, petal_width, color=colors[label])
legend_elements = [
    plt.Line2D([0], [0], color='r', lw=2, label='Sesota'),
    plt.Line2D([0], [0], color='b', lw=2, label='Versicolor'),
    plt.Line2D([0], [0], color='g', lw=2, label='Virginica'),
    plt.Line2D([0], [0], color='black', lw=2, label='Map nodes'),
]
for row in som.weights:
    for neuron in row:
        ax[1].scatter(neuron[0], neuron[1], color="black")
ax[1].legend(handles=legend_elements)

# Resulting classification
ax[2].set_title("Resulting classification")
for petal_length, petal_width in zip(petal_lengths, petal_wdiths):
    cluster = som.classify(np.array([petal_length, petal_width]))
    ax[2].scatter(petal_length, petal_width, color=colors[cluster])
plt.show()