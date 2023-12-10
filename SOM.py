import numpy as np
import random
from typing import Tuple
import tqdm


class SelfOrganazingMap:
    def __init__(self,
                 map_dim1: int,
                 map_dim2: int,
                 input_dim: int,
                 init_learning_rate: float = 0.2,
                 iterations: int = 1500) -> None:
        self.init_learning_rate = init_learning_rate
        self.iterations = iterations
        self.init_nbhood_raduis = max(map_dim1, map_dim2)
        self.time_constant = iterations / np.log(self.init_nbhood_raduis)

        # random weight initialization
        self.weights = np.random.rand(map_dim1, map_dim2, input_dim)

        np.random.seed(0)
        random.seed(0)

    def _find_bmu(self, input: np.ndarray) -> Tuple[np.ndarray, Tuple[int]]:
        """ Return BMU neuron & its index """
        bmu_idx = np.array([0, 0])
        sq_distances = np.sum((self.weights - input) ** 2, axis=2)
        bmu_idx = np.unravel_index(sq_distances.argmin(), sq_distances.shape)
        return self.weights[bmu_idx], bmu_idx

    def train(self, inputs: np.ndarray) -> None:
        for epoch in tqdm.tqdm(range(self.iterations)):
            # pick random input
            input = inputs[random.randint(0, inputs.shape[0] - 1)]

            # find the BMU (Best Matching Unit)
            bmu, bmu_idx = self._find_bmu(input)

            # evaluate epoch parameters
            raduis = self.raduis(epoch)
            learning_rate = self.learning_rate(epoch)

            # update weights
            rows, cols, _ = self.weights.shape
            for row in range(rows):
                for col in range(cols):
                    # calculate grid distance between neuron and BMU
                    distance = np.sqrt(
                        np.sum((np.array([row, col]) - bmu_idx) ** 2))
                    # update only if the distance is within the raduis
                    if distance <= raduis:
                        neuron = self.weights[row, col]
                        influence = self.neighboorhood_influance(
                            distance, raduis)
                        new_weight = neuron + learning_rate * \
                            influence * (input - neuron)

                        self.weights[row, col, :] = new_weight

    def neighboorhood_influance(self, distance: float, nbhod_raduis: float) -> float:
        """ Gaussian the neighborhood function """
        return np.exp(-distance**2 / 2 / nbhod_raduis**2)

    def learning_rate(self, iteration: int) -> float:
        return self.init_learning_rate * np.exp(-iteration / self.time_constant)

    def raduis(self, iteration: int) -> float:
        return self.init_nbhood_raduis * np.exp(-iteration / self.time_constant)

    def classify(self, input: np.ndarray) -> int:
        _, bmu_idx = self._find_bmu(input)
        return np.ravel_multi_index(bmu_idx, self.weights.shape[-2:])