import numpy as np

from layer import Layer


class NeuralNet:
    """
    A class representing a neural network.

    Attributes:
    - layers: List of layers in the neural network
    - input_dim: Dimension of the input data
    - output_dim: Dimension of the output data
    """

    def __init__(self):
        """
        Initialize the NeuralNet object.
        """
        self.layers = None
        self.input_dim = None
        self.output_dim = None

    def add_layer(self, size, output_size, activation, weights=None, biases=None) -> None:
        """
        Add a new layer to the neural network.

        Parameters:
        - size: Number of neurons in the layer
        - output_size: Dimension of the output of the layer
        - activation: Activation function for the layer
        - weights: Initial weights for the layer (default=None)
        - biases: Initial biases for the layer (default=None)
        """
        new_layer = Layer(size, output_size, activation, weights, biases)

        if self.layers is None:
            self.layers = [new_layer]
            self.input_dim = size
        else:
            self.layers.append(new_layer)

        self.output_dim = output_size

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the neural network.

        Parameters:
        - data: Input data for prediction

        Returns:
        - Output data after passing through the neural network
        """
        if len(data) != self.input_dim:
            raise ValueError("Input data dimension does not match the input dimension of the neural network")

        for layer in self.layers:
            outputs = layer.feed_forward(data)
            data = outputs

        return data

    def mutate(self) -> None:
        """
        Mutate the weights and biases of the neural network.
        """
        for layer in self.layers:
            layer.mutate_layer()
