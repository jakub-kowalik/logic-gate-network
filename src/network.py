from typing import List, NoReturn, Union, Dict

import numpy as np

from .gate import LogicGate
from .layer import Layer


class LogicGateNetwork:
    """
    Logic gate network with ability to specify input/output size and number of layers.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize network with specified input and output size.
        :param input_size: number of inputs
        :param output_size: number of outputs
        """
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []

    def add_layer(
        self, n_outputs: Union[int, List[LogicGate], Dict[str, int]]
    ) -> NoReturn:
        """
        Add layer to network.
        :param n_outputs: number of gates or list of gates or dictionary of gates and number of gates.
        :return: None
        """
        if len(self.layers) == 0:
            self.layers.append(Layer(self.input_size, n_outputs))
        else:
            self.layers.append(Layer(self.layers[-1].n_outputs, n_outputs))

    def _check_prerequisites(self, x) -> NoReturn:
        """
        Check if input/output are valid and if there are any layer present.
        :param x: input vector
        :return: None
        """
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size must be {self.input_size}")

        if len(self.layers) == 0:
            raise ValueError("No layers added to network")

        if self.layers[-1].n_outputs != self.output_size:
            raise ValueError(f"Last layer output size must match {self.output_size}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute output of network.
        :param x: input vector
        :return: output vector
        """
        self._check_prerequisites(x)

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def __str__(self):
        out_string = (
            self.__class__.__name__
            + f"(input_size:{self.input_size}, output_size:{self.output_size}"
            + ", layers:"
        )

        for layer in self.layers:
            out_string += "\n\t" + "\n\t".join(str(layer).split("\n"))

        out_string += "\n)"

        return out_string

    def __repr__(self):
        return self.__str__()
