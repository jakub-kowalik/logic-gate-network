import time
from typing import Union, List, Dict, NoReturn
import numpy as np

from .gate import LogicGate


class Layer:
    """
    Layer with ability to specify number of logical gates and number of inputs.
    """

    def __init__(self, n_inputs, n_gates: Union[int, List[LogicGate], Dict[str, int]]):
        """
        Initialize layer with number of inputs and number of gates.
        :param n_inputs: number of inputs
        :param n_gates: number of gates or list of gates or dictionary of gates and number of gates.
        """
        self.gates = []
        self.n_inputs = n_inputs

        self._initialize_gates(n_gates)

        self.n_outputs = len(self.gates)

    def _initialize_gates(
        self, n_outputs: Union[int, List[LogicGate], Dict[str, int]]
    ) -> NoReturn:
        """
        Initialize layer with specified number of gates.
        :param n_outputs: number of gates
        :return:
        """
        if isinstance(n_outputs, int):
            for _ in range(n_outputs):
                self.gates.append(self._initialize_gate())
        elif isinstance(n_outputs, list):
            for operator in n_outputs:
                self.gates.append(self._initialize_gate(operator))
        elif isinstance(n_outputs, dict):
            for operator, n in n_outputs.items():
                for _ in range(n):
                    self.gates.append(self._initialize_gate(operator))
        else:
            raise TypeError("n_outputs must be int, list or dict")

    def _initialize_gate(self, operator: str = None) -> LogicGate:
        """
        Initialize gate with specified operator.
        :param operator: operator to use for computing output. If not specified, operator is randomly selected
        :return: initialized gate
        """
        idx = np.random.choice(self.n_inputs, size=2, replace=False)
        return LogicGate(idx, operator)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute output of layer.
        :param x: input vector
        :return: output vector
        """
        samples, length = x.shape

        if length != self.n_inputs:
            raise ValueError(
                f"Input size must be equal to input size of layer. Expected {self.n_inputs}, got {length}"
            )

        out = np.zeros((samples, self.n_outputs), dtype=x.dtype)

        for j, gate in enumerate(self.gates):
            out[:, j] = gate.compute(x).reshape(-1)

        return out

    def __str__(self):
        output_string = (
            f"Layer(n_inputs:{self.n_inputs}, n_outputs:{self.n_outputs}, \n\t gates:"
        )
        for gate in self.gates:
            output_string += f"\n\t\t{gate}"
        output_string += "\n)"
        return output_string

    def __repr__(self):
        return self.__str__()
