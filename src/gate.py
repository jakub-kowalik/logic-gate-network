from typing import List, Tuple, Union, NoReturn
import numpy as np

from .logical_operations import (
    not_gate,
    and_gate,
    or_gate,
    nand_gate,
    nor_gate,
    xor_gate,
)


GATES = {
    "not": not_gate,
    "and": and_gate,
    "or": or_gate,
    "nand": nand_gate,
    "nor": nor_gate,
    "xor": xor_gate,
}


class LogicGate:
    """
    Logic gate with ability to specify indices and operator.
    Indices are used to select values from input vector.
    Operator is used to compute output over selected values.

    Valid operators are:

    ['not', 'and', 'or', 'nand', 'nor', 'xor']
    """

    def __init__(
        self, indices: Union[List[int], Tuple[int, ...]] = (0, 0), operator: str = None
    ):
        """
        Initialize logic gate with indices and operator.

        Valid operators are:

        ['not', 'and', 'or', 'nand', 'nor', 'xor'].

        NOTE: If not specified, operator is randomly selected
        :param indices: indices to select values from input vector. Number of indices is specified by operator.
        Currently all operators take 2 inputs. The 'not' operator takes 2 but use only first.
        :param operator: operator to use for computing output.
        """

        self.indices = indices

        self.operator = (
            GATES[operator]
            if operator in GATES
            else np.random.choice(list(GATES.values()))
        )

        self.input = None
        self.output = None

    def compute(self, inputs: np.ndarray) -> bool | np.ndarray[bool]:
        """
        Compute output of logic gate.
        :param inputs: input vector
        :return: output of logic gate
        """
        self.input = inputs[:, self.indices]
        self.output = self.operator(inputs[:, self.indices])
        return self.output

    def set_indices(self, indices: Union[List[int], Tuple[int, ...]]) -> NoReturn:
        """
        Set indices for logic gate.
        :param indices: indices to select values from input vector. Number of indices is specified by operator.
        :return: None
        """
        self.indices = indices

    def set_operator(self, operator: str) -> NoReturn:
        """
        Set operator for logic gate.
        :param operator: operator to use for computing output. NOTE: operator must be valid.
        :return: None
        """
        if not operator or operator not in GATES:
            raise ValueError("Operator must be valid")
        self.operator = GATES[operator]

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"operator:{self.operator.__name__}, "
            f"idx:{self.indices})"
        )

    def __repr__(self):
        return self.__str__()
