import numpy as np


def not_gate(x) -> np.ndarray[bool]:
    return np.logical_not(x[:, 0]).reshape(-1, 1)


def and_gate(x) -> np.ndarray[bool]:
    return np.logical_and(*x.T).reshape(-1, 1)


def or_gate(x) -> np.ndarray[bool]:
    return np.logical_or(*x.T).reshape(-1, 1)


def nand_gate(x) -> np.ndarray[bool]:
    return not_gate(and_gate(x))


def nor_gate(x) -> np.ndarray[bool]:
    return not_gate(or_gate(x))


def xor_gate(x) -> np.ndarray[bool]:
    return np.logical_xor(*x.T).reshape(-1, 1)
