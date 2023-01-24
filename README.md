# Logic gate network

This is a simple logic gate network implementation in python.

Implementation is based on description of logic gate network from paper [Deep Differentiable Logic Gate Networks](https://arxiv.org/pdf/2210.08277.pdf).

It lets users create logic gates and connect them together to create a network and feed forward data through it.
Currently, the network supports the following gates:
- AND
- OR
- XOR
- NOT
- NAND
- NOR

## Requirements

The project was developed using `python 3.10.9`. 
It should work with any python 3 version, but it was not tested.
This project also needs `numpy` to run.

## Usage

Example of how one can use this project is presented in `example.ipynb` notebook. 

**_Note:_** to run this example, one need to also install `jupyter`.
