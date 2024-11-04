# A simple DeepONet implementation

This project implements a Deep Operator Network (DeepONet) model to learn the antiderivative function from a given dataset. The model uses PyTorch to define and train the network, with separate branch and trunk sub-networks, as described in the paper *Learning Operators with DeepONet* by Lu et al.

## Problem and Data Source

The objective of this project is to predict the antiderivative function given input data, following the example provided in the [DeepXDE documentation](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html).

The dataset consists of unaligned antiderivative function values:

- **Input**: The dataset provides sample points for functions to learn the operator.
- **Output**: The target values are the antiderivatives corresponding to each sample.

The data files used are:

- `antiderivative_unaligned_train.npz`: Training data.
- `antiderivative_unaligned_test.npz`: Testing data.

## Setup and Requirements

**Dependencies**: Install the required packages using `pip`:

    ```bash
    pip install torch numpy matplotlib
    ```
    
## References

- Lu, Lu, et al. *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nature Machine Intelligence 3.3 (2021): 218-229.
- [DeepXDE Documentation - Antiderivative Example](https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html)
