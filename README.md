# Multi-Layer Perceptron (MLP) Classifier in C

A from-scratch implementation of a **Multi-Layer Perceptron (MLP)** neural network in pure C, trained with **backpropagation** and **gradient descent**. The network solves a three-class classification problem on 2D points, developed as part of a Computational Intelligence university course (Assignment 1, ak. year 2022–23).

No external machine-learning libraries are used — the forward pass, backpropagation, and weight updates are all implemented by hand.

## The Classification Problem

The dataset (`examplesSDT.txt`) consists of **8000 random points** `(x1, x2)` in the square `[-1, 1] × [-1, 1]`:

- **4000** points are used for **training**
- **4000** points are used for **testing**

Each point is assigned to one of three categories (**C1**, **C2**, **C3**) based on whether it falls inside one of several circular regions of radius² < 0.2 centered at `(±0.5, ±0.5)`, with additional constraints on `x2`. Points that match none of the regions are classified as **C3**.

This produces a non-linearly-separable problem, which is why a multi-layer (deep) network is required.

## Network Architecture

| Layer            | Neurons | Configurable |
| ---------------- | ------- | ------------ |
| Input            | 2 (`d`) | —            |
| Hidden layer 1   | 15 (`H1`) | yes (`#define`) |
| Hidden layer 2   | 15 (`H2`) | yes (`#define`) |
| Hidden layer 3   | 15 (`H3`) | yes (`#define`) |
| Output           | 3 (`K`) | —            |

- **Hidden layers:** activation function chosen by the user at runtime — **Sigmoid**, **ReLU**, or **Tanh** (selectable per layer).
- **Output layer:** Sigmoid activation, with one-hot encoded targets (`C1 = [1,0,0]`, `C2 = [0,1,0]`, `C3 = [0,0,1]`).
- **Weights/biases:** randomly initialized in the range `(-1, 1)`.
- **Learning rate:** `0.0001` (defined by `learning_rate`).

All architecture parameters are configurable through `#define` constants at the top of `neuralNetwork.c`.

## Training

The program supports three training modes through a single batch-size parameter **B** (which must be a divisor of the 4000 training examples):

- **B = 1** → **online / serial** update (weights updated after every example)
- **1 < B < 4000** → **mini-batch** update
- **B = 4000** → **full batch** update (weights updated once per epoch)

Training stops when the change in total training error between two consecutive epochs falls below a threshold (`0.01`), but only after **at least 700 epochs** have run.

The total training (sum-of-squares) error is printed each epoch and logged to `squad_error_per_epoch.txt`.

## Generalization & Output

After training, the program evaluates the network on the 4000-point test set and prints the **percentage of correct classifications** (generalization accuracy). Test results are written to `test.txt`, marking each point as correctly (`+`) or incorrectly (`-`) classified along with its true category.

Finally, `graphs.py` is invoked automatically to plot:

- The training error per epoch
- The test points colored by category and styled by correct/incorrect classification

## Requirements

- **Linux** operating system
- `gcc` with the math library (`-lm`)
- **Python 3** with `matplotlib` (for the plots)

```bash
pip install matplotlib
```

## Build & Run

A `makefile` is provided.

```bash
# Compile and run
make all

# To re-run from a clean state
make clean
make all
```

This is equivalent to compiling manually:

```bash
gcc -O2 -o neuralNetwork neuralNetwork.c -lm
./neuralNetwork
```

At startup the program prompts you to:

1. Choose an activation function for each hidden layer (`1` Sigmoid, `2` ReLU, `3` Tanh).
2. Enter the batch size **B** (must divide 4000).

When training and testing finish, the accuracy is printed and the plots are displayed.

## Project Structure

| File                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `neuralNetwork.c`       | Main implementation: MLP, forward pass, backpropagation, training, testing |
| `examplesSDT.txt`       | The 8000 generated 2D data points                                  |
| `graphs.py`             | Plots the training error and the classified test points            |
| `makefile`              | Build/run/clean targets                                            |
| `YN_project_2022_23.pdf`| Original assignment description (in Greek)                         |

## Generated Files (at runtime)

- `neuralNetwork` — compiled executable
- `squad_error_per_epoch.txt` — training error per epoch
- `test.txt` — test-set classification results

## Experimentation

The assignment encourages studying how generalization accuracy changes with:

- Different combinations of hidden-layer sizes `H1`, `H2`, `H3`
- Different hidden-layer activation functions (Sigmoid / Tanh / ReLU)
- Different batch sizes (e.g. `B = N/10` or `B = N/100`)

Adjust the `#define` constants in `neuralNetwork.c` and re-run to reproduce these experiments.

## License

Released for educational purposes.
