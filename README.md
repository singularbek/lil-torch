# lil_torch

A minimal PyTorch-inspired deep learning library built in Python. I wanted to implement the main components of PyTorch to better appreciate and understand how they work.

## What’s implemented so far

- `Tensor` class with NumPy backend including major mathematical operation including +, -, \*, /, @ (Matrix Multiplication),
  power (reciprocal with other operations) and tanh (Hyberbolic Tangent for use as a non-linearity later).
- Autograd engine via `lil_torch.autograd.Function`
- Optimizer abstraction and `SGD` optimizer using the strategy pattern

## Project structure

- `lil_torch/tensor.py` — `Tensor` class and tensor utilities
- `lil_torch/autograd/function.py` — base autograd `Function` class and gradient plumbing
- `lil_torch/autograd/ops_builtins.py` — built-in differentiable operations
- `lil_torch/optim/optimizer.py` — optimizer base class and parameter handling
- `lil_torch/optim/optimizers.py` — `SGD` optimizer implementation
- `main.py` — example usage or entry point for experiments

## Key concepts

### Tensor

The `Tensor` class wraps NumPy arrays and tracks the computational graph needed for backpropagation.

### Autograd

The autograd engine is based on `lil_torch.autograd.Function`, which defines the forward and backward behavior of differentiable operations. This enables automatic gradient computation for custom operations.

### Optimizer

`Optimizer` is designed as an extensible abstraction. The current implementation includes `SGD`, which updates parameters using gradient descent and demonstrates a strategy-pattern approach for optimizer behavior.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run examples or tests from `main.py`.

## Example usage

```python
from lil_torch.tensor import Tensor
from lil_torch.optim.optimizers import SGD

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()

print(x.grad)

optimizer = SGD([x], lr=0.01)
optimizer.step()
```

## Roadmap

Remaining pieces planned for future implementation:

- `nn.Module`
- `nn.Linear`
- `nn.Tanh`
- Additional layer and activation support
- Loss functions (like MSE and CrossEntropy) and training utilities
- More optimizer algorithms (like Adam)

## Goals

This project is intended for learning and experimentation rather than production use. It is a hands-on way to explore how PyTorch-like systems work internally and how the components connect.

---
