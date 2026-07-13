from .tensor import Tensor

# initialize a Tensor object with given data -> can take NumPy arrary or python list
def tensor(data, requires_grad=None):
    return Tensor(data, requires_grad=requires_grad)
