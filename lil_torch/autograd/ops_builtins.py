import numpy as np
from .function import Function

class Context:
    # initializes an empty Tensor list 
    def __init__(self):
        self.saved_tensors = []
    # method for extending the Tensor list
    def save_for_backward(self, *tensor):
        self.saved_tensors.extend(tensor)

# Add Function subclass 
class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, external_grad):
        dy = external_grad
        dx = external_grad
        return dx, dy
    
# Mul Function subclass 
class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, external_grad):
        x, y = ctx.saved_tensors
        dy = external_grad * x
        dx = external_grad * y
        return dx, dy
    
# MatMul Function subclass 
class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, external_grad):
        x, y = ctx.saved_tensors
        dx = external_grad @ y.T
        dy = x.T @ external_grad
        return dx, dy
    
# Pow Function subclass 
class Reciprocal(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x)
        return 1.0 / x 

    @staticmethod
    def backward(ctx, external_grad):
        x, = ctx.saved_tensors
        dx = external_grad * (-1.0 / (x ** 2.0))
        return dx
    
# Neg Function subclass 
class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, external_grad):
        dx = -external_grad
        return dx
    
# Tanh Function subclass 
class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        y = np.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, external_grad):
        y, = ctx.saved_tensors
        dx = external_grad * (1.0 - y ** 2)
        return dx