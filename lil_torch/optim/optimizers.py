from .optimizer import Optimizer

# Stochastic Gradient Descent implementation 
class SGD(Optimizer):
    def __init__(self, params, lr):
        lr = {'lr': lr}
        super().__init__(params=params, defaults=lr)

    def step(self):
        for param_group in self.param_groups:
            lr = param_group['lr']

            for p in param_group['params']:
                if p.grad is not None:
                    p._data -= lr * p.grad