from lil_torch import Tensor

class Parameter(Tensor):
    # always requires grad 
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Module:
    def __init__(self):
        self.__dict__('_parameters') = {}
        self.__dict__('_modules') = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value

        if isinstance(value, Module):
            self._modules[name] = value

        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()