from abc import ABC, abstractmethod

# abstract base class (ABC) definition for Fuction class 
class Function(ABC):
    @staticmethod
    @abstractmethod
    def forward(ctx, *args):
        '''Subclass must implement this method in its forward pass operations.'''
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, external_grad):
        '''Subclass must calculate VJP for backpropagation.'''
        pass