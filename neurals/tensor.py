from typing import List 
import numpy as np 
from .operations import Add, Neg, Mul, MatMul, Dot, Transpose

class Tensor:

    def __init__(self,data,_parents=() ,requires_grad=False, operation= None, _op= None, label= ' ' ) -> None:
        
        self._data = data
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.parents = _parents
        self.label = label
        self.shape = self._data.shape 
        self._op = _op
        if self.requires_grad: 
            self.grad = np.zeros_like(data).astype(float)


    def __repr__(self) -> str:
        return f"""({self._data}, requires_grad= {self.requires_grad})"""
    
    
    
    def data(self):
        ''' Returns the data stored in the tensor as a Numpy Array. '''
        return self._data
    

    def backward(self, grad = None, z = None):
        
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        if grad is None:
            grad = np.ones_like(self._data)

        self.grad += grad

        if z is not None:
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)
    

    def build_dag(self): 
        self.nodes = set()
        self.edges = set()
        def build_topo(v):
            if v not in self.nodes:
                self.nodes.add(v)
                for p in v.parents:
                    self.edges.add((p,v))
                    build_topo(p)
        build_topo(self)


    def tolist(self): 
        return self._data.tolist()
    
    def toarray(self): 
        return self._data
    

    def zero_grad(self): 
        self.grad = np.zeros_like(self._data).astype(float)

    def zero_grad_tree(self): 
        self.zero_grad()
        if self.operation:
            for parents in self.operation.parents: 
                parents.zero_grad_tree()
            self.operation = None 

    def __add__(self, other): 
        op = Add()
        return op.forward(self, tensor(other))
    

    def __radd__(self,other):
        op = Add()
        return op.forward(self,tensor(other))
    
    def __iadd__(self, other):
        op = Add()
        return op.forward(self, tensor(other))
    

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __isub__(self, other):
        return self + -other
    

    def __neg__(self):
        op = Neg()
        return op.forward(self) 

    def __mul__(self, other):
        op = Mul()
        return op.forward(self, tensor(other))

    def __rmul__(self, other):
        op = Mul()
        return op.forward(self, tensor(other))

    def __imul__(self, other):
        op = Mul()
        return op.forward(self, tensor(other))

    def __matmul__(self, other):
        op = MatMul()
        return op.forward(self, tensor(other))
    
    def dot(self, other): 
        op = Dot()
        return op.forward(self, tensor(other))
    

    def transpose(self, *dims):
        """
        Returns the original tensor with the two given dimentions transposed.
        Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8)

        @param *dims (integers): two dimentions to be transposed.
        """
        op = Transpose()
        return op.forward(self, *dims)
    
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)

