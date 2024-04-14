import numpy as np 
from itertools import zip_longest


class Operation():


    def __init__(self) -> None:
        pass


    def broadcast_axis(self, broadcasted_shape,original_shape):
        axes_to_be_summed = []
        zipped = list(zip_longest(tuple(reversed(broadcasted_shape)), tuple(reversed(original_shape)), fillvalue=None))
        for dim, (dim_broadcasted, dim_orig) in enumerate(reversed(zipped)):
            if dim_broadcasted!=dim_orig:
                axes_to_be_summed.append(dim)
        return tuple(axes_to_be_summed)


    def forward(self, a,b):
        NotImplementedError

    def backward(self,a,b):
        NotImplementedError

class Add(Operation): 

    def forward(self,a,b): 
        from .tensor import Tensor
        requires_grad = a.requires_grad or b.requires_grad 
        data = a._data + b._data
        label =  "("+ a.label +"+"+b.label + ")"
        z = Tensor(data,(a,b),requires_grad=requires_grad, operation= self, _op='+', label=label)
        self.parents = (a,b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a,b)
        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        if a.requires_grad: 
            da = 1.0 * dz
            broadcast_axes = self.broadcast_axis(da.shape,a.shape)
            da = da.sum(axis=broadcast_axes,keepdims=True) 
            a.backward(da,z)

        if b.requires_grad:
            db = 1.0 * dz
            broadcast_axes = self.broadcast_axis(db.shape,b.shape)
            db = db.sum(axis=broadcast_axes,keepdims=True) 
            b.backward(db, z)



class Neg(Operation): 
    def forward(self,a):
        from .tensor import Tensor
        requires_grad = a.requires_grad 
        data = - a._data 
        z = Tensor(data, requires_grad=requires_grad, operation=self,_op='-')
        self.parents = (a,)
        a.children.append(z)
        self.cache = a
        return z

    def backward(self, dz, z):
        a = self.cache

        if a.requires_grad: 
            da = -1.0 * dz
            broadcast_axes = self.broadcast_axis(da.shape,a.shape)
            da = da.sum(axis=broadcast_axes,keepdims=True) 
            a.backward(da,z)


class Mul(Operation):

    def forward(self, a, b):
        from .tensor import Tensor

        requires_grad = a.requires_grad or b.requires_grad
        # Get new Tensor's data:
        data = a._data * b._data
       
        # Create new Tensor:
        label =  "(" + a.label +"*"+b.label + ")"
        z = Tensor(data, (a,b),requires_grad=requires_grad, operation=self, _op='*', label=label) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(a*b) = b, apply chain rule:
            da = dz * b._data
            broadcast_axes = self.broadcast_axis(da.shape,a.shape)
            da = da.sum(axis=broadcast_axes,keepdims=True) 
            a.backward(da,z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # d/db(a*b) = a, apply chain rule:
            db = dz * a._data

            # Rescale gradient to have the same shape as "b":
            broadcast_axes = self.broadcast_axis(db.shape,b.shape)
            db = db.sum(axis=broadcast_axes,keepdims=True) 
            b.backward(db, z)



class MatMul(Operation):

    def forward(self, a, b):
        from .tensor import Tensor
        requires_grad = a.requires_grad or b.requires_grad
     
        # Get new Tensor's data:
        data = a._data @ b._data
      
        # Create new Tensor:
        label =  "(" + a.label +"@"+b.label + ")"
        z = Tensor(data, (a,b), requires_grad=requires_grad, operation=self,_op='@', label=label) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Backprop through the matmul:
            da = dz @ b._data.swapaxes(-1,-2)
            broadcast_axes = self.broadcast_axis(da.shape,a.shape)
            da = da.sum(axis=broadcast_axes,keepdims=True) 
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # Backprop through the matmul:
            db = a._data.swapaxes(-1,-2) @ dz
            broadcast_axes = self.broadcast_axis(db.shape,b.shape)
            db = db.sum(axis=broadcast_axes,keepdims=True) 
            b.backward(db, z)


class Dot(Operation):

    def forward(self, a, b):
        from .tensor import Tensor
        requires_grad = a.requires_grad or b.requires_grad
     
        # Get new Tensor's data:
        data = np.dot(a._data, b._data)
      
        # Create new Tensor:
        label =  "(" + a.label +u'\u2022' + b.label + ")"
        z = Tensor(data,(a,b) ,requires_grad=requires_grad, operation=self, _op='.', label=label) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Backprop through the matmul:
            da = np.dot(dz,b._data.T)
            broadcast_axes = self.broadcast_axis(da.shape,a.shape)
            da = da.sum(axis=broadcast_axes,keepdims=True) 
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # Backprop through the matmul:
            db = np.dot(a._data.T,dz)
            broadcast_axes = self.broadcast_axis(db.shape,b.shape)
            db = db.sum(axis=broadcast_axes,keepdims=True) 
            b.backward(db, z)



class Transpose:

    def forward(self, a, *dims):
        from .tensor import Tensor
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = a._data.swapaxes(*dims)
        # Create new Tensor:
        label =  a.label+".T"
        z = Tensor(data,(a,), requires_grad=requires_grad, operation=self, _op='T', label=label)
       
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Transpose upstream gradients:
            da = dz.swapaxes(*dims)
            a.backward(da, z)

