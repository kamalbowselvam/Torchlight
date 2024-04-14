from .tensor import * 
import numpy as np 


def tensor(data, requires_grad=False): 
    return Tensor(data, requires_grad=requires_grad)