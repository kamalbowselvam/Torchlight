import neurals as nn 
import numpy as np 



x = np.array([[1,1,1],[1,1,1]])
X = nn.Tensor(data=x, requires_grad=True, label='X')


y = np.array([[2,2,2],[2,2,2]])
Y = nn.Tensor(data=y, requires_grad=True, label='Y')
Z = X * Y 

e = np.array([[2,2,2],[2,2,2]])
E = nn.Tensor(data=e, requires_grad=True, label='E')

f = np.array([[2,2,2],[2,2,2]])
F = nn.Tensor(data=f, requires_grad=True, label='F')

R = Z + E

out = R + F

print(out.grad)
print(X.grad)


out.build_dag()
nn.draw_dag(out)

# print(out)
# print(out.grad)
# print(out.shape)
# print(X.grad)
# print(X.shape)

# print(Y.grad)
# print(Y.shape)