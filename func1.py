import neurals as nn 
import numpy as np 



x = np.array([[1,1,1],[1,1,1]])
X = nn.Tensor(data=x, requires_grad=True, label='X')

y = np.array([[2,2,2],[2,2,2]])
Y = nn.Tensor(data=y, requires_grad=True, label='Y')
 


Z = X * Y 


ZT = Z.transpose(-1,-2)
out = X.dot(ZT)
out.backward()



print(out.grad)
print(ZT.grad)
print(Z.grad)
print(X.grad)


out.build_dag()
nn.draw_dag(out)

