#Gauss-Seidel iterations
import numpy as np
import numpy.linalg as la
#
x = 0.0
y = 0.0
ni = 8
#
A = np.array([ [3.0, 2.0],
               [1.0, 4.0] ])
b = np.array([7.0, 9.0])
#
# does not converge for this:
#A = np.array([ [1.0, 2.0],
#               [3.0, 4.0] ])
#b = np.array([5.0, 11.0])
#
D = np.diag(np.diag(A))
L = np.tril(A) - D
U = np.triu(A) - D
A2 = np.matmul(la.inv(L+D), U)
normA2 = la.norm(A2)
print('||A2||=', normA2)
#
print('A=')
print(A)
print('L=')
print(L)
print('U=')
print(U)
print('D=')
print(D)
#
for i in range(ni):
    x = (b[0] - A[0,1]*y)/A[0,0]
    y = (b[1] - A[1,0]*x)/A[1,1]
    print('i=%i, x=%.2f, y=%.2f'%(i, x, y))
#
# this does converge, but slower...
#for i in range(ni):
#    x = (b[0] - A[0,1]*y0)/A[0,0]
#    y = (b[1] - A[1,0]*x0)/A[1,1]
#    print('i=%i, x=%.2f, y=%.2f'%(i, x, y))
#    x0 = x
#    y0 = y
print('done!')