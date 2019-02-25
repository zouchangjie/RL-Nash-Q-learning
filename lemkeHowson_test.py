import lemkeHowson
import matrix
import rational
import numpy as np

m0 = matrix.Matrix(2, 2)
m1 = matrix.Matrix(2, 2)

m0.setItem(1, 1, 1)
m0.setItem(1, 2, -1)
m0.setItem(2, 1, -1)
m0.setItem(2, 2, 1)

m1.setItem(1, 1, -1)
m1.setItem(1, 2, 1)
m1.setItem(2, 1, 1)
m1.setItem(2, 2, -1)

probprob = lemkeHowson.lemkeHowson(m0, m1)
prob0 = np.array(probprob[0])
prob1 = np.array(probprob[1])
prob0 = np.matrix(prob0)
prob1 = np.matrix(prob1).reshape((-1, 1))
print (prob0)
print (prob1)
q = []
for i in range(m0.getNumRows()):
    for j in range(m0.getNumCols()):
        q.append(m0.getItem(i+1, j+1))
q = np.matrix(q).reshape((2,2))

print (q)
c = prob0 * q * prob1
print (c[0,0].nom() / c[0,0].denom())