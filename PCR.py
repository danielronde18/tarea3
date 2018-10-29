import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg 

archivo= np.genfromtxt('WDBC.txt', delimiter=',',usecols=(0,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

def cov_matrix(mat):
    dim = np.shape(mat)[1]
    numpunt = np.shape(mat)[0]
    cov = np.ones([dim,dim])
    for i in range(dim):
        for j in range(dim):
            mediafila = np.mean(mat[:,i])
            mediacolum = np.mean(mat[:,j])
            cov[i,j] = np.sum((mat[:,i]-mediafila) * (mat[:,j]-mediacolum)) / (numpunt -1)
    return cov

print(cov_matrix(archivo))

def propios(archivo):
    cov = cov_matrix(archivo)
    vals, vecs = numpy.linalg.eig(cov)
    return vals, vecs

print(propios(archivo))
