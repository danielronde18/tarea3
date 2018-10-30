import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg 

archivo= np.genfromtxt('WDBC.txt', delimiter=',',dtype='U16')


arc=archivo.shape

for i in range(arc[0]):
	for j in range(arc[1]):
		if(j>=2):
			archivo[i][j].astype(float)

print(archivo)

datos=archivo[:,2:].astype(float)
indicador=archivo[:,:2]


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

print(cov_matrix(datos))

vals,vecs= np.linalg.eig(cov_matrix(datos))
columnas=vecs.shape[1]

for i in range(columnas):
	print ("eigenvalor",i+1,vals[i])
	print ("eigenvector",i+1,vecs[:,i])
