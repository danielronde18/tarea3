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

