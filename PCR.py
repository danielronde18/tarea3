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

	
normalizacion=np.zeros_like(datos)
def normalizador(datos):
	for j in range(normalizacion.shape[1]):
		promedio=np.mean(datos[:,j])
		desviacion=np.std(datos[:,j])
		for i in range(normalizacion.shape[0]):
			normalizacion[i][j]=(datos[i][j]-promedio)/desviacion
	return normalizacion
R=normalizador(datos)	
	
print("los parametros mas importantes son aquellos que presentan los autovectores positivos ya que se expanden hacia la componente 1 ")	
#-----------------intente separar benignos y malignos de esta forma pero no funciono-------
#-----------------por tal razon no pude correr los puntos posteriores del ejercicio --------
# -desconozco la forma de arreglar el metodos siguente o la parte en la que estoy cometiendo los errores por tal razon lo mando como un comentario

#tam=archivo.shape
#contadorB=0
#contadorM=0

#for i in range(tam[0]):
#		if(archivo[i][1]=='B'):		
#			contadorB=contadorB+1	
#		else:
#			contadorM=contadorM+1

#benigno=np.zeros((contadorB,tam[1]))
#maligno=np.zeros((contadorM,tam[1]))

#indicador=0
#indice=0

#for i in range(tam[0]):
#	for j in range(tam[1]):
#		if(archivo[i][1]=='B'):	
#				benigno[indicador][j]=datos[i][j]
#				indicador=indicador+1
#		elif(archivo[i][1]=='M'):
#				maligno[indice][j]=datos[i][j]	 
#				indice=indice+1
#print(maligno)
#print(benigno)

#------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------
#esta parte corresponde a la parte final del ejercicio creo que se desarrolla de esta manera pero al necesitar de la parte anteriorla cual no pude hacer  no lo puedo correr y por tal razon lo mando en comentario

#malo=[]
#bueno=[]
#for i in range(len(malignos)):
#	malo.append(R[malignos[i]])
#	bueno.append(R[benignos[i]])
		 
#B=np.matmul(bueno,vecs)
#M=np.matmul(malo,vecs)


#plt.figure()
#plt.scatter(B[:,0],B[:,1], color='B')
#plt.scatter(M[:,0],M[:,1], color='g')
#plt.xlabel("PC1")
#plt.ylabel("PC2")


#plt.savefig('ronderoscarlos_PCA.pdf')

#-------------------------------------------------------------------------------------------------
print("es util debido a que permite tener un mejor manejo de las variables y de una u otra forma despreciar ciertas variables que no sean muy relevantes para el diagnostico ademas que permite mostrar las variables sin ninguna correlacion ")
#--------------------------------------------------------------

	
	
	
