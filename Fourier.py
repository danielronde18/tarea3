import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.fftpack import fft, fftfreq,ifft
import math

archivo= np.genfromtxt('signal.txt', delimiter=',')

plt.figure()
plt.plot(archivo[:,0],archivo[:,1])
plt.title('señal')
plt.savefig('ronderoscarlos_signal.pdf')

x=archivo[:,0]
y=archivo[:,1]

def transformadafourier(x,y):
	N = len(x)
	num = np.linspace(0,0, N)
	for l in range(len(x)):
		for j in range(len(y)):
			num[l] = y[j]*np.exp((-1j)*2*np.pi*l*j/N)
	num=num/N
	return num


paso=x[1]-x[0]

frecuencia=fftfreq(len(x),paso)

plt.figure()
plt.plot(frecuencia,abs(transformadafourier(x,y)))
plt.title('transformada')
plt.savefig('ronderoscarlos_TF.pdf')

#print("las frecuencias principales de la señal son:",)



inco=np.genfromtxt("incompletos.txt",delimiter=",")

xgraf=inco[:,0]
ygraf=inco[:,1]


def interpolar (x,y):
	x2=np.linspace(min(x),max(x),512)
	cub=sp.interpolate.interp1d(x,y,kind='cubic')
	cuad=sp.interpolate.interp1d(x,y,kind='quadratic')
	cub1=cub(x2)
	cu1=cuad(x2)
	cuadra=sp.interpolate.splrep(x,y,k=2)
	cu=sp.interpolate.splev(x2,cuadra)
	cubica=sp.interpolate.splrep(x,y,k=3)
	cub=sp.interpolate.splev(x2,cubica)
	return cu,cub,cu1,cub1,x2

cu=interpolar(xgraf,ygraf)[0]
cu1=interpolar(xgraf,ygraf)[2]
cub=interpolar(xgraf,ygraf)[1]
cub1=interpolar(xgraf,ygraf)[3]
xinter=interpolar(xgraf,ygraf)[4]



