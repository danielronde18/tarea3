import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.fftpack import fft, fftfreq,ifft
import math

archivo= np.genfromtxt('signal.txt', delimiter=',',usecols=(0,1))

plt.figure()
plt.plot(archivo[:,0],archivo[:,1])
plt.title('señal')
plt.savefig('ronderoscarlos_signal.pdf')



def frequency(n, d = 1.0):
	f = np.arange(0, n//2 + 1)/(d*n)
	return f

def transformadafourier(archivo):
	N = len(archivo[:,0])
	num = np.linspace(0,0, N)
	for l in range(len(archivo[:,0])):
		for j in range(len(archivo[:,1])):
			num[l] = (archivo[j,1]*np.exp(-1j*2*np.pi*l*j/N)).sum()
		return num

paso=archivo[1,1]-archivo[0,1]
frecuencia=fftfreq(len(archivo[0]),paso)

plt.figure()
plt.plot(frecuencia,abs(transformadafourier(archivo)))
plt.title('transformada')
plt.savefig('')





