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





