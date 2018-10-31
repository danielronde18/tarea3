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







