import numpy as np
import matplotlib.pyplot as plt


imagen=plt.imread('/home/daniel/tarea3ronderoscarlos/Arboles.png').astype(float)

plt.figure()
plt.imshow(imagen,plt.cm.gray)
plt.title('imagen original')

from scipy import fftpack
imagen_fft=fftpack.fft2(imagen)

def plot_transformada(imagen_fft):
	from matplotlib.colors import LogNorm
	plt.imshow(np.abs(imagen_fft), norm=LogNorm(vmin=5))
	plt.colorbar()

plt.figure()
plot_transformada(imagen_fft)
plt.title('transformada de fourier')
plt.savefig('ronderoscarlos_FT2D.pdf.')

