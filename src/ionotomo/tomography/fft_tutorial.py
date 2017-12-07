import numpy as np
import pylab as plt

def f1(x):
    return np.exp(-x**2)

def g1(k):
    return np.sqrt(np.pi) * np.exp(-np.pi**2 * k**2)

T = 1.
N = 1000
x = np.linspace(-T,T,N)

f = f1(x)

dk = 1./T/N

k = np.arange(N)
g = g1(2*np.pi*k*dk)
g_ = T*np.fft.fftshift(np.fft.fft(f))
print(np.angle(g_/g))
plt.plot(g)
plt.plot(np.abs(g_))
plt.show()
