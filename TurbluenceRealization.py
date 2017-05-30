
# coding: utf-8

# In[11]:

import numpy as np
import pylab as plt
from scipy.special import gamma
from scipy.signal import convolve2d

n = 3
N = 100
x = np.linspace(0,1,N)
V = (x[-1] - x[0])**n
lvec = np.fft.fftfreq(N,d = x[1] - x[0])
B = np.random.normal(size=[N,N,N])
A = np.fft.fftn(B)*(2*np.pi)**n
theta1 = 0.5
theta2 = 0.2
theta3 = 3./2.

L,M,O = np.meshgrid(lvec,lvec,lvec)
r = np.sqrt(L**2 + M**2 + O**2)
gamma = theta1*theta2**n/gamma(theta3) /np.pi**(n/2.) * (1. + theta2**2 *r**2)**(-(theta3 + n/2.))
A *= np.sqrt(gamma)
A /= V
B = np.fft.ifftn(A).real/(2*np.pi)**n*N
#B *= theta1/np.max(B)

print(np.max(B))
print("Theta1: ",np.std(B.flatten()))


        

plt.imshow(np.mean(B,axis=0),extent=(x[0],x[-1],x[0],x[-1]))
plt.colorbar()
plt.show()
plt.imshow(np.mean(B,axis=1),extent=(x[0],x[-1],x[0],x[-1]))
plt.colorbar()
plt.show()
plt.imshow(np.mean(B,axis=2),extent=(x[0],x[-1],x[0],x[-1]))
plt.colorbar()
plt.show()


# In[40]:

help(np.fft)


# In[ ]:



