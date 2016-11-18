
# coding: utf-8

# In[15]:

import numpy as np
import pylab as plt
from scipy.optimize import minimize

def EQs(U,dU,ddU,S,dS,ddS,n,omega):
    '''At all times this must be valid'''
    c = 299792458.
    eq1 = ddU + (-dS**2 + (n*omega/c)**2) * U
    eq2 = 2*dS*dU + ddS*U
    return eq1,eq2

def chi(para,U_,dU_,ddU_,S_,dS_,ddS_,dz,n,omega):
    '''Args from last time step'''
    dU,ddU,dS,ddS = para
    #dU__ = dU_ + dz*ddU
    #dS__ = dS_ + dz*ddS
    U = U_ + dz*dU_ + dz**2/2. * ddU_
    S = S_ + dz*dS_ + dz**2/2. * ddS_
    eq1,eq2 = EQs(U,dU,ddU,S,dS,ddS,n,omega)
    return eq1-eq2

def N1(z):
    return 1.

def solve(z0,z,U0,omega,N):
    zp = z0
    U = [U0+1j]
    dU_ = 0
    ddU_ = 0
    S = [0+1j]
    dS_ = 0
    ddS_ = 0
    Z = [z0]
    dz= (z - z0)/1000.
    while zp <= z:
        zp += dz
        #solve for U,
        n = N(zp)
        res = minimize(chi,(dU_,ddU_,dS_,ddS_),args=(U[-1],dU_,ddU_,S[-1],dS_,ddS_,dz,n,omega))
        dU_,ddU_,dS_,ddS_ = res.x
        U.append(U[-1] + dz*dU_ + dz**2/2. * ddU_)
        S.append(S[-1] + dz*dS_ + dz**2/2. * ddS_)
        Z.append(zp)
        print U[-1],S[-1]
    plt.plot(Z,np.real(S))
    plt.show()

if __name__ == '__main__':
    solve(0.,1.,1.,1400e6,N1)
            
    


# In[ ]:



