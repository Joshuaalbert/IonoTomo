import numpy as np
from scipy.linalg import cho_solve

from scipy.integrate import simps
import dask.array as da

TECU = 1e13

def calc_S(CdCt, ne_tci, covariance, rays, i0):
    '''Compute Cd + G.Cm.Gt
    CdCt is shape (Na,Nt,Nd)
    ne_tci is TCI for ne in volume
    covariance is a ``ionotomo.ionosphere.covariance.Covariance`` class
    rays is shape (Na,Nt,Nd,4,Ns)
    Indexing is C-style with last element changing fastest.
    
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nr = Na*Nt*Nd
    res = np.zeros((Nr,Nr),dtype=float)

    #outer loop, ijk -> h1 = k + Nd*(j + Nt*i) C-order
    #S[h1,h1:] = this
    i = 0
    while i < Na:
        j = 0
        while j < Nt:
            k = 0
            while k < Nd:
                #inner integrand int Cm ne (ijk - i0jk)
                dx = rays[:,j:,k:,0,:] - rays[i,j,k,0,:]
                dy = rays[:,j:,k:,1,:] - rays[i,j,k,1,:]
                dz = rays[:,j:,k:,2,:] - rays[i,j,k,2,:]
                Cm = covariance(dx,dy,dz)
                Cmne = np.einsum('ijkl,l->ijkl',Cm,ne_tci.interp(rays[i,j,k,0,:],rays[i,j,k,1,:],rays[i,j,k,2,:]))
                integrand1 = simps(Cmne,rays[i,j,k,3,:],axis=3)
                integrand1 -= integrand1[i0,:,:]
                integrand2 = simps(integrand1*ne_tci.interp(
                res[k + Nd*(j + Nt*i),k + Nd*(j + Nt*i):] = integrand.flatten()
                res[k + Nd*(j + Nt*i): , k + Nd * (j + Nt * i)] = res[k + Nd*(j + Nt*i),k + Nd*(j + Nt*i):]
                k += 1
            j += 1
        i += 1
    res += np.diag(np.flatten(CdCt,order='C'))
    return res

def irls_step(m_n,m_prior,K_ne,rays,tci,covariance,CdCt, dobs,i0):
    '''Do the IRLS step about the current model point.
    Return the next model point.
    m_n+1 = m_p + epsilon_n C_m G^t T ( ( d_obs - g(m_n)) - G (m_p - m_n) )

    '''

    Na,Nt,Nd,_,Ns = rays.shape
    n_e = np.exp(m_n)
    n_e *= K_ne/TECU

    out = m_prior.copy()

    #dd = d_obs - g(m_n)
    tci.M = (n_e).reshape((tci.nx,tci.ny,tci.nz))

    S = calc_S(CdCt, tci, covariance, rays, i0)
    L = np.linalg.cholesky(S)

    g = simps(tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:]),rays[:,:,:,3,:],axis=3)
    g -= g[i0,:,:]
    dd = dobs - g

    # G (m_p - m_n) = int ray_ijk n_e dm - int ray_i0jk n_e dm
    dm = m_prior - m_n
    tci.M = (n_e*dm).reshape((tci.nx,tci.ny,tci.nz))
    Gdm = simps(tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:]),rays[:,:,:,3,:],axis=3)
    Gdm -= Gdm[i0,:,:]

    #dd - G dm
    ddGdm = (dd - Gdm).flatten()

    #wdd = S^-1 ddGdm
    wdd = cho_solve((L,True),ddGdm).reshape([Na,Nt,Nd])

    #Cm G
    X,Y,Z = tci.get_model_cooredinates()#nx*ny*nz

    j = 0
    while j < Nt:
        k = 0
        while k < Nd:
            dx = np.subtract.outer(X,rays[i0,j,k,0,:])
            #nx*ny*nz x Na*Nt*Nd*Ns
            dy = np.subtract.outer(Y,rays[i0,j,k,1,:])
            dz = np.subtract.outer(Z,rays[i0,j,k,2,:])
            Cm = covariance(dx,dy,dz)
            #nx*ny*nz x *Ns
            #int ray_ijk ne - int ray_i0jk ne Cm
            CmG = np.einsum('ij,i->ij',Cm,n_e)
            CmG0 = simps(CmG,rays[i0,j,k,3,:],axis=1)#nx*ny*nz x 1

            i = 0
            while i < Na:
                if i == i0:
                    i += 1
                    continue
                dx = np.subtract.outer(X,rays[i,j,k,0,:])
                #nx*ny*nz x Na*Nt*Nd*Ns
                dy = np.subtract.outer(Y,rays[i,j,k,1,:])
                dz = np.subtract.outer(Z,rays[i,j,k,2,:])
                Cm = covariance(dx,dy,dz)
                #nx*ny*nz x *Ns
                #int ray_ijk ne - int ray_i0jk ne Cm
                CmG = np.einsum('ij,i->ij',Cm,n_e)
                CmG = simps(CmG,rays[i,j,k,3,:],axis=1) - CmG0#nx*ny*nz x 1
                #add each ray
                out += np.einsum('ij,j->i',CmG,wdd[i,j,k])
                i += 1
            k += 1
        j += 1
    return out
