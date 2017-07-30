
# coding: utf-8

# In[1]:

import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get

def do_forward_equation(rays,K_ne,m_tci):
    N1,N2, _ , Ns = rays.shape
    nevec = np.zeros(Ns,dtype=np.double)
    g = np.zeros([N1,N2],dtype=np.double)
    j = 0
    while j < N1:
        k = 0
        while k < N2:
            x = rays[j,k,0,:]
            y = rays[j,k,1,:]
            z = rays[j,k,2,:]
            s = rays[j,k,3,:]
            nevec *= 0
            idx = 0
            while idx < Ns:
                nevec[idx] += m_tci.interp(x[idx],y[idx],z[idx])
                idx += 1
            
            np.exp(nevec,out=nevec)
            nevec *= K_ne/1e13
            g[j,k] += simps(nevec,s)
            k += 1
        j += 1
    return g
                

def forward_equation(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    Na,Nt,Nd, _, Ns = rays.shape
    if Na < Nd:
        #do over antennas
        tec = np.stack([do_forward_equation(rays[i,:,:,:,:],K_ne,m_tci) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = np.stack([do_forward_equation(rays[:,:,k,:,:],K_ne,m_tci) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec

def forward_equation_dask(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    Na,Nt,Nd, _, Ns = rays.shape
    if Na < Nd:
        #do over antennas
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[i,:,:,:,:],K_ne,m_tci),(Nt,Nd),dtype=np.double) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[:,:,k,:,:],K_ne,m_tci),(Na,Nt),dtype=np.double) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec.compute(get=get)

def test_forward_equation():
    from tri_cubic import TriCubic
    from real_data import DataPack
    from AntennaFacetSelection import selectAntennaFacets
    from CalcRays import calcRays
    import pylab as plt
    datapack = DataPack(filename="output/test/datapack_obs.hdf5")
    datapack_sel = selectAntennaFacets(15, datapack, ant_idx=-1, dir_idx=-1, time_idx = [0])
    antennas,antenna_labels = datapack_sel.get_antennas(ant_idx = -1)
    patches, patch_names = datapack_sel.get_directions(dir_idx = -1)
    times,timestamps = datapack_sel.get_times(time_idx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    ne_tci = TriCubic(filename="output/test/simulate/simulate_0/neModel.hdf5")
    rays = calcRays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000, 250)
    K_ne = np.mean(ne_tci.m)
    ne_tci.m = np.log(ne_tci.m/K_ne)
    m_tci = ne_tci
    i0 = 0
    
    g = forward_equation_dask(rays,K_ne,m_tci,i0)
    get_ipython().magic('time forward_equation_dask(rays,K_ne,m_tci,i0)')
    get_ipython().magic('timen 3 forward_equation(rays,K_ne,m_tci,i0)')
    
    m = m_tci.m.copy()
    i0 = 0
    rmse = []
    rmsef = []
    for error in np.linspace(0,0.25,40):
        m_tci.m = m + error*np.random.normal(size=m.size)
        g_ = forward_equation_dask(rays,K_ne,m_tci,i0)
        rmse.append(np.sqrt(np.mean((g_ - g)**2)))
        rmsef.append(np.sqrt(np.mean(((g_ - g)/(g+1e-15))**2)))
        #plt.hist(g.flatten())
        #plt.show()
    plt.plot(np.linspace(0,0.25,40),rmse)
    plt.xlabel('Gaussian Noise Level')
    plt.ylabel('RMS Error of forward equation (TECU)')
    plt.yscale('log')
    plt.savefig('forward_equationRMSE-noise.pdf',format='pdf')
    plt.show()
    plt.plot(np.linspace(0,0.25,40),rmsef)
    plt.xlabel('Gaussian Noise Level')
    plt.ylabel('RMS Error of forward equation (fractional)')
    plt.yscale('log')
    plt.savefig('forward_equationRMSEf-noise.pdf',format='pdf')
    plt.show()
    #datapack_sel.set_dtec(g,ant_idx=-1,time_idx=[0], dir_idx=-1,ref_ant=antenna_labels[i0])
    #datapack_sel.save("output/test/datapack_sim.hdf5")
    
    
    
    

if __name__ == '__main__':
    test_forward_equation()


# In[ ]:



