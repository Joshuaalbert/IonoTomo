
# coding: utf-8

# In[ ]:

'''Do a line search to find minimum in gradient direction'''
import numpy as np
import pylab as plt

from ForwardEquation import forward_equation, forward_equation_dask
from tri_cubic import TriCubic

def vertex(x1,x2,x3,y1,y2,y3):
    '''Given three pairs of (x,y) points return the vertex of the
         parabola passing through the points. Vectorized and common expression reduced.'''
    #Define a sequence of sub expressions to reduce redundant flops
    x0 = 1/x2
    x4 = x1 - x2
    x5 = 1/x4
    x6 = x1**2
    x7 = 1/x6
    x8 = x2**2
    x9 = -x7*x8 + 1
    x10 = x0*x1*x5*x9
    x11 = 1/x1
    x12 = x3**2
    x13 = x11*x12
    x14 = 1/(x0*x13 - x0*x3 - x11*x3 + 1)
    x15 = x14*y3
    x16 = x10*x15
    x17 = x0*x5
    x18 = -x13 + x3
    x19 = y2*(x1*x17 + x14*x18*x6*x9/(x4**2*x8))
    x20 = x2*x5
    x21 = x11*x20
    x22 = x14*(-x12*x7 + x18*x21)
    x23 = y1*(-x10*x22 - x21)
    x24 = x16/2 - x19/2 - x23/2
    x25 = -x17*x9 + x7
    x26 = x0*x1*x14*x18*x5
    x27 = 1/(-x15*x25 + y1*(x20*x7 - x22*x25 + x7) + y2*(-x17 + x25*x26))
    x28 = x24*x27
    return x28,x15 + x22*y1 + x24**2*x27 - x26*y2 + x28*(-x16 + x19 + x23)

def line_search(rays,K_ne,m_tci,i0,gradient,g,dobs,CdCt,figname=None):
    M = m_tci.get_shaped_array()
    #g = forward_equation_dask(rays,K_ne,m_tci,i0)
    dd = (g - dobs)**2/(CdCt + 1e-15)
    S0 = np.sum(dd)/2.
    ep_a = []
    S_a = []
    S = S0
    #initial epsilon_n 
    dd = (g - dobs)/(CdCt + 1e-15)
    ep = 1e-3
    g_ = forward_equation(rays,K_ne,TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,M - ep*gradient),i0)
    Gm = (g - g_)/ep
    #numerator
    dd *= Gm
    numerator = 2.*np.sum(dd)
    #denominator
    Gm *= Gm
    Gm /= (CdCt + 1e-15)
    denominator = np.sum(Gm)
    epsilon_n0 = np.abs(numerator/denominator)
    epsilon_n = epsilon_n0
    iter = 0
    while S >= S0 or iter < 3:
        epsilon_n /= 2.
        #m_tci.m = m - epsilon_n*gradient.ravel('C')
        g = forward_equation(rays,K_ne,TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,M - epsilon_n*gradient),i0)
        #print(np.mean(g),np.var(g))
        dd = (g - dobs)**2/(CdCt + 1e-15)
        S = np.sum(dd)/2.
        ep_a.append(epsilon_n)
        S_a.append(S)
        #print(epsilon_n,S)
        if not np.isnan(S):
            if S < 1<<64:
                iter += 1
    epsilon_n,S_p = vertex(*ep_a[-3:],*S_a[-3:])
    
    g = forward_equation_dask(rays,K_ne,TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,M - epsilon_n*gradient),i0)
    dd = (g - dobs)**2/(CdCt + 1e-15)
    S = np.sum(dd)/2.
    print("S0: {} | Estimated epsilon_n: {}".format(S0, epsilon_n0))
    print("Parabolic minimum | epsilon_n = {}, S = {}".format(epsilon_n,S_p))
    print("Actual | S = {}".format(S))
    print("Misfit Reduction: {:.2f}%".format(S/S0*100. - 100.))
    if figname is not None:
        plt.plot(ep_a,S_a)
        plt.scatter(epsilon_n,S,c='green',label='Final misfit')
        #plt.scatter(epsilon_n,S_p,c='red',label='Parabolic minimum')
        plt.yscale('log')
        plt.plot([min(epsilon_n,np.min(ep_a)),max(epsilon_n,np.max(ep_a))],[S0,S0],ls='--',c='red')
        plt.xscale('log')
        plt.legend(frameon=False)
        plt.savefig("{}.png".format(figname),format='png')
        
    return epsilon_n,S,(S/S0 - 1.)
    
def test_line_search():
    from real_data import DataPack
    from tri_cubic import TriCubic
    from CalcRays import calc_rays

    from InitialModel import create_initial_model
    from Gradient import compute_gradient, compute_gradient_dask
    i0 = 0
    datapack = DataPack(filename="output/test/datapack_sim.hdf5").clone()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=[0])
    datapack.set_reference_antenna(antenna_labels[i0])
    ne_tci = create_initial_model(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000.)
    dobs = datapack.get_dtec(ant_idx = -1, time_idx = [0], dir_idx = -1)
    CdCt = (0.15*np.abs(dobs))**2
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    print("Calculating rays...")
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000., 100)
    m_tci = ne_tci.copy()
    K_ne = np.mean(m_tci.m)
    m_tci.m /= K_ne
    np.log(m_tci.m,out=m_tci.m)
    #print(ne_tci.m)
    g = forward_equation(rays,K_ne,m_tci,i0)
    #gradient = compute_gradient(rays, g, dobs, 0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 3, 5.)
    gradient = compute_gradient_dask(rays, g, dobs,  i0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 5, 5.)
    import pylab as plt
    plt.hist(gradient.flatten())
    plt.yscale('log')
    plt.show()
    line_search(rays,K_ne,m_tci,i0,gradient,g,dobs,CdCt,figname=None)
    
def test_vertex():
    x = np.linspace(0,1,100)
    y = (x-0.25)**2 + 2*(x-0.75)**2- 1
    idx = np.random.randint(100,size=3)
    x0,y0 = vertex(*x[idx],*y[idx])
    import pylab as plt
    plt.plot(x,y)
    plt.scatter(x[idx],y[idx],c='red')
    plt.scatter(x0,y0,c='green')
    plt.xlim([0,1])
    plt.show()
    
    
if __name__=='__main__':
    test_line_search()
    
    #test_vertex()
    
    

