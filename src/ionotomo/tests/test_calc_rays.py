import numpy as np
from ionotomo import *
import os

def test_calc_rays():
    datapack = generate_example_datapack()
    ne_tci = create_initial_model(datapack)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=-1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    t1 = clock()
    rays1 = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    print('time serial',clock() - t1)
#    t1 = clock()
#    rays2 = calc_rays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
#    print('time parallel',clock() - t1)
#    print("Num of rays calculated: {}".format(rays1.shape[0]*rays1.shape[1]*rays1.shape[2]))
#    assert np.all(rays1==rays2),"Not same result"
#    assert rays1.shape[0] == Na and rays1.shape[1] == Nt and rays1.shape[2] == Nd and rays1.shape[3] == 4 and rays1.shape[4] == 1000
    Na,Nt,Nd,_,Ns = rays1.shape
    def index_inv(h, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Invert flattened index to the indices'''
        h = np.ndarray.astype(np.atleast_1d(h),float)
        s = np.mod(h, float(N4))
        h -= s
        h /= float(N4)
        k = np.mod(h, float(N3))
        h -= k
        h /= float(N3)
        j = np.mod(h, float(N2))
        h -= j
        h /= float(N2)
        i = h
        return np.ndarray.astype(i,int), np.ndarray.astype(j,int), np.ndarray.astype(k,int), np.ndarray.astype(s,int)

    rays = rays1
    from scipy.spatial import cKDTree
    kdt = cKDTree(np.array([rays[:,:,:,0,:].flatten(),
        rays[:,:,:,1,:].flatten(),
        rays[:,:,:,2,:].flatten()]).T)
    pairs = kdt.query_pairs(r=30,eps=0.,output_type='ndarray')
    i1,j1,k1,s1 = index_inv(pairs[:,0])
    i2,j2,k2,s2 = index_inv(pairs[:,1])
    print("Number of pairs within 30km of each other:",pairs.shape[0])
    print(pairs[:10,:])
    print(i1[:10],j1[:10],k1[:10],s1[:10])
    print(i2[:10],j2[:10],k2[:10],s2[:10])
    t1 = clock()
    for i in range(1):
        pairs = kdt.query_ball_point(rays[i,0,0,0:3,:].T,50)
    print("time per ray:",(clock() - t1)/10.)
    #print(pairs)


