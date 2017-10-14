import numpy as np
from ionotomo.geometry.slab_method import *
from ionotomo import *

def test_slab_method():
    r0 = np.array([0,0,0])
    n = np.array([0.05,0.05,0.05])
    ray = Ray(r0,n)
    xvec = np.linspace(0,1,100)
    yvec = np.linspace(0,1,100)
    zvec = np.linspace(0,10,1000)
    #res = slab_method_3d_dask([ray]*10,xvec,yvec,zvec,num_threads=8)       
    #for n in range(1,16):
    #    print(timeit(lambda : slab_method_3d_dask([ray]*20*15,xvec,yvec,zvec,num_threads=8),number = 1))
    t1 = clock()
    print([slab_method_ray_box(ray,xvec[0]-1/99.,yvec[0]-1/99.,zvec[0]-1/999.,xvec[0] + 1./99.,yvec[0]+1./99.,zvec[0]+10./999.) for i in range(100)])
    print(clock() - t1)

