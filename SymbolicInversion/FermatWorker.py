
# coding: utf-8

# In[ ]:

import pp
from FermatPrincipleThreaded import Fermat
import astropy.coordinates as ac
import astropy.units as au
#import astropy.time as at
from IRI import *
import numpy as np
from Geometry import *
import sympy
import time

import tempfile

def ppRayProp(file,inRays,N,pathlength,frequency):
    #enu = ENUFrame.ENU()
    fermat =  FermatPrincipleCartesian.Fermat(neFunc = None,type = 's',frequency = frequency)
    fermat.loadFunc(file)
    rays = {}
    for ray in inRays:
        datumIdx = ray.id
        #origin = astropy.coordinates.SkyCoord(*(ray.origin*astropy.units.km),frame=enu).transform_to('itrs').cartesian.xyz.to(astropy.units.km).value
        #direction = astropy.coordinates.SkyCoord(*ray.dir,frame=enu).transform_to('itrs').cartesian.xyz.value
        origin = ray.origin
        direction = ray.dir
        time = ray.time
        #print(ray)
        x,y,z,s = fermat.integrateRay(origin,direction,pathlength,time=time,N=N)
        rays[datumIdx] = {'x':x,'y':y,'z':z,'s':s}
    return rays
    
    
def test():
    sol = SolitonModel(5)
    neFunc = sol.generateSolitonsModel()
    theta = np.linspace(-np.pi/8.,np.pi/8.,25)
    inRays = []
    origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
    count = 0
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
            ray = Ray(origin,direction,id = count,time =0)
            inRays.append(ray)
            count += 1
    
    
    ncpus = 1
    t1 = time.time()
    N = len(inRays)/ncpus
    # Creates jobserver with ncpus workers
    jobs = []
    job_server = pp.Server(ncpus, ppservers=())
    for i in range(ncpus):
        file = 'neFunc-thread{0}.npz'.format(i)
        np.savez(file,neFunc=neFunc)
        #file = sol.saveNeFunc(neFunc)
        job = job_server.submit(ppRayProp,
                   args=(file,inRays[i:(i+1)*N],100,2000,120e6),
                   depfuncs=(),
                   modules=('FermatPrincipleCartesian',),
                   globals={})
        jobs.append(job)
    for job in jobs:
        result = job()
    print(time.time() - t1)
    job_server.print_stats()
if __name__ == '__main__':
    test()

