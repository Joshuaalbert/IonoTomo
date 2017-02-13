
# coding: utf-8

# In[7]:

import pp

from IRI import *
from Symbolic import *
from ENUFrame import ENU
from FermatPrincipleThreaded import *

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

def testThreadedFermat():
    sol = SolitonModel(5)
    neFunc = sol.generateSolitonsModel()
    f =  Fermat(neFunc = neFunc,type = 's')
    n = 1
    threads = []
    for i in range(n):
        threads.append(FermatIntegrationThread(f,i))
        threads[i].start()
    
    count = 0
    
    theta = np.linspace(-np.pi/8.,np.pi/8.,5)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
            threads[count % n].addJob(origin,direction,1000,0.,100,resultIdx = count)
            count += 1
    for i in range(n):
        threads[i].kill()
    #print('waiting for completion')
    for i in range(n):
        threads[i].join()
        #pass
        
ncpus = 4
ppservers = ()
job_server = pp.Server(ncpus, ppservers=ppservers)
print ("Starting pp with", job_server.get_ncpus(), "workers")

sol = SolitonModel(5)
neFunc = sol.generateSolitonsModel()
f =  Fermat(neFunc = neFunc,type = 's')
jobs = []
theta = np.linspace(-np.pi/8.,np.pi/8.,5)
#phi = np.linspace(0,2*np.pi,6)
rays = []
origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
for t in theta:
    for p in theta:
        direction = ac.SkyCoord(np.sin(t),
                                np.sin(p),
                                1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
        job = job_server.submit(f.integrateRay,
                                 args=(origin,direction,
                                  1000,
                                  0.,
                                  100),
                                 modules=("FermatPrincipleThreaded",))
        jobs.append(job)

for job in jobs:
    print(job())

job_server.print_stats()


# In[ ]:



