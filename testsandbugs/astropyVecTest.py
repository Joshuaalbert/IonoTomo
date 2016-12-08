
# coding: utf-8

# In[7]:

import astropy.coordinates as ac
import astropy.units as au
import time
import numpy as np

if __name__=='__main__':
    x = np.linspace(0,np.pi/2.,10)
    X,Y = np.meshgrid(x,x)
    print ("testing vectorized")
    t1 = time.time()
    loc_vec = ac.SkyCoord(ra=X*au.rad,dec=Y*au.rad)
    print("time: {}".format(time.time()-t1))
    print("testing non vectorized")
    loc_seq = []
    t1 = time.time()
    i = 0
    while i < len(x):
        row = []
        j = 0
        while j < len(x):
            row.append(ac.SkyCoord(ra=x[i]*au.rad,dec=x[j]*au.rad))
            j += 1
        loc_seq.append(row)
        i += 1
    print("time: {}".format(time.time()-t1))
    #print loc_vec,loc_seq
    print loc_vec.supergalactic
    


# In[ ]:



