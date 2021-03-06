
# coding: utf-8

# In[1]:

import pyrap.tables as pt
import numpy as np
import time

def generate(msfile,outfile):
    t = pt.table("{0}/ANTENNA".format(msfile))
    names = t.getcol('NAME')
    position = t.getcol('POSITION')
    diam = t.getcol('DISH_DIAMETER')
    f = file(outfile,'w')
    if f is None:
        print("Failed to create outfile")
        return
    f.write("# observatory=LOFAR\n# coordsys=XYZ\n# datum=WGS84\n\n")
    f.write("# created from {0}\n".format(msfile))
    f.write("# created on {0} by Joshua G. Albert\n\n".format(time.strftime("%d-%m-%Y",time.localtime())))
    f.write("#X Y Z Diam Station\n")
    
    i = 0
    while i < len(names):
        f.write("{0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4}\n".format(position[i,0],position[i,1],position[i,2],diam[i],names[i]))
        i += 1
    f.close()
    
if __name__=='__main__':
    ms = "/net/para11/data1/mandal/lockman294287/products/L294287_SBgr016-10_uv.dppp.pre-cal.ms"
    outfile = "arrays/lofar.hba.antenna.cfg"
    generate(ms,outfile)
        

