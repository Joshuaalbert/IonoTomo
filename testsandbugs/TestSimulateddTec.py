
# coding: utf-8

# In[3]:

from TricubicInversion import *

def testSweep():
    '''Test the full system.'''
    # The priori ionosphere 
    iri = IriModel()
    print("Creating priori model")
    eastVec,northVec,upVec,nePriori = createPrioriModel(iri)
    print("Creating perturbed model")
    nePert = perturbModel(eastVec,northVec,upVec,nePriori,([0,0,200.],),(40.,),(1e12,))
    print("creating TCI object")
    neTCI = TriCubic(eastVec,northVec,upVec,nePert)
    print("creating fermat object")
    f =  Fermat(neTCI = neTCI,type = 's')
    
    ### test interpolation in both
    for i in range(0):
        x = np.random.uniform(low=eastVec[0],high=eastVec[-1])
        y = np.random.uniform(low=northVec[0],high=northVec[-1])
        z = np.random.uniform(low=upVec[0],high=upVec[-1])
        n_,nx_,ny_,nz_,nxy_,nxz_,nyz_,nxyz_ = f.nTCI.interp(x,y,z,doDiff=True)
        
        ne,nex,ney,nez,nexy,nexz,neyz,nexyz = f.neTCI.interp(x,y,z,doDiff=True)
        A = - 8.98**2/f.frequency**2
        n = math.sqrt(1. + A*ne)
        ndot = A/(2.*n)
        nx = ndot * nex
        ny = ndot * ney
        nz = ndot * nez
        ndotdot = -(A * ndot)/(2. * n**2)
        nxy = ndotdot * nex*ney + ndot * nexy
        nxz = ndotdot * nex * nez + ndot * nexz
        nyz = ndotdot * ney * nez + ndot * neyz
        print(x,y,z)
        print(n,n_)
        print(nx,nx_)
        print(nxy,nxy)
        
    print("min and max n:",np.min(f.nTCI.m),np.max(f.nTCI.m))
    theta = np.linspace(-np.pi/15.,np.pi/15.,25)
    #phi = np.linspace(0,2*np.pi,6)
    rays = {}
    origin = ac.ITRS(iri.enu.location).transform_to(iri.enu).cartesian.xyz.to(au.km).value
    print(origin)
    rayIdx = 0
    t1 = tictoc()
    for t in theta:
        for p in theta:
            #print("integrating ray: {0}".format(rayIdx))
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=iri.enu).cartesian.xyz.value#.transform_to('itrs').cartesian.xyz.value
            x,y,z,s = f.integrateRay(origin,direction,1000,time=0.)
            rayIdx += 1
            rays[rayIdx] = {'x':x,'y':y,'z':z,'s':s}
    print("time per ray:",(tictoc()-t1)/len(rays))
    #print(rays)
    plotWavefront(neTCI,rays,save=False)
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol),save = False)
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)

if __name__ == '__main__':
    testSweep()

