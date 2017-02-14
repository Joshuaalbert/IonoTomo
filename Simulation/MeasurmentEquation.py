
# coding: utf-8

# In[1]:

'''This holds the machinery for making measurment equations for a given antenna layout and pointing and sky brightness.
Uses the linear equation formulation.'''

import numpy as np
import multiprocessing as mp
from itertools import product
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

from Geometry import itrs2uvw

def vec(A):
    '''Stack columns aka use F ordering (col-major)'''
    return np.ndarray.flatten(A,order='F')

def reshape(v,shape):
    return np.ndarray.reshape(v,shape,order='F')

def khatrirao(A,B,out=None):
    if len(A.shape) != len(B.shape):
        print("Not same size")
        return
    if A.shape[0] != B.shape[0]:
        print("Not same number of rows")
        return
    if A.shape[1] != B.shape[1]:
        print("Not same number of columns")
        return
    if out is not None:
        if out.shape[0] == A.shape[0] and out.shape[1] == A.shape[0]*A.shape[1]:
            K = out
        else:
            print("out matrix not right size")
            return
    else:
        K = np.zeros([A.shape[0]*B.shape[0],A.shape[1]],dtype=type(1j))
    n = 0
    while n<A.shape[1]:
        #print A[:,n].shape,B[:,n].shape,K[:,n*A.shape[0]:(n+1)*A.shape[0]].shape
        K[:,n:n+1] = np.kron(A[:,n],B[:,n]).reshape([A.shape[0]*B.shape[0],1])
        
        #K[:,n:n+1] = reshape(np.outer(A[:,n],B[:,n])[A.shape[0]*B.shape[0],1])
        n += 1
    return K

def createPointing(lvec,mvec):
    '''returns the pointings of all combinations of l,m in l m vec'''
    L,M = np.meshgrid(lvec,mvec)
    N = np.sqrt(1-L**2-M**2)
    P = np.array([vec(L),vec(M),vec(N)])
    return P

def phaseTracking(ra,dec,obsLoc,obsTime):
    s = ac.SkyCoord(ra=ra*au.deg,dec=dec*au.deg,frame='icrs')
    frame = ac.AltAz(location = obsLoc, obstime = obsTime, pressure=None, copy = True)
    s_ = s.transform_to(frame)
    return s_

def responseMatrix(Z,P):
    '''Returns the array response matrix, where each column is the array response vector:
    Z =[x_1,...,x_N] are antenna locations in lambda in uvw coordinates
    P = [p_1,...,p_Q] where each p_i is (l_i,m_i,sqrt(1-l_i^2-m_i^2))
    a_q = exp(-1j*Z^T.p_q)
    '''
    
    #Q = P.shape[1]#number of pointings
    #N = X.shape[1]#number of antennas
    
    A = np.exp(-1j*2*np.pi*Z.transpose().dot(P))
    return A

def measurementEquationMatrix(skyBrightness,X,P,noiseCovariance=None):
    if np.size(skyBrightness) != np.size(P)/3:
        print("Not equal components")
        return
    if noiseCovariance is not None:
        if np.size(noiseCovariance) == 1:
            sigmaN = np.eye(X.shape[1])*noiseCovariance
        else:
            if np.size(noiseCovariance) == np.size(skyBrightness):
                sigmaN = np.diag(noiseCovariance)
            else:
                print("sigma sizes not equal")
                return
    else:
        sigmaN = 0.
    sigmaS = np.diag(skyBrightness)
    A = responseMatrix(X,P)
    R = A.dot(sigmaS).dot(A.conjugate().transpose()) + sigmaN
    return R

def measurementEquationVector(skyBrightness,X,P,noiseCovariance=None):
    if np.size(skyBrightness) != P.shape[1]:
        print("Not equal components")
        return
    sigmaS = skyBrightness
    A = responseMatrix(X,P)
    r = np.zeros(A.shape[0]*A.shape[0],dtype=type(1j))
    n = 0
    #pool = mp.Pool(4)
    #print pool
    #r = pool.map(lambda p: np.kron(p[0].conjugate(),p[0])*p[1],product(list(A.transpose()),list(sigmaS)))
    #print r
    while n<A.shape[1]:
        #print A[:,n].shape,B[:,n].shape,K[:,n*A.shape[0]:(n+1)*A.shape[0]].shape
        r += np.kron(A[:,n],A[:,n])*sigmaS[n]
        #r += vec(np.outer(A[:,n],A[:,n]))*sigmaS[n]
        n += 1
    #Ms = khatrirao(A.conjugate(),A)
    if noiseCovariance is not None:
        if np.size(noiseCovariance) == np.size(skyBrightness):
            sigmaN = noiseCovariance
            Eye = np.eye(np.size(skyBrightness))
            Mn = khatrirao(Eye,Eye)
        else:
            if np.size(noiseCovariance) == 1:
                sigmaN = np.ones_like(skyBrightness)*noiseCovariance
                Mn = vec(np.eye(np.size(skyBrightness)))
            else:
                print("sigma sizes not equal")
                return
        r += Mn.dot(sigmaN)
    return r

def catalog2ms(simTk,catalogName):
    #load catalog
    cat = np.genfromtxt(catalogName,comments='#',skip_header=5,names=True)
    print (cat)
    ra = cat['RA']
    
    dec = cat['DEC']
    skyBrightness = cat['Total_flux']
    if np.size(ra) == 1:
        ra = np.array([ra])
        dec = np.array([dec])
        skyBrightness = np.array([skyBrightness])
    #Maj = cat['Maj']
    noise = np.mean(cat['E_Total_flux'])
    #pointings of sources in catalog
    P = ac.SkyCoord(ra=ra*au.deg,dec = dec*au.deg,frame='icrs')
    # RadioArray with all frame info
    radioArray = simTk.radioArray
    #compare with ms 0-1 baseline at first timestamp, make sure fieldId is correct in json
    print(radioArray.baselines[:,0,1,:])
    s = radioArray.pointing
    lon = radioArray.center.earth_location.geodetic[0].rad
    lat = radioArray.center.earth_location.geodetic[1].rad
    #response matrix, visibilities
    R = np.zeros([len(simTk.timeSlices),radioArray.Nantenna,radioArray.Nantenna],dtype=type(1j))
    plotBl = []
    times = []
    i = 0
    while i < len(simTk.timeSlices):
        #print('Calculating visibilities for: {0}'.format(simTk.timeSlices[i]))
        lmst = simTk.timeSlices[i].sidereal_time('mean',lon)
        hourangle = lmst.to(au.rad).value - s.ra.rad
        Ruvw = itrs2uvw(hourangle,s.dec.rad,lon,lat)
        #l,m,n 3xM
        responseDirections = Ruvw.dot(P.itrs.cartesian.xyz)
        #uvw in lambda
        X = radioArray.antennaLocs[i,:,:].transpose()/simTk.wavelength
        R[i,:,:] = measurementEquationMatrix(skyBrightness,X,responseDirections,noiseCovariance=noise)
        times.append(simTk.timeSlices[i].gps)
        plotBl.append(R[i,0,1])
        #print R[i,0,:]
        i += 1
    
    import pylab as plt
    #baselines = np.sqrt(np.sum(radioArray.baselines[i,:,1,:]**2,axis=1))
    plt.plot(times,np.abs(plotBl))
    #plt.title("UV:{0}".format(P.itrs.cartesian.xyz[]))
    plt.show()
    
    
    
if __name__=='__main__':
    N = 5*10

    M = 25
    X = np.random.uniform(size=[3,N])
    s = np.array([0,0,1])
    lvec = np.linspace(-1./np.sqrt(2),1./np.sqrt(2),int(np.sqrt(M)))
    mvec = np.copy(lvec)
    P = createPointing(lvec,mvec)
    wavelength = 1
    
    skyBrightness = np.random.uniform(size=M)
    #R = measurementEquationMatrix(skyBrightness,X,s,P,wavelength)
    #r = measurementEquationVector(skyBrightness,X,s,P,wavelength)
    #print r,R
    obsTime = at.Time('2016-12-31T00:00:00.000',format='isot',scale='utc')
    obsLoc = ac.ITRS(x=0*au.m,y=0*au.m,z=0*au.m)#stick your location here
    
    from SimulationToolkit import Simulation
    
    simTk = Simulation(simConfigJson='SimulationConfig.json',logFile='logs/0.log')
    #catalogName = '/net/para34/data1/albert/casa/images/plckg004-19_150.fits.gaul'
    catalogName = 'test.gaul'
    catalog2ms(simTk,catalogName)
    
    

