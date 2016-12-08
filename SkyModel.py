
# coding: utf-8

# In[18]:

'''
We represent a sky model as:
id,ra,dec,S(nu=nu0),spectralindex=-0.73
ICRS frame
'''
import numpy as np
import astropy.coordinates as ac
import astropy.units as au
import os

def specCalc(A,nu,nu_ref):
    '''Calculates flux at given freq [Jy]'''
    Sout = A[0]*np.ones_like(nu)
    N = len(A)
    i = 1
    while i < N:
        Sout *= 10**(A[i]*(np.log10(nu/nu_ref)**i))
        i += 1
    return Sout

class SkyModel(object):
    def __init__(self,skyModelFile=None,log = None,nu0=150e6):
        if log is not None:
            self.log = log
        self.skyModel = None
        self.nu0 = nu0
        
        self.ra = np.array([])
        self.dec = np.array([])
        self.S = np.array([])
        self.alpha = np.array([])
        
        if skyModelFile is not None:
            if os.path.isfile(skyModelFile):
                try:
                    self.loadSkyModel(skyModelFile)
                except:
                    pass
                
    def getSource(self,id):
        '''Get the source with id'''
        icrsLoc = ac.SkyCoord(ra=self.ra[id]*au.deg,dec=self.dec[id]*au.deg,frame='icrs')
        return icrsLoc,self.S[id],self.alpha[id]
    
    def getFullSky(self):
        icrsLocs = ac.SkyCoord(ra=self.ra*au.deg,dec = self.dec*au.deg,frame='icrs')
        return icrsLocs,self.S,self.alpha
    
    def addSource(self,icrsLoc,S,alpha,nu0=None):
        if alpha is None:
            alpha = -0.7#default
        try:
            ra = icrsLoc.ra.deg
            dec = icrsLoc.dec.deg
        except:
            ra = icrsLoc[0]
            dec = icrsLoc[1]
        if nu0 is not None:
            #transform from nu0 to self.nu0
            self.nu0 = nu0
        self.ra = np.append(self.ra,ra)
        self.dec = np.append(self.dec,dec)
        self.S = np.append(self.S,S)
        self.alpha = np.append(self.alpha,alpha)
            
    def loadSkyModel(self,filename):
        '''load skymodel from file. Perhaps replace with the thing from directions.py'''
        skyModel = np.genfromtxt(filename,comments='#',delimiter=',',names=True)
        self.nu0 = float(self.skyModel.dtype.names[3].split('Hz')[0].split('S')[1])
        self.id = skyModel[:,0]
        self.ra = skyModel[:,1]
        self.dec = skyModel[:,2]
        self.S = skyModel[:,3]
        self.alpha = skyModel[:,4]
        
    def saveSkyModel(self,filename):
        '''Save skymodel to file.'''
        skyModel = np.array([np.arange(np.size(self.ra)),self.ra,self.dec,self.S,self.alpha]).transpose()
        np.savetxt(filename,skyModel,fmt='%-5d,%5.10f,%5.10f,%5.10f,%+5.5f',delimiter=',',header="id,ra,dec,S({0}Hz),alpha".format(int(self.nu0)),comments='#')

    def addRandom(self,pointing,fov,N):
        '''add a scattering of point sources around fov of pointing'''
        try:
            ra = pointing.ra.deg
            dec = pointing.dec.deg
        except:
            ra = pointing[0]
            dec = pointing[1]
        x,y = np.random.uniform(low=0,high = fov/2.,size=N),np.random.uniform(low=0,high = fov/2.,size=N)
        r = np.sqrt(x**2 + y**2)
        theta = np.random.uniform(0,2*np.pi,N)
        self.ra = ra+r * np.cos(theta*np.pi/180.)
        self.dec = dec+r * np.sin(theta*np.pi/180.)
        self.S = np.abs(np.random.normal(loc= 1e-2,scale = 1.,size=N)**2)*5.#up to 5Jy
        self.alpha = np.random.normal(loc= -0.7,scale = 0.5,size=N)#-2 to 0 alpha
    def angularIntensity(self,L,M,frame,pointing,frequency):
        '''Create the angular intensity of the sky'''
        I = np.zeros_like(L)
        #add only sources above the horizon (min el)
        locs = ac.SkyCoord(ra=self.ra*au.deg,dec=self.dec*au.deg,frame='icrs').transform_to(frame)
        for ra,dec,s,alpha,loc in zip(self.ra,self.dec,self.S,self.alpha,locs):
            if loc.alt.deg > 0:
                mask = np.argmin((L - (ra-pointing[0]))**2,axis=1)*np.argmin((M - (dec - pointing[1]))**2,axis=0)  
                I[mask] +=  specCalc([s,alpha],frequency,self.nu0)
        return I
        
if __name__=='__main__':
    SM = SkyModel(nu0=150e6)
    SM.addRandom((64.,12.),1.,1000)
    SM.saveSkyModel('SkyModels/testSM.csv')

    
        


# In[ ]:



