import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np
import sys

import os

class RadioArray(object):
    '''Handles the radio array object.'''
    lofar_array = os.path.join(os.path.dirname(__file__),'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(os.path.dirname(__file__),'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(os.path.dirname(__file__),'arrays/gmrtPos.csv')
    
    def __init__(self,array_file = None,antenna_pos=None,name = None,msFile=None,numAntennas=0,earthLocs=None,frequency=120e6):
        self.frequency = frequency#can be widebandwidth later
        self.Nantenna = 0
        if array_file is not None:
            self.array_file = array_file
            self.load_array_file(array_file)
        if antenna_pos is not None:
            self.load_pos_array(antenna_pos)
            
    def load_array_file(self,array_file):
        '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''
        
        try:
            types = np.dtype({'names':['X','Y','Z','diameter','station_label'],
                             'formats':[np.double,np.double,np.double,np.double,'S16']})
            d = np.genfromtxt(array_file,comments = '#',dtype=types)
            self.diameters = d['diameter']
            self.labels = d['station_label'].astype(str)
            self.locs = ac.SkyCoord(x=d['X']*au.m,y=d['Y']*au.m,z=d['Z']*au.m,frame='itrs')
            self.Nantenna = int(np.size(d['X']))
        except:
            d = np.genfromtxt(array_file,comments = '#',usecols=(0,1,2))
            self.locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
            self.Nantenna = d.shape[0]
            self.labels = [str(i) for i in range(self.Nantenna)]
            self.diameters = None
        self.calc_center()
    
    def get_antenna_locs(self):
        return self.locs
    def get_antenna_labels(self):
        return self.labels
        
    def get_fov(self):
        '''get the field of view in radians'''
        return 4.*np.pi/180.
    
    
    def save_array_file(self,array_file):
        import time
        locs = self.locs.cartesian.xyz.to(au.m).value.transpose()
        f = open(array_file,'w')
        f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c",time.localtime())))
        f.write('# ITRS(m)\n')
        f.write('# X\tY\tZ\tdiameter\tlabels\n')
        i = 0
        while i < self.Nantenna:
            if self.diameters is not None:
                f.write('{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:1.4e}\t{4}'.format(locs[i,0],locs[i,1],locs[i,2],self.diameters[i],self.labels[i]))
            else:
                f.write('{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i,0],locs[i,1],locs[i,2],-1,self.labels[i]))
            if i < self.Nantenna-1:
                f.write('\n')
            i += 1
        f.close()
        
    def load_pos_array(self,antenna_pos,antenna_labels=None):
        '''Load pos is shape (N,3), typically grabbed from a ms/ANTENNA table.
        Assumes it is in ITRS(m)'''
        self.locs = ac.SkyCoord(x=antenna_pos[:,0]*au.m,y=antenna_pos[:,1]*au.m,z=antenna_pos[:,2]*au.m,frame='itrs')
        self.Nantenna = antenna_pos.shape[0]
        if antenna_labels is not None:
            assert len(antenna_labels) == self.Nantenna
            self.labels = np.array([str(lab) for lab in antenna_labels])
        else:
            self.labels = np.array(["ant{:02d}".format(i) for i in range(self.Nantenna)])
        self.calc_center()

    def calc_center(self):
        '''calculates the centroid of the array based on self.locs returns the ITRS of center'''
        center = np.mean(self.locs.cartesian.xyz,axis=1)
        self.center = ac.SkyCoord(x=center[0],y=center[1],z=center[2],frame='itrs')
        
        #n = self.center.itrs.earth_location.geocentric.to(au.m).value
        #self.n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        return self.center
    
    def get_center(self):
        '''Return the ITRS center of the array'''
        try:
            return self.center
        except:
            self.calc_center()
            self.log("Center of array: {0}".format(self.center))
            return self.center
    
    def get_antenna_idx(self,name):
        '''Retrieve the index of the given name from labels.
        Returns None if no match.'''
        i = 0
        while i < self.Nantenna:
            if self.labels[i] == name:
                return i
            i += 1
        return None
    
    def get_sun_zenith_angle(self,time):
        '''Return the solar zenith angle in degrees at the given time.'''
        frame = ac.AltAz(location=self.get_center().earth_location,obstime=time)
        sun = ac.get_sun(time).transform_to(frame)
        return 90. - sun.alt.deg
    
    def __repr__(self):
        return "Radio Array: {0:1.5e} MHz, Longitude {1:.2f} Latitude {2:.2f} Height {3:.2f}".format(self.frequency,*self.get_center().earth_location.to_geodetic('WGS84'))

def generate_example_radio_array(Nant=10):
    radio_array = RadioArray()
    location = [np.random.uniform(low=0,high=2*np.pi),np.random.uniform(low=-np.pi/3., high = np.pi/3.)]
    scatter = [20./6371.]*2
    ant_pos = np.random.multivariate_normal(mean=location,cov=np.diag(scatter)**2,size=Nant)
    ant_pos = ac.EarthLocation.from_geodetic(ant_pos[:,0]*au.rad,ant_pos[:,1]*au.rad,np.zeros(Nant)*au.m,"WGS84").to_geocentric()
    ant_pos = np.array([ant_pos[0].si.value,ant_pos[1].si.value,ant_pos[2].si.value]).T
    radio_array.load_pos_array(ant_pos)
    return radio_array 
