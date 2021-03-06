import glob
import numpy as np
from scipy.interpolate import interp2d
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import h5py
import os
import pylab as plt
from time import gmtime, strftime
from ionotomo.astro.radio_array import *
from ionotomo.astro.frames.uvw_frame import UVW
from ionotomo.astro.frames.pointing_frame import Pointing


class DataPack(object):
    '''data_dict = {'radio_array':radio_array,'antennas':outAntennas,'antenna_labels':outAntennaLabels,
                    'times':outTimes,'timestamps':outTimestamps,
                    'directions':outDirections,'patch_names':outPatchNames,'dtec':outDtec}
    '''
    def __init__(self,data_dict=None,filename=None,ref_ant=None):
        '''get the astropy object defining rays and then also the dtec data'''
        if data_dict is not None:
            self.add_data_dict(**data_dict)
        else:
            if filename is not None:
                self.load(filename)
                return
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        #print("Loaded {0} antennas, {1} times, {2} directions".format(self.Na,self.Nt,self.Nd))
    
    def __repr__(self):
        return "DataPack: num_antennas = {}, num_time = {}, num_directions = {}\nReference Antenna = {}".format(self.Na,self.Nt,self.Nd,self.ref_ant)
    def clone(self):
        datapack = DataPack({'radio_array':self.radio_array, 'antennas':self.antennas, 'antenna_labels':self.antenna_labels,
                        'times':self.times, 'timestamps':self.timestamps, 'directions':self.directions,
                         'patch_names' : self.patch_names, 'dtec':self.dtec})
        datapack.set_reference_antenna(self.ref_ant)
        return datapack
    
    def save(self,filename):
        dt = h5py.special_dtype(vlen=str)
        f = h5py.File(filename,'w')
        antenna_labels = f.create_dataset("datapack/antennas/labels",(self.Na,),dtype=dt)
        f["datapack/antennas"].attrs['frequency'] = self.radio_array.frequency
        antennas = f.create_dataset("datapack/antennas/locs",(self.Na,3),dtype=np.double)
        antenna_labels[...] = self.antenna_labels
        antennas[:,:] = self.antennas.cartesian.xyz.to(au.m).value.transpose()#to Nax3 in m
        patch_names = f.create_dataset("datapack/directions/patchnames",(self.Nd,),dtype=dt)
        ra = f.create_dataset("datapack/directions/ra",(self.Nd,),dtype=np.double)
        dec = f.create_dataset("datapack/directions/dec",(self.Nd,),dtype=np.double)
        patch_names[...] = self.patch_names
        ra[...] = self.directions.ra.deg
        dec[...] = self.directions.dec.deg
        timestamps = f.create_dataset("datapack/times/timestamps",(self.Nt,),dtype=dt)
        gps = f.create_dataset("datapack/times/gps",(self.Nt,),dtype=np.double)
        timestamps[...] = self.timestamps
        gps[...] = self.times.gps
        dtec = f.create_dataset("datapack/dtec",(self.Na,self.Nt,self.Nd),dtype=np.double)
        dtec[:,:,:] = self.dtec
        dtec.attrs['ref_ant'] = str(self.ref_ant)
        f.close()
        
    def load(self,filename):
        f = h5py.File(filename,'r')
        self.antenna_labels = f["datapack/antennas/labels"][:].astype(str)
        antennas = f["datapack/antennas/locs"][:,:]
        frequency = f["datapack/antennas"].attrs['frequency']
        self.radio_array = RadioArray(antenna_pos = antennas,frequency = frequency)
        self.antennas = ac.SkyCoord(antennas[:,0]*au.m,antennas[:,1]*au.m,antennas[:,2]*au.m,frame='itrs')
        self.patch_names = f["datapack/directions/patchnames"][:].astype(str)
        ra = f["datapack/directions/ra"][:]
        dec = f["datapack/directions/dec"][:]
        self.directions = ac.SkyCoord(ra*au.deg,dec*au.deg,frame='icrs')
        self.timestamps = f["datapack/times/timestamps"][:].astype(str)
        self.times = at.Time(self.timestamps,format='isot',scale='tai')
        self.dtec = f["datapack/dtec"][:,:,:]
        self.ref_ant = np.array(f["datapack/dtec"].attrs['ref_ant']).astype(str).item(0)
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.set_reference_antenna(self.ref_ant)
        f.close()
        
    
    def add_data_dict(self,**args):
        '''Set up variables here that will hold references throughout'''
        for attr in args.keys():
            try:
                setattr(self,attr,args[attr])
            except:
                print("Failed to set {0} to {1}".format(attr,args[attr]))
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.antenna_labels = np.array(self.antenna_labels)
        self.patch_names = np.array(self.patch_names)
        self.timestamps = np.array(self.timestamps)
        if 'ref_ant' in args.keys():
            self.set_reference_antenna(args['ref_ant'])

                
    def set_dtec(self,dtec,ant_idx=[],time_idx=[], dir_idx=[],ref_ant=None):
        '''Set the specified dtec solutions corresponding to the requested indices.
        value of -1 means all.'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    self.dtec[ant_idx[i],time_idx[j],dir_idx[k]] = dtec[i,j,k]
                    k += 1
                j += 1
            i += 1
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        else:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)
                

    def get_dtec(self,ant_idx=[],time_idx=[], dir_idx=[]):
        '''Retrieve the specified dtec solutions corresponding to the requested indices.
        value of -1 means all.'''
        #assert self.ref_ant is not None, "set reference antenna first"
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        output = np.zeros([Na,Nt,Nd],dtype=np.double)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    output[i,j,k] = self.dtec[ant_idx[i],time_idx[j],dir_idx[k]]
                    k += 1
                j += 1
            i += 1
        return output
    
    def get_antennas(self,ant_idx=[]):
        '''Get the list of antenna locations in itrs'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        ant_idx = np.sort(ant_idx)
        output = self.antennas[ant_idx]
        Na = len(ant_idx)
        output_labels = []
        i = 0
        while i < Na:
            output_labels.append(self.antenna_labels[ant_idx[i]])
            i += 1
        return output, output_labels
    
    def get_times(self,time_idx=[]):
        '''Get the gps times'''
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        time_idx = np.sort(time_idx)
        output = self.times[time_idx]
        Nt = len(time_idx)
        output_labels = []
        j = 0
        while j < Nt:
            output_labels.append(self.timestamps[time_idx[j]])
            j += 1
        return output, output_labels
    
    def get_directions(self, dir_idx=[]):
        '''Get the array of directions in itrs'''
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        dir_idx = np.sort(dir_idx)
        output = self.directions[dir_idx]
        Nd = len(dir_idx)
        output_labels = []
        k = 0
        while k < Nd:
            output_labels.append(self.patch_names[dir_idx[k]])
            k += 1
        return output, output_labels
    
    def get_antenna_idx(self,ant):
        assert ant in self.antenna_labels, "{} not a valid label".format(ant)
        i = 0
        while i < self.Na:
            if self.antenna_labels[i] == ant:
                return i
            i += 1
    
    def set_reference_antenna(self,ref_ant):
        if ref_ant is None:
            return
        ref_ant_idx = None
        i = 0
        while i < self.Na:
            if self.antenna_labels[i] == ref_ant:
                ref_ant_idx = i
                break
            i += 1          
        assert ref_ant_idx is not None, "{} is not a valid antenna. Choose from {}".format(ref_ant,self.antenna_labels)
        #print("Setting ref_ant: {}".format(ref_ant))
        self.ref_ant = ref_ant
        self.dtec = self.dtec - self.dtec[ref_ant_idx,:,:]
        
    def get_center_direction(self):
        ra_mean = np.mean(self.directions.transform_to('icrs').ra)
        dec_mean = np.mean(self.directions.transform_to('icrs').dec)
        phase = ac.SkyCoord(ra_mean,dec_mean,frame='icrs')
        return phase

    def find_flagged_antennas(self):
        '''Determine which antennas are flagged'''
        assert self.ref_ant is not None, "Set a ref_ant before finding flagged (zeroed) antennas"
        mask = np.sum(np.sum(self.dtec,axis=2),axis=1) == 0
        i = 0
        while i < self.Na:
            if self.antenna_labels[i] == self.ref_ant:
                ref_ant_idx = i
                break
            i += 1   
        mask[ref_ant_idx] = False
        return list(self.antenna_labels[mask])
        
    def flag_antennas(self,antenna_labels):
        '''remove data corresponding to the given antenna names if it exists'''
        if not hasattr(antenna_labels,'__iter__'):
            antenna_labels = [antenna_labels]
        mask = np.ones(len(self.antenna_labels), dtype=bool)
        antennas_found = 0
        i = 0
        while i < self.Na:
            if self.antenna_labels[i] in antenna_labels:
                antennas_found += 1
                mask[i] = False
                if self.antenna_labels[i] == self.ref_ant:
                    self.ref_ant = None
            i += 1
        assert antennas_found < self.Na, "Must leave at least one antenna"
        #some flags may have not existed in data
        self.antenna_labels = self.antenna_labels[mask]
        self.antennas = self.antennas[mask]
        self.dtec = self.dtec[mask,:,:]
        self.Na = len(self.antennas)
        
    def flag_directions(self,patch_names):
        '''remove data corresponding to the given direction names if it exists'''
        if not hasattr(patch_names,'__iter__'):
            patch_names = [patch_names]
        mask = np.ones(len(self.patch_names), dtype=bool)
        patches_found = 0
        i = 0
        while i < self.Nd:
            if self.patch_names[i] in patch_names:
                patches_found += 1
                mask[i] = False
            i += 1
        assert patches_found < self.Nd, "Must leave at least one direction"
        #some flags may have not existed in data
        self.patch_names = self.patch_names[mask]
        self.directions = self.directions[mask]
        self.dtec = self.dtec[:,:,mask]
        self.Nd = len(self.directions)

    def flag_times(self,timestamps):
        '''remove data corresponding to the given timestamps if it exists'''
        if not hasattr(timestamps,'__iter__'):
            timestamps = [timestamps]
        mask = np.ones(len(self.timestamps), dtype=bool)
        times_found = 0
        i = 0
        while i < self.Nt:
            if self.timestamps[i] in timestamps:
                times_found += 1
                mask[i] = False
            i += 1
        assert times_found < self.Nt, "Must leave at least one time"
        #some flags may have not existed in data
        self.timestamps = self.timestamps[mask]
        self.times = self.times[mask]
        self.dtec = self.dtec[:,mask,:]
        self.Nt = len(self.times)        

def generate_example_datapack(Nant = 10, Ntime = 1, Ndir = 10, fov = 4., alt = 90., az=0., time = None, radio_array=None):
    '''Generate a datapack suitable for testing purposes, if time is None then use current time. The dtec is randomly distributed.'''
    if radio_array is None:
        radio_array = generate_example_radio_array(Nant=Nant)
    if time is None:
        time = at.Time(strftime("%Y-%m-%dT%H:%M:%S",gmtime()),format='isot')
    else:
        if isinstance(time,str):
            time = at.Time(time,format='isot')
    antennas = radio_array.get_antenna_locs()
    antenna_labels = radio_array.get_antenna_labels()
    Nant = len(antennas)
    times = at.Time(np.arange(Ntime)*8. + time.gps, format='gps')
    phase = ac.AltAz(alt=alt*au.deg,az=az*au.deg,location=radio_array.get_center(),obstime=time).transform_to(ac.ICRS)
    uvw = UVW(location = radio_array.get_center(),obstime=time,phase=phase)
    phi = np.random.uniform(low=-fov/2.,high=fov/2.,size=Ndir)*np.pi/180.
    theta = np.random.uniform(low=0,high=360.,size=Ndir)*np.pi/180.
    dirs = np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)]).T
    dirs = ac.SkyCoord(u = dirs[:,0], v = dirs[:,1], w = dirs[:,2],frame=uvw).transform_to(ac.ICRS)
    patch_names = np.array(["facet_patch_{}".format(i) for i in range(Ndir)])
    
    dtec = np.random.normal(size=[Nant,Ntime,Ndir])
    data_dict = {'radio_array':radio_array,'antennas':antennas,'antenna_labels':antenna_labels,
                    'times':times,'timestamps':times.isot,
                    'directions':dirs,'patch_names':patch_names,'dtec':dtec}
    datapack = DataPack(data_dict=data_dict)
    datapack.set_reference_antenna(antenna_labels[0])
    return datapack


def phase_screen_datapack(N,Nant = 10, Ntime = 1, fov = 4., alt = 90., az=0., time = None, radio_array=None,datapack=None):
    """Generate an empty datapack with N points in a grid pointing at alt (90 deg) and az (0 deg).
    The number of antennas and times are given by Nant and Ntime.
    field of view (fov) is by default 4 degrees.
    If time is None use current time."""
    if datapack is None:
        datapack = generate_example_datapack(Nant = Nant, Ntime = Ntime, Ndir = 1, fov = fov, alt = alt, az=az, time = time, radio_array=radio_array)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    times,timestamps = datapack.get_times(time_idx=-1)
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    uvw = UVW(location = datapack.radio_array.get_center(),obstime=fixtime,phase=phase)
    uvec = np.linspace(-fov/2.,fov/2.,N)*np.pi/180.
    dirs = []
    for theta1 in uvec:
        for theta2 in uvec:
            dir = np.array([np.sin(theta1),np.sin(theta2),1.])
            dir /= np.linalg.norm(dir)
            dirs.append(dir)
    dirs = np.array(dirs)
    dirs = ac.SkyCoord(u = dirs[:,0], v = dirs[:,1], w = dirs[:,2],frame=uvw).transform_to(ac.ICRS)
    patch_names = np.array(["facet_patch_{}".format(i) for i in range(len(dirs))])
    
#    dtec = np.random.normal(size=[Na,Nt,len(dirs)])
    dtec = np.zeros([Na,Nt,len(dirs)])
    data_dict = {'radio_array':datapack.radio_array,'antennas':antennas,'antenna_labels':antenna_labels,
                    'times':times,'timestamps':times.isot,
                    'directions':dirs,'patch_names':patch_names,'dtec':dtec}
    datapack = DataPack(data_dict=data_dict)
    datapack.set_reference_antenna(antenna_labels[0])
    return datapack
