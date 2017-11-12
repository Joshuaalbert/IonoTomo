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
                    'directions':outDirections,'patch_names':outPatchNames,'freqs':freqs,'phase':outDtec}
    '''
    def __init__(self,data_dict=None,filename=None,ref_ant=None):
        '''get the astropy object defining rays and then also the phase data'''
        if data_dict is not None:
            self.add_data_dict(**data_dict)
        else:
            if filename is not None:
                self.load(filename)
                return
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
    
    def __repr__(self):
        return "DataPack: num_antennas = {}, num_time = {}, num_directions = {}, num_freqs = {}\nReference Antenna = {}".format(self.Na,self.Nt,self.Nd,self.Nf,self.ref_ant)
    def get_data_dict(self):
        return {'radio_array':self.radio_array, 'antennas':self.antennas, 'antenna_labels':self.antenna_labels,
                        'times':self.times, 'timestamps':self.timestamps, 'directions':self.directions,
                        'patch_names' : self.patch_names, 'freqs':self.freqs,'phase':self.phase,'ref_ant':self.ref_ant, 'const':self.const, 'clock':self.clock, 'variance': self.variance}

    def clone(self):
        datapack = DataPack(self.get_data_dict())
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
        freqs = f.create_dataset("datapack/freqs",(self.Nf,),dtype=np.double)
        freqs[...] = self.freqs
        phase = f.create_dataset("datapack/phase",(self.Na,self.Nt,self.Nd,self.Nf),dtype=np.double)
        phase[:,:,:,:] = self.phase
        variance = f.create_dataset("datapack/variance",(self.Na,self.Nt,self.Nd,self.Nf),dtype=np.double)
        variance[:,:,:,:] = self.variance

        clock = f.create_dataset("datapack/clock",(self.Na,self.Nt),dtype=np.double)
        clock[:,:] = self.clock

        const = f.create_dataset("datapack/const",(self.Na,),dtype=np.double)
        const[:] = self.const

        phase.attrs['ref_ant'] = str(self.ref_ant)

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
        self.freqs = f["datapack/freqs"][:]
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.Nf = len(self.freqs)
        try:
            self.phase = f["datapack/phase"][:,:,:,:]
        except:
            self.phase = np.zeros([self.Na,self.Nt,self.Nd,self.Nf])
        try:
            self.variance = f["datapack/variance"][:,:,:,:]
        except:
            self.variance = np.zeros([self.Na,self.Nt,self.Nd,self.Nf])
        try:
            self.clock = f["datapack/clock"][:,:]
        except:
            self.clock = np.zeros([self.Na,self.Nt])
        try:
            self.const = f["datapack/const"][:]
        except:
            self.const = np.zeros(self.Na)
        try:
            self.ref_ant = np.array(f["datapack/phase"].attrs['ref_ant']).astype(str).item(0)
        except:
            self.ref_ant = None
        self.set_reference_antenna(self.ref_ant)
        f.close()

    def define_params(self):
        '''Define the params that define the datapack'''
        params = {'radio_array' : (RadioArray, "The radio array object"),
                'antennas' : (ac.ITRS,"The positions of the antennas in itrs"),
                'antenna_labels' : ([str],"The labels of antennas"),
                'times' : (at.Time, "The time objects of timestamps"),
                'timestamps' : ([str],"The ISOT timestamps"),
                'directions' : (ac.ICRS,"The directions in ICRS"),
                'patch_names' : ([str],"The patch names as in calibration"),
                'freqs' : (np.array,"The frequencies in Hz"),
                'phase' : (np.array,"The phase for each antenna, time, direction, and frequency in radians"),
                'const' : (np.array,"Constant phase offset param"),
                'clock' : (np.array,"Clock term"),
                'prop' : (np.array,"Propagation term (depreciated)"),
                'ref_ant' : (str,"The reference antenna label"),
                'variance' : (np.array,"The variance of measurements")}
        return params

    def help(self):
        """Print the parameter definitions"""
        params = self.define_params()
        print("Params for DataPack:")
        for key in params.keys():
            print("\t{} : {}\n\
                    \t{}".format(key,*params[key]))
    
    def add_data_dict(self,**args):
        '''Set up variables here that will hold references throughout'''
        params = self.define_params()

        for attr in params:
            setattr(self,attr,args.get(attr,None))
        
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.Nf = len(self.freqs)
        self.antenna_labels = np.array(self.antenna_labels)
        self.patch_names = np.array(self.patch_names)
        self.timestamps = np.array(self.timestamps)
        if self.phase is not None:
            assert self.phase.shape == (self.Na,self.Nt,self.Nd,self.Nf), "Invalid shape {} {}".format(self.phase.shape, (self.Na,self.Nt,self.Nd,self.Nf))
        if self.const is not None:
            assert self.const.shape == (self.Na,)
        if self.clock is not None:
            assert self.clock.shape == (self.Na,self.Nt)
        if self.variance is not None:
            assert self.variance.shape == (self.Na,self.Nt,self.Nd,self.Nf), "Invalid shape {} {}".format(self.variance.shape, (self.Na,self.Nt,self.Nd,self.Nf))
        
        if 'ref_ant' in args.keys():
            self.set_reference_antenna(args['ref_ant'])

    def set_slot(self, param, A, indices, set_ref_ant):
        """Set slots in datapack.
        param : str
            the key of slot
        A : array of corrct shape
            the data to insert
        indices : tuple of index array
            Must be (ant_idx, time_idx, dir_idx, freq_idx)
            Ignore an index by setting to None
        set_ref_ant : bool
            Whether to set the ref ant
        """
        assert isinstance(indices,(tuple,list))
        ant_idx, time_idx, dir_idx, freq_idx = indices
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)

        set_indices = []
        if ant_idx is not None:
            ant_idx = np.sort(ant_idx)
            set_indices.append(ant_idx)
        if time_idx is not None:
            time_idx = np.sort(time_idx)
            set_indices.append(time_idx)
        if dir_idx is not None:
            dir_idx = np.sort(dir_idx)
            set_indices.append(dir_idx)
        if freq_idx is not None:
            freq_idx = np.sort(freq_idx)
            set_indices.append(freq_idx)

        indices = np.meshgrid(*set_indices, indexing='ij')

        array = getattr(self,param,None)
        if array is not None:
            array[indices] = A
        else:
            raise ValueError("param does not exist {}".format(param))

        if set_ref_ant:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)

    def get_slot(self, param, indices):
        """Get slots in datapack.
        param : str
            the key of slot
        indices : tuple of index array
            Must be (ant_idx, time_idx, dir_idx, freq_idx)
            Ignore an index by setting to None
        Return an array of shape determined by indices
        """
        assert isinstance(indices,(tuple,list))
        ant_idx, time_idx, dir_idx, freq_idx = indices
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)

        set_indices = []
        if ant_idx is not None:
            ant_idx = np.sort(ant_idx)
            set_indices.append(ant_idx)
        if time_idx is not None:
            time_idx = np.sort(time_idx)
            set_indices.append(time_idx)
        if dir_idx is not None:
            dir_idx = np.sort(dir_idx)
            set_indices.append(dir_idx)
        if freq_idx is not None:
            freq_idx = np.sort(freq_idx)
            set_indices.append(freq_idx)

        indices = np.meshgrid(*set_indices, indexing='ij')

        array = getattr(self,param,None)
        if array is not None:
            return array[indices]
        else:
            raise ValueError("param does not exist {}".format(param))

    def set_const(self,const,ant_idx=[], ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        self.set_slot("const",const,(ant_idx,None,None,None),False)
        self.set_reference_antenna(ref_ant)


    def get_const(self,ant_idx=[]):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        return self.get_slot("const",(ant_idx,None,None,None))

    def set_clock(self,clock,ant_idx=[], time_idx=[],ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        self.set_slot("clock",clock,(ant_idx,time_idx,None,None),False)
        self.set_reference_antenna(ref_ant)


    def get_clock(self,ant_idx=[],time_idx=[]):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        return self.get_slot("clock",(ant_idx,time_idx,None,None))

    def set_phase(self,phase,ant_idx=[], time_idx=[],dir_idx=[],freq_idx=[],ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        self.set_slot("phase",phase,(ant_idx,time_idx,dir_idx,freq_idx),False)
        self.set_reference_antenna(ref_ant)

    def get_phase(self,ant_idx=[],time_idx=[],dir_idx=[],freq_idx=[]):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        return self.get_slot("phase",(ant_idx,time_idx,dir_idx,freq_idx))

    def set_variance(self,variance,ant_idx=[], time_idx=[],dir_idx=[],freq_idx=[]):
        '''Set the specified variance solutions corresponding to the requested indices.
        value of -1 means all.'''
        self.set_slot("variance",variance,(ant_idx,time_idx,dir_idx,freq_idx),False)

    def get_variance(self,ant_idx=[],time_idx=[],dir_idx=[],freq_idx=[]):
        '''Retrieve the specified variance solutions corresponding to the requested indices.
        value of -1 means all.'''
        return self.get_slot("variance",(ant_idx,time_idx,dir_idx,freq_idx))

    def set_prop(self,prop,ant_idx=[], time_idx=[],dir_idx=[],freq_idx=[],ref_ant=None):
        '''Set the specified prop solutions corresponding to the requested indices.
        value of -1 means all.
        @depreciated
        '''
        self.set_slot("prop",prop,(ant_idx,time_idx,dir_idx,freq_idx),False)
        self.set_reference_antenna(ref_ant)

    def get_prop(self,ant_idx=[],time_idx=[],dir_idx=[],freq_idx=[]):
        '''Retrieve the specified prop solutions corresponding to the requested indices.
        value of -1 means all.
        @depreciated
        '''
        return self.get_slot("prop",(ant_idx,time_idx,dir_idx,freq_idx))

    def get_antennas(self,ant_idx=[]):
        '''Get the list of antenna locations in itrs'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        ant_idx = np.sort(ant_idx)
        output = self.antennas[ant_idx]
        output_labels = self.antenna_labels[ant_idx]
        return output, output_labels
    
    def get_times(self,time_idx=[]):
        '''Get the gps times'''
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        time_idx = np.sort(time_idx)
        output = self.times[time_idx]
        output_labels = self.timestamps[time_idx]
        return output, output_labels
    
    def get_directions(self, dir_idx=[]):
        '''Get the array of directions in itrs'''
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        dir_idx = np.sort(dir_idx)
        output = self.directions[dir_idx]
        output_labels = self.patch_names[dir_idx]
        return output, output_labels

    def get_freqs(self,freq_idx=[]):
        '''Get the list of antenna locations in itrs'''
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        freq_idx = np.sort(freq_idx)
        output = self.freqs[freq_idx]
        return output
    
    def get_antenna_idx(self,ant):
        '''Get the antenna index of the given label
        ant : str
            the label of antenna (case sensitive)
        return int
        '''
        assert ant in self.antenna_labels, "{} not a valid label".format(ant)
        return np.where(self.antenna_labels == ant)[0][0]
    
    def set_reference_antenna(self,ref_ant):
        if ref_ant is None:
            return
        try:
            ref_ant_idx = self.get_antenna_idx(ref_ant)
        except:
            raise ValueError("{} is not a valid antenna. Choose from {}".format(ref_ant,self.antenna_labels))
        #print("Setting ref_ant: {}".format(ref_ant))
        self.ref_ant = ref_ant
        self.phase -= self.phase[ref_ant_idx,:,:,:]
        #self.prop -= self.prop[ref_ant_idx,:,:,:]
        self.clock -= self.clock[ref_ant_idx,:]
        self.const -= self.const[ref_ant_idx]

    def get_center_direction(self):
        """Get the mean direction in ICRS"""
        ra_mean = np.mean(self.directions.transform_to('icrs').ra)
        dec_mean = np.mean(self.directions.transform_to('icrs').dec)
        phase = ac.SkyCoord(ra_mean,dec_mean,frame='icrs')
        return phase

    def find_flagged_antennas(self):
        '''Determine which antennas are flagged by assuming all zeros means flagged.
        Ignore the reference antenna if it is set.'''
        assert self.ref_ant is not None, "Set a ref_ant before finding flagged (zeroed) antennas"
        mask = np.sum(np.sum(np.sum(self.phase,axis=3),axis=2),axis=1) == 0
        flagged = []
        for i,m in enumerate(mask):
            if m and self.antenna_labels[i] != self.ref_ant:
                flagged.append(self.antenna_labels[i])
        return list(self.antenna_labels[flagged])

        
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
        self.phase = self.phase[mask,:,:,:]
        self.variance = self.variance[mask,:,:,:]
        self.clock = self.clock[mask,:]
        self.const = self.const[mask]
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
        self.phase = self.phase[:,:,mask,:]
        self.variance = self.variance[:,:,mask,:]
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
        self.phase = self.phase[:,mask,:,:]
        self.variance = self.variance[:,mask,:,:]
        self.clock = self.clock[:,mask]
        self.Nt = len(self.times)        
    def flag_freqs(self,freq_idx=[]):
        '''remove data corresponding to the given timestamps if it exists'''
        mask = np.ones(self.Nf, dtype=bool)
        for l in range(self.Nf):
            if l not in freq_idx:
                mask[l] = True
            else:
                mask[l] = False
        assert np.sum(mask) > 0, "Must leave at least one freq"
        #some flags may have not existed in data
        self.freqs = self.freqs[mask]
        self.phase = self.phase[:,:,:,mask]
        self.variance = self.variance[:,:,:,mask]
        self.Nf = len(self.freqs)        

def generate_example_datapack(Nant = 10, Ntime = 1, Ndir = 10, Nfreqs=4, fov = 4., alt = 90., az=0., time = None, radio_array=None):
    '''Generate a datapack suitable for testing purposes, if time is None then use current time. The phase is randomly distributed.'''
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
    freqs = np.linspace(-0.5,0.5,Nfreqs)*Nfreqs*2e6 + radio_array.frequency

    tec = np.random.normal(size=[Nant,Ntime,Ndir])*0.01*1e16
    #print(tec/1e16)
    clock = np.random.normal(size=[Nant,Ntime])*5e-9
    const = np.random.normal(size=[Nant])*2*np.pi
    #prop = np.zeros([Nant,Ntime,Ndir,Nfreqs])
    phase = np.einsum("i,j,k,l,i->ijkl",np.ones(Nant),np.ones(Ntime),np.ones(Ndir),np.ones(Nfreqs),const)
    variance = np.zeros_like(phase)
    for l in range(Nfreqs):
        a_ = 2*np.pi * freqs[l]
        dg = a_ * np.einsum("ij,k->ijk",clock,np.ones(Ndir))
        dg -= 8.4480e-7/freqs[l]*tec
        #prop[:,:,:,l] += 8.4480e-7/freqs[l]*tec
        phase[:,:,:,l] += dg
    phase += np.random.normal(size=phase.shape)*5*np.pi/180.
    data_dict = {'radio_array':radio_array,'antennas':antennas,'antenna_labels':antenna_labels,
                    'times':times,'timestamps':times.isot,
                    'directions':dirs,'patch_names':patch_names,
                    'freqs':freqs,'phase':phase,'clock':clock,'const':const,
                    'variance':variance}
    datapack = DataPack(data_dict=data_dict)
    datapack.set_reference_antenna(antenna_labels[0])
    return datapack


def phase_screen_datapack(N,ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=-1,Nant = 10, Ntime = 1, Nfreqs = 1, fov = 4., alt = 90., az=0., time = None, radio_array=None,datapack=None):
    """Generate an empty datapack with N points in a grid pointing at alt (90 deg) and az (0 deg).
    The number of antennas and times are given by Nant and Ntime.
    field of view (fov) is by default 4 degrees.
    If time is None use current time."""
    if datapack is None:
        datapack = generate_example_datapack(Nant = Nant, Ntime = Ntime, Ndir = 1, Nfreqs=Nfreqs,fov = fov, alt = alt, az=az, time = time, radio_array=radio_array)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    freqs = datapack.get_freqs(freq_idx=freq_idx)
    Na = len(antennas)
    Nt = len(times)
    Nf = len(freqs)
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
    
#    phase = np.random.normal(size=[Na,Nt,len(dirs)])
    phase = np.zeros([Na,Nt,len(dirs),Nf])
    #prop = np.zeros([Na,Nt,len(dirs),Nf])
    clock = np.zeros([Na,Nt])
    const = np.zeros(Na)
    variance = np.zeros_like(phase)
    data_dict = datapack.get_data_dict()
    data_dict.update({'antennas':antennas,'antenna_labels':antenna_labels,'times':times,'timestamps':timestamps,
        'directions':dirs,'patch_names':patch_names,'phase':phase,'clock':clock,'const':const,'freqs':freqs,'variance':variance})
    datapack = DataPack(data_dict=data_dict)
    datapack.set_reference_antenna(antenna_labels[0])
    return datapack

if __name__ == '__main__':
    datapack = generate_example_datapack(Nant=20,Ntime=40,Ndir=42,Nfreqs=100)
    datapack.save("Test_Save.hdf5")
    data = DataPack(filename='Test_Save.hdf5')
