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
                        'patch_names' : self.patch_names, 'freqs':self.freqs,'phase':self.phase,'ref_ant':self.ref_ant, 'const':self.const, 'clock':self.clock, 'prop':self.prop}

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
        phase = f.create_dataset("datapack/phase",(self.Na,self.Nt,self.Nd,self.Nf),dtype=np.double)
        phase[:,:,:,:] = self.phase
        prop = f.create_dataset("datapack/prop",(self.Na,self.Nt,self.Nd,self.Nf),dtype=np.double)
        prop[:,:,:,:] = self.prop

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
        self.phase = f["datapack/phase"][:,:,:,:]
        self.prop = f["datapack/prop"][:,:,:,:]
        self.clock = f["datapack/clock"][:,:]
        self.const = f["datapack/const"][:]
        self.ref_ant = np.array(f["datapack/phase"].attrs['ref_ant']).astype(str).item(0)
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.Nf = len(self.freqs)
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
                'prop' : (np.array,"Propagation term"),
                'ref_ant' : (str,"The reference antenna label")}
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
        
        for attr in args.keys():
            if attr in params.keys():
                #not list testing
                #assert isinstance(args[attr],params[attr][0], "Invalid attribute type {}. Help says {} is {}".format(type(args[attr]),attr,params[attr][1])
                #print(attr)
                try:
                    setattr(self,attr,args[attr])
                except:
                    print("Failed to set {0} to {1}".format(attr,args[attr]))
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.Nf = len(self.freqs)
        assert self.phase.shape == (self.Na,self.Nt,self.Nd,self.Nf)
        assert self.const.shape == (self.Na,)
        assert self.clock.shape == (self.Na,self.Nt)
        assert self.prop.shape == (self.Na,self.Nt,self.Nd,self.Nf)
        self.antenna_labels = np.array(self.antenna_labels)
        self.patch_names = np.array(self.patch_names)
        self.timestamps = np.array(self.timestamps)
        if 'ref_ant' in args.keys():
            self.set_reference_antenna(args['ref_ant'])
    
    def set_const(self,const,ant_idx=[], ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        ant_idx = np.sort(ant_idx)
        Na = len(ant_idx)
        i = 0
        while i < Na:
            self.clock[ant_idx[i]] = const[i]
            i += 1
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        else:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)


    def get_const(self,ant_idx=[]):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        #assert self.ref_ant is not None, "set reference antenna first"
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        ant_idx = np.sort(ant_idx)
        Na = len(ant_idx)
        output = np.zeros([Na],dtype=np.double)
        i = 0
        while i < Na:
            output[i] = self.const[ant_idx[i]]
            i += 1
        return output

    def set_clock(self,clock,ant_idx=[],time_idx=[], ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
               self.clock[ant_idx[i],time_idx[j]] = clock[i,j]
               j += 1
            i += 1
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        else:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)


    def get_clock(self,ant_idx=[],time_idx=[]):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        #assert self.ref_ant is not None, "set reference antenna first"
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        output = np.zeros([Na,Nt],dtype=np.double)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                output[i,j] = self.clock[ant_idx[i],time_idx[j]]
                j += 1
            i += 1
        return output

    def set_prop(self,prop,ant_idx=[],time_idx=[], dir_idx=[],freq_idx = -1, ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        freq_idx = np.sort(freq_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        Nf = len(freq_idx)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    for l in range(Nf):
                        self.prop[ant_idx[i],time_idx[j],dir_idx[k],freq_idx[l]] = prop[i,j,k,l]
                    k += 1
                j += 1
            i += 1
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        else:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)


    def get_prop(self,ant_idx=[],time_idx=[], dir_idx=[],freq_idx=-1):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        #assert self.ref_ant is not None, "set reference antenna first"
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        freq_idx = np.sort(freq_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        Nf = len(freq_idx)
        output = np.zeros([Na,Nt,Nd,Nf],dtype=np.double)
        indices = np.meshgrid(ant_idx,time_idx,dir_idx,freq_idx,indexing='ij')
        output = self.prop[indices]
#        i = 0
#        while i < Na:
#            j = 0
#            while j < Nt:
#                k = 0
#                while k < Nd:
#                    for l in range(Nf):
#                        output[i,j,k,l] = self.prop[ant_idx[i],time_idx[j],dir_idx[k],freq_idx[l]]
#                    k += 1
#                j += 1
#            i += 1
        return output


                
    def set_phase(self,phase,ant_idx=[],time_idx=[], dir_idx=[],freq_idx = -1, ref_ant=None):
        '''Set the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        freq_idx = np.sort(freq_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        Nf = len(freq_idx)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    for l in range(Nf):
                        self.phase[ant_idx[i],time_idx[j],dir_idx[k],freq_idx[l]] = phase[i,j,k,l]
                    k += 1
                j += 1
            i += 1
        if ref_ant is not None:
            self.set_reference_antenna(ref_ant)
        else:
            if self.ref_ant is not None:
                self.set_reference_antenna(self.ref_ant)
                

    def get_phase(self,ant_idx=[],time_idx=[], dir_idx=[],freq_idx=-1):
        '''Retrieve the specified phase solutions corresponding to the requested indices.
        value of -1 means all.'''
        #assert self.ref_ant is not None, "set reference antenna first"
        if ant_idx is -1:
            ant_idx = np.arange(self.Na)
        if time_idx is -1:
            time_idx = np.arange(self.Nt)
        if dir_idx is -1:
            dir_idx = np.arange(self.Nd)
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        ant_idx = np.sort(ant_idx)
        time_idx = np.sort(time_idx)
        dir_idx = np.sort(dir_idx)
        freq_idx = np.sort(freq_idx)
        Na = len(ant_idx)
        Nt = len(time_idx)
        Nd = len(dir_idx)
        Nf = len(freq_idx)
        output = np.zeros([Na,Nt,Nd,Nf],dtype=np.double)
        indices = np.meshgrid(ant_idx,time_idx,dir_idx,freq_idx,indexing='ij')
        output = self.phase[indices]
#        i = 0
#        while i < Na:
#            j = 0
#            while j < Nt:
#                k = 0
#                while k < Nd:
#                    for l in range(Nf):
#                        output[i,j,k,l] = self.phase[ant_idx[i],time_idx[j],dir_idx[k],freq_idx[l]]
#                    k += 1
#                j += 1
#            i += 1
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

    def get_freqs(self,freq_idx=[]):
        '''Get the list of antenna locations in itrs'''
        if freq_idx is -1:
            freq_idx = np.arange(self.Nf)
        freq_idx = np.sort(freq_idx)
        output = self.freqs[freq_idx]
        Nf = len(freq_idx)
        return output
    
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
        self.phase -= self.phase[ref_ant_idx,:,:,:]
        self.prop -= self.prop[ref_ant_idx,:,:,:]
        self.clock -= self.clock[ref_ant_idx,:]
        self.const -= self.const[ref_ant_idx]

    def get_center_direction(self):
        ra_mean = np.mean(self.directions.transform_to('icrs').ra)
        dec_mean = np.mean(self.directions.transform_to('icrs').dec)
        phase = ac.SkyCoord(ra_mean,dec_mean,frame='icrs')
        return phase

    def find_flagged_antennas(self):
        '''Determine which antennas are flagged'''
        assert self.ref_ant is not None, "Set a ref_ant before finding flagged (zeroed) antennas"
        mask = np.sum(np.sum(self.phase,axis=2),axis=1) == 0
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
        self.phase = self.phase[mask,:,:,:]
        self.prop = self.prop[mask,:,:,:]
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
        self.prop = self.prop[:,:,mask,:]
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
        self.prop = self.prop[:,mask,:,:]
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
        self.prop = self.prop[:,:,:,mask]
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
    prop = np.zeros([Nant,Ntime,Ndir,Nfreqs])
    phase = np.einsum("i,j,k,l,i->ijkl",np.ones(Nant),np.ones(Ntime),np.ones(Ndir),np.ones(Nfreqs),const)
    for l in range(Nfreqs):
        a_ = 2*np.pi * freqs[l]
        dg = a_ * np.einsum("ij,k->ijk",clock,np.ones(Ndir))
        dg -= 8.4480e-7/freqs[l]*tec
        prop[:,:,:,l] += 8.4480e-7/freqs[l]*tec
        phase[:,:,:,l] += dg
    phase += np.random.normal(size=phase.shape)*5*np.pi/180.
    data_dict = {'radio_array':radio_array,'antennas':antennas,'antenna_labels':antenna_labels,
                    'times':times,'timestamps':times.isot,
                    'directions':dirs,'patch_names':patch_names,
                    'freqs':freqs,'phase':phase,'clock':clock,'const':const,'prop':prop}
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
    prop = np.zeros([Na,Nt,len(dirs),Nf])
    clock = np.zeros([Na,Nt])
    const = np.zeros(Na)
    data_dict = datapack.get_data_dict()
    data_dict.update({'antennas':antennas,'antenna_labels':antenna_labels,'times':times,'timestamps':timestamps,
        'directions':dirs,'patch_names':patch_names,'phase':phase,'prop':prop,'clock':clock,'const':const})
    datapack = DataPack(data_dict=data_dict)
    datapack.set_reference_antenna(antenna_labels[0])
    return datapack

if __name__ == '__main__':
    datapack = generate_example_datapack(Nant=20,Ntime=40,Ndir=42,Nfreqs=100)
    datapack.save("Test_Save.hdf5")
    print(datapack)
    phase = datapack.get_phase(ant_idx=[0,1],time_idx=-1,dir_idx=[0],freq_idx=-1)
    prop = datapack.get_prop(ant_idx=[0,1],time_idx=-1,dir_idx=[0],freq_idx=-1)
    phase = np.angle(np.exp(1j*phase))
    prop = np.angle(np.exp(1j*prop))
    #print(phase)
    import pylab as plt
    plt.plot(datapack.freqs,prop[1,0,0,:])
    plt.show()
    plt.imshow(prop[1,:,0,:].T,cmap='rainbow')
    plt.show()


