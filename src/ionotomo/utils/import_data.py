import h5py
import os
import numpy as np

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac


def transfer_patch_data(infoFile, data_folder, hdf5Out):
    '''transfer old numpy format to hdf5. Only run with python 2.7'''
    
    assert os.path.isdir(data_folder), "{0} is not a directory".format(data_folder)
    dt = h5py.special_dtype(vlen=str)
    f = h5py.File(hdf5Out,"w")
    
    info = np.load(infoFile)
    #these define the direction order
    patches = info['patches']#names
    radec = info['directions']#astrpy.icrs
    Nd = len(patches)
    print("Loading {} patches".format(Nd))
    namesds = f.create_dataset("dtec_observations/patch_names",(Nd,),dtype=dt)
    #rads = f.create_dataset("dtec_observations/patches/ra",(Nd,),dtype=np.double)
    #dec = f.create_dataset("dtec_observations/patches/dec",(Nd,),dtype=np.double)
    dset = f['dtec_observations']
    dset.attrs['frequency'] = 150e6
    namesds[...] = patches
    #rads[...] = radec.ra.deg
    #decds[...] = radec.dec.deg
    
    patch_idx = 0
    while patch_idx < Nd:
        patch = patches[patch_idx]
        #find the appropriate file (this will be standardized later)
        files = glob.glob("{0}/*_{1}_*.npz".format(data_folder,patch))
        if len(files) == 1:
            patchFile = files[0]
        else:
            print('Too many files found. Could not find patch: {0}'.format(patch))
            patch_idx += 1
            continue
        try:
            d = np.load(patchFile)
            print("Loading data file: {0}".format(patchFile))
        except:
            print("Failed loading data file: {0}".format(patchFile))
            return  
        if "dtec_observations/antenna_labels" not in f:
            antenna_labels = d['antennas']#labels
            Na = len(antenna_labels)
            antenna_labelsds = f.create_dataset("dtec_observations/antenna_labels",(Na,),dtype=dt)
            antenna_labelsds[...] = antenna_labels
        if "dtec_observations/timestamps" not in f:
            times = d['times']#gps tai
            timestamps = at.Time(times,format='gps',scale='tai').isot
            Nt = len(times)
            print(len(timestamps[0]))
            timeds = f.create_dataset("dtec_observations/timestamps",(Nt,),dtype=dt)
            timeds[...] = timestamps
        patchds = f.create_dataset("dtec_observations/patches/{}".format(patch),(Nt,Na),dtype=np.double)
        patchds.attrs['ra'] = radec[patch_idx].ra.deg
        patchds.attrs['dec'] = radec[patch_idx].dec.deg
        patch_idx += 1
    f.close()
    
def prepare_datapack(hdf5Datafile,timeStart=0,timeEnd=-1,array_file='arrays/lofar.hba.antenna.cfg'):
    '''Grab real data from soltions products. 
    Stores in a DataPack object.'''
    
    f = h5py.File(hdf5Datafile,'r')
    dset = f['dtec_observations']
    frequency = dset.attrs['frequency']
    print("Using radio array file: {}".format(array_file))
    #get array stations (they must be in the array file to be considered for data packaging)
    radio_array = RadioArray(array_file,frequency=frequency)#set frequency from solutions todo
    print("Created {}".format(radio_array))
    patch_names = f["dtec_observations/patch_names"][:].astype(str)
    Nd = len(patch_names)
    ra = np.zeros(Nd,dtype= np.double)
    dec = np.zeros(Nd,dtype=np.double)
    antenna_labels = f["dtec_observations/antenna_labels"][:].astype(str)
    Na = len(antenna_labels)
    antennas = np.zeros([3,Na],dtype=np.double)
    ant_idx = 0#index in solution table
    while ant_idx < Na:
        ant = antenna_labels[ant_idx]
        labelIdx = radio_array.get_antenna_idx(ant)  
        if labelIdx is None:
            print("failed to find {} in {}".format(ant,radio_array.labels))
            return
        #ITRS WGS84
        stationLoc = radio_array.locs[labelIdx]
        antennas[:,ant_idx] = stationLoc.cartesian.xyz.to(au.km).value.flatten()
        ant_idx += 1
    antennas = ac.SkyCoord(antennas[0,:]*au.km,antennas[1,:]*au.km,
                          antennas[2,:]*au.km,frame='itrs')
    timestamps = f["dtec_observations/timestamps"][:].astype(str)
    times = at.Time(timestamps,format="isot",scale='tai')
    Nt = len(timestamps)
    dtec = np.zeros([Na,Nt,Nd],dtype=np.double)
    patch_idx = 0
    while patch_idx < Nd:
        patchName = patch_names[patch_idx]
        patchds = f["dtec_observations/patches/{}".format(patchName)]
        ra[patch_idx] = patchds.attrs['ra']
        dec[patch_idx] = patchds.attrs['dec']
        dtec[:,:,patch_idx] = patchds[:,:].transpose()#from NtxNa to NaxNt
        patch_idx += 1
    f.close()
    directions = ac.SkyCoord(ra*au.deg,dec*au.deg,frame='icrs')
    data_dict = {'radio_array':radio_array,'antennas':antennas,'antenna_labels':antenna_labels,
                'times':times,'timestamps':timestamps,
                'directions':directions,'patch_names':patch_names,'dtec':dtec}
    return DataPack(data_dict)


