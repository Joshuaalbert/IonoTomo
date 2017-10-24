"""This loads the data from Reinout's reduction into a DataPack"""
from ionotomo.astro.real_data import DataPack
from ionotomo.astro.radio_array import RadioArray
import h5py

import numpy as np
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at

import os

import sys
import logging as log

if sys.hexversion >= 0x3000000:
    def str_(s):
        return str(s,'utf-8')
else:
    def str_(s):
        return str(s)

#TECU = 1e16
tec_conversion = -8.4480e9# rad Hz/tecu


def import_data(dd_file, di_file, datapack_file, clobber=False):
    """Create a datapack from the direction (de)independent files.
    dd_file : str
        path to dd solutions made by NDPPP
    di_file : str
        path to di solutions made with losoto
    datapack_file : str
        path to store resulting DataPack

    Note: assumes dtypes in hdf5 files are float
    """
    f_dd = h5py.File(dd_file,"r",libver="earliest")
    f_di = h5py.File(di_file,"r")

    assert (os.path.isfile(datapack_file) and clobber) or (not os.path.isfile(datapack_file)), "datapack file {} already exists and clobber is False".format(datapack_file)

    antenna_labels = []
    antenna_positions = []
    for row in f_dd['/sol000/antenna']:
        antenna_labels.append(str_(row[0]))
        antenna_positions.append(row[1])
    antenna_labels = np.array(antenna_labels)
    antenna_positions = np.array(antenna_positions)
#    antenna_positions = ac.SkyCoord(antenna_positions[:,0]*au.m, 
#            antenna_positions[:,1]*au.m, 
#            antenna_positions[:,2]*au.m,frame='itrs')
    Na = len(antenna_labels)

    radio_array = RadioArray(array_file = RadioArray.lofar_array)
    ant_idx = []#location in radio_array (we don't go by file)
    for n, lab in enumerate(antenna_labels):
        i = radio_array.get_antenna_idx(lab)
        assert i is not None, "{} doesn't exist in radio array, please add entry: {} {}".format(lab,lab,antenna_positions[n])
        ant_idx.append(i)

    patch_names = []
    directions = []
    for row in f_dd['/sol000/source']:
        patch_names.append(str_(row[0]))
        directions.append(row[1])
    patch_names = np.array(patch_names)
    directions = np.array(directions)
    directions = ac.SkyCoord(directions[:,0]*au.rad, directions[:,1]*au.rad,frame='icrs')
    Nd = len(patch_names)

    times = at.Time(f_dd['/sol000/tec000/time'][...]/86400., format='mjd',scale='tai')
    timestamps = times.isot
    Nt = len(times)

    freqs = f_di['/sol000/phasesoffset000/freq'][...]#Hz
    Nf = len(freqs)

    data_dict = {"radio_array":radio_array,"times":times, "timestamps": timestamps, "directions": directions, "patch_names": patch_names, "antennas": radio_array.get_antenna_locs(), "antenna_labels": radio_array.get_antenna_labels(), 'freqs': freqs}

    #now construct data 
    #Direction1: scalarphase000(dir1) + tec000(dir1) + clock000+ tec000+ phaseoffset000
    phase = np.zeros([Na,Nt,Nd,Nf],dtype=float)
    freqs_inv = 1./freqs
#    for i in range(Na):
#        for j in range(Nt):
#            for k in range(Nd):
                #di part

    phaseoffset_di = np.einsum('ilj,k->ijkl',
            f_di['/sol000/phasesoffset000/val'][0,0,:,:,:],
            np.ones(Nd,dtype=float))

    tec_di = tec_conversion*np.einsum('ji,k,l->ijkl',
            f_di['/sol000/tec000/val'][:,:],
            np.ones(Nd,dtype=float),
            freqs_inv)

    clock_di = (2*np.pi)*np.einsum('ji,k,l->ijkl',
            f_di['/sol000/clock000/val'][:,:], 
            np.ones(Nd,dtype=float), 
            freqs)

    #dd part
    scalarphase_dd = np.einsum("jik,l->ijkl",
            f_dd['/sol000/scalarphase000/val'][:,:,:,0],
            np.ones(Nf,dtype=float))
    tec_dd = tec_conversion*np.einsum("jik,l->ijkl",
            f_dd['/sol000/tec000/val'][:,:,:,0], 
            freqs_inv)

    phase = phaseoffset_di + tec_di + clock_di + scalarphase_dd + tec_dd

    #fill out some of the values
    prop = tec_dd# + scalarphase_dd#tec_di + tec_dd#radians
    clock = f_di['/sol000/clock000/val'][:,:].T#seconds
    const = np.mean(np.mean(np.mean((phaseoffset_di + scalarphase_dd)[:,:,:,:],axis=3),axis=2),axis=1)#radians
    data_dict.update({'phase':phase, 'prop':prop, 'clock':clock, 'const':const})
    datapack = DataPack(data_dict)
    datapack.set_reference_antenna(antenna_labels[0])

    datapack.save(datapack_file)
    f_dd.close()
    f_di.close()
    return datapack

if __name__=='__main__':
    import_data("../../data/NsolutionsDDE_2.5Jy_tecandphasePF_correctedlosoto.hdf5",
            "../../data/DI.circ.hdf5",
            "rvw_datapack_only_tec_dd.hdf5",
            clobber=True)
#    from ionotomo.plotting.plot_tools import plot_datapack, animate_datapack
#    datapack = DataPack(filename="rvw_datapack.hdf5")
##    animate_datapack(datapack,"rvw_datapack_animation_perdirection", num_threads=1,mode='perdirection')
##    animate_datapack(datapack,"rvw_datapack_animation_phase", num_threads=1,mode='perantenna',observable='phase')
#    animate_datapack(datapack,"rvw_datapack_animation_prop", num_threads=1,mode='perantenna',observable='prop')




