'''Choose optimal facet and antenna layout from a datapack object'''

from ionotomo.astro.real_data import DataPack
import numpy as np
import astropy.units as au
from ionotomo.astro.frames.uvw_frame import UVW
import logging as log

def select_random_facets(N,datapack,dir_idx=-1,time_idx=-1):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Nd, "Requested number of directions {} to large {}".format(N,datapack.Na)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    flag_patches = []
    unflagged = list(patch_names)
    k = 0
    while len(unflagged) > N:
        idx = np.random.randint(len(unflagged))
        flag_patches.append(unflagged[idx])
        del unflagged[idx]
        k += 1
    out_datapack = datapack.clone()
    out_datapack.flag_directions(flag_patches)
    log.info("flagged {}".format(flag_patches))
    return out_datapack



def select_facets(N,datapack,dir_idx=-1,time_idx=-1):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Nd, "Requested number of directions {} to large {}".format(N,datapack.Na)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Nd = len(patches)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #log.info("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.get_center_direction()
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #
    dirs = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    center = phase.transform_to(uvw).cartesian.xyz.value.transpose()
    mag = np.arccos(np.dot(dirs, center))
    #mag and get first
    argsort = np.argsort(mag)
    outdirs = [dirs[argsort[0],:]]
    outidx = [argsort[0]]
    i = 1
    while i < N:
        center = np.mean(outdirs,axis=0)
        mag = np.arccos(np.dot(dirs,center))
        argsort = np.argsort(mag)
        j = len(argsort) - 1
        while j >= 0:
            if argsort[j] not in outidx:
                outidx.append(argsort[j])
                outdirs.append(dirs[argsort[j],:])
                break
            j -= 1
        i += 1
    # flag all others
    flag = []
    i = 0
    while i < len(patch_names):
        if i not in outidx:
            flag.append(patch_names[i])
        i += 1
    out_datapack = datapack.clone()
    out_datapack.flag_directions(flag)
    log.info("flagged {}".format(flag))
    return out_datapack

def select_antennas_idx(N,datapack,ant_idx=-1,time_idx=-1):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Na, "Requested number of antennas {} to large {}".format(N,datapack.Na)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #log.info("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.get_center_direction()
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #
    center = datapack.radio_array.get_center().transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    ants = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dist = ants - center
    #mag and get first
    mag = np.linalg.norm(dist,axis=1)
    argsort = np.argsort(mag)
    outants = [ants[argsort[0],:]]
    outidx = [argsort[0]]
    i = 1
    while i < N:
        center = np.mean(outants,axis=0)
        dist = np.subtract(ants,center)
        mag = np.linalg.norm(dist,axis=1)
        argsort = np.argsort(mag)
        j = len(argsort) - 1
        while j >= 0:
            if argsort[j] not in outidx:
                outidx.append(argsort[j])
                outants.append(ants[argsort[j],:])
                break
            j -= 1
        i += 1
    # flag all others
    ant_idx = []
    flag = []
    i = 0
    while i < len(antenna_labels):
        if i not in outidx:
            flag.append(antenna_labels[i])
        else:
            ant_idx.append(i)
        i += 1
    return ant_idx


def select_antennas(N,datapack,ant_idx=-1,time_idx=-1):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Na, "Requested number of antennas {} to large {}".format(N,datapack.Na)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #log.info("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.get_center_direction()
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #
    center = datapack.radio_array.get_center().transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    ants = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dist = ants - center
    #mag and get first
    mag = np.linalg.norm(dist,axis=1)
    argsort = np.argsort(mag)
    outants = [ants[argsort[0],:]]
    outidx = [argsort[0]]
    i = 1
    while i < N:
        center = np.mean(outants,axis=0)
        dist = np.subtract(ants,center)
        mag = np.linalg.norm(dist,axis=1)
        argsort = np.argsort(mag)
        j = len(argsort) - 1
        while j >= 0:
            if argsort[j] not in outidx:
                outidx.append(argsort[j])
                outants.append(ants[argsort[j],:])
                break
            j -= 1
        i += 1
    # flag all others
    flag = []
    i = 0
    while i < len(antenna_labels):
        if i not in outidx:
            flag.append(antenna_labels[i])
        i += 1
    out_datapack = datapack.clone()
    out_datapack.flag_antennas(flag)
    log.info("flagged {}".format(flag))
    out_datapack.set_reference_antenna(antenna_labels[outidx[0]])
    return out_datapack

def select_antennas_facets(N,datapack,ant_idx=-1,dir_idx=-1,time_idx=-1):
    datapack = select_antennas(N,datapack,ant_idx=ant_idx,time_idx=time_idx)
    datapack = select_facets(N,datapack,dir_idx=dir_idx,time_idx=time_idx)
    return datapack

