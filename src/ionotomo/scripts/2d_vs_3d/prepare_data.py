from ionotomo import *
import h5py
import numpy as np

if __name__=="__main__":
    datapack = DataPack(filename="../rvw_data_analysis/rvw_datapack.hdf5")
    antennas,antenna_labels = datapack.get_antennas(ant_idx=-1)
    times,timestamps = datapack.get_times(time_idx=range(400))
    directions,patch_names = datapack.get_directions(dir_idx=-1)
    freqs = datapack.get_freqs(freq_idx=-1)
    phase = datapack.get_phase(ant_idx=-1,time_idx=range(400),dir_idx=-1,freq_idx=-1)

    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    Nf = len(freqs)

    output = h5py.File("datapack_vanWeeren_partial_v1.hdf5",'w')
    output['/data/rays_uvw'] = np.zeros((Na,Nt,Nd,Nf,8),dtype=float)
    output['/data/phase'] = phase
    
    dt = h5py.special_dtype(vlen=str)
    antenna_labels_ = output.create_dataset("/data/antenna_labels",(Na,),dtype=dt)
    timestamps_ = output.create_dataset("/data/timestamps",(Nt,),dtype=dt)
    patch_names_ = output.create_dataset("/data/patch_names",(Nd,),dtype=dt)
    antenna_labels_[:] = antenna_labels
    timestamps_[:] = timestamps
    patch_names_[:] = patch_names

    for i,t in enumerate(times):
        print("Working on {}".format(timestamps[i]))
        phase_center = datapack.get_center_direction()
        fixtime = times[0]
        obstime = t
        center = datapack.radio_array.get_center()
        uvw = Pointing(location=center, phase= phase_center, fixtime=fixtime, obstime=t)
        ants_uvw = antennas.transform_to(uvw)
        dirs_uvw = directions.transform_to(uvw)
        output['/data/rays_uvw'][:,i:i+1,:,:,:] = np.stack(np.meshgrid(ants_uvw.u, 
            ants_uvw.v, 
            ants_uvw.w, 
            times[i:i+1], 
            dirs_uvw.u,
            dirs_uvw.v,
            dirs_uvw.w, 
            freqs,
            indexing='ij'),axis=-1)
    output.close()
