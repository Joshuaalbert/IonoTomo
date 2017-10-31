from ionotomo import *
import h5py
import numpy as np
import astropy.units as au
from rathings.phase_unwrap import phase_unwrapp1d

def plot(filename):
    output = h5py.File(filename,'r')
    x = output['/data/rays_uvw'][...]
    output.close()
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(x.shape[1]):
        #i = np.random.randint(x.shape[0])
        #j = np.random.randint(x.shape[1])
        #k = np.random.randint(x.shape[2]) 
        for k in range(4):
            ant = x[0,j,k,0,0:3]
            dir = x[0,j,k,0,4:7]
            ax.plot(*zip(ant,ant+dir*100))
    plt.show()




if __name__=="__main__":
#    plot("datapack_vanWeeren_partial_v1.hdf5")
#    exit()
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
    for i in range(Na):
        for k in range(Nd):
            for l in range(Nf):
                phase[i,:,k,l] = phase_unwrapp1d(phase[i,:,k,l])

    print(Na*Nt*Nd*Nf*8*8/1024/1024)

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
    def _replicate(before, after,v):
        """bring from (N,) to (before,N,after)"""
        for _ in range(len(before)):
            v = np.expand_dims(v,0)
        for _ in range(len(after)):
            v = np.expand_dims(v,-1)
        v = np.tile(v,tuple(before) + (1,) + tuple(after))
        return v

    for i,t in enumerate(times):
        print("Working on {}".format(timestamps[i]))
        phase_center = datapack.get_center_direction()
        fixtime = times[0]
        obstime = t
        center = datapack.radio_array.get_center()
        uvw = Pointing(location=center, phase= phase_center, fixtime=fixtime, obstime=t)
        ants_uvw = antennas.transform_to(uvw)
        dirs_uvw = directions.transform_to(uvw)

        #Na,1,Nd,Nf,8
        output['/data/rays_uvw'][:,i:i+1,:,:,:] = np.stack([
            _replicate((),(1, Nd, Nf),ants_uvw.u.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.v.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.w.to(au.km).value),
            _replicate((Na,),(Nd,Nf),times[i:i+1].gps),
            _replicate((Na,1),(Nf,),dirs_uvw.u.value),
            _replicate((Na,1),(Nf,),dirs_uvw.v.value),
            _replicate((Na,1),(Nf,),dirs_uvw.w.value), 
            _replicate((Na,1,Nd),(),freqs)],axis=-1)

    datapack_screen = phase_screen_datapack(10,datapack=datapack)

    antennas,antenna_labels = datapack_screen.get_antennas(ant_idx=-1)
    times,timestamps = datapack_screen.get_times(time_idx=-1)
    directions,patch_names = datapack_screen.get_directions(dir_idx=-1)
    freqs = datapack_screen.get_freqs(freq_idx=-1)
    phase = datapack_screen.get_phase(ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=-1)

    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    Nf = len(freqs)
    print(Na*Nt*Nd*Nf*8*8/1024/1024)

    output['/data/rays_uvw_star'] = np.zeros((Na,Nt,Nd,Nf,8),dtype=float)
    output['/data/phase_ystar'] = np.zeros_like(phase)
    output['/data/phase_var'] = np.zeros_like(phase)
    
    dt = h5py.special_dtype(vlen=str)
    antenna_labels_ = output.create_dataset("/data/antenna_labels_star",(Na,),dtype=dt)
    timestamps_ = output.create_dataset("/data/timestamps_star",(Nt,),dtype=dt)
    patch_names_ = output.create_dataset("/data/patch_names_star",(Nd,),dtype=dt)
    antenna_labels_[:] = antenna_labels
    timestamps_[:] = timestamps
    patch_names_[:] = patch_names

    for i,t in enumerate(times):
        print("Working on {}".format(timestamps[i]))
        phase_center = datapack_screen.get_center_direction()
        fixtime = times[0]
        obstime = t
        center = datapack_screen.radio_array.get_center()
        uvw = Pointing(location=center, phase= phase_center, fixtime=fixtime, obstime=t)
        ants_uvw = antennas.transform_to(uvw)
        dirs_uvw = directions.transform_to(uvw)

        #Na,1,Nd,Nf,8
        output['/data/rays_uvw_star'][:,i:i+1,:,:,:] = np.stack([
            _replicate((),(1, Nd, Nf),ants_uvw.u.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.v.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.w.to(au.km).value),
            _replicate((Na,),(Nd,Nf),times[i:i+1].gps),
            _replicate((Na,1),(Nf,),dirs_uvw.u.value),
            _replicate((Na,1),(Nf,),dirs_uvw.v.value),
            _replicate((Na,1),(Nf,),dirs_uvw.w.value), 
            _replicate((Na,1,Nd),(),freqs)],axis=-1)

    output.close()
