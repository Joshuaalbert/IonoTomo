from ionotomo import *
from ionotomo.bayes.phase_screen_interp import *
from rathings.phase_unwrap import phase_unwrapp1d
#from ionotomo.utils.gaussian_process import *
import h5py
import numpy as np
import astropy.units as au
import os


def _replicate(before, after,v):
    """bring from (N,) to (before,N,after)"""
    for _ in range(len(before)):
        v = np.expand_dims(v,0)
    for _ in range(len(after)):
        v = np.expand_dims(v,-1)
    v = np.tile(v,tuple(before) + (1,) + tuple(after))
    return v

def fit_datapack(datapack,template_datapack,ant_idx = -1, time_idx=-1, dir_idx=-1, freq_idx=-1):
    """Fit a datapack to a template datapack using Bayesian optimization
    of given kernel. Conjugation occurs at 350km."""

    antennas,antenna_labels = datapack.get_antennas(ant_idx)
    directions, patch_names = datapack.get_directions(dir_idx)
    times,timestamps = datapack.get_times(time_idx)
    freqs = datapack.get_freqs(freq_idx)
    phase = datapack.get_phase(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx,freq_idx=freq_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions) 
    Nf = len(freqs)
    for i in range(Na):
        for k in range(Nd):
            for l in range(Nf):
                phase[i,:,k,l] = phase_unwrapp1d(phase[i,:,k,l])
    phase -= np.mean(phase.flatten())


    X = np.zeros((Na,Nt,Nd,Nf,8),dtype=float)

    for i,t in enumerate(times):
        print("Working on {}".format(t.isot))
        phase_center = datapack.get_center_direction()
        fixtime = times[0]
        obstime = t
        center = datapack.radio_array.get_center()
        uvw = Pointing(location=center, phase= phase_center, fixtime=fixtime, obstime=t)
        ants_uvw = antennas.transform_to(uvw)
        dirs_uvw = directions.transform_to(uvw)

        #Na,1,Nd,Nf,8
        X[:,i:i+1,:,:,:] = np.stack([
            _replicate((),(1, Nd, Nf),ants_uvw.u.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.v.to(au.km).value),
            _replicate((),(1, Nd, Nf),ants_uvw.w.to(au.km).value),
            _replicate((Na,),(Nd,Nf),times[i:i+1].gps),
            _replicate((Na,1),(Nf,),dirs_uvw.u.value),
            _replicate((Na,1),(Nf,),dirs_uvw.v.value),
            _replicate((Na,1),(Nf,),dirs_uvw.w.value), 
            _replicate((Na,1,Nd),(),freqs)],axis=-1)

    X = X.reshape((-1,8))

    template_datapack = template_datapack.clone()

    antennas_,antenna_labels_ = template_datapack.get_antennas(ant_idx)
    directions_, patch_names_ = template_datapack.get_directions(dir_idx)
    times_,timestamps_ = template_datapack.get_times(time_idx)
    freqs_ = template_datapack.get_freqs(freq_idx)

    Na_ = len(antennas_)
    Nt_ = len(times_)
    Nd_ = len(directions_)
    Nf_ = len(freqs_)

    Xstar = np.zeros((Na_,Nt_,Nd_,Nf_,8),dtype=float)

    for i,t in enumerate(times_):
        print("Working on {}".format(t.isot))
        phase_center = template_datapack.get_center_direction()
        fixtime = times_[0]
        obstime = t
        center = template_datapack.radio_array.get_center()
        uvw = Pointing(location=center, phase= phase_center, fixtime=fixtime, obstime=t)
        ants_uvw = antennas_.transform_to(uvw)
        dirs_uvw = directions_.transform_to(uvw)

        Xstar[:,i:i+1,:,:,:] = np.stack([
            _replicate((),(1, Nd_, Nf_),ants_uvw.u.to(au.km).value),
            _replicate((),(1, Nd_, Nf_),ants_uvw.v.to(au.km).value),
            _replicate((),(1, Nd_, Nf_),ants_uvw.w.to(au.km).value),
            _replicate((Na_,),(Nd_, Nf_),times_[i:i+1].gps),
            _replicate((Na_,1),(Nf_,),dirs_uvw.u.value),
            _replicate((Na_,1),(Nf_,),dirs_uvw.v.value),
            _replicate((Na_,1),(Nf_,),dirs_uvw.w.value), 
            _replicate((Na_,1,Nd_),(),freqs_)],axis=-1)

    Xstar = Xstar.reshape((-1,8))
    y = phase.flatten()
    sigma_y = np.ones_like(y)*0.01
#    plot_datapack(datapack,ant_idx=ant_idx,time_idx=time_idx, dir_idx=-1,freq_idx=freq_idx,figname=None,vmin=None,vmax=None,mode='perantenna',observable='phase')
#    plot_datapack(datapack,ant_idx=ant_idx,time_idx=time_idx, dir_idx=-1,freq_idx=freq_idx,figname=None,vmin=None,vmax=None,mode='perantenna',observable='variance')
    
    K = PhaseScreen()
    #K = SquaredExponential()
    #K.set_hyperparams_bounds('l',[1e-5,10])
    P = Pipeline(1, X.shape[0],K, multi_dataset = False, share_x = True)
    neg_log_mar_like = P.level2_optimize(X,y,sigma_y,delta=0.001,patience=5,epochs=1000)
    win_idx = np.argmin(neg_log_mar_like)

    ystar,cov,lml = P.level1_predict(X,y,sigma_y,Xstar=Xstar,smooth=False,batch_idx = win_idx)
    ystar = ystar[0,:].reshape([Na_,Nt_,Nd_,Nf_])
    cov = cov[0,:,:]
    var = np.diag(cov).reshape([Na_,Nt_,Nd_,Nf_])
    template_datapack.set_phase(ystar,ant_idx = ant_idx, time_idx = time_idx, dir_idx = -1, freq_idx = freq_idx, ref_ant = datapack.ref_ant)
    template_datapack.set_variance(var,ant_idx = ant_idx, time_idx = time_idx, dir_idx = -1, freq_idx = freq_idx)
    plot_datapack(template_datapack,ant_idx=ant_idx,time_idx=time_idx, dir_idx=-1,freq_idx=freq_idx,figname=None,vmin=None,vmax=None,mode='perantenna',observable='phase')
    plot_datapack(template_datapack,ant_idx=ant_idx,time_idx=time_idx, dir_idx=-1,freq_idx=freq_idx,figname=None,vmin=None,vmax=None,mode='perantenna',observable='variance')
    return template_datapack



def run(output_folder, time_idx = range(1), freq_idx= range(1)):
    output_folder = os.path.join(os.getcwd(),output_folder)

    try:
        os.makedirs(output_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    datapack_folder = os.path.join(output_folder,"datapacks")
    datapack_rec_folder = os.path.join(output_folder,"datapacks_rec")

    try:
        os.makedirs(datapack_rec_folder)
    except:
        pass

    datapack_path = os.path.abspath("../rvw_data_analysis/rvw_datapack.hdf5")
    #ant_idx = select_antennas_idx(Na,datapack,ant_idx=-1,time_idx=-1)
    ant_idx = -1
    print('Loading original datapack')
    datapack = DataPack(filename=datapack_path)
    print('Creating phase screen')
    datapack_screen = phase_screen_datapack(10,datapack=datapack,ant_idx = ant_idx, time_idx=time_idx, dir_idx=-1, freq_idx=freq_idx)
    datapack_screen = fit_datapack(datapack,datapack_screen,ant_idx = ant_idx, time_idx=time_idx, dir_idx=-1, freq_idx=freq_idx)
    screen_path = os.path.join(datapack_rec_path,os.path.basename(datapack_path.replace('.hdf5','-rec.hdf5')))
    datapack_screen.save(screen_path)
    return
    log.info("Solving on generated ionospheres")
#    #using lofar configuration generate for a number of random pointings and times
#    radio_array = generate_example_radio_array(config='lofar')
    info_file = os.path.join(output_folder,"info")
    if os.path.exists(info_file) and os.path.isfile(info_file):
        info = open(info_file,'a')
    else:
        info = open(info_file,'w')
        info.write("#time_idx timestamp factor corr\n")
    times,timestamps = datapack_.get_times(time_idx=time_idx)
    freqs = datapack_.get_freqs(freq_idx=freq_idx)
   
    Nt = len(times)
    for factor in [2.,4.,8.,16.]:
        for corr in [10.,20.,40.,70.]:
            datapack_path = os.path.join(datapack_folder,"datapack_factr{:d}_corr{:d}.hdf5".format(int(factor),int(corr)))
            datapack = DataPack(filename=datapack_path)
            screen_path = os.path.join(datapack_folder,"datapack_screen_factr{:d}_corr{:d}.hdf5".format(int(factor),int(corr)))
            datapack_screen = DataPack(filename=screen_path)
            datapack_screen = fit_datapack(datapack,datapack_screen,ant_idx = ant_idx, time_idx=time_idx, dir_idx=-1, freq_idx=freq_idx)
            datapack_screen.save(osdatapack_path.replace('.hdf5','-rec.hdf5'))

if __name__ == '__main__':
    run("output")
