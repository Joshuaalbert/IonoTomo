from ionotomo import *
from ionotomo.utils.gaussian_process import *
from rathings.phase_unwrap import *
import pylab as plt
import numpy as np
import logging as log
import os
import h5py
import sys
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au

if sys.hexversion >= 0x3000000:
    def str_(s):
        return str(s,'utf-8')
else:
    def str_(s):
        return str(s)

tec_conversion = -8.4480e9# rad Hz/tecu


def prepare_phase(phase,axis=0,center=True):
    """unwrap phase and mean center
    phase : array
        phase to be unwrapped. 
    axis : int
        the axis to unwrap down (default 0)
    center : bool
        whether to mean center (defualt True)
    """
    phase = phase_unwrapp1d(phase,axis=axis)
    if center:
        phase -= np.mean(phase)
    return phase

def opt_kernel(times, phase, K, sigma_y=0, n_random_start=0):
    """Bayes Optimization of kernel wrt hyper params
    times : array
        array of times in seconds most likely
    phase : array
        1D array of phases already prepared.
    K : NDKernel
        the kernel for the level 2 optimization.
    sigma_y : float or array
        if float then measurement uncertainty for all phase.
        if array then measurement uncertainty for each phase array element
    n_random_start : int
        number of random initializations to use in optimization (default 0)
    """
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    X = times.reshape((-1,1))
    y = phase
    K.hyperparams = level2_solve(X,y,sigma_y,K,n_random_start=n_random_start)
    return K

def multi_opt_kernel(times, phase, K, sigma_y=0, n_random_start=0):
    """Bayes Optimization of kernel wrt hyper params over multiple directions
    times : array (num_times,)
        time array
    phase : array (num_times, num_directions)
        phases in several directions
    K : NDKernel
        the kernel for the level 2 optimization.
    sigma_y : float or array
        if float then measurement uncertainty for all phase.
        if array then measurement uncertainty for each phase array element
    n_random_start : int
        number of random initializations to use in optimization (default 0)
    """
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    num_directions = phase.shape[1]

    X = [times.reshape((-1,1))]*num_directions
    y = phase
    K.hyperparams = level2_multidataset_solve(X,y,[sigma_y]*num_directions,K,n_random_start=10)
    return K

def plot_prediction(times_predict, times, phase, K, sigma_y = 0,phase_true=None,figname=None,ant_label=None,patch_name=None):
    """Level1 predictive and plot
    times_predict : array
        the times to predict at
    times : array
        times for training set
    phase : array
        phase for training set
    K : NDKernel
        optimized kernel
    sigma_y : float of array
        if float then measurement uncertainty for all phase.
        if array then measurement uncertainty for each phase array element
    phase_true : array (optional)
        if given then the phases for `times_predict`
    ant_label : str (optional)
        if given plots the label
    """
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    X = times.reshape((-1,1))
    #smooth
    Xstar = X
    y = phase
    ystar, cov, lml = level1_solve(X,y,sigma_y,Xstar,K)

    plt.plot(X[:,0],y,c='teal',label='data')
    plt.plot(Xstar[:,0],ystar,c='red',ls='--')
    plt.plot(Xstar[:,0],ystar+np.sqrt(np.diag(cov)),c='green',ls='--')
    plt.plot(Xstar[:,0],ystar-np.sqrt(np.diag(cov)),c='blue',ls='--')
    if ant_label is not None:
        plt.title(ant_label)
    plt.xlabel('time (s)')
    plt.ylabel('phase (rad)')

    #y_true = prepare_phase(phase_true)
    Xstar = times_predict.reshape((-1,1))
    ystar, cov, lml = level1_solve(X,y,sigma_y,Xstar,K)
    std = np.sqrt(np.diag(cov))
    plt.plot(Xstar[:,0],ystar,c='red',ls='-',label='pred')
    plt.plot(Xstar[:,0],ystar+std,c='green',ls='-',label=r'$+\sigma$')
    plt.plot(Xstar[:,0],ystar-std,c='blue',ls='-',label=r'$-\sigma$')
    
    if phase_true is not None:
        y_true = phase_true
        plt.plot(Xstar[:,0],y_true,c='orange',label="true")
        
    plt.legend(frameon=False)
    plt.tight_layout()
    #plt.show()

def plot_bayes_smoothed(times, data, smoothed, std, figname,ant_label,patch_name,type):
    """Plot the smoothed
    times : array
        times for training set
    data : array
        tec_dd for training set
    smoothed : array
        smoothed version of tec
    figname : str
        figure name to save to
    ant_label : str
        antenna label
    patch_name : str
        patch name
    """
    
    plt.plot(times,data,c='orange',label='data')
    plt.plot(times,smoothed,c='red',ls='--',label='mean')
    plt.plot(times,smoothed + std,c='green',ls='--',label=r'$+\sigma$')
    plt.plot(times,smoothed - std,c='blue',ls='--',label=r'$-\sigma$')

    plt.title("{} | {}".format(ant_label,patch_name))
    plt.xlabel('time (s)')
    if type == 'tec':
        plt.ylabel('TEC (TEC units)')
    if type == 'cs':
        plt.ylabel('Scalar Phase (radians)')


    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figname,format='png')
    plt.close()
    #plt.show()

def smooth_data(times, phase, K, sigma_y = 0):
    """Level1 predictive of data
    times : array
        times for training set
    phase : array
        phase for training set
    K : NDKernel
        optimized kernel
    sigma_y : float of array
        if float then measurement uncertainty for all phase.
        if array then measurement uncertainty for each phase array element
    """
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    X = times.reshape((-1,1))
    #smooth
    Xstar = X
    y = phase
    ystar, cov, lml = level1_solve(X,y,sigma_y,Xstar,K)
    std = np.sqrt(np.diag(cov))
    return ystar, std

def smooth_dd_tec(dd_file,output_folder):
    """Use optima bayesian filtering.
    dd_file : str
        the hdf5 file containing direction dependent solutions
    """

    output_folder = os.path.join(os.getcwd(),output_folder)
    diagnostic_folder = os.path.join(output_folder,'diagnostics')
    try:
        os.makedirs(diagnostic_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)

    f_dd = h5py.File(dd_file,"r",libver="earliest")

    antenna_labels = []
    antenna_positions = []
    for row in f_dd['/sol000/antenna']:
        antenna_labels.append(str_(row[0]))
        antenna_positions.append(row[1])
    antenna_labels = np.array(antenna_labels)
    antenna_positions = np.array(antenna_positions)
    Na = len(antenna_labels)

    
    times = at.Time(f_dd['/sol000/tec000/time'][...]/86400., format='mjd',scale='tai')
    timestamps = times.isot
    times = times.gps
    Nt = len(times)

    patch_names = []
    directions = []
    for row in f_dd['/sol000/source']:
        patch_names.append(str_(row[0]).replace('[','').replace(']',''))
        directions.append(row[1])
    patch_names = np.array(patch_names).flatten()
    directions = np.array(directions)
    directions = ac.SkyCoord(directions[:,0]*au.rad, directions[:,1]*au.rad,frame='icrs')
    Nd = len(patch_names)

    #times, antennas, direction -> ijk
    tec_dd = np.einsum("jik->ijk",
            f_dd['/sol000/tec000/val'][:,:,:,0])

    scalarphase_dd = np.einsum("jik->ijk",
            f_dd['/sol000/scalarphase000/val'][:,:,:,0])

    coherence_time = 200.#seconds
    dt = times[1]-times[0]
    num_opt = int(coherence_time / dt * 4)

    sigma_y = 0#0.14/8.4480e9*120e6 #0.14 rad in TEC at 120MHz for approximation

    K1 = Diagonal(1, sigma = sigma_y / 2.)
    K1.set_hyperparams_bounds([1e-5,0.2],name='sigma')
    K2 = SquaredExponential(1,l=20)
    K2.set_hyperparams_bounds([dt*2,70],name='l')
    K2.set_hyperparams_bounds([1e-5,1],name='sigma')
    K3 = SquaredExponential(1,l=220)
    K3.set_hyperparams_bounds([70,300],name='l')
    K3.set_hyperparams_bounds([1e-5,1],name='sigma')

    K = K2 * K3 + K1

    tec_dd_smoothed = np.zeros([Na,Nt,Nd],dtype=float)
    tec_dd_std = np.zeros([Na,Nt,Nd],dtype=float)

    sc_dd_smoothed = np.zeros([Na,Nt,Nd],dtype=float)
    sc_dd_std = np.zeros([Na,Nt,Nd],dtype=float)


    for i in range(Na):
        for k in range(Nd):
            log.info("Working on {} | {}".format(antenna_labels[i],patch_names[k]))
            slices = range(0,Nt,num_opt>>1)
            count = np.zeros(Nt)
            tec_m = np.mean(tec_dd[i,:,k])
            scalarphase_dd[i,:,k] = phase_unwrapp1d(scalarphase_dd[i,:,k])
            for s in slices:
                start = s
                stop = min(Nt,start+num_opt)
                X = times[start:stop]
                count[start:stop] += 1
                y = tec_dd[i,start:stop,k]-tec_m
                K = opt_kernel(X,y, K, sigma_y=sigma_y, n_random_start=1)
                log.info(K)                
                ystar,std = smooth_data(X, y, K, sigma_y = 0)
                ystar += tec_m
                tec_dd_smoothed[i,start:stop,k] += ystar
                tec_dd_std[i,start:stop,k] += std**2

                y = scalarphase_dd[i,start:stop,k]
                K = opt_kernel(X,y, K, sigma_y=sigma_y, n_random_start=1)
                log.info(K)                
                ystar,std = smooth_data(X, y, K, sigma_y = 0)
                sc_dd_smoothed[i,start:stop,k] += ystar
                sc_dd_std[i,start:stop,k] += std
            tec_dd_smoothed[i,:,k] /= count
            tec_dd_std[i,:,k] /= count
            sc_dd_smoothed[i,:,k] /= count
            sc_dd_std[i,:,k] /= count
            np.sqrt(tec_dd_std[i,:,k],out=tec_dd_std[i,:,k])
            np.sqrt(sc_dd_std[i,:,k],out=sc_dd_std[i,:,k])

            figname=os.path.join(diagnostic_folder,"tec_bayes_smoothed_{}_{}.png".format(antenna_labels[i],patch_names[k]))
            plot_bayes_smoothed(times, tec_dd[i,:,k], tec_dd_smoothed[i,:,k], tec_dd_std[i,:,k], 
                    figname,antenna_labels[i],patch_names[k],type='tec')
            figname=os.path.join(diagnostic_folder,"scalarphase_bayes_smoothed_{}_{}.png".format(antenna_labels[i],patch_names[k]))
            plot_bayes_smoothed(times, scalarphase_dd[i,:,k], sc_dd_smoothed[i,:,k], sc_dd_std[i,:,k], 
                    figname,antenna_labels[i],patch_names[k],type='cs')
    f_dd.close()
    os.system("cp {} {}".format(dd_file,os.path.join(output_folder,dd_file.split('/')[-1].replace('.hdf5','_bayes_smoothed.hdf5'))))
    f_dd = h5py.File(os.path.join(output_folder,dd_file.split('/')[-1].replace('.hdf5','_bayes_smoothed.hdf5')),"r",libver="earliest")
    f_dd['/sol000/tec000/val'][:,:,:,0] = tec_dd_smoothed
    f_dd['/sol000/scalarphase000/val'][:,:,:,0] = sc_dd_smoothed
    f_dd.close()


if __name__=='__main__':
    dd_file = "../../data/NsolutionsDDE_2.5Jy_tecandphasePF_correctedlosoto.hdf5"
    smooth_dd_tec(dd_file,'output_bayes_smoothing')

