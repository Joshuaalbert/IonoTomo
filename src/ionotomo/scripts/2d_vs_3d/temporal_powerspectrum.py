
# coding: utf-8

# In[1]:


from ionotomo import *
from ionotomo.utils.gaussian_process import *
from rathings.phase_unwrap import *
import pylab as plt
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10,10)




# In[2]:


# data intake ~ 28GB

datapack = DataPack(filename="../scripts/rvw_datapack.hdf5")
print("Loaded : {}".format(datapack))
antennas,antenna_labels = datapack.get_antennas(ant_idx=-1)
times,timestamps = datapack.get_times(time_idx=-1)
directions, patch_names = datapack.get_directions(dir_idx=-1)
phase = datapack.get_phase(ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=[0])
   


# In[3]:


## functions

def prepare_phase(phase):
    phase = phase_unwrapp1d(phase,axis=0)
    phase -= np.mean(phase)
    return phase

def opt_kernel(times, phase, K, sigma_y=0):
    """Bayes Optimization"""
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    X = np.array([times.gps]).T
    y = prepare_phase(phase)
    K.hyperparams = level2_solve(X,y,sigma_y,K,n_random_start=0)
    return K

def multi_opt_kernel(times, phase, K, sigma_y=0):
    """Bayes Optimization over multiple directions
    times : array (num_times,)
        time array
    phase : array (num_times, num_directions)
        phases in several directions
    """
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    num_directions = phase.shape[1]
    
    X = [np.array([times.gps]).T]*num_directions
    y = prepare_phase(phase).T
    K.hyperparams = level2_multidataset_solve(X,y,[sigma_y]*num_directions,K,n_random_start=10)
    return K

def plot_prediction(times_predict, times, phase, K, sigma_y = 0,phase_true=None,ant_label=None):
    """Level1 predictive"""
    assert len(times) < np.sqrt(1e6), "Don't want to do too many ops"
    X = np.array([times.gps]).T
    
    Xstar = X
    #y = prepare_phase(phase)
    y = phase
    ystar, cov, lml = level1_solve(X,y,sigma_y,Xstar,K)
    
    plt.plot(X[:,0],y,label='data')
    plt.plot(Xstar[:,0],ystar,c='red',ls='--')
    plt.plot(Xstar[:,0],ystar+np.sqrt(np.diag(cov)),c='green',ls='--')
    plt.plot(Xstar[:,0],ystar-np.sqrt(np.diag(cov)),c='blue',ls='--')
    if ant_label is not None:
        plt.title(ant_label)
    plt.xlabel('time (s)')
    plt.ylabel('phase (rad)')
    
    if phase_true is not None:
        #y_true = prepare_phase(phase_true)
        Xstar = np.array([times_predict.gps]).T
        ystar, cov, lml = level1_solve(X,y,sigma_y,Xstar,K)
        y_true = phase_true
        plt.plot(Xstar[:,0],y_true,label="true")
        plt.plot(Xstar[:,0],ystar,c='red',ls='-',label='pred')
        plt.plot(Xstar[:,0],ystar+np.sqrt(np.diag(cov)),c='green',ls='-',label=r'$+\sigma$')
        plt.plot(Xstar[:,0],ystar-np.sqrt(np.diag(cov)),c='blue',ls='-',label=r'$-\sigma$')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# In[15]:


ant_id = 52

sigma_y = 2*np.pi/180.

for ant_id in range(62):
    print("Using : {}".format(antenna_labels[ant_id]))
    phases = prepare_phase(phase[ant_id,:,:,0])
    # plt.imshow(phases,aspect='auto')
    # plt.colorbar()
    # plt.show()

    K1 = Diagonal(1)
    K1.set_hyperparams_bounds([1e-5,10*np.pi/180.],name='sigma')
    K2 = SquaredExponential(1,l=20)
    K2.set_hyperparams_bounds([8,50],name='l')
    K2.set_hyperparams_bounds([1e-5,5],name='sigma')
    K3 = SquaredExponential(1,l=220)
    K3.set_hyperparams_bounds([50,1000],name='l')
    K3.set_hyperparams_bounds([1e-5,5],name='sigma')
    K = K1 + K2 + K3

    K = multi_opt_kernel(times[:200],phases[:200],K,sigma_y=sigma_y)
    print(K)
    #plot_prediction(times[200:300],times[:200:2],phases[:200:2], K,sigma_y=0.03,phase_true=phases[200:300],ant_label=antenna_labels[ant_id])






# In[ ]:


for ant_id in range(62):

    print("Using : {}".format(antenna_labels[ant_id]))
    print(phase.shape)
    phases = prepare_phase(phase[ant_id,:,0,0])

    K1 = Diagonal(1)
#     K2 = SquaredExponential(1)
#     K2.set_hyperparams_bounds([50,1000],name='l')
    K3 = RationalQuadratic(1)
    K3.set_hyperparams_bounds([50,500],name='l')
#     K4 = DotProduct(1,c=times[0].gps)
    K = K1 + K3

    K = opt_kernel(times[:200:2],phases[:200:2],K,sigma_y=0.03)
    print(K)
    plot_prediction(times[200:300],times[:200:2],phases[:200:2], K,sigma_y=0.03,phase_true=phases[200:300],ant_label=antenna_labels[ant_id])


# In[ ]:




