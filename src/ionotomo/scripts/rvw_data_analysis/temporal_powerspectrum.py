from ionotomo import *
from ionotomo.utils.gaussian_process import *

if __name__ == '__main__':
    time_idx = range(200)
    datapack = DataPack(filename="rvw_datapack.hdf5")
    antennas,antenna_labels = datapack.get_antennas(ant_idx=-1)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    directions, patch_names = datapack.get_directions(dir_idx=-1)
    phase = datapack.get_phase(ant_idx=-1,time_idx=time_idx,dir_idx=-1)

    def get_kernel(times, phase, K):
        X = np.array([times.gps]).T
        y = phase
        K.hyperparams = level2_solve(X,y,0,K)
        return K

    K1 = Diagonal(1)
    K2 = SquaredExponential(1)
    K2.set_hyperparams_bounds([2,40],name='l')
    K3 = SquaredExponential(1)
    K3.set_hyperparams_bounds([50,1000],name='l')
    K = K1 + K2 + K3


    K = get_kernel(times,phase[1,:,0,0])
    print(K)

    
