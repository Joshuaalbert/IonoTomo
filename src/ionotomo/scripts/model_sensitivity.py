from ionotomo.inversion.forward_equation import *
from ionotomo.geometry.calc_rays import *
#from ionotomo.inversion.forward_equation import *
from ionotomo.inversion.initial_model import *
from ionotomo.astro.real_data import generate_example_datapack
import numpy as np
import os
import pylab as plt
import logging as log
def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)

    try:
        os.makedirs(output_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    i0 = 0
    datapack = generate_example_datapack()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx = -1)
    times,timestamps = datapack.get_times(time_idx=-1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    ne_tci = create_initial_model(datapack)
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000, ne_tci.nz) 
    m_tci = ne_tci.copy()
    K_ne = np.median(m_tci.M)
    m_tci.M = np.log(m_tci.M/K_ne)
    m_tci_ = m_tci.copy()
    d0 = forward_equation(rays,K_ne,m_tci,i0)
    model_noise = []
    obs_noise = []
    dne = []
    for noise in 10**np.linspace(-3,0,100):
        m_tci_.M = m_tci.M + noise*np.random.normal(size=m_tci_.M.shape)
        d = forward_equation(rays,K_ne,m_tci_,i0)
        model_noise.append(noise)
        obs_noise.append(np.mean((d - d0)**2))
        dne.append(np.median(np.abs(K_ne * np.exp(m_tci_.M) - ne_tci.M)))
    plt.plot(model_noise,obs_noise)
    plt.xlabel("Model noise [log(ne/K)]")
    plt.ylabel("RMSE (TECU)")
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(output_folder,"noise_due_to_imperfect_model.png"),format='png')
    plt.show()
    plt.clf()
    plt.plot(model_noise,dne)
    plt.xlabel("Model noise [log(ne/K)]")
    plt.ylabel(r"$\Delta n_e$ (m^-3)")
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(output_folder,"median_dne_on_model_pert.png"),format='png')
    plt.show()

if __name__=='__main__':
    run('output')
