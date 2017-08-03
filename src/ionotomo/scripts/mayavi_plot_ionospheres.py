from ionotomo.plotting.mayavi_tools import plot_tci
from ionotomo.geometry.tri_cubic import TriCubic
import numpy as np
import os
import logging as log

def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)
    tci_folder = os.path.join(output_folder,"ionospheres")
    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.INFO)
    log.info("Using output folder {}".format(output_folder))
    plot_folder = os.path.join(output_folder,'tci_plots')
    try:
        os.makedirs(plot_folder)
    except:
        pass
    log.info("Plotting generated ionospheres")
    files = np.genfromtxt(os.path.join(output_folder,"info"),dtype='str',usecols=[5])
    for file in files:
    #    file = str(file,'utf-8')
        tci = TriCubic(filename=os.path.join(tci_folder,file))
        plot_tci(tci,filename=os.path.join(plot_folder,file.split('.hdf5')[0]))
if __name__=='__main__':
    run('output')
