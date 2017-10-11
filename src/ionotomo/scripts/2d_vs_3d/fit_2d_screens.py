from ionotomo.astro.real_data import DataPack
from ionotomo.plotting.plot_tools import plot_datapack
from ionotomo.astro.fit_tec_2d import fit_datapack
from ionotomo.astro.antenna_facet_selection import select_random_facets
import numpy as np
import os
import logging as log

def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)
    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    plot_folder = os.path.join(output_folder,'datapack_rec')
    try:
        os.makedirs(plot_folder)
    except:
        pass
    log.info("Starting fit_tec_2d method on datapacks")
    time_idx = [0]#only first time slot
    info_file = os.path.join(output_folder,"info")
    datapack_files = np.genfromtxt(info_file,dtype='str',usecols=[4])
    alt_array = np.genfromtxt(info_file,dtype=float,usecols=[1])
    corr_array = np.genfromtxt(info_file,dtype=float,usecols=[2])
    max_residual = []
    corr = []
    alt = []
    thin = []
    num_facets = []
    rec_file = os.path.join(output_folder,"rec_info")
    if os.path.exists(info_file) and os.path.isfile(info_file):
        rec_info = open(rec_file,'a')
    else:
        rec_info = open(rec_file,'w')
        rec_info.write("#alt corr thin num_facets max_res template_datapack select_datapack fitted_datapack\n")
    log.info("Fitting...")
    for (alt_,corr_,datapack_file) in zip(alt_array,corr_array,datapack_files):
        log.info("Running on {} {} {}".format(alt_,corr_,datapack_file))
        datapack = DataPack(filename=datapack_file)
        antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
        patches, patch_names = datapack.get_directions(dir_idx = -1)
        times,timestamps = datapack.get_times(time_idx=time_idx)
        #flag directions 
        for N in [5,10,20,30,40,50]:#(np.arange(len(patches)-3)+4)[::4]:
            #datapack has full phase screen, call it template
            template_datapack = datapack.clone()
            #full
            dtec_template = template_datapack.get_dtec(ant_idx = -1,dir_idx=-1,time_idx=time_idx)
            corr.append(corr_)
            alt.append(alt_)
            thin.append('thin' in datapack_file)
            num_facets.append(N)
            #selection
            select_datapack = select_random_facets(N,template_datapack.clone(),dir_idx=-1,time_idx=time_idx)
            save_select_file = os.path.join(os.path.split(datapack_file)[0], os.path.split(datapack_file)[-1].replace('datapack','datapack_sel_{}_facets'.format(N)))
            select_datapack.save(save_select_file)
            #replace template
            fitted_datapack = fit_datapack(select_datapack,template_datapack, ant_idx=-1,time_idx=time_idx,dir_idx=-1)
            dtec_fitted = fitted_datapack.get_dtec(ant_idx = -1,dir_idx=-1,time_idx=time_idx)
            max_res = np.max(dtec_fitted - dtec_template)
            max_residual.append(max_res)
            save_fitted_file = os.path.join(os.path.split(datapack_file)[0], os.path.split(datapack_file)[-1].replace('datapack','datapack_rec_{}_facets'.format(N)))
            fitted_datapack.save(save_fitted_file)
            datapack_fitted_plot = os.path.join(plot_folder,save_fitted_file.split(os.sep)[-1].split('.hdf5')[0])
            datapack_select_plot = os.path.join(plot_folder,save_select_file.split(os.sep)[-1].split('.hdf5')[0])
            plot_datapack(fitted_datapack,ant_idx=-1,time_idx=time_idx,dir_idx=-1,figname=datapack_fitted_plot)
            plot_datapack(select_datapack,ant_idx=-1,time_idx=time_idx,dir_idx=-1,figname=datapack_select_plot)
            rec_info.write("{} {} {} {} {} {} {} {}\n".format(alt[-1],corr[-1],thin[-1],num_facets[-1],max_residual[-1],datapack_file, save_select_file, save_fitted_file))
            rec_info.flush()
    rec_info.close()
        
if __name__=='__main__':
    run('output')
