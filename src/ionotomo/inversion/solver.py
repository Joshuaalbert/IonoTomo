from ionotomo.astro.real_data import DataPack
from ionotomo.plotting.plot_tools import plot_datapack
from ionotomo.inversion.initial_model import create_initial_model
from ionotomo.geometry.calc_rays import calc_rays
import numpy as np
import os
import logging as log

class Solver(object):
    def __init__(self,datapack,output_folder,diagnostic_folder,**kwargs):
        self.datapack = datapack
        self.output_folder = output_folder
        self.diagnostic_folder = diagnostic_folder
        log.info("Initializing inversion for {}".format(self.datapack))

    @property
    def output_folder(self):
        return self._output_folder
    @output_folder.setter
    def output_folder(self,folder):
        self._output_folder = os.path.join(os.getcwd(),folder)
        try:
            os.makedirs(self._output_folder)
            log.basicConfig(filename=os.path.join(self.output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
        except:
            pass
    @property
    def diagnostic_folder(self):
        return self._diagnostic_folder
    @diagnostic_folder.setter
    def diagnostic_folder(self,folder):
        self._diagnostic_folder = os.path.join(self.output_folder, folder)
        try:
            os.makedirs(self._diagnostic_folder)
        except:
            pass

    @property
    def datapack(self):
        return self._datapack
    @datapack.setter
    def datapack(self,datapack):
        assert isinstance(datapack,DataPack)
        self._datapack = datapack
    
    def setup(ref_ant_idx = 0, tmax = 1000., L_ne = 20., size_cell = 5., straight_line_approx = True,**kwargs):
        time_idx = -1
        dir_idx = -1
        antennas,antenna_labels = self.datapack.get_antennas(ant_idx = ant_idx)
        patches, patch_names = self.datapack.get_directions(dir_idx = dir_idx)
        times,timestamps = self.datapack.get_times(time_idx=time_idx)
        self.datapack.set_reference_antenna(antenna_labels[ref_ant_idx])
        dobs = self.datapack.get_dtec(ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx)
        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches) 
        fixtime = times[Nt>>1]
        phase = self.datapack.get_center_direction()
        array_center = self.datapack.radio_array.get_center()
        #Get dtec error
        #Average time axis down and center on fixtime
        if Nt == 1:
            var = 0.01**2
            Cd = np.ones([Na,1,Nd],dtype=np.double)*var
            Ct = (np.abs(dobs)*0.01)**2
            CdCt = Cd + Ct        
        else:
            dt = times[1].gps - times[0].gps
            log.info("Averaging down window of length {} seconds [{} timestamps]".format(dt*Nt, Nt))
            Cd = np.stack([np.var(dobs,axis=1)]*Nt,axis=1)
            #dobs = np.stack([np.mean(dobs,axis=1)],axis=1)
            Ct = (np.abs(dobs)*0.01)**2
            CdCt = Cd + Ct
            #time_idx = [Nt>>1]
            #times,timestamps = datapack.get_times(time_idx=time_idx)
            #Nt = len(times)
        log.info("E[S/N]: {} +/- {}".format(np.mean(np.abs(dobs)/np.sqrt(CdCt+1e-15)),np.std(np.abs(dobs)/np.sqrt(CdCt+1e-15))))
        log.debug("CdCt = {}".format(CdCt))
        vmin = np.percentile(dobs,5)
        vmax = np.percentile(dobs,95)
        plot_datapack(self.datapack,ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx,
            figname='{}/dobs'.format(self.diagnostic_folder), vmin = vmin, vmax = vmax)
        ne_tci = create_initial_model(self.datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx, zmax = tmax,spacing=size_cell)
        #save the initial model?
        ne_tci.save(os.path.join(self.diagnostic_folder,"ne_initial.hdf5"))
        rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, self.datapack.radio_array.frequency, straight_line_approx, tmax, ne_tci.nz)
        m_tci = ne_tci.copy()
        K_ne = np.mean(ne_tci.M)
        m_tci.M /= K_ne
        np.log(m_tci.M,out=m_tci.M)
        self.K_ne,self.m_tci,self.rays,self.CdCt = K_ne,m_tci,rays,CdCt
            
    def go():
        raise NotImplementedError("Needs implementation")
