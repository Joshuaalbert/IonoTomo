'''
Specifies the inversion pipeline.
'''
import logging as log
from dask.distributed import Client
from ionotomo import *
import os
import numpy as np
from functools import partial
import astropy.time as at
import astropy.units as au

class InversionPipeline(object):
    def __init__(self, datapack, output_folder = 'output', diagnostic_folder = 'diagnostics', **kwargs):
        self.output_folder = os.path.join(os.getcwd(),output_folder)
        self.diagnostic_folder = os.path.join(self.output_folder,diagnostic_folder)
        log.basicConfig(filename=os.path.join(self.output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
        log.info("Initializing inversion for {}".format(datapack))
        self.datapack = datapack
        self.default_params()
        for key in kwargs.keys():
            try:
                cur_val = getattr(self,key)
                log.info("Setting {} from {} to {}".format(key,cur_val,kwargs[key]))
                setattr(self,key,kwargs[key])
            except:
                log.debug("denied: trying to set invalid param {} {}".format(key,kwargs[key]))

    def default_params(self):
        '''Set the default params for the pipeline.
        Each can be changed by passing as kwargs to __init__'''
        self.tmax = 1000. #length of rays
        self.coherence_time = 32. #seconds
        self.num_threads_per_solve = None #all (dangerous if using num_parallel_solves more than 1)
        self.num_parallel_solves = 1 #serial solve
        self.stateful = False #if True then use result of previous timestep as initial point for next
        self.diagnostic_period = 1 #how often to save intermediate results and report
    def preprocess(self):
        """Prepare the model"""
        #split into time chunks, assumes they are contiguous
        antennas,antenna_labels = self.datapack.get_antennas(ant_idx=-1)
        patches, patch_names = self.datapack.get_directions(dir_idx=-1)
        times,timestamps = self.datapack.get_times(time_idx=-1)
        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches)  
        dobs = self.datapack.get_dtec(ant_idx = -1, dir_idx = -1, time_idx = -1)
        if len(times) == 1:
            self.time_window = 1
        else:
            dt = times[1].gps - times[0].gps
            if dt <= self.coherence_time:
                log.debug("Time sampling is larger than coherence time")
            self.time_window = int(np.ceil(self.coherence_time/dt))
        #average chunks
        Navg = int(np.ceil(float(Nt)/self.time_window))
        times_new = np.zeros(Navg,dtype=float)
        dobs_new = np.zeros([Na,Navg,Nd],dtype=float)
        m = np.zeros(Navg,dtype=float)
        for i in range(self.time_window):
            t_tmp = times[i::self.time_window]
            d_tmp = dobs[:,i::self.time_window,:]
            if len(t_tmp) == Navg:
                times_new += t_tmp.gps
                dobs_new += d_tmp
                m += 1.
            else:
                times_new[:-1] += t_tmp.gps
                dobs_new[:,:-1,:] += d_tmp
                m[:-1] += 1.
        times_new /= m
        minv = 1./m
        dobs_new = np.einsum('ijk,j->ijk',dobs_new,minv)
        times_new = at.Time(times_new,format='gps',scale='tai')
        data_dict = {'radio_array':self.datapack.radio_array, 'antennas':self.datapack.antennas, 'antenna_labels':self.datapack.antenna_labels,
                        'times':times_new, 'timestamps':times_new.isot, 'directions':self.datapack.directions,
                         'patch_names' : self.datapack.patch_names, 'dtec':dobs_new}
        datapack = DataPack(data_dict)
        datapack.set_reference_antenna(self.datapack.ref_ant)
        self.datapack = datapack

    def run():
        antennas,antenna_labels = self.datapack.get_antennas(ant_idx = -1)
        patches, patch_names = self.datapack.get_directions(dir_idx=-1)
        times,timestamps = self.datapack.get_times(time_idx=-1)
        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches)
        chunk_size = int(np.ceil(Nt/self.num_parallel_solves))
        dsk = {}
        dsk["datapack"] = self.datapack
        dsk["antennas"] = antennas
        dsk["patches"] = patches
        dsk["array_center"] = self.datapack.radio_array.get_center()
        dsk["phase"] = self.datapack.get_center_direction()
        dsk["covariance"] = Covariance()
        objective = []
        for i in range(self.num_parallel_solves):
            for j in range(chunk_size):
                dsk["ne_prior_{:02d}_{:04d}".format(i,j)] = (partial(create_initial_solution,ant_idx=-1,time_idx=[j],dir_idx=-1,zmax=self.tmax,spacing=5.,padding=20,thin_f=False),"datapack")
                if j == 0 or not self.stateful:
                    dsk["ne_0_{:02d}_{:04d}".format(i,j)] = ["ne_prior_{:02d}_{:04d}".format(i,j)]
                elif self.stateful and j > 0:
                    dsk["ne_0_{:02d}_{:04d}".format(i,j)] = (transfer_solutions,"sol_{:02d}_{:04d}".format(i,j-1),"ne_prior_{:02d}_{:04d}_".format(i,j))
                    dsk["time_{:02d}_{:04d}".format(i,j)] = times[chunk_size*i + j]
                dsk["rays_{:02d}_{:04d}".format(i,j)] = (calc_rays,"antennas","patches","time_{:02d}_{:04d}".format(i,j), "array_center", "time_{:02d}_{:04d}".format(i,j), "phase", "ne_0_{:02d}_{:04d}".format(i,j), self.datapack.radio_array.frequency, True, self.tmax, None)
#                dsk["sol_{:02d}_{:04d}".format(i,j)] = (irls_solve,"ne_0_{:02d}_{:04d}".format(i,j),"ne_prior_{:02d}_{:04d}".format(i,j),"rays_{:02d}_{:04d}".format(i,j))
                dsk["sol_{:02d}_{:04d}".format(i,j)] = (lambda x: print('irls_solve'),"ne_0_{:02d}_{:04d}".format(i,j),"ne_prior_{:02d}_{:04d}".format(i,j),"rays_{:02d}_{:04d}".format(i,j))
                objective.append("sol_{:02d}_{:04d}".format(i,j))
        client = Client()
        client.get(dsk,objective,num_workers = None)

    @property
    def num_threads_per_solve(self):
        return self._num_threads_per_solve
    @num_threads_per_solve.setter
    def num_threads_per_solve(self,num):
        if num is not None:
            assert num > 0
            self._num_threads_per_solve = int(num)
        else:
            self._num_threads_per_solve = None

    @property
    def coherence_time(self):
        return self._coherence_time
    @coherence_time.setter
    def coherence_time(self,num):
        assert num > 0
        self._coherence_time = float(num)
    
    @property
    def num_parallel_solves(self):
        return self._num_parallel_solves
    @num_parallel_solves.setter
    def num_parallel_solves(self,num):
        assert num > 0
        self._num_parallel_solves = int(num)
    
    @property
    def stateful(self):
        return self._stateful
    @stateful.setter
    def stateful(self,num):
        self._stateful = bool(num)
