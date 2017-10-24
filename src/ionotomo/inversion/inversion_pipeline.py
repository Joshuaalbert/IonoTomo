'''
Specifies the inversion pipeline.
'''
import logging as log
from dask.distributed import Client
import dask
from dask.multiprocessing import get
from ionotomo.geometry.calc_rays import calc_rays
from ionotomo.inversion.initial_model import *
from ionotomo.inversion.solution import transfer_solutions
from ionotomo.inversion.iterative_newton import iterative_newton_solve
from ionotomo.astro.real_data import DataPack
from ionotomo.ionosphere.covariance import Covariance
from ionotomo.plotting.plot_tools import plot_datapack
import os
import numpy as np
from functools import partial
import astropy.time as at
import astropy.units as au

class InversionPipeline(object):
    def __init__(self, datapack, output_folder = 'output', diagnostic_folder = 'diagnostics', **kwargs):
        self.output_folder = os.path.join(os.getcwd(),output_folder)
        self.diagnostic_folder = os.path.join(self.output_folder,diagnostic_folder)
        try:
            os.makedirs(self.diagnostic_folder)
        except:
            pass
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
        self.coherence_time = 32. #seconds where we consider the ionosphere to be the same
        self.num_threads_per_solve = None #all (dangerous if using num_parallel_solves more than 1)
        self.num_parallel_solves = 1 #serial solve
        self.stateful = False #if True then use result of previous timestep as initial point for next
        self.diagnostic_period = 1 #how often to save intermediate results and report
        self.spacing = 10.#km spacing in model

    def preprocess(self):
        """Prepare the model"""
        #split into time chunks, assumes they are contiguous
        antennas,antenna_labels = self.datapack.get_antennas(ant_idx=-1)
        patches, patch_names = self.datapack.get_directions(dir_idx=-1)
        times,timestamps = self.datapack.get_times(time_idx=-1)
        freqs = self.datapack.get_freqs(freq_idx=-1)
        clock = self.datapack.get_clock(ant_idx=-1,time_idx=-1)
        pointing = Pointing(location = self.datapack.radio_array.get_center().earth_location,
                            obstime = times[0], fixtime = times[0], phase = self.datapack.get_center_direction())
        #print(antennas.transform_to(pointing).cartesian.xyz.to(au.km).value)
        
        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches)  
        Nf = len(freqs)
        dobs = self.datapack.get_phase(ant_idx = -1, dir_idx = -1, time_idx = -1, freq_idx = -1)
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
        dobs_new = np.zeros([Na,Navg,Nd,Nf],dtype=float)
        prop_new = np.zeros([Na,Navg,Nd,Nf],dtype=float)
        clock_new = np.zeros([Na,Navg],dtype=float)
        m = np.zeros(Navg,dtype=float)
        for i in range(self.time_window):
            t_tmp = times[i::self.time_window]
            d_tmp = dobs[:,i::self.time_window,:,:]
            p_tmp = dobs[:,i::self.time_window,:,:]
            c_tmp = clock[:,i::self.time_window]
            if len(t_tmp) == Navg:
                times_new += t_tmp.gps
                dobs_new += d_tmp
                prop_new += d_tmp
                clock_new += c_tmp
                m += 1.
            else:
                times_new[:-1] += t_tmp.gps
                dobs_new[:,:-1,:,:] += d_tmp
                prop_new[:,:-1,:,:] += d_tmp
                clock_new[:,:-1] += c_tmp
                m[:-1] += 1
        times_new /= m
        minv = 1./m
        dobs_new = np.einsum('ijkl,j->ijkl',dobs_new,minv)
        prop_new = np.einsum('ijkl,j->ijkl',prop_new,minv)
        clock_new = np.einsum('ij,j->ij',clock_new,minv)
        times_new = at.Time(times_new,format='gps',scale='tai')
        data_dict = self.datapack.get_data_dict()
        data_dict.update({'times':times_new, 'timestamps':times_new.isot, 'prop':prop_new,'phase':dobs_new, 'clock':clock_new})
        datapack = DataPack(data_dict)
        datapack.set_reference_antenna(self.datapack.ref_ant)
        self.datapack = datapack
                
    def run(self):
        antennas,antenna_labels = self.datapack.get_antennas(ant_idx = -1)
        patches, patch_names = self.datapack.get_directions(dir_idx=-1)
        times,timestamps = self.datapack.get_times(time_idx=-1)
        freqs = self.datapack.get_freqs(freq_idx=-1)
        dobs = self.datapack.get_phase(ant_idx = -1,time_idx=-1,dir_idx=-1,freq_idx=-1)

        clock_prior = self.datapack.get_clock(ant_idx = -1, time_idx = -1)
        const_prior = self.datapack.get_const(ant_idx = -1)

        Cd = np.ones(dobs.shape)*(5*np.pi/180.)**2
        Ct = 0#(np.abs(dobs)*0.01)**2
        CdCt = Cd + Ct

        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches)
        Nf = len(freqs)
        chunk_size = int(np.ceil(float(Nt)/self.num_parallel_solves))
        dsk = {}
        dsk["datapack"] = self.datapack
        dsk["antennas"] = antennas
        dsk["patches"] = patches
        dsk["freqs"] = freqs
        dsk["array_center"] = self.datapack.radio_array.get_center()
        dsk["phase"] = self.datapack.get_center_direction()
        print("Setting covariance")
        dsk["covariance"] = (lambda *x: x,Covariance(dx=self.spacing,dy=self.spacing,dz=self.spacing), (5e-9)**2)
        objective = []
        print("running")
        indices = np.arange(Nt,dtype=int)
        for i in range(chunk_size):
            for thread_num, time_idx in enumerate(indices[i::chunk_size]):

                save_folder = os.path.join(self.diagnostic_folder,"thread_{}_time_{}".format(thread_num,time_idx))
                try:
                    os.makedirs(save_folder)
                except:
                    pass

                #observables of this step
                dsk["dobs_{}_{}".format(thread_num,time_idx)] = dobs[:,time_idx:time_idx+1,:]

                dsk["CdCt_{}_{}".format(thread_num,time_idx)] = CdCt[:,time_idx:time_idx+1,:]

                #time of this step
                dsk["time_{}_{}".format(thread_num,time_idx)] = times[time_idx:time_idx+1]
                dsk["fixtime_{}_{}".format(thread_num,time_idx)] = times[time_idx]

                #initial model for time step
                dsk["clock_prior_{}_{}".format(thread_num, time_idx)] = clock_prior[:,time_idx:time_idx+1]
                dsk["const_prior_{}_{}".format(thread_num, time_idx)] = const_prior[:]
                dsk["ne_prior_{}_{}".format(thread_num, time_idx)] = (partial(create_initial_solution,
                    ant_idx=-1,time_idx=[time_idx],dir_idx=-1,zmax=self.tmax,spacing=self.spacing,padding=20),"datapack")
                dsk["model_prior_{}_{}".format(thread_num, time_idx)] = (lambda ne, clock, const: (ne, clock, const), "ne_prior_{}_{}".format(thread_num, time_idx),
                        "clock_prior_{}_{}".format(thread_num, time_idx),
                        "const_prior_{}_{}".format(thread_num, time_idx))


                #if stateful then use solution from last time step (transfer solutions)
                if self.stateful and i > 0:
                    dsk["ne_0_{}_{}".format(thread_num,time_idx)] = (lambda sol, ne_prior: transfer_solutions(sol[0],ne_prior),
                            "sol_{}_{}".format(thread_num,time_idx-1),"ne_prior_{}_{}".format(thread_num,time_idx))
                    dsk["clock_0_{}_{}".format(thread_num,time_idx)] = (lambda sol: sol[1],
                            "sol_{}_{}".format(thread_num,time_idx-1))
                    dsk["const_0_{}_{}".format(thread_num,time_idx)] = (lambda sol: sol[2],
                            "sol_{}_{}".format(thread_num,time_idx-1))


                #else take the a priori as starting point
                elif i == 0 or not self.stateful:
                    dsk["ne_0_{}_{}".format(thread_num,time_idx)] = "ne_prior_{}_{}".format(thread_num,time_idx)
                    dsk["clock_0_{}_{}".format(thread_num,time_idx)] = "clock_prior_{}_{}".format(thread_num,time_idx)
                    dsk["const_0_{}_{}".format(thread_num,time_idx)] = "const_prior_{}_{}".format(thread_num,time_idx)

                    
                dsk["model_0_{}_{}".format(thread_num, time_idx)] = (lambda ne, clock, const: (ne, clock, const), 
                        "ne_0_{}_{}".format(thread_num, time_idx),
                        "clock_0_{}_{}".format(thread_num, time_idx),
                        "const_0_{}_{}".format(thread_num, time_idx))


                #calculate the rays
                dsk["rays_{}_{}".format(thread_num, time_idx)] = (calc_rays,
                        "antennas","patches","time_{}_{}".format(thread_num, time_idx), "array_center", "fixtime_{}_{}".format(thread_num,time_idx),
                        "phase", "ne_0_{}_{}".format(thread_num,time_idx), self.datapack.radio_array.frequency, True, self.tmax, None)

                #irls solve
#                dsk["sol_{}_{}".format(thread_num,time_idx)] = (lambda *x: x[0],"ne_0_{}_{}".format(thread_num,time_idx),"ne_prior_{}_{}".format(thread_num,time_idx),"rays_{}_{}".format(thread_num,time_idx))
#                
                dsk["plot_datapack_{}_{}".format(thread_num,time_idx)] = (partial(plot_datapack,ant_idx=-1,time_idx=[time_idx], dir_idx=-1,freq_idx=-1,figname=os.path.join(save_folder,"dobs"),vmin=None,vmax=None), "datapack")
                dsk["sol_{}_{}".format(thread_num,time_idx)] = (iterative_newton_solve,
                        "model_0_{}_{}".format(thread_num,time_idx),
                        "model_prior_{}_{}".format(thread_num,time_idx),
                        "rays_{}_{}".format(thread_num,time_idx),
                        "freqs",
                        "covariance",
                        "CdCt_{}_{}".format(thread_num,time_idx),
                        "dobs_{}_{}".format(thread_num,time_idx),
                        self.num_threads_per_solve,
                        save_folder)

                #Add result of list of computations                
                objective.append("plot_datapack_{}_{}".format(thread_num,time_idx))
                objective.append("sol_{}_{}".format(thread_num,time_idx))
                from dask.callbacks import Callback
        class PrintKeys(Callback):
            def _pretask(self, key, dask, state):
                """Print the key of every task as it's started"""
                print("Computing: {0}!".format(repr(key)))
        with PrintKeys():
            vals = dask.get(dsk,objective,num_workers = None)
        print(vals)

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
