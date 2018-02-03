import tensorflow as tf
import numpy as np
from ionotomo import *
from ionotomo.settings import *
import logging as log

from ionotomo.tomography.linear_operators import TECForwardEquation
from ionotomo.tomography.pipeline import Pipeline
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

from ionotomo.tomography.model import calc_rays_and_initial_model

def turbulent_perturbation(grid,sigma=1.,corr = 20., seed=None):    
    Sim = IonosphereSimulation(grid[0], grid[1],grid[2], sigma, corr, type='m52')
    B = Sim.realization(seed=seed)
    return B

class SimulateTec(object):
    def __init__(self,datapack,ant_idx=-1,time_idx=-1,dir_idx=-1,spacing=10.,res_n=201,sess=None):
        self.datapack = datapack
        self.antennas,_ = datapack.get_antennas(-1)
        self.times, _ = datapack.get_times(-1)
        self.directions, _ = datapack.get_directions(-1)
        self.rays, self.grid, self.model0 = calc_rays_and_initial_model(self.antennas,self.directions,
                self.times, zmax=1000.,res_n=res_n,spacing=spacing)
        #self.pipeline = Pipeline()
        #self.pipeline.add_graph('simulate_tec')
        self.graph = tf.Graph()
        self.pipeline = Pipeline()
        self.pipeline.add_graph('simulate_tec',self.graph)
        if sess is None:
            self.sess = tf.Session(graph=self.graph)
            log.info("Remember to call close on object")
        else:
            self.sess = sess
        self.model_scope = "simulate_tec"
        with self.graph.as_default():
            with tf.variable_scope(self.model_scope):
                grid_placeholders = [tf.placeholder(shape=(None,),dtype=TFSettings.tf_float,name='grid_{}'.format(i)) for i in range(3)]
#                grid_shapes = [tf.shape(grid_placeholders[i]) for i in range(3)]
#                model_shape = tf.concat(grid_shapes,axis=0)
                model_placeholder = \
                        tf.placeholder(TFSettings.tf_float,shape=(None,None,None),name='model')
                simulate_tec_op = self._build_simulate_tec(grid_placeholders,model_placeholder)
                self.pipeline.initialize_graph(self.sess)
                tf.add_to_collection(self.model_scope+"model", model_placeholder)
                tf.add_to_collection(self.model_scope+"grid", grid_placeholders)
                tf.add_to_collection(self.model_scope+"simulate_tec_op", simulate_tec_op)
    
    def set_grid(self, grid):
        self.grid = tuple(grid)

    def set_model0(self,model0):
        self.model0 = model0

    def close(self):
        self.sess.close()

    def _build_simulate_tec(self, grid_placeholders, model_placeholder):
        rays = self.pipeline.add_variable("rays",init_value=self.rays)
        #grid = [self.pipeline.add_variable("grid{}".format(i),init_value=self.grid[i]) for i in range(3)]
        integrator = TECForwardEquation(0,grid_placeholders,model_placeholder,rays)
        tec = integrator.matmul(tf.ones_like(model_placeholder))/1e13#km m^2
        return tec

    def simulate_tec(self):
        """Run the simulation for current model"""
        with self.graph.as_default():
            model_placeholder = tf.get_collection(self.model_scope+"model")[0]
            grid_placeholders = tf.get_collection(self.model_scope+"grid")[0]
            simulate_tec_op = tf.get_collection(self.model_scope+"simulate_tec_op")[0]
            tec = self.sess.run(simulate_tec_op, feed_dict={model_placeholder:self.model,
                                                            grid_placeholders[0]:self.grid[0],
                                                            grid_placeholders[1]:self.grid[1],
                                                            grid_placeholders[2]:self.grid[2]
                                                            })
        return tec

    def generate_model(self,factor=1.01,corr=10.,seed=None):
        log.info("Generating turbulent ionospheric model, correlation scale : {}".format(corr))
        if seed is not None:
            np.random.seed(seed)
            log.info("Seeding random seed to : {}".format(seed))
        assert self.model0.shape == (self.grid[0].shape[0],self.grid[1].shape[0],self.grid[2].shape[0]), "Your grid and model0 shapes are mismatched"
        dm = turbulent_perturbation(self.grid, sigma=np.log(factor), corr=corr,seed=seed)
        np.exp(dm,out=dm)
        self.model = self.model0*dm
        self.n = np.sqrt(1 - 8.98**2 * self.model/self.datapack.radio_array.frequency**2)
        log.info("Refractive index stats:\n\
                max(n) : {}\n\
                min(n) : {}\n\
                median(n) : {} \n\
                mean(n) : {}\n\
                std(n) : {}".format(np.max(self.n), np.min(self.n),np.median(self.n), np.mean(self.n), np.std(self.n)))
      
def test_simulate_tec():
    datapack = generate_example_datapack()
    antennas,_ = datapack.get_antennas(-1)
    times, _ = datapack.get_times(-1)
    directions, _ = datapack.get_directions(-1)
    Sim = SimulateTec(datapack,ant_idx=-1,time_idx=-1,dir_idx=-1,spacing=1.,res_n=201)
    Sim.generate_model()
    print(Sim.simulate_tec())
    model0 = Sim.model0[1:,:,:]
    grid = list(Sim.grid)
    grid[0] = grid[0][1:]
    Sim.set_model0(model0)
    Sim.set_grid(grid)
    
    print(Sim.simulate_tec())

if __name__ == '__main__':
    test_simulate_tec()
