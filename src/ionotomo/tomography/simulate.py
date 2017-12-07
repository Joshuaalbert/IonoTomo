import tensorflow as tf
import numpy as np
from ionotomo import *
from ionotomo.settings import *

from ionotomo.tomography.linear_operators import TECForwardEquation
from ionotomo.tomography.pipeline import Pipeline
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

from ionotomo.tomography.model import calc_rays_and_initial_model


def simulate_tec(antennas, directions,times,pipeline=None):
    rays, grid, model = calc_rays_and_initial_model(antennas,directions,times,
            zmax=1000.,res_n=201,spacing=10.)
    if pipeline is None:
        pipeline = Pipeline()
        pipeline.add_graph('simulate_tec')
    graph = pipeline.get_graph()
    with graph.as_default():
        with tf.name_scope("simulate_tec"):
            _rays = pipeline.add_variable('rays',init_value=rays)
            _grid = [pipeline.add_variable('grid{}'.format(i), init_value=grid[i]) for i in range(len(grid))]
            _model = pipeline.add_variable('model',init_value=model)
            integrator = TECForwardEquation(0,_grid,_model,_rays)
            tec = integrator.matmul(tf.ones_like(_model))/1e13#km m^2
            pipeline.add_result('tec',tec)
        sess = tf.Session()
        pipeline.initialize_graph(sess)
        res = pipeline.run_graph(None,sess)
        sess.close()
        return res['tec']
    
def test_simulate_tec():
    datapack = generate_example_datapack()
    antennas,_ = datapack.get_antennas(-1)
    times, _ = datapack.get_times(-1)
    directions, _ = datapack.get_directions(-1)
    print(simulate_tec(antennas,directions,times))

if __name__ == '__main__':
    test_simulate_tec()
