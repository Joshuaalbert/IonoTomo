import tensorflow as tf
from ionotomo.settings import TFSettings

class Pipeline(object):
    def __init__(self):
        self.placeholders = {}
        self.results = {}
        self.current_graph = None
        self.graphs = {}
        self.add_graph('default_graph')

    def add_graph(self,name):
        self.graphs[name] = tf.Graph()
        self.current_graph = name
    def choose_graph(self,graph_name):
        assert graph_name in self.graphs.keys()
        self.current_graph = graph_name

    def get_graph(self,graph_name = None):
        if graph_name is not None:
            assert graph_name in self.graphs.keys()
            graph = self.graphs[graph_name]
        else:
            graph = self.graphs[self.current_graph]
        return graph

    def add_variable(self,name, shape=(), init_value=None, dtype = TFSettings.tf_float,trainable=False,persistent=True,graph_name=None):
        """Add to graph given by graph_name or self.current_graph 
        a variable with initializer if persistent is True, or just a placeholder if False"""
            
        with self.get_graph(graph_name).as_default():
            if persistent:
                assert init_value is not None
                init = tf.placeholder(shape=init_value.shape,dtype=dtype,name="{}_init".format(name))
                var = tf.get_variable(name,initializer=init,trainable=trainable)
                self.placeholders[name] = (init,var,init_value)
                return var
            else:
                var = tf.placeholder(shape=shape,dtype=dtype,name="{}".format(name))
                self.placeholders[name] = (var,None,None)
                return var
    def initialize_graph(self,sess,graph_name=None):
        inits = self.get_initializers(graph_name)
        with self.get_graph(graph_name).as_default():
            sess.run(tf.global_variables_initializer(), feed_dict = inits)

    def run_graph(self,inputs,sess,graph_name=None):
        results = self.get_result(graph_name)
        #print(results)
        inputs_graph = self.get_inputs(graph_name)
        if isinstance(inputs,dict):
            feed_dict = {inputs_graph[name] : inputs[name] for name in inputs}
        with self.get_graph(graph_name).as_default():
            res = sess.run(results, feed_dict = inputs)
        return res

    def get_inputs(self,graph_name=None):
        g = self.get_graph(graph_name)
        with g.as_default():
            inputs = {}
            for name in self.placeholders:
                if self.placeholders[name][1] is None:
                     if g.is_fetchable(self.placeholders[name][0]):
                        inputs[self.placeholders[name][0]] = self.placeholders[name][2]
            return inputs

    def get_initializers(self,graph_name=None):
        g = self.get_graph(graph_name)
        with g.as_default():
            inits = {}
            for name in self.placeholders:
                if self.placeholders[name][1] is not None:
                    if g.is_fetchable(self.placeholders[name][0]):
                        inits[self.placeholders[name][0]] = self.placeholders[name][2]
            return inits

    def add_result(self,name,result):
        """Add a tensor or op as result with given name"""
        self.results[name] = result
    def get_result(self,graph_name=None):
        g = self.get_graph(graph_name)
        with g.as_default():
            res = {}
            for key in self.results:
                if g.is_fetchable(self.results[key]):
                    res[key] = self.results[key]
            return res

from ionotomo import *
from ionotomo.settings import *

def test_pipeline():
    import tensorflow as tf
    import numpy as np

    from ionotomo.tomography.linear_operators import TECForwardEquation
    import astropy.coordinates as ac
    import astropy.units as au
    import astropy.time as at

    from ionotomo.tomography.model import calc_rays_and_initial_model

    datapack = generate_example_datapack()
    antennas,_ = datapack.get_antennas(-1)
    times, _ = datapack.get_times(-1)
    directions, _ = datapack.get_directions(-1)

    rays, grid, model = calc_rays_and_initial_model(antennas,directions,times,
            zmax=1000.,res_n=201,spacing=10.)
    pipeline = Pipeline()
    graph = pipeline.get_graph()
    with graph.as_default():
        with tf.name_scope("simulate_tec"):
            _rays = pipeline.add_variable('rays',init_value=rays)
            _grid = [pipeline.add_variable('grid{}'.format(i), init_value=grid[i]) for i in range(len(grid))]
            _model = pipeline.add_variable('model',init_value=model)
            integrator = TECForwardEquation(0,_grid,_model,_rays)
            tec = integrator.matmul(tf.ones_like(_model))/1e13
            pipeline.add_result('tec',tec)
        sess = tf.Session()
        pipeline.initialize_graph(sess)
        res = pipeline.run_graph(None,sess)
        sess.close()
        print(res)

if __name__ == '__main__':
    test_pipeline()
        
