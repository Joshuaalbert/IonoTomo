'''
Specifies the inversion pipeline.
This controls the flow and parallelization of the inversion
across multiple machines.
'''
import logging as log

from distributed import Client
from ionotomo.inversion.solver import Solver
import os
import numpy as np

class InversionPipeline(object):
    def __init__(self, datapack, solver_cls, output_folder = 'output', diagnostic_folder = 'diagnostics', num_threads = None,**solver_kwargs):
        self.num_threads = num_threads
        self.client = Client()
        self.output_folder = output_folder
        self.diagnostic_folder = diagnostic_folder
        log.basicConfig(filename=os.path.join(self.output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
        log.info("Initializing inversion for {}".format(self.datapack))
        assert isinstance(solver_cls,Solver)
        self.solver = solver_cls(datapack, self.output_folder, self.diagnostic_folder,**solver_kwargs)

    @property
    def num_threads(self):
        return self._num_threads
    @num_threads.setter
    def num_threads(self,num):
        if num is not None:
            assert num > 0
        self._num_threads = num

    def run():
        self.solver.setup()
        self.solver.go()
