
# coding: utf-8

# In[29]:

import dask.array as da
import numpy as np
from dask.dot import dot_graph
from dask import delayed
from dask.threaded import get
from functools import partial
from time import sleep, clock
import dask

def requires(*args,time=0):
    '''time is in days to do task'''
    #print("Spend: approx. {} days".format(time))
    sleep(time/100.)
    #print(args)
    return time

def input(time=0):
    '''time is in days to do task'''
    #print("Spend: approx. {} days".format(time))
    sleep(time/100.)
    #print(args)
    return time

def createDask(depDict,timeDict={}):
    '''Input a dictionary of dependancies:
    e.g. if task_A requires task_B and task_C then add 
    depDict["task_A"] = ["task_B","task_C"]
    
    e.g. if task_A requires nothing then add
    depDict["task_A"] = []
    
    Try not to include loops like "task_A":["task_A","task_B"].
    Doing so will allow the graph to be constructed but an optimal order can not be determined.
       
    If a certain task will take a certain number of days then put this in the timeDict.
    e.g. if task_A will take 4 days add
    timeDict["task_A"] = 4
    
    e.g. if task_A is unknown add nothing (0 time)
    
    e.g. if task_A is a question then typically add nothing (0 time) because typically time is taken for 
    requirements for the question.
    
    This returns a dask object for further processing.    
    '''
    dsk = {}
    for key in depDict.keys():
        if key in timeDict.keys():
            time = timeDict[key]
        else:
            time = 0
        if len(depDict[key]) == 0:
            dsk[key] = (partial(input, time=time),)
        else:
            dsk[key] = (partial(requires, time=time),depDict[key])
            
    return dsk

def getOrderOfExecution(dsk,finalTask, numConcurrentTasks=1):
    '''Get the optimal order to minimizing backtracking and memory footprint.
    numConcurrentTasks can be more than 1 if you think you can do two things at once.
    
    Put the final task you want to achieve (one of the keys in depDict)'''
    from dask.callbacks import Callback
    #from multiprocessing.pool import ThreadPool
    from dask.threaded import get
    #dask.set_options(pool=ThreadPool(numConcurrentTasks))
    class PrintKeys(Callback):
        def _start(self,dsk):
            self.startTime = clock()
        def _pretask(self, key, dask, state):
            """Print the key of every task as it's started"""
            pass
        def _posttask(self,key,result,dsk,state,id):
            print("Do {}, approx. {} days".format(repr(key),repr(result)))
        def _finish(self,dsk,state,errored):
            self.endTime = clock()
            dt = (self.endTime - self.startTime)*100.
            print("Approximate time to complete: {} days".format(dt))
    with PrintKeys():
        get(dsk,finalTask,num_workers=numConcurrentTasks)
    
def printGraph(dsk,outfile):
    dot_graph(dsk,filename=outfile,format='pdf')
    
depDict = {'URSI Present':['Make Presentation',"Arrange the trip"],
       "Arrange the trip": ["funding","flights","accomadations"],
       "funding":[],
       "flights":[],
       "accomadations":[],
      'Make Presentation':['Show Works on Real','Survey ionosphere literature'],
      'Show Works on Real':['Does inversions show time coherence?','How well does it improve image?'],
      'How well does it improve image?':['Get phase screens','Apply a-term projections'],
      'Apply a-term projections':['Find an imager that can do a-term'],
      'Find an imager that can do a-term': ['Talk to people'],
      'Talk to people':[],
      'Does inversions show time coherence?':[ 'Get well Behaved Data','Trusted on Sim Data'],
      'Get well Behaved Data':['Calm Ionosphere','Trusted dTEC'],
      'Calm Ionosphere':["Jit's script"],
      'Trusted dTEC':['Is Robust to dTEC measurement uncertainty?','Validity checks'],
      'Validity checks':['Refractive scale consistent','Compare with literature','Compare with a priori'],
      'Trusted on Sim Data':['Stress Test All Internals','Make Robust','Quantify Reconstruction Resolution',"Get working on Sim Data"],
      'Quantify Reconstruction Resolution':['Information Completeness in FOV','Measured ne RMS Error'],
      'Measured ne RMS Error':['Optimal Antenna/facet selection','Choose required resolution'],
      'Optimal Antenna/facet selection':[ 'Bayesian optimization'],
      'Information Completeness in FOV':['Optimal Antenna/facet selection'],
      'Stress Test All Internals':['Stress Test Interpolation'],
      'Stress Test Interpolation':['Test RMS Error',"Write interpolation"],
      'Handle with proper padding':[],#'Stress Test Interpolation',
      'What happens near boundary':['Handle with proper padding'],
      'What effects of cell size':['Handle with proper padding'],
      'Choose required resolution':['What effects of cell size','Choose coherence Scale hyper parameter'],
      'Test RMS Error':['What happens near boundary','What effects of cell size'],
      'Make Robust':['Is Robust to dTEC measurement uncertainty?'],
      'Refractive scale consistent':["Jit's script"],
      "Jit's script":[],
      'Compare with literature': ['Survey ionosphere literature'],
      'Survey ionosphere literature':[],
      'Derive a priori ionosphere':['Survey ionosphere literature','Choose coherence Scale hyper parameter'],
      'Choose coherence Scale hyper parameter': ['Survey ionosphere literature','Bayesian optimization'],
      'Bayesian optimization':['Loop requires Trusted on Sim Data'],
      'Loop requires Trusted on Sim Data':[],
      'Compare with a priori':['Derive a priori ionosphere'],
      'Get phase screens':['Trusted on Sim Data'],
      'Is Robust to dTEC measurement uncertainty?':["quasi-Newton method","try different starting points"],
      "quasi-Newton method":["speed","limited memory","Gradient"],
      "try different starting points":["sampling turbulent realizations"],
      "sampling turbulent realizations":['Survey ionosphere literature','speed'],
      "limited memory":["BFGS"],
      "speed":["Parallelization","Optimization","Approximations"],
       "Approximations":[],
      "Parallelization":["Use dask"],
      "Use dask":[],
      "Optimization": ['Bayesian optimization'],
      "BFGS": ["Gradient","Preconditioning"],
       "Preconditioning":['Information Completeness in FOV'],
      "Get working on Sim Data":["quasi-Newton method",'Derive a priori ionosphere',"Forward equation"],
      "Forward equation":["Calc Rays",'framework to handle data'],
      "Calc Rays":["Fermats Principle"],
      "Fermats Principle":["Write interpolation"],
      "Write interpolation":[],
      "framework to handle data":["UVW frame","DataPack"],
      "DataPack":[],
      "UVW frame":[],
      "Gradient":["Forward equation"]}

timeDict = {'URSI Present':0.5,
       "Arrange the trip": 4,
       "funding":7,
       "flights":2,
       "accomadations":3,
      'Make Presentation':7,
      'Show Works on Real':4,
      'How well does it improve image?':5,
      'Apply a-term projections':3,
      'Find an imager that can do a-term': 15,
      'Talk to people':0,
      'Does inversions show time coherence?':3,
      'Get well Behaved Data':0.5,
      'Calm Ionosphere':1,
      'Trusted dTEC':4,
      'Validity checks':2,
      'Trusted on Sim Data':2,
      'Quantify Reconstruction Resolution':5,
      'Measured ne RMS Error':2,
      'Optimal Antenna/facet selection':2,
      'Information Completeness in FOV':2,
      'Stress Test All Internals':2,
      'Stress Test Interpolation':5,
      'Handle with proper padding':1,#'Stress Test Interpolation',
      'What happens near boundary':3,
      'What effects of cell size':3,
      'Choose required resolution':1,
      'Test RMS Error':1,
      'Make Robust':7,
      'Refractive scale consistent':4,
      "Jit's script":5,
      'Compare with literature': 6,
      'Survey ionosphere literature':15,
      'Derive a priori ionosphere':7,
      'Choose coherence Scale hyper parameter': 5,
      'Bayesian optimization':0,
      'Loop requires Trusted on Sim Data':0,
      'Compare with a priori':2,
      'Get phase screens':3,
      'Is Robust to dTEC measurement uncertainty?':14,
      "quasi-Newton method":14,
      "try different starting points":2,
      "sampling turbulent realizations":2,
      "limited memory":0,
      "speed":0,
       "Approximations":7,
      "Parallelization":0,
      "Use dask":14,
      "Optimization": 7,
      "BFGS": 5,
       "Preconditioning":2,
      "Get working on Sim Data":5,
      "Forward equation":2,
      "Calc Rays":2,
      "Fermats Principle":2,
      "Write interpolation":5,
      "framework to handle data":7,
      "DataPack":4,
      "UVW frame":7,
      "Gradient":5}

dsk = createDask(depDict,timeDict=timeDict)
getOrderOfExecution(dsk,"URSI Present",numConcurrentTasks=3)
printGraph(dsk,"URSI Roadmap")



# In[97]:

import dask.array as da
import numpy as np
from dask.dot import dot_graph
from dask import delayed
from dask.threaded import get
from functools import partial
from time import sleep
import dask

def merge(*args,time=0):
    '''time is in days to do task'''
    sleep(time/100.)
    #print(args)
    return args

n = 4
dsk = {'input':'input_',
      'state':'state_',
       'control stack':(merge,'state','input'),
       'min control':'control stack',
       'max control':'control stack',
       'mean control':'control stack',
       'diff control':'control stack',
       'pattern stack':(merge,'input','control stack',['diff-{}'.format(i) for i in range(n)],['mean-{}'.format(i) for i in range(n)]),
      'entry':'pattern stack',
      'exit':'pattern stack',
      'stop loss':'pattern stack',
      'timeout':'pattern stack'}
for i in range(n):
    dsk['diff-{}'.format(i)] = (merge,'min-{}'.format(i),'max-{}'.format(i),'diff control')
    dsk['min-{}'.format(i)] = (merge,'input','min control')
    dsk['max-{}'.format(i)] = (merge,'input','max control')
    dsk['mean-{}'.format(i)] = (merge,'input','mean control')

from dask.callbacks import Callback
class PrintKeys(Callback):
    def _pretask(self, key, dask, state):
        """Print the key of every task as it's started"""
        print("{0}".format(repr(key)))
with PrintKeys():
    dask.get(dsk,['entry','exit','stop loss','timeout'])
dot_graph(dsk,filename="ValueRangeNetwork",format='pdf')
dot_graph(dsk)


#dot_graph(dsk,filename="Roadmap to URSI",format="svg")


# In[98]:

import dask.array as da
import numpy as np
from dask.dot import dot_graph
from dask import delayed
from dask.threaded import get
from functools import partial
from time import sleep
import dask

def merge(*args,time=0):
    '''time is in days to do task'''
    sleep(time/100.)
    #print(args)
    return args

n = 4
dsk = {'input':'input_',
      'state':'state_',

       'pattern stack':(merge,'input',['diff-{}'.format(i) for i in range(n)],['mean-{}'.format(i) for i in range(n)]),
      'entry':'pattern stack',
      'exit':'pattern stack',
      'stop loss':'pattern stack',
      'timeout':'pattern stack'}
for i in range(n):
    dsk['diff-{}'.format(i)] = (merge,'min-{}'.format(i),'max-{}'.format(i))
    dsk['min-{}'.format(i)] = 'input'
    dsk['max-{}'.format(i)] = 'input'
    dsk['mean-{}'.format(i)] = 'input'

from dask.callbacks import Callback
class PrintKeys(Callback):
    def _pretask(self, key, dask, state):
        """Print the key of every task as it's started"""
        print("{0}".format(repr(key)))
with PrintKeys():
    dask.get(dsk,['entry'])
dot_graph(dsk,filename="ValueRangeNetwork-task0",format='pdf')
dot_graph(dsk)


#dot_graph(dsk,filename="Roadmap to URSI",format="svg")


# In[ ]:

dsk = {'URSI Present':['Make Presentation',"Arrange the trip"],
       "Arrange the trip": ["funding","flights","accomadations"],
       "funding":[],
       "flights":[],
       "accomadations":[],
      'Make Presentation':['Show Works on Real','Survey ionosphere literature'],
      'Show Works on Real':['Does inversions show time coherence?','How well does it improve image?'],
      'How well does it improve image?':['Get phase screens','Apply a-term projections'],
      'Apply a-term projections':['Find an imager that can do a-term'],
      'Find an imager that can do a-term': ['Talk to people'],
      'Talk to people':[],
      'Does inversions show time coherence?':[ 'Get well Behaved Data','Trusted on Sim Data'],
      'Get well Behaved Data':['Calm Ionosphere','Trusted dTEC'],
      'Calm Ionosphere':["Jit's script"],
      'Trusted dTEC':['Is Robust to dTEC measurement uncertainty?','Validity checks'],
      'Validity checks':['Refractive scale consistent','Compare with literature','Compare with a priori'],
      'Trusted on Sim Data':['Stress Test All Internals','Make Robust','Quantify Reconstruction Resolution',"Get working on Sim Data"],
      'Quantify Reconstruction Resolution':['Information Completeness in FOV','Measured ne RMS Error'],
      'Measured ne RMS Error':['Optimal Antenna/facet selection','Choose required resolution'],
      'Optimal Antenna/facet selection':[ 'Bayesian optimization'],
      'Information Completeness in FOV':['Optimal Antenna/facet selection'],
      'Stress Test All Internals':['Stress Test Interpolation'],
      'Stress Test Interpolation':['Test RMS Error',"Write interpolation"],
      'Handle with proper padding':[],#'Stress Test Interpolation',
      'What happens near boundary':['Handle with proper padding'],
      'What effects of cell size':['Handle with proper padding'],
      'Choose required resolution':['What effects of cell size','Choose coherence Scale hyper parameter'],
      'Test RMS Error':['What happens near boundary','What effects of cell size'],
      'Make Robust':['Is Robust to dTEC measurement uncertainty?'],
      'Refractive scale consistent':["Jit's script"],
      "Jit's script":[],
      'Compare with literature': ['Survey ionosphere literature'],
      'Survey ionosphere literature':[],
      'Derive a priori ionosphere':['Survey ionosphere literature','Choose coherence Scale hyper parameter'],
      'Choose coherence Scale hyper parameter': ['Survey ionosphere literature','Bayesian optimization'],
      'Bayesian optimization':['Loop requires Trusted on Sim Data'],
      'Loop requires Trusted on Sim Data':[],
      'Compare with a priori':['Derive a priori ionosphere'],
      'Get phase screens':['Trusted on Sim Data'],
      'Is Robust to dTEC measurement uncertainty?':["quasi-Newton method","try different starting points"],
      "quasi-Newton method":["speed","limited memory","Gradient"],
      "try different starting points":["sampling turbulent realizations"],
      "sampling turbulent realizations":['Survey ionosphere literature','speed'],
      "limited memory":["BFGS"],
      "speed":["Parallelization","Optimization","Approximations"],
       "Approximations":[],
      "Parallelization":["Use dask"],
      "Use dask":[],
      "Optimization": ['Bayesian optimization'],
      "BFGS": ["Gradient","Preconditioning"],
       "Preconditioning":['Information Completeness in FOV'],
      "Get working on Sim Data":["quasi-Newton method",'Derive a priori ionosphere',"Forward equation"],
      "Forward equation":["Calc Rays",'framework to handle data'],
      "Calc Rays":["Fermats Principle"],
      "Fermats Principle":["Write interpolation"],
      "Write interpolation":[],
      "framework to handle data":["UVW frame","DataPack"],
      "DataPack":[],
      "UVW frame":[],
      "Gradient":["Forward equation"]}

