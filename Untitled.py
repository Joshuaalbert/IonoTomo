
# coding: utf-8

# In[5]:

import numpy as np
import dask.array as da
from multiprocessing.pool import ThreadPool

pool = ThreadPool()
da.set_options(pool=pool)

x = da.ones((5,15),chunks=(5,5))
d = (x+1)
from dask.dot import dot_graph
dot_graph(d)


# In[ ]:



