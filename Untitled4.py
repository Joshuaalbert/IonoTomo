
# coding: utf-8

# In[17]:

import numpy as np
from scipy.linalg import lu, cholesky
a = np.random.uniform(size=[500,500])
from time import clock
t1 = clock()
np.linalg.inv(a)
print(clock() - t1)
t1 = clock()
L=lu(a)

import pylab as plt
plt.imshow(P)
plt.show()
print(np.isclose(P.dot(L).dot(U),a))
print(clock() - t1)


# In[ ]:



