import numpy as np
import dask.array as da

if __name__ == '__main__':
    x = np.random.uniform(size=[10,11,12])
    for i in range(10):
        for j in range(11):
            for k in range(12):
                print(x.flatten()[k + 12*(j + 11*i)], x[i,j,k])
#    x1 = np.random.uniform(size=100*100*100)
#    x2 = np.random.uniform(size=[47,1,30])
#    y = np.subtract.outer(x1,x2)
#    print(y.shape)

