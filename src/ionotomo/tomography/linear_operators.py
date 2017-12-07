import tensorflow as tf
import numpy as np
from ionotomo.settings import TFSettings
from ionotomo.tomography.interpolation import RegularGridInterpolator
from ionotomo.tomography.integrate import simps

class RayOp(object):
    r"""Linear operator that performs for any v(x)
    
        h[i1,...,ir] = \int_R[i1,...,ir] ds M(x) v(x)

    grid : tuple of ndim Tensors specifying grid coordinates used for interpolation
    M : the function over V to integrate, defined on the *grid*
    rays : Tensor with *r* ray index dimensions and last dim is size ndim
        Defines the ray trajectories over which to integrate.
        Shape (i1,...,ir, ndim, N)
    transpose : bool
        If True then Av represents \sum_R \Delta_R(x) v_R M(x)
    """
    def __init__(self,grid,M,rays,dx = None,
            weight = None, transpose = False,
            dtype=TFSettings.tf_float):
        self.dtype = dtype
        self.grid = grid
        self.rays = tf.cast(rays,TFSettings.tf_float)
        if dx is None:
            self.dx = tf.sqrt(tf.reduce_sum(tf.square(self.rays[...,1:] - self.rays[...,:-1]),axis=-2))
            self.dx = tf.cumsum(tf.concat([tf.zeros_like(self.dx[...,0:1]),self.dx],axis=-1),axis=-1)
        else:
            nd = tf.size(tf.shape(rays))
            dxshape = tf.concat([tf.ones_like(tf.shape(rays)[0:-2]),
                tf.shape(rays)[nd-1:nd]],axis=0)
            self.dx = tf.reshape(dx,dxshape)
        if weight is not None:
            self.weight = tf.reshape(tf.cast(weight,self.dtype),self.range_shape())
        else:
            self.weight = None
        self.M = tf.cast(M,self.dtype)
        self.transpose = transpose

    def domain_shape(self):
        return tf.shape(self.M)

    def range_shape(self):
        return tf.shape(self.rays)[:-2]

    def shape(self):
        return tf.concat([self.range_shape(),self.domain_shape()],axis=0)

    def matmul(self,x,adjoint=False,adjoint_arg=False):
        '''Transform [batch] matrix x with left multiplication: x --> Ax.

        x: Tensor with compatible shape and same dtype as self. 
        See class docstring for definition of compatibility.
        adjoint: Python bool. If True, left multiply by the adjoint: A^H x.
        adjoint_arg: Python bool. 
        If True, compute A x^H where x^H is the hermitian transpose 
        (transposition and complex conjugation).
        name: A name for this `Op.
        
        Returns:
        A Tensor with shape [..., M, R] and same dtype as self.
        '''

        x = tf.cast(x,self.dtype)
        Ax = self.M * x
        Ax = RegularGridInterpolator(self.grid,Ax,method='linear')
        if self.weight is None:
            Ax = Ax(tf.unstack(self.rays,axis=-2))
        else:
            Ax = self.weight*Ax(tf.unstack(self.rays,axis=-2))
        Ax = simps(Ax, self.dx,axis = -1)
        return Ax

class TECForwardEquation(RayOp):
    def __init__(self,i0, grid,M,rays,dx = None,
            weight = None, transpose = False,
            dtype=TFSettings.tf_float):
        super(TECForwardEquation,self).__init__(grid,M,rays,dx,
            weight, transpose, dtype)
        self.i0 = tf.cast(i0,TFSettings.tf_int)
    def matmul(self,x,adjoint=False,adjoint_arg=False):
        '''Transform [batch] matrix x with left multiplication: x --> Ax.

        x: Tensor with compatible shape and same dtype as self. 
        See class docstring for definition of compatibility.
        adjoint: Python bool. If True, left multiply by the adjoint: A^H x.
        adjoint_arg: Python bool. 
        If True, compute A x^H where x^H is the hermitian transpose 
        (transposition and complex conjugation).
        name: A name for this `Op.
        
        Returns:
        A Tensor with shape [..., M, R] and same dtype as self.
        '''
        Ax = super(TECForwardEquation,self).matmul(x)
        Ax = Ax - Ax[self.i0:self.i0+1, ...]
        return Ax

if __name__ == '__main__':
    rays = np.sort(np.random.uniform(size=[2,2,3,6]),axis=-1)
    M = np.random.normal(size=(100,100,100))
    grid = (np.linspace(0,1,100),)*3
    op = TECForwardEquation(0,grid, M, rays)
    x = np.random.normal(size=(100,100,100))
    sess = tf.Session()
    print(sess.run(op.matmul(x)))
    sess.close()
