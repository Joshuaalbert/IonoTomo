import tensorflow as tf
import numpy as np
from ionotomo.settings import TFSettings
import itertools

def _bisection(array,value):
    '''Find the index of value in sorted array.
    array : Tensor (N,) 
        absissa to sort into
    value : Tensor (M,)
        The values to sort into array
    Note: inserts on left side.
    '''
    array = tf.cast(array,TFSettings.tf_float)
    value = tf.cast(value,TFSettings.tf_float)

    M = tf.shape(value)[0]
    N = tf.shape(array)[0]

    def _cond(jl,ju,value,array):
        """Loop for bin search
        ju_l : Tensor (M,)
            ju - jl
        """
        cond_vec = (ju - jl) > 1
        return tf.reduce_any(cond_vec)

    def _body(jl,ju,value,array):
        jm=tf.truncatediv(ju+jl, 2)# compute a midpoint,
        #jm = tf.Print(jm,[jl,jm,ju])
        value_ = tf.gather(array,jm)
        #array[jl] <= value < array[ju]
        #value_ = tf.Print(value_,[value_,value])
        jl = tf.where(value >= value_, jm, jl)
        ju = tf.where(value < value_, jm, ju)
        return (jl, ju, value, array)

    jl = tf.zeros((M,),M.dtype)
    ju = (N-1)*tf.ones((M,),N.dtype)

    jl, ju, _, _ = tf.while_loop(_cond, _body, (jl,ju,value,array), back_prop = False)
    
    
    jl = tf.where(value < array[0],-tf.ones_like(jl), jl)
    jl = tf.where(value >= array[-1], (N-1)*tf.ones_like(jl), jl)
    
    return jl

#def _ndim_coords_from_arrays(points):
#    """
#    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
#    """
#    if isinstance(points, tuple) and len(points) == 1:
#        # handle argument tuple
#        points = points[0]
#    if isinstance(points, tuple):
#        p = tf.meshgrid(points,indexing='ij')
##        points = tf.zeros_like(p[0])
##        points = tf.expand_dims(points,-1)
##        points = tf.tile(points,(1,)*len(p) + (len(p),))
#        points = tf.stack(p,-1)
#    else:
#        raise ValueError("coords should be tuple")    
#    return points

class RegularGridInterpolator(object):
    """
    Batched interpolation on a regular grid in arbitrary dimensions
    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.
    Parameters
    ----------
    points : tuple of Tensors of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.
    values : Tensor, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".
    
        Values outside the domain are extrapolated.

    Methods
    -------
    __call__
    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.
    """

    def __init__(self, points, values, method="linear"):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method

        #assume type already tf_float for both
        self.ndim = len(points)

        self.grid = tuple([tf.cast(p,TFSettings.tf_float) for p in points])
        self.values = tf.cast(values,TFSettings.tf_float)

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates
        Parameters
        ----------
        xi : batched Tensor of shape (B1,...,Bb, ndim)
            or tuple of ndim coords ( (B1,..., Bb)_1, ..., (B1,...,Bb)_ndim )
            The coordinates to sample the gridded data at
        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".
        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        if isinstance(xi,(list,tuple)):
            xi = tf.stack([tf.cast(x,TFSettings.tf_float) for x in xi],axis=-1)
        else:
            xi = tf.cast(xi,TFSettings.tf_float)
        

        #xi = _ndim_coords_from_arrays(xi)        
        
        xi_shape = tf.shape(xi)
        
        xi = tf.reshape(xi,(-1, self.ndim))
       

        indices, norm_distances, out_of_bounds = self._find_indices(tf.transpose(xi))
        #indices[0] = tf.Print(indices[0], [indices,norm_distances,out_of_bounds])

        if method == "linear":
            result = self._evaluate_linear(indices, norm_distances, out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices, norm_distances, out_of_bounds)

        return tf.reshape(result,xi_shape[:-1])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = tf.zeros(tf.shape(indices[0]),TFSettings.tf_float)
        for edge_indices in edges:
            with tf.control_dependencies([values]):
                weight = tf.ones(tf.shape(indices[0]),TFSettings.tf_float)
                for k in range(self.ndim):
                    ei = edge_indices[k]
                    i = indices[k]
                    yi = norm_distances[k]
                    with tf.control_dependencies([weight]):
                        weight *= tf.where(ei == i, 1 - yi, yi)
                values += tf.gather_nd(self.values,tf.transpose(edge_indices)) * weight
        return values

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(tf.where(yi <= .5, i, i + 1))
        return tf.gather_nd(self.values,tf.stack(idx_res,axis=-1))

    def _find_indices(self, xi):
        """Find the index of abcissa for each coord in xi
        xi : Tensor shape (ndim, M)
        """
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        M = tf.shape(xi)[1]
        out_of_bounds = tf.zeros((M,), dtype=TFSettings.tf_int)
        xi = tf.unstack(xi,axis=0)
        control = None
        # iterate through dimensions
        for d in range(self.ndim):
            with tf.control_dependencies(control):
                x = xi[d]
                grid = self.grid[d]
                i = _bisection(grid, x)
                #i = tf.Print(i,[i])
                i = tf.where(i < 0, tf.zeros_like(i),i)
                ub = tf.shape(grid)[0] - 2
                i = tf.where(i > ub, tf.ones_like(i) * ub, i)
                indices.append(i) 
                norm_distances.append((x - tf.cast(tf.gather(grid,i),TFSettings.tf_float)) / tf.cast(tf.gather(grid,i+1) - tf.gather(grid,i), TFSettings.tf_float))

                out_of_bounds += tf.cast(x < grid[0],TFSettings.tf_int)
                out_of_bounds += tf.cast(x > grid[-1],TFSettings.tf_int)
                control = [indices[-1],norm_distances[-1],out_of_bounds]
        return indices, norm_distances, tf.cast(out_of_bounds,TFSettings.tf_bool)


def test_regular_grid_interpolator():
    points = [np.linspace(0,1,100) for i in range(3)]
    M = np.random.normal(size=[100]*3)
    y = tuple([ np.random.uniform(size=100) for i in range(3)])
    r = RegularGridInterpolator(tuple([tf.constant(p) for p in points]),tf.constant(M),method='linear')
    from scipy.interpolate import RegularGridInterpolator as rgi
    r_ = rgi(points,M, method='linear',fill_value=None,bounds_error=False)
    sess = tf.Session()
    u = sess.run(r(y))
    sess.close()
    u_ = r_(y)
#    import pylab as plt
#    print(u-u_)
#    plt.hist(u-u_,bins=100)
#    plt.show()
    #print(u-u_)
    assert np.all(np.isclose(u,u_,atol=1e-4))

def test_bisection():  
    array = np.linspace(0,1,100)
    values = np.linspace(-1,2,100)
    #print(values[34])
    i_np = np.searchsorted(array,values)-1
    #print(array[i_np]<values)
    sess = tf.Session()
    i_tf = sess.run(_bisection(array,values))
#    print(i_tf,i_np)
#    print(values[np.where(i_np!=i_tf)])
#    print(array[i_tf[np.where(i_np!=i_tf)]])
    #print(array[i_np],array[i_tf],values)
#    assert np.all(i_np == i_tf)
    sess.close()


if __name__ == '__main__':
    test_bisection()
    test_regular_grid_interpolator()
    
    
