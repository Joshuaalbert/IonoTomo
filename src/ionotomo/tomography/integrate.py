import tensorflow as tf
import numpy as np
from ionotomo.settings import TFSettings


def swap_axes(a,axis,start=0):
    """Swap axis *axis* of tensor *a* with location *start*"""
    nd = tf.size(tf.shape(a))
    axis = tf.cond(tf.less(axis,0),lambda: nd + axis, lambda: axis)
    #perm is (axis, 1 ,...,axis-1, 0, axis + 1, ..., nd - 1)
    def _do():
        def _min_range(a,b):
            return tf.range(a,tf.reduce_max([a,b],axis=0))
        perm = tf.concat([tf.range(axis,axis+1),
            _min_range(1,axis), 
            tf.range(0,1),
            _min_range(axis+1, nd)],axis=0)
        #perm = tf.Print(perm,[perm])
        #perm = tf.reshape(perm,(nd,))
        return tf.transpose(a,perm=perm)
    return tf.cond(tf.equal(axis,start), lambda: a, _do)

def diff(x,n=1,axis=-1):
    '''Calculate the n-th discrete difference along given axis.
    
    The first difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.'''
    #nd = tf.size(tf.shape(x))
    x = swap_axes(x, axis)
    res = x[1:, ...] - x[:-1, ...]
    res = swap_axes(res,axis)
    return res

def _slice(x,axis,start,stop,step=1):
    x = swap_axes(x,axis)
    res = x[start:stop:step,...]
    res = swap_axes(res,axis)
    return res

def _get(x,axis,idx):
    nd = tf.size(tf.shape(x))
    idx = tf.cond(tf.less(idx,0),lambda: nd + idx, lambda: idx)
    x = swap_axes(x,axis)
    res = x[idx:idx+1,...]
    res = tf.squeeze(swap_axes(res,axis),axis=axis)
    return res


def _basic_simps(y,start,stop,x,dx,axis):
    nd = tf.size(tf.shape(y))
    if start is None:
        start = 0
    step = 2

    if x is None:  # Even spaced Simpson's rule.
        result = tf.reduce_sum(dx/3.0 * (_slice(y,axis,start,stop,step) \
                +4*_slice(y,axis,start+1,stop+1,step) \
                +_slice(y,axis,start+2,stop+2,step)),
                                    axis=axis)
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = diff(x,axis=axis)
        h0 = _slice(h,axis,start,stop,step)
        h1 = _slice(h,axis,start+1,stop+1,step)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1
        result = tf.reduce_sum(hsum/6.0*(_slice(y,axis,start,stop,step)*(2-1.0/h0divh1) \
                + _slice(y,axis,start+1,stop+1,step)*hsum*hsum/hprod \
                + _slice(y,axis,start+2,stop+2,step)*(2-h0divh1)),
                axis=axis)
    return result

def simps(y,x=None,dx=1.,axis=-1,even='avg'):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule.  If x is None, spacing of dx is assumed.
    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals.  The parameter 'even' controls how this is handled.
    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : int, optional
        Spacing of integration points along axis of `y`. Only used when
        `x` is None. Default is 1.
        if y is (n1,...,nN, m) and x is (n1,...,nN,m) then integrate
        y[..., m] x[..., m].
        if x is (m,) then integrate all y along given axis.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : {'avg', 'first', 'str'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.
    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less.  If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.
    """
    y = tf.cast(y,TFSettings.tf_float)
    nd = tf.size(tf.shape(y))
    N = tf.shape(y)[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if not x is None:
        x = tf.cast(x,TFSettings.tf_float)
        def _yes0(x=x):
            shapex = tf.concat([tf.reshape(tf.shape(x)[0], (1,)),
                tf.ones((nd-1,),dtype=nd.dtype)],axis=0)
            x = tf.reshape(x,shapex)
            x = swap_axes(x,axis)
            return x
        def _no0(x=x):
            return x        
        x = tf.cond(tf.equal(tf.size(tf.shape(x)), 1), _yes0, _no0)
    assert even in ['avg','last','first']
    def _even():
        val = 0.0
        result = 0.0
        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            if x is not None:
                last_dx = _get(x,axis,-1) - _get(x,axis,-2)
            #A_ = tf.Print(A_,[tf.shape(A_)])
            val += 0.5*last_dx*(_get(y,axis,-1)+_get(y,axis,-2))
            result = _basic_simps(y,0,N-3,x,dx,axis)
            #result = tf.Print(result,[tf.shape(result)])
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            if not x is None:
                first_dx = _get(x,axis,1) - _get(x,axis,0)
            val += 0.5*first_dx*(_get(y,axis,1) + _get(y,axis,0))
            result += _basic_simps(y,1,N-2,x,dx,axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
        return result

    return tf.cond(tf.equal(tf.truncatemod(N,2),0),_even,lambda:_basic_simps(y,0,N-2,x,dx,axis))

def test_simps():
    from scipy.integrate import simps as simps_
    x = np.random.uniform(size=[40,5,31,20])
    x = np.sort(x,axis=-1)
    y = x**2
    res_np = simps_(y,x,axis=-1)
    res_tf = simps(y,x,axis=-1)
    sess = tf.Session()
    print(res_np-sess.run(res_tf))
    assert np.all(np.isclose(res_np,sess.run(res_tf),atol=1e-2))
    sess.close()

if __name__ == '__main__':
    test_simps()
