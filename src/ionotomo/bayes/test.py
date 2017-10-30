import tensorflow as tf
import numpy as np

def cdist(x,y):
    """do pdist
    x : Tensor (batch_size,num_points,ndims)"""
    #D[:,i,j] = (a[:,i] - b[:,j]) (a[:,i] - b[:,j])'
    #=   a[:,i,p] a[:,i,p]' - b[:,j,p] a[:,i,p]' - a[:,i,p] b[:,j,p]' + b[:,j,p] b[:,j,p]'
    # batch_size,num_points,1
    r1 = tf.reduce_sum(x*x,axis=-1,keep_dims=True)
    r2 = tf.reduce_sum(y*y,axis=-1,keep_dims=True)
    out = r1 - 2*tf.matmul(x,tf.transpose(y,perm=[0,2,1])) +  tf.transpose(r2,perm=[0,2,1])
    
    return out

def pdist(x):
    """do pdist
    x : Tensor (batch_size,num_points,ndims)"""
    #D[:,i,j] = a[:,i] a[:,i]' - a[:,i] a[:,j]' -a[:,j] a[:,i]' + a[:,j] a[:,j]'
    #       =   a[:,i,p] a[:,i,p]' - a[:,i,p] a[:,j,p]' - a[:,j,p] a[:,i,p]' + a[:,j,p] a[:,j,p]'
    # batch_size,num_points,1
    r = tf.reduce_sum(x*x,axis=-1,keep_dims=True)
    #batch_size,num_points,num_points
    A = tf.matmul(x,tf.transpose(x,perm=[0,2,1]))
    B = r - 2*A
    out = B + tf.transpose(r,perm=[0,2,1])
    return out

sess = tf.Session()
A = tf.constant([[[1, 1], [2, 2], [3, 3]]])
B = tf.constant([[[1, 1], [2, 2], [3, 4]]])
print(sess.run(cdist(A,B)))
