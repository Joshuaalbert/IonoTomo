
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pylab as plt

@tf.RegisterGradient('WrapGrad')
def _wrap_grad(op,grad):
    phi = op.inputs[0]
    return tf.ones_like(phi)*grad

def wrap(phi):
    out = tf.atan2(tf.sin(phi),tf.cos(phi))
    with tf.get_default_graph().gradient_override_map({'Identity': 'WrapGrad'}):
        return tf.identity(out)

