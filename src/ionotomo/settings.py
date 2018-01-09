import tensorflow as tf
import numpy as np

class TFSettings(object):
    tf_float = tf.float64
    tf_complex = tf.complex128
    tf_int = tf.int64
    tf_bool = tf.bool
    np_float = np.float64
    np_complex = np.complex128
    np_int = np.int64
    np_bool = np.bool

if __name__ == '__main__':
    print(TFSettings.tf_float)
