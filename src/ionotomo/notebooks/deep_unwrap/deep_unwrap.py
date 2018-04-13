import numpy as np
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings
from timeit import default_timer
from ionotomo import *
import astropy.coordinates as ac
import astropy.units as au
import gpflow as gp
import sys
import h5py
import threading
from timeit import default_timer
#%matplotlib notebook
from concurrent import futures
from functools import partial
from threading import Lock
import astropy.units as au
import astropy.time as at
from collections import deque
from doubly_stochastic_dgp.dgp import DGP

from ionotomo.bayes.gpflow_contrib import GPR_v2,Gaussian_v2
from scipy.cluster.vq import kmeans2

from scipy.spatial.distance import pdist,squareform
import os

def get_only_vars_in_model(variables, model):
    reader = tf.train.NewCheckpointReader(model)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_in_model = [k for k in sorted(var_to_shape_map)]
    out_vars = []
    for var in variables:
        v = var.name.split(":")[0]
        if v in vars_in_model:
            if tuple(var.shape.as_list()) != reader.get_tensor(v).shape:
                logging.warning("{} has shape mis-match: {} {}".format(v,
                    tuple(var.shape.as_list()), reader.get_tensor(v).shape))
                continue
            out_vars.append(var)
    return out_vars

class AIUnwrap(object):
    def __init__(self,project_dir,datapack):
        self.project_dir = os.path.abspath(project_dir)
        try:
            os.makedirs(self.project_dir)
        except:
            pass
    
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack
        X = np.array([self.datapack.directions.ra.deg,self.datapack.directions.dec.deg]).T
        self.Nd = X.shape[0]
        self.Nt = 10
        coords = np.zeros([self.Nt, self.Nd,3])
        for j in range(self.Nt):
            for k in range(X.shape[0]):
                coords[j,k,0] = j*8
                coords[j,k,1:] = X[k,:]
        self.X = coords.reshape((self.Nt*self.Nd,3))
        
    def train(self,run_id, num_examples, max_steps=1000, minibatch_size=32, keep_prob=0.9,
             learning_rate=1e-3,disp_period=5.,patience=10,load_model=None,test_split=0.25):
        
        ls = np.array([np.random.uniform(low=40.,high=120,size=num_examples),
                      np.random.uniform(low=0.25,high=1.,size=num_examples)]).T
        variance = 10**np.random.uniform(low=-2,high=-0.5,size=num_examples)
        noise = 10**np.random.uniform(low=-3,high=-1,size=num_examples)
        
        tf.reset_default_graph()
        
        graph = self._build_train_graph()
        model_folder = os.path.join(self.project_dir,"model_{}".format(run_id))
        
        model_name = os.path.join(model_folder,"model")

        with tf.Session(graph=graph) as sess,\
            tf.summary.FileWriter(os.path.join(self.project_dir,"summary_{}".format(run_id)), graph) as writer:

            sess.run(tf.global_variables_initializer())
            
            if load_model is not None:
                try:
                    self.load_params(sess,load_model)
                except:
                    logging.warning("Could not load {} saved model".format(load_model))

            num_test =int(test_split*num_examples)
            num_train = num_examples - num_test
            logging.warning("Using num_train: {} num_test {}".format(num_train,num_test))
            
            last_train_loss = np.inf
            last_test_loss = np.inf
            
            train_losses = deque(maxlen=num_train//minibatch_size)
            predict_losses = deque(maxlen=num_train//minibatch_size)
            patience_cond = deque(maxlen=patience)
                
            step = sess.run(self.global_step)
            proceed = True
            test_h_val, train_h_val = sess.run([self.test_h,self.train_h])
            while proceed and step < max_steps:
                feed_dict = {
                    self.ls_pl : ls,
                    self.variance_pl : variance,
                    self.noise_pl : noise,
                    self.num_test : num_test,
                    self.minibatch_size : minibatch_size}
                
                sess.run([self.train_init,self.test_init],
                        feed_dict = feed_dict)
                # Train loop
                t0 = default_timer()
                t = t0
                while True:
                    lr_feed = self._get_learning_rate(last_test_loss, learning_rate)
                    feed_dict = {
                            self.learning_rate: lr_feed,
                            self.keep_prob : keep_prob,
                            self.handle: train_h_val
                            }
                    sess.run(self.metric_initializer)
                    try:
                        
                        train_loss, step, summary,  _, acc = sess.run([self.total_loss, self.global_step,
                                                            self.train_summary,self.train_op, self.acc], feed_dict=feed_dict)
                        train_losses.append(train_loss)
                        writer.add_summary(summary, global_step=step)

                        if default_timer() - t > disp_period:
                            logging.warning("Minibatch \tStep {:5d}\tloss {:.2e}\tacc {:.2e}".format(step, np.mean(train_losses),acc))
                            t = default_timer()

                    except tf.errors.OutOfRangeError:
                        break
                last_train_loss = np.mean(train_losses)
                samples_per_sec = num_train / (default_timer() - t0)
                ms_per_sample = 1000./samples_per_sec
                logging.warning("Speed\t{:.1f} samples/sec. [{:.1f} ms/sample]"\
                        .format(samples_per_sec,ms_per_sample))

                # Test loop
                test_losses = []
                while True:
                    feed_dict = {self.handle: test_h_val,
                            self.keep_prob : 1.
                                }
                    sess.run(self.metric_initializer)
                    try:
                        test_loss, step, summary, acc = sess.run([self.total_loss, self.global_step, self.test_summary, self.acc], feed_dict=feed_dict)
                        test_losses.append(test_loss)
                        writer.add_summary(summary, global_step=step)                               
                    except tf.errors.OutOfRangeError:
                        break
                last_test_loss = -acc#np.mean(test_losses)
                patience_cond.append(last_test_loss)
                if len(patience_cond) == patience and np.min(patience_cond) == patience_cond[0]:
                    proceed = False
                logging.warning("Validation\tStep {:5d}\tloss {:.2e}\tacc {:.2e}".format(step, np.mean(test_losses),acc))
                save_path = self.save_params(sess,model_name)
        
    def _build_train_graph(self,graph=None):
        graph = graph or tf.Graph()
        with graph.as_default():
            
            self.ls_pl = tf.placeholder(tf.float32,shape=(None,2),
                    name='ls')
            self.variance_pl = tf.placeholder(tf.float32,shape=(None,),
                    name='variance')
            self.noise_pl = tf.placeholder(tf.float32,shape=(None,),
                    name='noise')

            
            self.create_iterators(self.ls_pl,self.variance_pl,self.noise_pl)
            self.f_latent, self.labels,self.weights = self.data_tensors
            self.f_latent.set_shape([None,self.X.shape[0]])
            self.f_latent = tf.reshape(self.f_latent,(-1, self.Nt, self.Nd))
            self.labels.set_shape([None,self.X.shape[0]])
            self.weights.set_shape([None,self.X.shape[0]])
            
            
            
            with tf.variable_scope("predict") as scope:
                self.keep_prob = tf.placeholder(tf.float32, shape=(),name='keep_prob')
                cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.DropoutWrapper(
                        (
                            tf.nn.rnn_cell.LSTMCell(self.Nd*3,activation=tf.nn.relu)),
                        output_keep_prob=self.keep_prob),
                    tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.ResidualWrapper(
                            tf.nn.rnn_cell.LSTMCell(self.Nd*3,activation=tf.nn.relu)),
                        output_keep_prob=self.keep_prob),
                    tf.contrib.rnn.DropoutWrapper(
                        (
                            tf.nn.rnn_cell.LSTMCell(self.Nd*13,activation=tf.nn.relu)),
                        output_keep_prob=self.keep_prob),
                    tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.ResidualWrapper(
                            tf.nn.rnn_cell.LSTMCell(self.Nd*13,activation=tf.nn.relu)),
                        output_keep_prob=self.keep_prob)])
                predict, state = tf.nn.dynamic_rnn(cell,self.f_latent,dtype=tf.float32)
                self.predict = tf.reshape(predict,(-1,self.Nt*self.Nd,13))
#                 self.predict = tf.reshape(tf.layers.dense(predict,self.Nt*self.Nd*13),(-1,self.Nt*self.Nd,13))
                    
            self._build_metrics(self.labels,self.predict)
            self.acc = tf.identity(self.acc_update)
            
            with tf.variable_scope('train') as scope:
                self.total_loss = tf.reduce_mean(self.weights * \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict,labels=self.labels))
                self.learning_rate = tf.placeholder(tf.float32, shape=(),name='learning_rate')
                opt = tf.train.AdamOptimizer(self.learning_rate)
                train_vars = tf.trainable_variables()
                
                grad_and_vars = opt.compute_gradients(self.total_loss, train_vars)
                clipped,_ = tf.clip_by_global_norm([g for g,_ in grad_and_vars], 1.)
                grad_and_vars = zip(clipped, train_vars)
                self.global_step = tf.train.get_or_create_global_step()
                self.train_op = opt.apply_gradients(grad_and_vars,self.global_step)
                
            with tf.variable_scope("summaries"):
                self.train_summary = tf.summary.merge([
                    ###
                    # scalars
                    tf.summary.scalar("acc",self.acc,family='train'),
                    tf.summary.scalar("prec",self.prec_update,family='train'),
                    tf.summary.scalar("loss",self.total_loss,family='train')
                ])
                self.test_summary = tf.summary.merge([
                    ###
                    # scalars
                    tf.summary.scalar("acc",self.acc,family='test'),
                    tf.summary.scalar("prec",self.prec_update,family='test'),
                    tf.summary.scalar("loss",self.total_loss,family='test')
                ])
    
            return graph
        
    def _build_metrics(self,true,predict):
        with tf.variable_scope("metrics") as scope:
            predict_labels = tf.cast(tf.argmax(predict,axis=-1), tf.int32)
            self.acc, self.acc_update = tf.metrics.accuracy(true,predict_labels)
            self.prec, self.prec_update = tf.metrics.precision(true,predict_labels)
            
            #self.metric_update_op = tf.group([acc_update,prec_update])
            
            self.metric_initializer = \
            tf.variables_initializer(
                var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope=scope.name))
            
    def create_iterators(self,lengthscale, variance, noise):
        """Will take the first num_test from batch axis as test set"""
        with tf.name_scope("datasets"):
            self.num_test = tf.placeholder(tf.int64,shape=(),name='num_test')
            self.minibatch_size = tf.placeholder(tf.int64,shape=(),name='minibatch_size')

            #img, mask, border_weights, prior, bb, num_masks
            output_types = [tf.float32,tf.int32,tf.float32]
            dataset = tf.data.Dataset.from_tensor_slices((lengthscale, variance, noise))
            dataset = dataset.map(lambda lengthscale, variance, noise: \
                                  tuple(tf.py_func(self.load_train_example,[lengthscale, variance, noise],
                                                       output_types)),num_parallel_calls=None)
            
            test_dataset = dataset.take(self.num_test).batch(self.num_test)
            train_dataset = dataset.skip(self.num_test).batch(self.minibatch_size)
            
            test_iterator = test_dataset.make_initializable_iterator()
            train_iterator = train_dataset.make_initializable_iterator()
            self.train_init, self.test_init = train_iterator.initializer,test_iterator.initializer

            self.handle = tf.placeholder(tf.string,shape=[])
            iterator = tf.data.Iterator.from_string_handle(self.handle, train_dataset.output_types, train_dataset.output_shapes)

            self.data_tensors = iterator.get_next()
            self.test_h, self.train_h = test_iterator.string_handle(),train_iterator.string_handle()
    
    def _get_learning_rate(self, rec_loss, lr):
        if np.sqrt(rec_loss) < 0.5:
            return lr/2.
        elif np.sqrt(rec_loss) < 0.4:
            return lr/3.
        elif np.sqrt(rec_loss) < 0.3:
            return lr/4.
        elif np.sqrt(rec_loss) < 0.2:
            return lr/5.
        elif np.sqrt(rec_loss) < 0.15:
            return lr/6.
        elif np.sqrt(rec_loss) < 0.1:
            return lr/10.   
        return lr
    
    def load_params(self,sess, model):
        with sess.graph.as_default():
            all_vars = tf.trainable_variables()
            load_vars = get_only_vars_in_model(all_vars,model)
            saver = tf.train.Saver(load_vars)
            saver.restore(sess,model)

    def save_params(self,sess, model):
        with sess.graph.as_default():
            all_vars = tf.trainable_variables()
            saver = tf.train.Saver(all_vars)
            save_path = saver.save(sess,model)
            return save_path
            
    def load_train_example(self, ls, variance, noise):
        tec_conversion = -8.4480e9# rad Hz/tecu

        X = self.X.copy()
        X[:,0] /= ls[0]
        X[:,1:] /= ls[1]
        pd = pdist(X,metric='sqeuclidean')
        pd *= -1.
        K = variance*np.exp(squareform(pd))
        tec = np.random.multivariate_normal(np.zeros(self.X.shape[0]),cov = K)
        tec += noise*np.random.normal(size=tec.shape)
        phase = tec*tec_conversion / 150e6
        phase = phase.reshape((self.Nt,self.Nd))
        phase -= phase.mean(axis=1)[:,None]
        phase = phase.reshape((self.Nt*self.Nd,))
        phase_wrap = np.angle(np.exp(1j*phase))
        jumps = ((phase - phase_wrap)/(2*np.pi)) + 6
        jumps[jumps < 0] = 0
        jumps[jumps > 12 ] = 12
        where = jumps!=6
        weights = np.ones(tec.shape)
        #weights[where] += tec.size - np.sum(where)
        return phase.astype(np.float32), jumps.astype(np.int32), weights.astype(np.float32)

am = AIUnwrap("projects", "../../data/rvw_datapack_full_phase_dec27_unwrap.hdf5")
#am.load_train_example([100,1],0.1**2,0.001)
am.train(0, num_examples = 10000, max_steps=10000, minibatch_size=200, 
         keep_prob=0.9, learning_rate=1e-3,disp_period=5.,patience=5,load_model='projects/model_0/model',
         test_split=200./10000.)
