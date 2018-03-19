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

from ionotomo.bayes.gpflow_contrib import GPR_v2,Gaussian_v2
from scipy.cluster.vq import kmeans2



class Smoothing(object):
    """
    Class for all types of GP smoothing/conditioned prediction
    """

    def __init__(self,datapack):
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack

    def _make_coord_array(t,d,f):
        """Static method to pack coordinates
        """
        Nt,Nd,Nf = t.shape[0],d.shape[0], f.shape[0]
        X = np.zeros([Nt,Nd,Nf,4],dtype=np.float64)
        for j in range(Nt):
            for k in range(Nd):
                for l in range(Nf):
                    X[j,k,l,0:2] = d[k,:]
                    X[j,k,l,2] = t[j]   
                    X[j,k,l,3] = f[l]
        X = np.reshape(X,(Nt*Nd*Nf,4))
        return X

    def _solve_svgp(coords, data, var=None, M = 100, minibatch_size=500, iterations=1000, ARD=True, lock = None):
        assert len(coords) == len(data.shape)-1
        num_latent = data.shape[-1]
        data_mean = [data[...,i].mean() for i in range(num_latent)]
        data_std = [data[...,i].std() for i in range(num_latent)]
        data = np.stack([(data[...,i] - d_m)/d_s for i,(d_m,d_s) in enumerate(zip(data_mean,data_std))],axis=-1)
        Y = data.reshape((-1,num_latent))
        x_mean = [c.mean() for c in coords]
        x_std = [c.std() for c in coords]
        coords = [(c-c_m)/c_s for c,c_m,c_s in zip(coords, x_means,x_std)]

        X = Smoothing._make_coords_array(*coords)
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            if lock is not None:
                lock.acquire()
            try:
                with gp.defer_build():
                    kern = gp.kernels.RBF(len(coords),ARD=True)
                    kern.lengthscales.prior = gp.priors.Gaussian(0.,1/3.)
                    
                    mean = gp.mean_functions.Constant()

                    m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                            likelihood=gp.likelihoods.Gaussian() if var is None else Gaussian_v2(Y_var=var, trainable=False), 
                            Z=Z, num_latent=num_latent, minibatch_size=minibatch_size, whiten=True)
                    m.feature.set_trainable(False)
                    m.compile()
            finally:
                if lock is not None:
                    lock.release()
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
        
            ystar,varstar = m.predict_y(X)
            ystar = ystar.reshape(data.shape)
            ystar = np.stack([ystar[...,i]*d_s + d_m for i,(d_m,d_s) in enumerate(zip(data_mean,data_std))],axis=-1)
            varstar = varstar.reshape(data.shape)
            varstar = np.stack([varstar[...,i]*d_s**2 for i,(d_m,d_s) in enumerate(zip(data_mean,data_std))],axis=-1)
            return ystar, varstar



    def _solve_block_svgp(phase, error, coords, lock, init=(None,None),pargs=None,verbose=False):
        try:
            if verbose:
                logging.warning("{}".format(pargs))
            error_scale = np.mean(np.abs(phase))*0.1/np.mean(error)
            if verbose:
                logging.warning("Error scaling {}".format(error_scale))

            y_mean = np.mean(phase)
            y_scale = np.std(phase) + 1e-6
            y = (phase - y_mean)/y_scale
            y = y.flatten()[:,None]
            var = (error/y_scale*error_scale)**2
            var = var.flatten()
            
            t,d,f = coords
            t_scale = np.max(t) - np.min(t) + 1e-6
            d_scale = np.std(d - np.mean(d,axis=0)) + 1e-6
            f_scale = np.max(f) - np.min(f) + 1e-6
            t = (t - np.mean(t))/(t_scale+1e-6)
            d = (d - np.mean(d,axis=0))/(d_scale+1e-6)
            f = (f - np.mean(f))/(f_scale+1e-6)
            X = Smoothing._make_coord_array(t,d,f)

            M = 100
            Z = kmeans2(X, M, minit='points')[0]

            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire()
                try:
                    with gp.defer_build():
                        if init[0] is None:
                            k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.3])
                        else:
                            k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[init[0]/d_scale])
                            logging.warning('Using spatial scale: {}'.format(init[0]))
                            k_space.lengthscales.set_trainable(False)
                        if init[1] is None:
                            k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.3])
                        else:
                            k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[init[1]/t_scale])
                            logging.warning('Using spatial scale: {}'.format(init[1]))
                            k_time.lengthscales.set_trainable(False)

                        k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[10.])
                        
                        #k_white = gp.kernels.White(4)
                        kern = k_space * k_time * k_freq# + k_white
                        mean = gp.mean_functions.Zero()#Constant()

                        m = gp.models.svgp.SVGP(X, y, kern, mean_function = mean, 
                                likelihood=Gaussian_v2(Y_var=var, trainable=False), 
                                Z=Z, num_latent=1, minibatch_size=100, whiten=True)
                        m.feature.set_trainable(False)
                        m.kern.rbf_1.lengthscales.prior = gp.priors.Gaussian(1./d_scale,0.5/d_scale)
                        m.kern.rbf_2.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
                        m.kern.rbf_3.lengthscales.set_trainable(False)
                        m.compile()
                finally:
                    lock.release()
                iterations=200
                gp.train.AdamOptimizer(0.1).minimize(m, maxiter=iterations)
                
                if verbose:
                    logging.warning(m)
                kern_lengthscales = (
                        m.kern.rbf_1.lengthscales.value[0]*d_scale,
                        m.kern.rbf_2.lengthscales.value[0]*t_scale,
                        m.kern.rbf_3.lengthscales.value[0]*f_scale
                        )
                kern_variance = m.kern.rbf_1.variance.value*m.kern.rbf_2.variance.value*m.kern.rbf_3.variance.value*y_scale**2
                if verbose:
                    logging.warning(kern_lengthscales)
                    logging.warning(kern_variance)
                return kern_lengthscales, kern_variance
        except Exception as e:
            print(e)


    def _solve_block(phase, error, coords, lock, pargs=None,verbose=False):
        try:
            if verbose:
                logging.warning("{}".format(pargs))
            error_scale = np.mean(np.abs(phase))*0.1/np.mean(error)
            if verbose:
                logging.warning("Error scaling {}".format(error_scale))

            y_mean = np.mean(phase)
            y_scale = np.std(phase) + 1e-6
            y = (phase - y_mean)/y_scale
            y = y.flatten()[:,None]
            var = (error/y_scale*error_scale)**2
            var = var.flatten()
            
            t,d,f = coords
            t_scale = np.max(t) - np.min(t) + 1e-6
            d_scale = np.std(d - np.mean(d,axis=0)) + 1e-6
            f_scale = np.max(f) - np.min(f) + 1e-6
            t = (t - np.mean(t))/(t_scale+1e-6)
            d = (d - np.mean(d,axis=0))/(d_scale+1e-6)
            f = (f - np.mean(f))/(f_scale+1e-6)
            X = Smoothing._make_coord_array(t,d,f)

            ###
            # stationary points
            d_slice = np.s_[:]
            t_slice = np.s_[len(t)>>1:(len(t)>>1) + 10]
            f_slice = np.s_[len(f)>>1:(len(f)>>1)+1]

            Z = Smoothing._make_coord_array(t[t_slice],d[d_slice,:],f[f_slice])
            Zy = ((phase[t_slice,d_slice,f_slice] - y_mean)/y_scale).flatten()[:,None]
            Zvar = (error[t_slice,d_slice,f_slice].flatten() / y_scale * error_scale)**2

            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire()
                try:
                    with gp.defer_build():
                        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.1])
                        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.25])
                        k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[10.0])
                        
                        #k_white = gp.kernels.White(4)
                        kern = k_space * k_time * k_freq# + k_white
                        mean = gp.mean_functions.Constant()

                        m = GPR_v2(X, y, kern, Z=Z,Zy=Zy,Zvar=Zvar, mean_function=mean,var=var,trainable_var=False, minibatch_size=400)
                        m.kern.rbf_3.lengthscales.set_trainable(False)
                        m.compile()
                finally:
                    lock.release()
                o = gp.train.ScipyOptimizer(method='BFGS')
                #o = gp.train.AdamOptimizer(0.01)
                sess = m.enquire_session()
                with sess.as_default():
                    marginal_log_likelihood = [m.objective.eval()]
                    for i in range(3):
                        o.minimize(m,maxiter=4)
                        #marginal_log_likelihood.append(m.objective.eval())
                        #print(marginal_log_likelihood[-1])
                    #plt.plot(marginal_log_likelihood)
                    #plt.show()

                if verbose:
                    logging.warning(m)
                kern_lengthscales = (
                        m.kern.rbf_1.lengthscales.value[0]*d_scale,
                        m.kern.rbf_2.lengthscales.value[0]*t_scale,
                        m.kern.rbf_3.lengthscales.value[0]*f_scale
                        )
                kern_variance = m.kern.rbf_1.variance.value*m.kern.rbf_2.variance.value*m.kern.rbf_3.variance.value*y_scale**2
                if verbose:
                    logging.warning(kern_lengthscales)
                    logging.warning(kern_variance)
                return kern_lengthscales, kern_variance
        except Exception as e:
            print(e)

    def _ref_distance(self,antennas,i0=0):
        x = antennas.x.to(au.km).value
        y = antennas.y.to(au.km).value
        z = antennas.z.to(au.km).value
        dist = np.sqrt((x-x[i0])**2 + (y-y[i0])**2 + (z-z[i0])**2)
        return dist

    def refine_statistics_timeonly(self,results_file):
        plt.style.use('ggplot')
        antennas,antenna_labels = self.datapack.get_antennas(-1)
        data = np.load(results_file)
        # antenna, time
        length_scales = data['kern_ls'][:,:,0]
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,1],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
#                X[j,k,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,1))
        Xs = (times[:,None]-time_mean)/time_std
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                kern = k_time                
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(Xs)
            y = y*y_std + y_mean
            y = y.reshape((Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(length_scales.shape[0])]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            ax.plot(times,y[:,0],color='blue',label='Bayes')
            ax.fill_between(times,y[:,0]+std[:,0],y[:,0]-std[:,0],alpha=0.25,color='blue')
            ax.set_ylim([0.25,2.75])
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen directional correlation scale (deg)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_directional_scale_timeonly.png'))
            plt.show()
            # antenna, time, 1
            l_space = y.copy()

        # antenna, time
        length_scales = data['kern_ls'][:,:,1]
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,1],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
#                X[j,k,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,1))
        Xs = (times[:,None] - time_mean)/time_std
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                kern = k_time                
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(Xs)
            y = y*y_std + y_mean
            y = y.reshape((Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(length_scales.shape[0])]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            ax.plot(times,y[:,0],color='blue',label='Bayes')
            ax.fill_between(times,y[:,0]+std[:,0],y[:,0]-std[:,0],alpha=0.25,color='blue')
            ax.set_ylim([0.,700.])
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen temporal correlation scale (seconds)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_temporal_scale_timeonly.png'))
            plt.show()
            # antenna, time, 1
            l_time = y.copy()


        ###
        # var scale

        # antenna, time
        length_scales = np.log10(data['kern_var'][:,:,0])
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,1],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
#                X[j,k,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,1))
        Xs = (times[:,None] - time_mean)/time_std
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                kern = k_time                
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(Xs)
            y = y*y_std + y_mean
            y = y.reshape((Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(length_scales.shape[0])]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            ax.plot(times,y[:,0],color='blue',label='Bayes')
            ax.fill_between(times,y[:,0]+std[:,0],y[:,0]-std[:,0],alpha=0.25,color='blue')
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen log-variance correlation scale (mag.rad.)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_variance_scale_timeonly.png'))
            plt.show()


        return np.concatenate([l_space,l_time],axis=-1)

    def refine_statistics(self,results_file):
        plt.style.use('ggplot')
        antennas,antenna_labels = self.datapack.get_antennas(-1)
        data = np.load(results_file)
        # antenna, time
        length_scales = data['kern_ls'][:,:,0]
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,3],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
                X[k,j,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,3))
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                kern = k_time*k_space                
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                k_space.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=2000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(X)
            y = y*y_std + y_mean
            y = y.reshape((Np,Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Np,Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(y.shape[0])]
            [ax.plot(times,y[i,:,0],color='blue',lw = 2.,label='Bayes'if i == 0 else None,alpha=0.5) for i in range(61)]
#            [ax.fill_between(times,y[i,:,0]+0.5*std[i,:,0],y[i,:,0]-0.5*std[i,:,0],alpha=0.1,color='blue',label='Bayes'if i == 0 else None) for i in range(61)]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            #ax.fill_between(times,y[0,:,0]+std[0,:,0],y[0,:,0]-std[0,:,0],alpha=0.25,color='blue')
            ax.set_ylim([0.25,2.75])
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen directional correlation scale (deg)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_directional_scale.png'))
            plt.show()

        # antenna, time
        length_scales = data['kern_ls'][:,:,1]
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,3],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
                X[k,j,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,3))
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                kern = k_time*k_space             
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                k_space.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=2000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(X)
            y = y*y_std + y_mean
            y = y.reshape((Np,Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Np,Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(y.shape[0])]
            
            [ax.plot(times,y[i,:,0],color='blue',lw = 2.,label='Bayes'if i == 0 else None,alpha=0.5) for i in range(61)]
#            [ax.fill_between(times,y[i,:,0]+0.5*std[i,:,0],y[i,:,0]-0.5*std[i,:,0],alpha=0.1,color='blue',label='Bayes'if i == 0 else None) for i in range(61)]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            #ax.fill_between(times,y[0,:,0]+std[0,:,0],y[0,:,0]-std[0,:,0],alpha=0.25,color='blue')
            ax.set_ylim([0.,700.])
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen temporal correlation scale (seconds)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_temporal_scale.png'))
            plt.show()

        ###
        # var correlation scale

        # antenna, time
        length_scales = np.log10(data['kern_var'][:,:,0])
        y_mean = length_scales.mean()
        y_std = length_scales.std()

        times = data['time']
        time_mean = times.mean()
        time_std = times.std()

        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        pos_mean = positions.mean(0)
        positions -= pos_mean
        pos_std = positions.std(0).mean()
        positions /= pos_std

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Np,Nt,3],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[k,j,0] = (times[j] - time_mean)/time_std
                X[k,j,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,3))
        Y = (length_scales.reshape((-1,1)) - y_mean)/y_std

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                kern = k_time*k_space             
                mean = gp.mean_functions.Zero()#Constant()
                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                k_space.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.likelihood.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=2000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            y,var = m.predict_y(X)
            y = y*y_std + y_mean
            y = y.reshape((Np,Nt,1))
            var = var*y_std**2
            std = np.sqrt(var).reshape((Np,Nt,1))

            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
            [ax.scatter(times,length_scales[i,:],marker='+',c='black',alpha=0.15) for i in range(y.shape[0])]
            
            [ax.plot(times,y[i,:,0],color='blue',lw = 2.,label='Bayes'if i == 0 else None,alpha=0.5) for i in range(61)]
#            [ax.fill_between(times,y[i,:,0]+0.5*std[i,:,0],y[i,:,0]-0.5*std[i,:,0],alpha=0.1,color='blue',label='Bayes'if i == 0 else None) for i in range(61)]
            ax.plot(times,length_scales.mean(0),lw=2,ls='--',color='red',label='antenna average')
            #ax.fill_between(times,y[0,:,0]+std[0,:,0],y[0,:,0]-std[0,:,0],alpha=0.25,color='blue')
            #ax.set_ylim([0.,700.])
            ax.set_xlabel('Time (mjd)')
            ax.set_ylabel('Phase screen log-variance correlation scale (mag.rad.)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(results_file.replace('.npz','_variance_scale.png'))
            plt.show()
        return


    
    def solve_time_intervals(self, save_file, ant_idx, time_idx, dir_idx, freq_idx, interval, shift, num_threads=1,verbose=False, refined_params = None):
        """
        Solve for kernel characteristics over given domain.
        ant_idx, time_idx, dir_idx, freq_idx: the domain selectors
        interval: int interval in time to solve.
        shift: int the shift in time between solves.
        num_threads: int (default 1) the number of parallel solvers.
        Return interval start array, interval end array,
        the kernel length scales per antenna and variances per antenna 
        """

        datapack = self.datapack
        directions, patch_names = datapack.get_directions(dir_idx)
        times,timestamps = datapack.get_times(time_idx)
        antennas,antenna_labels = datapack.get_antennas(ant_idx)
        freqs = datapack.get_freqs(freq_idx)

        if ant_idx is -1:
            ant_idx = range(len(antennas))
        if time_idx is -1:
            time_idx = range(len(times))
        if freq_idx is -1:
            freq_idx = range(len(freqs))
        if dir_idx is -1:
            dir_idx = range(len(directions))

        phase = datapack.get_phase(ant_idx,time_idx,dir_idx,freq_idx)
        Na,Nt,Nd,Nf = phase.shape
        logging.warning("Working on shapes {}".format(phase.shape))

        assert interval <= Nt

        variance = datapack.get_variance(ant_idx,time_idx,dir_idx,freq_idx)
        error = np.sqrt(variance)
        data_mask = variance < 0
        error[data_mask] = 10.
        logging.warning("Total masked phases: {}".format(np.sum(data_mask)))

        uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
        dirs_uvw = directions.transform_to(uvw)
        #already centered on zero
#        d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
#                     np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        t = times.gps
        f = freqs
        directional_sampling = 1
        time_sampling = 1
        freq_sampling = 1
        directional_slice = slice(0,Nd,directional_sampling)
        freq_slice = slice(0,Nf,freq_sampling)

        lock = Lock()

        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*shift
                    stop = min(start+interval,Nt)
                    time_slice = slice(start,stop,time_sampling)

                    if refined_params is not None:
                        init = refined_params[j,:]
                    else:
                        init = [None,None]
                    
                    jobs.append(executor.submit(
                        Smoothing._solve_block_svgp,
                        phase[i,time_slice,directional_slice,freq_slice],
                        error[i,time_slice,directional_slice,freq_slice],
                        (t[time_slice],d[directional_slice],f[freq_slice]),
                        lock,
                        init=init,
                        pargs="Working on {} time chunk ({}) {} to ({}) {}".format(antenna_labels[i],
                            start,timestamps[start],stop-1,timestamps[stop-1]),
                        verbose=verbose
                        )
                        )
            ref_dist = []
            for j,aj in enumerate(time_idx[::shift]):
                start = j*shift
                stop = min(start+interval,Nt)
                ref_dist.append(self._ref_distance(
                        antennas, i0=0))
            ref_dist = np.stack(ref_dist,axis=0)
            results = futures.wait(jobs)
            if verbose:
                logging.warning(results)
            Nt_ = len(jobs)//Na
            kern_lengthscales = np.zeros([Na,Nt_,3])
            kern_variances = np.zeros([Na,Nt_,1])
            mean_time = np.zeros(Nt_)
            results = [j.result() for j in jobs]
            res_idx = 0
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*interval
                    stop = min((j+1)*interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    mean_time[j] = np.mean(t[time_slice])
                    res = results[res_idx]
                    res_idx += 1
                    kern_lengthscales[i,j,:] = res[0]
                    kern_variances[i,j,0] = res[1]
            np.savez(save_file,**{"kern_ls":kern_lengthscales,"kern_var":kern_variances,"time":mean_time,"antenna":antenna_labels,"ref_dist":ref_dist})

    def _apply_block_svgp(phase, error, coords, lock, kern_params,pargs=None,verbose=False):
        try:
            if verbose:
                logging.warning("{}".format(pargs))
            error_scale = np.mean(np.abs(phase))*0.1/np.mean(error)
            if verbose:
                logging.warning("Error scaling {}".format(error_scale))

            Nt,Nd,Nf = phase.shape

            y_mean = np.mean(phase)
            y_scale = np.std(phase) + 1e-6
            y = (phase - y_mean)/y_scale
            y = y.flatten()[:,None]
            var = (error/y_scale*error_scale)**2
            var = var.flatten()
            
            t,d,f = coords
            assert len(t) == Nt and len(d) == Nd and len(f) == Nf
            t_scale = np.max(t) - np.min(t) + 1e-6
            d_scale = np.std(d - np.mean(d,axis=0),axis=0).mean() + 1e-6
            f_scale = np.max(f) - np.min(f) + 1e-6
            t = (t - np.mean(t))/(t_scale)
            d = (d - np.mean(d,axis=0))/(d_scale)
            f = (f - np.mean(f))/(f_scale)
            X = Smoothing._make_coord_array(t,d,f)

            M = 100
            Z = kmeans2(X, M, minit='points')[0]

            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire()
                try:
                    with gp.defer_build():
                        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[kern_params[0]/d_scale])
                        k_space.lengthscales.set_trainable(False)
                        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[kern_params[1]/t_scale])
                        k_time.lengthscales.set_trainable(False)
                        k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[kern_params[2]/f_scale])
                        k_freq.lengthscales.set_trainable(False)
                        ## just set k_space, rest to 1.0
                        k_space.variance = kern_params[3]
                        k_space.variance.set_trainable(False)
                        k_time.variance = 1.0
                        k_time.variance.set_trainable(False)
                        k_freq.variance = 1.0
                        k_freq.variance.set_trainable(False)
                        
                        kern = k_space * k_time * k_freq
                        mean = gp.mean_functions.Zero()#Constant()

                        m = gp.models.svgp.SVGP(X, y, kern, mean_function = mean, 
                                likelihood=Gaussian_v2(Y_var=var, trainable=False), 
                                Z=Z, num_latent=1, minibatch_size=100, whiten=True)
                        m.feature.set_trainable(False)
                        m.kern.rbf_1.lengthscales.prior = gp.priors.Gaussian(1./d_scale,0.5/d_scale)
                        m.kern.rbf_2.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
                        m.kern.rbf_3.lengthscales.set_trainable(False)
                        m.compile()
                finally:
                    lock.release()
                iterations=200
                gp.train.AdamOptimizer(0.09).minimize(m, maxiter=iterations)
                
                if verbose:
                    logging.warning(m)

                for l,fs in enumerate(f):
                    if verbose:
                        logging.warning("Predicting freq {} MHz".format(coords[2][l]/1e6))
                    Xs = Smoothing._make_coord_array(t,d,np.array([fs]))
                    logging.warning("{}".format(Xs.shape))
                    ystar,varstar = m.predict_y(Xs)
                    logging.warning("{} {}".format(ystar.shape,varstar.shape))
                    ystar = ystar.reshape([Nt,Nd,1]) * y_scale + y_mean
                    varstar = varstar.reshape([Nt,Nd,1]) * y_scale**2

                    # set in the originial array (use locking)
                    lock.acquire()
                    try:
                        phase[...,l] = ystar
                        error[...,l] = np.sqrt(varstar)
                    finally:
                        lock.release()
                        
                return phase, error**2
        except Exception as e:
            print(e)


    def apply_solutions(self, save_datapack, solution_params, ant_idx, time_idx, dir_idx, freq_idx, interval, shift, num_threads=1,verbose=False):
        data = np.load(solution_params)

        kern_ls = data['kern_ls']
        kern_var = data['kern_var']
        kern_times = data['time']
        kern_antenna_labels = data['antenna']
        
        datapack = self.datapack
        directions, patch_names = datapack.get_directions(dir_idx)
        times,timestamps = datapack.get_times(time_idx)
        antennas,antenna_labels = datapack.get_antennas(ant_idx)
        freqs = datapack.get_freqs(freq_idx)

        if ant_idx is -1:
            ant_idx = range(len(antennas))
        if time_idx is -1:
            time_idx = range(len(times))
        if freq_idx is -1:
            freq_idx = range(len(freqs))
        if dir_idx is -1:
            dir_idx = range(len(directions))

        phase = datapack.get_phase(ant_idx,time_idx,dir_idx,freq_idx)
        Na,Nt,Nd,Nf = phase.shape
        logging.warning("Working on shapes {}".format(phase.shape))

        assert interval <= Nt

        variance = datapack.get_variance(ant_idx,time_idx,dir_idx,freq_idx)
        error = np.sqrt(variance)
        data_mask = variance < 0
        error[data_mask] = 10.
        logging.warning("Total masked phases: {}".format(np.sum(data_mask)))

        uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
        dirs_uvw = directions.transform_to(uvw)
        #already centered on zero
#        d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
#                     np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        t = times.gps
        f = freqs
        directional_sampling = 1
        time_sampling = 1
        freq_sampling = 1
        directional_slice = slice(0,Nd,directional_sampling)
        freq_slice = slice(0,Nf,freq_sampling)

        lock = Lock()

        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            mean_count = np.zeros(phase.shape)

            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*shift
                    stop = min(start+interval,Nt)
                    time_slice = slice(start,stop,time_sampling)

                    ###
                    # interpolate kern_params with this interval/shift
                    mean_time = np.mean(times.gps[time_slice])
                    
                    # d, t, f, v
                    kern_params = [
                            np.interp(mean_time, kern_times, kern_ls[i,:,0]),
                            np.interp(mean_time, kern_times, kern_ls[i,:,1]),
                            np.interp(mean_time, kern_times, kern_ls[i,:,2]),
                            np.interp(mean_time, kern_times, kern_var[i,:,0])
                            ]

                    mean_count[i,time_slice,directional_slice,freq_slice] += 1
                    
                    
#                    for l,al in enumerate(freq_idx):
#                        freq_slice = slice(l,l+1)
                    jobs.append(executor.submit(
                        Smoothing._apply_block_svgp,
                        phase[i,time_slice,directional_slice,freq_slice].copy(),
                        error[i,time_slice,directional_slice,freq_slice].copy(),
                        (t[time_slice],d[directional_slice],f[freq_slice]),
                        lock,
                        kern_params=kern_params,
                        pargs="Working on {} time chunk ({}) {} to ({}) {} at {} to {} MHz".format(antenna_labels[i],
                            start,timestamps[start],stop-1,timestamps[stop-1], freqs[0]/1e6, freqs[-1]/1e6),
                        verbose=verbose
                        )
                        )
            results = futures.wait(jobs)
            if verbose:
                logging.warning(results)
            results = [j.result() for j in jobs]

            phase_mean = np.zeros(phase.shape)
            variance_mean = np.zeros(variance.shape)
            res_idx = 0
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*interval
                    stop = min((j+1)*interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    res = results[res_idx]
                    phase_mean[i,time_slice,directional_slice,freq_slice] += res[0]
                    variance_mean[i,time_slice,directional_slice,freq_slice] += res[1]
                    res_idx += 1
            phase_mean /= mean_count
            variance_mean /= mean_count
            datapack.set_phase(phase_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.set_variance(variance_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)

            datapack.save(save_datapack)
    
if __name__=='__main__':
    import os

    if len(sys.argv) == 2:
        starting_datapack = sys.argv[1]
    else:
        starting_datapack = "../data/rvw_datapack_full_phase_dec27.hdf5"
    smoothing = Smoothing(starting_datapack)
    #smoothing.solve_time_intervals("gp_params.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True)
#    refined_params = smoothing.refine_statistics_timeonly('gp_params.npz')
#    print(refined_params.shape)
#    smoothing.solve_time_intervals("gp_params_fixed_scales.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True,refined_params=refined_params)
#    plt.ion()
#    smoothing.refine_statistics_timeonly('gp_params.npz')
#    smoothing.refine_statistics('gp_params.npz')
#    smoothing.refine_statistics_timeonly('gp_params_fixed_scales.npz')
#    smoothing.refine_statistics('gp_params_fixed_scales.npz')
#    plt.ioff()
    smoothing.apply_solutions(starting_datapack.replace('.hdf5','_refined_smoothed.hdf5'), 
            "gp_params_fixed_scales.npz",range(1,62), -1, -1, range(0,20), 32, 32, num_threads=1,verbose=True)
