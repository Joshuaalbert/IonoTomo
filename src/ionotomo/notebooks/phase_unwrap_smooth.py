
# coding: utf-8

# In[10]:

import numpy as np
import tensorflow as tf
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings
from timeit import default_timer
from ionotomo import *
import gpflow as gp
import sys
#%matplotlib notebook


class PhaseUnwrap(object):
    """The phase unwrapper object.
    Unwraps 2D phase in batches.
    Args:
    directions : array (num_directions, 2) the directions in the 2D plane wth phases assigned
    shape : tuple of int (num_directions, batch_dim)
    redundancy: int (see self._create_triplets) (leave 2)
    sess : tf.Session object to use or None for own
    """
    def __init__(self,directions, shape, redundancy=2, graph = None,max_wrap_K=2, max_epochs=1000):
        self.max_epochs = int(max_epochs)
        self.num_logits = int(max_wrap_K)*2 + 1
        self.directions = directions
        self.redundancy = redundancy
        self.shape = tuple(shape)
        assert len(self.shape)==2, "shape should be (num_dir, batch_size)"
        self.path, self.triplets,self.adjoining_map = self._create_triplets(self.directions,redundancy=self.redundancy, adjoining=True)
        self.pairs =                 np.unique(np.sort(np.concatenate([self.triplets[:,[0,1]],
                    self.triplets[:,[1,2]], self.triplets[:,[2,0]]],axis=0),axis=1),axis=0)
        
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        self.pipeline = Pipeline()
        self.pipeline.add_graph('phase_unwrap',self.graph)
        
        self.model_scope = "phase_unwrap"
        with self.graph.as_default():
            with tf.variable_scope(self.model_scope):
                phi_wrap_placeholder =                         tf.placeholder(tf.float32,shape=shape,name='phi_wrap')
                phi_mask_placeholder =                         tf.placeholder(tf.float32,shape=shape[1:],name='phi_mask')
                keep_prob_placeholder = tf.placeholder(tf.float32,shape=(),name='keep_prob')
                ent_w = tf.placeholder(tf.float32,shape=(),name='ent_w')
                learning_rate_placeholder = tf.placeholder(tf.float32,shape=(),name='learning_rate')
                directions = self.pipeline.add_variable(name='directions',trainable=False,init_value=self.directions,dtype=tf.float32)
                adjoining_map = self.pipeline.add_variable(name='adjoining_map',trainable=False,init_value=self.adjoining_map,dtype=tf.int32)
                
                phase_unwrap_op, losses, K_greedy = self._build_phase_unwrap(phi_wrap_placeholder,phi_mask_placeholder,keep_prob_placeholder,learning_rate_placeholder,ent_w,
                                                                          directions,adjoining_map)
                
                self.pipeline.store_ops([adjoining_map,directions,ent_w,phi_wrap_placeholder,phi_mask_placeholder, phase_unwrap_op, losses, K_greedy, keep_prob_placeholder, learning_rate_placeholder],
                                        ["adjoining_map","directions","ent_w","phi_wrap","phi_mask","phase_unwrap","losses","K_greedy","keep_prob","learning_rate"], 
                                        self.model_scope)
                
    def set_logits(self,sess,logits):
        """Set logics which are of shape (num_dir, batch_size, self.num_logits)"""
        _ = sess.run(self.zero_logits_op,feed_dict={self.K_zero_logits_placeholder: logits})
        
    def maybe_unwrap(self,sess,phi_wrap):
        """Return a mask of shape (batch_size) with True where
        unwrapping needed"""
        with self.graph.as_default():
            losses,phi_wrap_placeholder,phi_mask_placeholder, keep_prob_placeholder =                     self.pipeline.grab_ops(["losses","phi_wrap","phi_mask","keep_prob"], self.model_scope)
            #self.sess.run(tf.global_variables_initializer())
            self.pipeline.initialize_graph(sess)
            K_zero_logits = np.zeros(self.shape+(self.num_logits,),dtype=np.float32)
            K_zero_logits[...,self.num_logits>>1] += 1000
            
            self.set_logits(sess,K_zero_logits)
            
            consistency_val,losses_val = sess.run([self.consistency,losses],
                            feed_dict={phi_wrap_placeholder:phi_wrap,
                                phi_mask_placeholder:np.ones(self.shape[1:]),
                                keep_prob_placeholder:1.})
            
            mean_con = []
            for k in range(self.shape[0]):
                c1 = consistency_val[self.pairs[:,0] == k,:]
                c2 = consistency_val[self.pairs[:,1] == k,:]
                c = np.zeros(self.shape[1])
                if c1.size > 0:
                    c += np.mean(c1,axis=0)
                if c2.size > 0:
                    c += np.mean(c2,axis=0)
                mean_con.append(c)
                
                
            mask = ((losses_val[0] + losses_val[1]) > 0.01).astype(np.bool)
            #mask_2 = (np.sum(consistency_val>0.1,axis=0) > 0).astype(np.bool)
            return mask,np.stack(mean_con,axis=0)
        
    def stuck_directions(self,sess, phi):
        _,mean_con = self.maybe_unwrap(sess,phi)
        return np.abs(mean_con) > 0.1
        
        
    def phase_unwrap(self, sess, phi_wrap, phi_mask = None):
        """Runs the training loop until termination"""
        
        if len(phi_wrap.shape) == 1:
            phi_wrap = phi_wrap[:,None,None]
        if phi_mask is None:
            phi_mask,mean_con = self.maybe_unwrap(sess,phi_wrap)
            phi_mask = np.any(np.abs(mean_con) > 0.1,axis=0)
            #[print(m,c) for m,c in zip(phi_mask,mean_con.T)]
            #print(phi_mask)
            print("Number that need unwrapping: {}".format(np.sum(phi_mask)))
            print("Indices: {}".format(np.where(phi_mask)))
        assert phi_mask.shape == self.shape[1:]
        #self._numpy_unwrap(phi_wrap, phi_mask)
        with self.graph.as_default():
            ent_w, phi_wrap_placeholder,phi_mask_placeholder, phase_unwrap_op, losses, K_greedy,keep_prob_placeholder, learning_rate_placeholder =                     self.pipeline.grab_ops(["ent_w","phi_wrap","phi_mask","phase_unwrap","losses","K_greedy","keep_prob","learning_rate"], self.model_scope)
            #self.sess.run(tf.global_variables_initializer())
            self.pipeline.initialize_graph(sess)

            last_losses_val_max = [np.inf]*4
            last_losses_val_95 = [np.inf]*4
            last_losses_val_median = [np.inf]*4
            last_losses_val_min = [np.inf]*4
            
            t0 = default_timer()-10.
            for epoch in range(self.max_epochs):
                lr = 0.1
                dp = 0.5
                ew = 1e-3
                last_lse,last_residue = last_losses_val_max[0:2]
                if last_lse < 0.5 and last_residue < 0.5:
                    ew = 1e-2
                    lr = 0.1
                    dp = 0.5
                if last_lse < 0.3 and last_residue < 0.3:
                    ew = 1e-2
                    lr = 0.06
                    dp = 0.5
                if last_lse < 0.15 and last_residue < 0.15:
                    ew = 1e-2
                    lr = 0.03
                    dp = 1.
                if last_lse < 0.05 and last_residue < 0.05:
                    ew = 5e-1
                    lr = 0.01
                    dp = 1.
                
                _, losses_val,K_greedy_val = sess.run([phase_unwrap_op,losses,K_greedy],
                        feed_dict={phi_wrap_placeholder:phi_wrap,
                phi_mask_placeholder:phi_mask,keep_prob_placeholder:dp,
                learning_rate_placeholder:lr,
                                  ent_w:ew})
                #print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_val),*losses_val))
                losses_val_max = [np.max(l) for l in losses_val]
                losses_val_95 = [np.percentile(l,95) for l in losses_val]
                losses_val_median = [np.percentile(l,50) for l in losses_val]
                losses_val_min = [np.min(l) for l in losses_val]
                
                if (epoch + 1) % 1000 == 0 or default_timer() - t0 > 10.:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f}".format(epoch,np.sum(losses_val_max),*losses_val_max))
                    t0 = default_timer()
                
                def _relative_change(cur,last):
                    if last is np.inf:
                        return np.inf
                    return np.abs((cur[0]+cur[1] - last[0]-last[1])/(last[0]+last[1]))
                
                no_change = _relative_change(losses_val_max, last_losses_val_max) < 1e-3                     and _relative_change(losses_val_median, last_losses_val_median) < 1e-3                     and _relative_change(losses_val_min, last_losses_val_min) < 1e-3
                
                if losses_val_max[0]+losses_val_max[1] < 0.1:# or no_change:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f}".format(epoch,np.sum(losses_val_max),*losses_val_max))
                    break
                    
                last_losses_val_max = losses_val_max
                last_losses_val_95 = losses_val_95
                last_losses_val_median = losses_val_median
                last_losses_val_min = losses_val_min
                                                            
                         
            f_rec = np.zeros_like(phi_wrap)
            f_rec[self.path[0][0],...] = phi_wrap[self.path[0][0],...]
            for i,p in enumerate(self.path):
                df = phi_wrap[p[1],...] - phi_wrap[p[0],...] + K_greedy_val[p[1],...] - K_greedy_val[p[0],...]
                f_rec[p[1],...] = f_rec[p[0],...] + df
            same_mask = np.bitwise_not(phi_mask)
            f_rec[:,same_mask] = phi_wrap[:,same_mask]
#             sc=plt.scatter(self.directions[:,0],self.directions[:,1],c=K_cum.flatten())
#             plt.colorbar(sc)
#             plt.show()
        return f_rec

    def _create_triplets(self,X,redundancy=2,adjoining=True):
        """Create the path of closed small triplets.
        Args:
            X : array (N,2) the coordinates in image
            redundancy : int the number of unique triplets 
            to makes for each segment of path.
            adjoining: return a list of triplets that share same leg of path.
                currently requires redundance = 2
        Returns:
            path, triplets
            path, triplets, adjoining_map if adjoining=True
        """
        kt = cKDTree(X)
        #get center of map
        C = np.mean(X,axis=0)
        _,idx0 = kt.query(C,k=1)
        #define unique path
        dist, idx = kt.query(X[idx0,:],k=2)
        path = [(idx0, idx[1])]
        included = [idx0, idx[1]]
        while len(included) < X.shape[0]:
            dist,idx = kt.query(X[included,:],k = len(included)+1)
            mask = np.where(np.isin(idx,included,invert=True))
            argmin = np.argmin(dist[mask])
            idx_from = included[mask[0][argmin]]
            idx_to = idx[mask[0][argmin]][mask[1][argmin]]
            path.append((idx_from,idx_to))
            included.append(idx_to)
        #If redundancy is 2 then build adjoining triplets
        if adjoining:
            assert redundancy == 2
            adjoining_map = []
        M = np.mean(X[path,:],axis=1)
        _,idx = kt.query(M,k=2 + redundancy)
        triplets = []
        for i,p in enumerate(path):
            count = 0
            for c in range(2 + redundancy):
                if idx[i][c] not in p:
                    triplets.append(p + (idx[i][c],))
                    count += 1
                    if count == redundancy:
                        if adjoining:
                            adjoining_map.append([len(triplets)-2,len(triplets)-1])
                        break
        adjoining_map = np.sort(adjoining_map,axis=1)
        adjoining_map = np.unique(adjoining_map,axis=0)
        
        triplets = np.sort(triplets,axis=1)
        triplets, _, unique_inverse = np.unique(triplets,return_index=True,
                                                             return_inverse=True,axis=0)
        if adjoining:
            #map i to j
            for i,ui in enumerate(unique_inverse):
                adjoining_map = np.where(adjoining_map == i, ui, adjoining_map)
            return path,triplets,adjoining_map
            
        return path,triplets
    
    def _numpy_unwrap(self,phi_wrap,phi_mask):
        """diagnostic method"""
        
        def _wrap(array):
            return np.angle(np.exp(1j*array))
        
        K = np.zeros_like(phi_wrap,dtype=np.float32)
        f = phi_wrap + (2*np.pi)*K
        lse = np.inf
        while lse > 0.1:
            for p in self.path:
                df = _wrap(_wrap(phi_wrap[p[1],...]) - _wrap(phi_wrap[p[0],...]))
                f[p[1],...] += df
            lse = (f - phi_wrap)/(2*np.pi)
            K = np.around(lse)
            #f = phi_wrap + (2*np.pi)*K
            lse = np.max(np.abs(lse))
            print(lse)
        plt.hist(K.flatten(),bins=25)
        plt.show()
        print(K)
        return lse
        
            
                           
                

    def _build_phase_unwrap(self, phi_wrap_placeholder,phi_mask_placeholder,
                            keep_prob_placeholder,learning_rate_placeholder,ent_w,
                            directions,adjoining_map):
        """Main phase unwrapping ops.
        Performs the unwrap with a smooth representation of K
        then minimizes entropy to converge on an integer.
        Returns also the greedy solution for termination checking.
        """
        with self.graph.as_default():
            with tf.name_scope("unwrapper"):
                
                twopi = tf.constant(2*np.pi,shape=(),dtype=tf.float32)
                
                def _wrap(a):
                    """wraps given function"""
                    return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,tf.complex64))),tf.float32)
                def trip_arc(f,i,j):
                    return _wrap(tf.gather(f,triplets[:,j]) - tf.gather(f,triplets[:,i]))

                triplets = self.pipeline.add_variable("triplets",init_value=self.triplets,
                        dtype=tf.int32,trainable=False)
                pairs = self.pipeline.add_variable("pairs",init_value=self.pairs,
                        dtype=tf.int32,trainable=False)

                K_logits = tf.get_variable("K",shape=self.shape + (self.num_logits,),
                        dtype=tf.float32,initializer=tf.zeros_initializer)
                
                
                ### set logits op
                self.K_zero_logits_placeholder =                     tf.placeholder(shape=self.shape+(self.num_logits,), dtype=tf.float32)
                self.zero_logits_op = tf.assign(K_logits,self.K_zero_logits_placeholder)
                
                ### distribution representation
                K_dist = tf.nn.softmax(K_logits)

                indices =                     tf.constant((np.arange(self.num_logits)-(self.num_logits>>1))[None,None,:], dtype=tf.float32)
                K = tf.reduce_sum(K_dist*indices,axis=-1)*twopi
                
                K_greedy = tf.cast(tf.argmax(K_logits,axis=-1) - (self.num_logits>>1),tf.float32)*twopi
                
                ### absoulte phase representation
                f_noise = tf.get_variable("f_noise",shape=self.shape,dtype=tf.float32,initializer=tf.zeros_initializer)
                f = phi_wrap_placeholder + K + f_noise
                f_greedy = phi_wrap_placeholder + K_greedy + f_noise

#                 ##Plates
#                 #get the x1,x2,x3 of each triplet
#                 pos = tf.stack([tf.tile(directions[:,0,None,None],(1,)+self.shape[1:]),
#                        tf.tile(directions[:,1,None,None],(1,)+self.shape[1:]),
#                        K_cum],axis=-1)
#                 x1 = tf.gather(pos,triplets[:,0],axis=0)
#                 x2 = tf.gather(pos,triplets[:,1],axis=0)
#                 x3 = tf.gather(pos,triplets[:,2],axis=0)
#                 x1x2 = x1-x2
#                 x1x3 = x1-x3
#                 n = tf.cross(x1x2,x1x3)
#                 nmag = tf.norm(n,axis=-1,keep_dims=True)
#                 n /= nmag
#                 n1 = tf.gather(n,adjoining_map[:,0],axis=0)
#                 n2 = tf.gather(n,adjoining_map[:,1],axis=0)
#                 plate_tension_loss = tf.square(tf.abs(tf.matmul(n1[:,:,:,None,:],n2[:,:,:,:,None]))-1.)
                
                ### Consistency condition
                def _consistency(f):
                    df = tf.gather(f,pairs[:,1]) - tf.gather(f,pairs[:,0])
                    consistency = tf.sqrt(1.+tf.square(_wrap(tf.gather(phi_wrap_placeholder,pairs[:,1]) - tf.gather(phi_wrap_placeholder,pairs[:,0])) - df)) - 1.
                    return consistency
                
                self.consistency = _consistency(f)
                consistency = tf.nn.dropout(self.consistency,keep_prob_placeholder)
                consistency_greedy = _consistency(f_greedy)
                
                ### Residue condition
                def _residue(f):
                    Wf = _wrap(f)
                    df01_ = trip_arc(Wf,0,1) 
                    df01 = _wrap(df01_) 
                    df12_ = trip_arc(Wf,1,2)
                    df12 = _wrap(df12_)
                    df20_ = trip_arc(Wf,2,0)
                    df20 = _wrap(df20_)
                    residue = tf.sqrt(1. + tf.square(df01 + df12 + df20))-1.
                    return residue
                self.residue = _residue(f)
                residue = tf.nn.dropout(self.residue,keep_prob_placeholder)
                residue_greedy = _residue(f_greedy)
                
                #trainable lse and residue
                loss_residue = tf.reduce_mean(residue,axis=0)
                loss_lse = tf.reduce_mean(consistency,axis=0)
                loss_tv = tf.reduce_mean(tf.square(f_noise),axis=0)
                entropy = - tf.reduce_mean(tf.reduce_sum(K_dist*tf.log(K_dist+1e-10),axis=-1),axis=0)
                total_loss = loss_lse+loss_residue+loss_tv+ent_w*entropy
                total_loss *= phi_mask_placeholder
                
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
                train_op = opt.minimize(tf.reduce_mean(tf.reduce_mean(total_loss)))
                
                loss_residue_greedy = tf.reduce_mean(residue_greedy,axis=0)
                loss_lse_greedy = tf.reduce_mean(consistency_greedy,axis=0)

                losses = [loss_lse_greedy ,loss_residue_greedy, entropy, loss_tv]
                #with tf.control_dependencies([train_op]):
                return train_op, losses, K_greedy

    def plot_triplets(self,figname=None):
        fig = plt.figure(figsize=(8,8))
        for id,(i,j,k) in enumerate(self.triplets):
            plt.plot([self.directions[i,0],self.directions[j,0],self.directions[k,0],
                self.directions[i,0]],[self.directions[i,1],self.directions[j,1],
                    self.directions[k,1],self.directions[i,1]])
            plt.text((self.directions[i,0]+self.directions[j,0]+self.directions[k,0])/3.,
                    (self.directions[i,1]+self.directions[j,1]+self.directions[k,1])/3.,"{}".format(id))
        if figname is not None:
            plt.savefig(figname)
            plt.close("all")
        else:
            plt.show()
            
    def plot_edge_dist(self,):
        dist = np.linalg.norm(np.concatenate([self.directions[self.triplets[:,1],:]                - self.directions[self.triplets[:,0],:], self.directions[self.triplets[:,2],:]                - self.directions[self.triplets[:,1],:],self.directions[self.triplets[:,0],:]                - self.directions[self.triplets[:,2],:]],axis=0),axis=1)
        plt.hist(dist,bins=20)
        plt.show()

    
def generate_data_aliased(noise=0.,sample=100):
    """Generate Gaussian bump in phase.
    noise : float
        amount of gaussian noise to add as fraction of peak height
    sample : int
        number to sample
    """
    #max gradient at b
    a = 50
    b = 1
    max_slope = np.abs(a/np.sqrt(np.exp(1))/b)
    
    #in dx want max_slope*dx > np.pi
    dx = 1.5*np.pi/max_slope
    
    N = 10
    xvec = np.linspace(-dx*N, dx*N, N*2 + 1)
    X,Y = np.meshgrid(xvec,xvec,indexing='ij')
    phi = a * np.exp(-(X**2 + Y**2)/2./b**2)
    X = np.array([X.flatten(),Y.flatten()]).T
    
    phi += a*noise*np.random.normal(size=phi.shape)
    phi = phi.flatten()
    if sample != 0:
        mask = np.random.choice(phi.size,size=min(sample,phi.size),replace=False)
        return X[mask,:],phi[mask]
    return X,phi

def generate_data_nonaliased(noise=0.,sample=100):
    """Generate Gaussian bump in phase.
    noise : float
        amount of gaussian noise to add as fraction of peak height
    sample : int
        number to sample
    """
    #max gradient at b
    a = 15
    b = 1
    max_slope = np.abs(a/np.sqrt(np.exp(1))/b)
    
    #in dx want max_slope*dx < np.pi
    dx = 0.5*np.pi/max_slope
    
    N = 10
    xvec = np.linspace(-dx*N, dx*N, N*2 + 1)
    X,Y = np.meshgrid(xvec,xvec,indexing='ij')
    phi = a * np.exp(-(X**2 + Y**2)/2./b**2)
    X = np.array([X.flatten(),Y.flatten()]).T
    
    phi += a*noise*np.random.normal(size=phi.shape)
    phi = phi.flatten()
    if sample != 0:
        mask = np.random.choice(phi.size,size=min(sample,phi.size),replace=False)
        return X[mask,:],phi[mask]
    return X,phi

def generate_data_nonaliased_nonsquare(noise=0.,sample=100):
    """Generate Gaussian bump in phase.
    noise : float
        amount of gaussian noise to add as fraction of peak height
    sample : int
        number to sample
    """
    #max gradient at a
    assert sample > 0
    
    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))

    a = 2*np.pi/dx/(2*(np.cos(2.*3/8*np.pi) - np.sin(2.*3/8*np.pi)))
    
    #in dx want max_slope*dx < np.pi (nyquist limit)
    
    
    X = np.random.uniform(low=-np.pi,high=np.pi,size=(sample,2))
    
    phi = a * (np.sin(2*X[:,0]) + np.cos(2*X[:,1])) 
    
    phi += a*noise*np.random.normal(size=phi.shape)

    return X,phi

def generate_data_nonaliased_nonsquare(noise=0.,sample=100,a_base=0.1):
    """Generate Gaussian bump in phase.
    noise : float
        amount of gaussian noise to add as fraction of peak height
    sample : int
        number to sample
    """
    #max gradient at a
    assert sample > 0
    
    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))

    a = a_base*np.pi/dx
    
    #in dx want max_slope*dx < np.pi (nyquist limit)
    
    
    X = np.random.uniform(low=0,high=1,size=(sample,2))
    
    phi = a * X[:,0] + a * X[:,1]
    
    phi += a*noise*np.random.normal(size=phi.shape)
    
    phi [sample >> 1] += np.pi

    return X,phi

def generate_data_nonaliased_nonsquare_many(noise=0.,sample=100,batch_size=100):
    """Generate fake (num_direction, cumulant, indepdenent)"""
    #max gradient at a
    assert sample > 0
    
    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))

    a = np.pi/dx*np.ones([1,batch_size])*5#*np.random.normal(size=[1,batch_size])
    
    #in dx want max_slope*dx < np.pi (nyquist limit)
    
    
    X = np.random.uniform(low=0,high=1,size=(sample,2))
    
    phi = a * X[:,0,None] + a * X[:,1,None]
    
    phi += a*noise*np.random.normal(size=phi.shape)
    
    #phi [sample >> 1,:,:] += np.pi

    return X,phi

def plot_phase(X,phi,label=None,figname=None):
    """Plot the phase.
    X : array (num_points, 2)
        The coords
    phi : array (num_points,)
        The phases
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:,0],X[:,1],phi,c=np.angle(np.exp(1j*phi)),cmap=cmocean.cm.phase,s=10,vmin=-np.pi,vmax=np.pi,label=label or "")
    plt.colorbar(sc)
    if label is not None:
        plt.legend(frameon=False)
    if figname is not None:
        plt.savefig(figname)
        plt.close("all")
    else:
        plt.show()

def test_phase_unwrap():
    #X,phi = generate_data_nonaliased_nonsquare(0.1,sample=100)
    X,phi = generate_data_nonaliased_nonsquare_many(noise=0.05,sample=42,batch_size=1)
    phi_wrap = np.angle(np.exp(1j*phi))
    real_mask = np.sum(np.abs(phi_wrap - phi) > np.pi/2.,axis=0)
#     graph = tf.Graph()
#     with tf.Session(graph = graph) as sess:
#         pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, sess = sess, graph = graph)
#         c_mask = pu.maybe_unwrap(phi_wrap)
        
    graph = tf.Graph()
    pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, graph = graph)
    pu._numpy_unwrap(phi_wrap,np.ones_like(phi_wrap))
    pu.plot_triplets()
    pu.plot_edge_dist()
    with tf.Session(graph = pu.graph) as sess:
        f_rec = pu.phase_unwrap(sess,phi_wrap,phi_mask=None)

    plot_phase(X,phi_wrap[:,0],label='phi_wrap',figname=None)
    plot_phase(X,f_rec[:,0],label='f_rec',figname=None)
    plot_phase(X,phi[:,0],label='true',figname=None)
    plot_phase(X,(f_rec-phi)[:,0],label='f_rec - true',figname=None)
    plot_phase(X,(f_rec-np.angle(np.exp(1j*f_rec)))[:,0]/(2*np.pi),label='jumps',figname=None)
    plot_phase(X,(phi-phi_wrap)[:,0]/(2*np.pi),label='true jumps',figname=None)

def unwrap_script(datapack,save_file,ant_idx=-1,time_idx=-1,freq_idx=-1):
    datapack = DataPack(filename=datapack)
    reject_list = ['CS001HBA0']
    datapack_smooth = datapack.clone()
    directions, patch_names = datapack.get_directions(-1)
    times,timestamps = datapack.get_times(time_idx)
    antennas,antenna_labels = datapack.get_antennas(ant_idx)
    freqs = datapack.get_freqs(freq_idx)
    if ant_idx is -1:
        ant_idx = range(len(antennas))
    if time_idx is -1:
        time_idx = range(len(times))
    if freq_idx is -1:
        freq_idx = range(len(freqs))
    phase = datapack.get_phase(ant_idx,time_idx,-1,freq_idx)
    phase = np.angle(np.exp(1j*phase))
    phase_smooth = phase.copy()#np.angle(1j*phase)
    variance_smooth = datapack.get_variance(ant_idx,time_idx,-1,freq_idx)
    Na,Nt,Nd,Nf = phase.shape
    uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
    dirs_uvw = directions.transform_to(uvw)
    X = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
                 np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
    
    graph = tf.Graph()
    pu = PhaseUnwrap(X, (Nd,Nt), redundancy=2, graph = graph)
    pu.plot_triplets()
    with tf.Session(graph = pu.graph) as sess:
        for i in range(Na):
            if antenna_labels[i] in reject_list:
                continue
            print("Working on antenna {}".format(antenna_labels[i]))
            for l in range(Nf):
                print("Working on freq {}".format(freqs[l]))
                phi_wrap = phase[i,:,:,l].transpose()
                f_rec = pu.phase_unwrap(sess,phi_wrap,phi_mask=None).transpose()
                stuck_directions = pu.stuck_directions(sess, f_rec)
                phase_smooth[i,:,:,l] = f_rec
                variance_smooth[i,:,:,l][stuck_directions.T] = -2*np.pi
                K = f_rec - phase[i,:,:,l]
                ##Difference applied to rest of freqs
                phase[i,:,:,l+1:] += K[:,:,None]
            datapack_smooth.set_phase(phase_smooth[i,:,:,:],ant_idx=[ant_idx[i]],time_idx=time_idx,dir_idx=-1,freq_idx=freq_idx)
            datapack_smooth.set_variance(variance_smooth[i,:,:,:],ant_idx=[ant_idx[i]],time_idx=time_idx,dir_idx=-1,freq_idx=freq_idx)
            datapack_smooth.save(save_file)

from scipy.special import erf
def outlier_detection(k, y_star, var_star, y_obs, obs_noise = 1.0):
    def _pdf(z):
        return np.exp(-0.5*z**2)/np.sqrt(np.pi*2)
    def _cdf(z):
        return 0.5*(1+erf(z/np.sqrt(2)))
    residual = np.abs(y_obs - y_star)
    
    error_star = np.sqrt(var_star)
    z = (k*obs_noise - residual)/error_star
    prob = 1 - _cdf(z)
    return prob

def presmooth_outlier(phase,error,k=2):
    mean = np.mean(phase,axis=2,keepdims=True)
    std = np.std(phase,axis=2,keepdims=True)
    return np.abs(mean-phase) > k*std
    prob_mask = outlier_detection(k,mean, std**2,phase,error)
    return prob_mask

def calibrate_presmooth(datapack,ant_idx=-1,time_idx=-1,freq_idx=-1):
    datapack = DataPack(filename=datapack)
    reject_list = ['CS001HBA0']
    directions, patch_names = datapack.get_directions(-1)
    times,timestamps = datapack.get_times(time_idx)
    antennas,antenna_labels = datapack.get_antennas(ant_idx)
    freqs = datapack.get_freqs(freq_idx)
    phase = datapack.get_phase(ant_idx,time_idx,-1,freq_idx)
    error = np.sqrt(datapack.get_variance(ant_idx,time_idx,-1,freq_idx))
    plot_datapack(datapack,ant_idx,time_idx,-1,freq_idx,phase_wrap=True,observable='phase',
                     plot_facet_idx=True,plot_crosses=False)
    variance_smooth = error**2
    uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
    dirs_uvw = directions.transform_to(uvw)
    d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
                 np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
    spatial_scale = np.std(d)
    d /= spatial_scale
    X = d
    Y = phase
    with  gp.defer_build():
        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.1])
        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.25])
        #+gp.kernels.White(1,variance=0.01)
    #     k.lengthscales.transform = gp.transforms.Logistic(0.1,1)
        #white = gp.kernels.White(1,variance=0.01)
        #white.variance.set_trainable(False)
        kern = k_space*k_time
        mean = gp.mean_functions.Zero()
        m = gp.models.GPR(X, Y, kern, mean_function=mean,var=var)
        m.compile()
        print(o.minimize(m,maxiter=1000))
        print(m)
        ystar,varstar = m.predict_f(X)
        
    for k in np.linspace(2.0,2.5,10):
        
        print(k)
        prob_mask = presmooth_outlier(phase,error,k=k)
        datapack.set_variance(prob_mask,ant_idx,time_idx,-1,freq_idx)
        plot_datapack(datapack,ant_idx,time_idx,-1,freq_idx,phase_wrap=False,observable='variance',
                     plot_facet_idx=True,plot_crosses=False)
    

def smooth_script(datapack, save_file, param_file=None, ant_idx=-1,time_idx=-1,freq_idx=-1):
    """Script to smooth a datapack"""
    datapack = DataPack(filename=datapack)
    reject_list = ['CS001HBA0']
    datapack_smooth = datapack.clone()
    directions, patch_names = datapack.get_directions(-1)
    times,timestamps = datapack.get_times(time_idx)
    antennas,antenna_labels = datapack.get_antennas(ant_idx)
    freqs = datapack.get_freqs(freq_idx)
    if ant_idx is -1:
        ant_idx = range(len(antennas))
    if time_idx is -1:
        time_idx = range(len(times))
    if freq_idx is -1:
        freq_idx = range(len(freqs))
    if param_file is None:
        param_file = 'param_file_{}_{}'.format(time_idx[0],time_idx[-1])
    phase = datapack.get_phase(ant_idx,time_idx,-1,freq_idx)
    error = np.sqrt(datapack.get_variance(ant_idx,time_idx,-1,freq_idx))
    variance_smooth = error**2
    
#     ###
#     # pre-outlier detection (which facets failed to unwrap or are large errors)
#     mean = np.mean(phase,axis=2,keepdims=True)
#     std = np.std(phase,axis=2,keepdims=True)
#     mask = np.abs(phase-mean) > 3*std
#     prob_mask = outlier_detection(2,mean, std**2,phase,error)
    
#     error[mask] = 2*np.pi # not contributing much 1 radian error
    phase_smooth = phase.copy()#np.angle(1j*phase)
    Na,Nt,Nd,Nf = phase.shape
    uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
    dirs_uvw = directions.transform_to(uvw)
    d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
                 np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
    spatial_scale = np.std(d)
    d /= spatial_scale
    
    t = times.gps
    t -= np.mean(t)
    time_scale = np.std(t)
    t /= time_scale
    
    X = np.zeros([Nt,Nd,3],dtype=np.float64)
    for j in range(Nt):
        for k in range(Nd):
            X[j,k,0:2] = d[k,:]
            X[j,k,2] = t[j]   
    X = np.reshape(X,(Nt*Nd,3))
    data = {'antenna_labels':antenna_labels, 
            'freqs':freqs, 
            'length_scale':np.zeros([Na,Nt,Nf]), 
            'kernel_variance':np.zeros([Na,Nt,Nf]),
            'time_scale':np.zeros([Na,Nt,Nf]), 
            'variance_scale':np.zeros([Na,Nt,Nf])}
    
    o = gp.train.ScipyOptimizer(method='BFGS')
    with  gp.defer_build():
        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.1])
        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.25])
        #+gp.kernels.White(1,variance=0.01)
    #     k.lengthscales.transform = gp.transforms.Logistic(0.1,1)
        #white = gp.kernels.White(1,variance=0.01)
        #white.variance.set_trainable(False)
        kern = k_space*k_time
        mean = gp.mean_functions.Zero()
    #     m = gp.models.GPR(X, Y, kern=kern, mean_function=mean)
    #     m.likelihood = TorusGaussian()
        for i, ai in enumerate(ant_idx):
            if antenna_labels[i] in reject_list:
                continue
            print("Working on antenna {}".format(antenna_labels[i]))
            for l, fi in enumerate(freq_idx):
                print("Working on freq {}".format(freqs[l]))
                Y = phase[i,:,:,l].flatten()[:,None]
                y_mean = np.mean(Y)
                y_scale = np.std(Y)
                Y -= y_mean
                Y /= y_scale
                var = (error[i,:,:,l].flatten()/y_scale)**2
                var[error[i,:,:,l].flatten() < 0] = 100.
                
                
                m = gp.models.GPR(X, Y, kern, mean_function=mean,var=var)
                m.likelihood.variance.set_trainable(False)
                m.compile()
                print(o.minimize(m,maxiter=1000))
                print(m)
                data['length_scale'][i,:,l] = m.kern.rbf_1.lengthscales.value[0]*spatial_scale
                data['time_scale'][i,:,l] = m.kern.rbf_2.lengthscales.value[0]*time_scale
                data['kernel_variance'][i,:,l] = m.kern.rbf_1.variance.value*m.kern.rbf_2.variance.value*y_scale**2
                data['variance_scale'][i,:,l] = m.likelihood.variance.value*y_scale**2
                ystar,varstar = m.predict_f(X)
                
                phase_smooth[i,:,:,l] = (ystar*y_scale + y_mean).reshape((Nt,Nd))
                variance_smooth[i,:,:,l] = (varstar*y_scale**2).reshape((Nt,Nd))
                
                
            with h5py.File(param_file,'w') as pf:
                for key in data:
                    pf[key] = data[key]
            datapack_smooth.set_phase(phase_smooth[i,:,:,:],ant_idx=[ai],time_idx=time_idx,dir_idx=-1,freq_idx=freq_idx)
            datapack_smooth.set_variance(variance_smooth[i,:,:,:],ant_idx=[ai],time_idx=time_idx,dir_idx=-1,freq_idx=freq_idx)
            
            datapack_smooth.save(save_file)
        
    
def animate_datapack(datapack,**kwargs):
    from ionotomo.plotting.plot_tools import animate_datapack
    datapack = DataPack(filename=datapack)
    phase = datapack.get_phase(-1,range(20),-1,-1)
#     Na,Nt,Nd,Nf = phase.shape
#     vmin = np.min(phase,axis=2,keepdims=True)
#     K = np.zeros_like(phase)
#     while np.any(vmin < -np.pi):
#         dK = np.where(vmin < -np.pi, 2*np.pi, 0.)
#         phase += dK
#         K += dK
#         vmin = np.min(phase,axis=2,keepdims=True)
#     vmax = np.max(phase,axis=2,keepdims=True)
#     while np.any(vmax > np.pi):
#         dK = np.where(vmax > np.pi, -2*np.pi, 0.)
#         phase += dK
#         K += dK
#         vmax = np.max(phase,axis=2,keepdims=True)
#     import pylab as plt
#     plt.figure(figsize=(16,16))
#     [plt.hist(K[i,...].flatten(),alpha=0.1,label="D{}".format(i)) for i in range(Na)]
#     plt.show()
#     datapack.set_phase(phase,-1,-1,-1,-1)
#     #datapack.save("../data/rvw_datapack_full_phase_dec27_smooth_shianimate_datapackmate_datapackmate_datapackhdf5")
    animate_datapack(datapack,"phase_unwrapped_animation_RS210_std_ex", 
                     ant_idx=-1,time_idx=range(30),dir_idx=-1,num_threads=1,mode='perantenna',observable='std',
                     phase_wrap=True,vmin=None,vmax=None,**kwargs)
    animate_datapack(datapack,"phase_unwrapped_animation_RS210_phase_ex", 
                     ant_idx=-1,time_idx=range(30),dir_idx=-1,num_threads=1,mode='perantenna',observable='phase',
                     phase_wrap=True,vmin=None,vmax=None,**kwargs)


if __name__=='__main__':
#    calibrate_presmooth("../data/rvw_datapack_full_phase_dec27.hdf5",ant_idx=-1,time_idx=[0],freq_idx=-1)
    
#    test_phase_unwrap()
    if len(sys.argv) == 2:
        starting_datapack = sys.argv[1]
    else:
        starting_datapack = "../data/rvw_datapack_full_phase_dec27.hdf5"
    unwrap_script(starting_datapack,starting_datapack.replace('.hdf5','_unwrap.hdf5'),
                 ant_idx=[51],time_idx=range(30),freq_idx=[0])
    smooth_script(starting_datapack.replace('.hdf5','_unwrap.hdf5'),starting_datapack.replace('.hdf5','_unwrap_smooth.hdf5'), 
                  ant_idx=[51],time_idx=range(30),freq_idx=[10])
#     animate_datapack("../data/rvw_datapack_full_phase_dec27_smooth_v2_bayes_ex.hdf5",freq_idx=[10])

