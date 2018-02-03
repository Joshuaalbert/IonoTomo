#import numpy as np
#import tensorflow as tf
#import pylab as plt
#import cmocean
#from scipy.spatial import cKDTree
#from ionotomo.tomography.pipeline import Pipeline
#from ionotomo.settings import TFSettings
#
#
#class PhaseUnwrap(object):
#    """The phase unwrapped object.
#    Unwraps 2D phase as well as cumulatively down one axis and indepedently down another.
#    Args:
#    directions : array (num_directions, 2) the directions in the 2D plane wth phases assigned
#    shape : array or list of int (num_directions, cumulative_dim, independent_dim)
#    redundancy: int (see self._create_triplets)
#    sess : tf.Session object to use or None for own
#    """
#    def __init__(self,directions, shape, redundancy=2, sess = None, graph = None):
#        self.directions = directions
#        self.redundancy = redundancy
#        self.shape = shape
#        self.path, self.triplets = self._create_triplets(self.directions,redundancy=self.redundancy)
#        self.pairs = \
#                np.unique(np.sort(np.concatenate([self.triplets[:,[0,1]],
#                    self.triplets[:,[1,2]], self.triplets[:,[2,0]]],axis=0),axis=1),axis=0)
#        
#        if graph is None:
#            self.graph = tf.Graph()
#        else:
#            self.graph = graph
#        self.pipeline = Pipeline()
#        self.pipeline.add_graph('phase_unwrap',self.graph)
#        if sess is None:
#            self.sess = tf.Session(graph=self.graph)
#            log.info("Remember to call close on object")
#        else:
#            self.sess = sess
#        self.model_scope = "phase_unwrap"
#        with self.graph.as_default():
#            with tf.variable_scope(self.model_scope):
#                phi_wrap_placeholder = \
#                        tf.placeholder(tf.float32,shape=shape,name='phi_wrap')
#                phi_var_placeholder = \
#                        tf.placeholder(tf.float32,shape=shape,name='phi_var')
#                keep_prob_placeholder = tf.placeholder(tf.float32,shape=(),name='keep_prob')
#                ent_w = tf.placeholder(tf.float32,shape=(),name='ent_w')
#                learning_rate_placeholder = tf.placeholder(tf.float32,shape=(),name='learning_rate')
#                
#                phase_unwrap_op, losses, K_dist = self._build_phase_unwrap(phi_wrap_placeholder,phi_var_placeholder,keep_prob_placeholder,learning_rate_placeholder,ent_w)
#                self.pipeline.store_ops([ent_w,phi_wrap_placeholder,phi_var_placeholder, phase_unwrap_op, losses, K_dist, keep_prob_placeholder, learning_rate_placeholder],
#                                        ["ent_w","phi_wrap","phi_var","phase_unwrap","losses","K_dist","keep_prob","learning_rate"], 
#                                        self.model_scope)
#
#    def phase_unwrap(self, phi_wrap, phi_var = None):
#        """Run the simulation for current model"""
#        if len(phi_wrap.shape) == 1:
#            phi_wrap = phi_wrap[:,None,None]
#        if phi_var is None:
#            phi_var = np.ones_like(phi_wrap)
#        if len(phi_var.shape) == 1:
#            phi_var = phi_var[:,None,None]
#        with self.graph.as_default():
#            ent_w, phi_wrap_placeholder,phi_var_placeholder, phase_unwrap_op, losses, K_dist,keep_prob_placeholder, learning_rate_placeholder = \
#                    self.pipeline.grab_ops(["ent_w","phi_wrap","phi_var","phase_unwrap","losses","K_dist","keep_prob","learning_rate"], self.model_scope)
#            #self.sess.run(tf.global_variables_initializer())
#            self.pipeline.initialize_graph(self.sess)
#            loss_sum = np.inf
#            for epoch in range(25000):
#                lr = 0.1
#                dp = 0.2
#                ew = 0.
#                if epoch > 1000:
#                    lr = 0.1
#                    dp = 0.3
#                if epoch > 5000:
#                    lr = 0.05
#                    dp = 0.3
#                if epoch > 10000:
#                    lr = 0.03
#                    dp = 0.5
#                if epoch > 15000:
#                    lr = 0.01
#                    dp = 0.5
#                if epoch > 20000:
#                    lr = 0.001
#                    dp = 0.8
#                if loss_sum < 1.:
#                    ew = 1e-6
#                if loss_sum < 0.5:
#                    ew = 1e-4
#                if loss_sum < 0.1:
#                    ew = 1e-2
#                _, losses_val,K_dist_val = self.sess.run([phase_unwrap_op,losses,K_dist],
#                        feed_dict={phi_wrap_placeholder:phi_wrap,
#                phi_var_placeholder:phi_var,keep_prob_placeholder:dp,
#                learning_rate_placeholder:lr,
#                                  ent_w:ew})
#    
#                if np.sum(losses_val) < 0.05 or (epoch + 1) % 1000 == 0:
#                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_val),*losses_val))
#                if np.sum(losses_val) < 0.05:
#                    break
#                         
#            f_rec = np.zeros_like(phi_wrap)
#            f_rec[self.path[0][0],:,:] = phi_wrap[self.path[0][0],:,:]
#            K_cum = np.cumsum((np.argmax(K_dist_val,axis=-1)-2)*2*np.pi,axis=1)
#            for i,p in enumerate(self.path):
#                df = phi_wrap[p[1],:,:] - phi_wrap[p[0],:,:] + K_cum[p[1],:,:] - K_cum[p[0],:,:]
#                f_rec[p[1],:,:] = f_rec[p[0],:,:] + df
#        return f_rec
#
#
#    def _create_triplets(self,X,redundancy=2):
#        """Create the path of closed small triplets.
#        Args:
#            X : array (N,2) the coordinates in image
#            redundancy : int the number of unique triplets 
#            to makes for each segment of path.
#        """
#        kt = cKDTree(X)
#        #get center of map
#        C = np.mean(X,axis=0)
#        _,idx0 = kt.query(C,k=1)
#        #define unique path
#        dist, idx = kt.query(X[idx0,:],k=2)
#        path = [(idx0, idx[1])]
#        included = [idx0, idx[1]]
#        while len(included) < X.shape[0]:
#            dist,idx = kt.query(X[included,:],k = len(included)+1)
#            mask = np.where(np.isin(idx,included,invert=True))
#            argmin = np.argmin(dist[mask])
#            idx_from = included[mask[0][argmin]]
#            idx_to = idx[mask[0][argmin]][mask[1][argmin]]
#            path.append((idx_from,idx_to))
#            included.append(idx_to)
#
#        M = np.mean(X[path,:],axis=1)
#        _,idx = kt.query(M,k=2 + redundancy)
#        triplets = []
#        for i,p in enumerate(path):
#            count = 0
#            for c in range(2 + redundancy):
#                if idx[i][c] not in p:
#                    triplets.append(p + (idx[i][c],))
#                    count += 1
#                    if count == redundancy:
#                        break
#        triplets = np.sort(triplets,axis=1)
#        triplets = np.unique(triplets,axis=0)
#        return path,triplets
#
#
#    def _build_phase_unwrap(self, phi_wrap_placeholder,phi_var_placeholder,keep_prob_placeholder,learning_rate_placeholder,ent_w):
#        with self.graph.as_default():
#            with tf.name_scope("unwrapper"):
#                def _wrap(a):
#                    return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,tf.complex64))),tf.float32)
#
#                triplets = self.pipeline.add_variable("triplets",init_value=self.triplets,
#                        dtype=tf.int32,trainable=False)
#                pairs = self.pipeline.add_variable("pairs",init_value=self.pairs,
#                        dtype=tf.int32,trainable=False)
#
#                K_logits = tf.get_variable("K",shape=self.shape + (5,),
#                        dtype=tf.float32,initializer=tf.zeros_initializer)
#                K_dist = tf.nn.softmax(K_logits)        
#
#                indices = tf.constant((np.arange(5)-2.)[None,None,None,:],dtype=tf.float32)
#                K = tf.reduce_sum(K_dist*indices,axis=-1)*2*np.pi
#                entropy = - tf.reduce_mean(tf.reduce_sum(K_dist*tf.log(K_dist),axis=-1))
#
#                f_noise = tf.get_variable("f_noise",shape=self.shape,dtype=tf.float32,initializer=tf.zeros_initializer)
#                K_cum = K#tf.cumsum(K,axis=1)
#                f = phi_wrap_placeholder + K_cum + f_noise
#                
#                df = tf.gather(f,pairs[:,1]) - tf.gather(f,pairs[:,0])
#                consistency = tf.sqrt(1.+tf.square(_wrap(tf.gather(phi_wrap_placeholder,pairs[:,1]) - tf.gather(phi_wrap_placeholder,pairs[:,0])) - df)) - 1.
#                consistency = tf.nn.dropout(consistency,keep_prob_placeholder)
#                loss_lse = tf.reduce_mean(consistency)
#                loss_tv = tf.reduce_mean(tf.square(f_noise)/phi_var_placeholder)
#                
#                Wf = _wrap(f)
#
#                df01 = tf.gather(Wf,triplets[:,1]) - tf.gather(Wf,triplets[:,0])
#                df01 = _wrap(df01) 
#                df12 = tf.gather(Wf,triplets[:,2]) - tf.gather(Wf,triplets[:,1])
#                df12 = _wrap(df12)
#                df20 = tf.gather(Wf,triplets[:,0]) - tf.gather(Wf,triplets[:,2])
#                df20 = _wrap(df20)
#
#                residue = tf.sqrt(1. + tf.square(df01 + df12 + df20))-1.
#                residue = tf.nn.dropout(residue,keep_prob_placeholder)
#                loss_residue = tf.reduce_mean(residue)
#
#                
#                opt = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
#                train_op = opt.minimize(loss_lse+ent_w*entropy+loss_residue+loss_tv)
#
#                losses = [loss_lse ,loss_residue,entropy,loss_tv]
#                return train_op, losses, K_dist
#
#    def close(self):
#        self.sess.close()
#    def plot_triplets(self,figname=None):
#        fig = plt.figure(figsize=(8,8))
#        for i,j,k in self.triplets:
#            plt.plot([self.directions[i,0],self.directions[j,0],self.directions[k,0],
#                self.directions[i,0]],[self.directions[i,1],self.directions[j,1],
#                    self.directions[k,1],self.directions[i,1]])
#        if figname is not None:
#            plt.savefig(figname)
#            plt.close("all")
#        else:
#            plt.show()
#    def plot_edge_dist(self,):
#        dist = np.linalg.norm(np.concatenate([self.directions[self.triplets[:,1],:]\
#                - self.directions[self.triplets[:,0],:], self.directions[self.triplets[:,2],:]\
#                - self.directions[self.triplets[:,1],:],self.directions[self.triplets[:,0],:]\
#                - self.directions[self.triplets[:,2],:]],axis=0),axis=1)
#        plt.hist(dist,bins=20)
#        plt.show()
#
#    
#def generate_data_aliased(noise=0.,sample=100):
#    """Generate Gaussian bump in phase.
#    noise : float
#        amount of gaussian noise to add as fraction of peak height
#    sample : int
#        number to sample
#    """
#    #max gradient at b
#    a = 50
#    b = 1
#    max_slope = np.abs(a/np.sqrt(np.exp(1))/b)
#    
#    #in dx want max_slope*dx > np.pi
#    dx = 1.5*np.pi/max_slope
#    
#    N = 10
#    xvec = np.linspace(-dx*N, dx*N, N*2 + 1)
#    X,Y = np.meshgrid(xvec,xvec,indexing='ij')
#    phi = a * np.exp(-(X**2 + Y**2)/2./b**2)
#    X = np.array([X.flatten(),Y.flatten()]).T
#    
#    phi += a*noise*np.random.normal(size=phi.shape)
#    phi = phi.flatten()
#    if sample != 0:
#        mask = np.random.choice(phi.size,size=min(sample,phi.size),replace=False)
#        return X[mask,:],phi[mask]
#    return X,phi
#
#def generate_data_nonaliased(noise=0.,sample=100):
#    """Generate Gaussian bump in phase.
#    noise : float
#        amount of gaussian noise to add as fraction of peak height
#    sample : int
#        number to sample
#    """
#    #max gradient at b
#    a = 15
#    b = 1
#    max_slope = np.abs(a/np.sqrt(np.exp(1))/b)
#    
#    #in dx want max_slope*dx < np.pi
#    dx = 0.5*np.pi/max_slope
#    
#    N = 10
#    xvec = np.linspace(-dx*N, dx*N, N*2 + 1)
#    X,Y = np.meshgrid(xvec,xvec,indexing='ij')
#    phi = a * np.exp(-(X**2 + Y**2)/2./b**2)
#    X = np.array([X.flatten(),Y.flatten()]).T
#    
#    phi += a*noise*np.random.normal(size=phi.shape)
#    phi = phi.flatten()
#    if sample != 0:
#        mask = np.random.choice(phi.size,size=min(sample,phi.size),replace=False)
#        return X[mask,:],phi[mask]
#    return X,phi
#
#def generate_data_nonaliased_nonsquare(noise=0.,sample=100):
#    """Generate Gaussian bump in phase.
#    noise : float
#        amount of gaussian noise to add as fraction of peak height
#    sample : int
#        number to sample
#    """
#    #max gradient at a
#    assert sample > 0
#    
#    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))
#
#    a = 2*np.pi/dx/(2*(np.cos(2.*3/8*np.pi) - np.sin(2.*3/8*np.pi)))
#    
#    #in dx want max_slope*dx < np.pi (nyquist limit)
#    
#    
#    X = np.random.uniform(low=-np.pi,high=np.pi,size=(sample,2))
#    
#    phi = a * (np.sin(2*X[:,0]) + np.cos(2*X[:,1])) 
#    
#    phi += a*noise*np.random.normal(size=phi.shape)
#
#    return X,phi
#
#def generate_data_nonaliased_nonsquare(noise=0.,sample=100,a_base=0.1):
#    """Generate Gaussian bump in phase.
#    noise : float
#        amount of gaussian noise to add as fraction of peak height
#    sample : int
#        number to sample
#    """
#    #max gradient at a
#    assert sample > 0
#    
#    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))
#
#    a = a_base*np.pi/dx
#    
#    #in dx want max_slope*dx < np.pi (nyquist limit)
#    
#    
#    X = np.random.uniform(low=0,high=1,size=(sample,2))
#    
#    phi = a * X[:,0] + a * X[:,1]
#    
#    phi += a*noise*np.random.normal(size=phi.shape)
#    
#    phi [sample >> 1] += np.pi
#
#    return X,phi
#
#def generate_data_nonaliased_nonsquare_many(noise=0.,sample=100,n_cum=2,n_ind=3):
#    """Generate fake (num_direction, cumulant, indepdenent)"""
#    #max gradient at a
#    assert sample > 0
#    
#    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))
#
#    a = np.pi/dx*np.random.normal(size=[1,n_cum,n_ind])*2
#    
#    #in dx want max_slope*dx < np.pi (nyquist limit)
#    
#    
#    X = np.random.uniform(low=0,high=1,size=(sample,2))
#    
#    phi = a * X[:,0,None,None] + a * X[:,1,None,None]
#    
#    phi += a*noise*np.random.normal(size=phi.shape)
#    
#    #phi [sample >> 1,:,:] += np.pi
#
#    return X,phi
#
#def plot_phase(X,phi,label=None,figname=None):
#    """Plot the phase.
#    X : array (num_points, 2)
#        The coords
#    phi : array (num_points,)
#        The phases
#    """
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_subplot(111, projection='3d')
#    sc = ax.scatter(X[:,0],X[:,1],phi,c=np.angle(np.exp(1j*phi)),cmap=cmocean.cm.phase,s=10,vmin=-np.pi,vmax=np.pi,label=label or "")
#    plt.colorbar(sc)
#    if label is not None:
#        plt.legend(frameon=False)
#    if figname is not None:
#        plt.savefig(figname)
#        plt.close("all")
#    else:
#        plt.show()
#
#
#
#
#def test_phase_unwrap():
#    #X,phi = generate_data_nonaliased_nonsquare(0.1,sample=100)
#    X,phi = generate_data_nonaliased_nonsquare_many(noise=0.05,sample=100,n_cum=1,n_ind=100)
#    phi_wrap = np.angle(np.exp(1j*phi))
#    graph = tf.Graph()
#    with tf.Session(graph = graph) as sess:
#        pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, sess = sess, graph = graph)
#        f_rec = pu.phase_unwrap(phi_wrap)
#
#    plot_phase(X,phi_wrap[:,0,0],label='phi_wrap',figname=None)
#    plot_phase(X,f_rec[:,0,0],label='f_rec',figname=None)
#    plot_phase(X,phi[:,0,0],label='true',figname=None)
#    plot_phase(X,(f_rec-phi)[:,0,0],label='f_rec - true',figname=None)
#    plot_phase(X,(f_rec-np.angle(np.exp(1j*f_rec)))[:,0,0]/(2*np.pi),label='jumps',figname=None)
#    plot_phase(X,(phi-phi_wrap)[:,0,0]/(2*np.pi),label='true jumps',figname=None)
#
#if __name__=='__main__':
#    test_phase_unwrap()

import numpy as np
import tensorflow as tf
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings
from timeit import default_timer


class PhaseUnwrap(object):
    """The phase unwrapped object.
    Unwraps 2D phase as well as cumulatively down one axis and indepedently down another.
    Args:
    directions : array (num_directions, 2) the directions in the 2D plane wth phases assigned
    shape : array or list of int (num_directions, cumulative_dim, independent_dim)
    redundancy: int (see self._create_triplets)
    sess : tf.Session object to use or None for own
    """
    def __init__(self,directions, shape, redundancy=2, sess = None, graph = None):
        self.directions = directions
        self.redundancy = redundancy
        self.shape = tuple(shape)
        self.path, self.triplets,self.adjoining_map = self._create_triplets(self.directions,redundancy=self.redundancy, adjoining=True)
        self.pairs = \
                np.unique(np.sort(np.concatenate([self.triplets[:,[0,1]],
                    self.triplets[:,[1,2]], self.triplets[:,[2,0]]],axis=0),axis=1),axis=0)
        
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph
        self.pipeline = Pipeline()
        self.pipeline.add_graph('phase_unwrap',self.graph)
        if sess is None:
            self.sess = tf.Session(graph=self.graph)
            log.info("Remember to call close on object")
        else:
            self.sess = sess
        self.model_scope = "phase_unwrap"
        with self.graph.as_default():
            with tf.variable_scope(self.model_scope):
                phi_wrap_placeholder = \
                        tf.placeholder(tf.float32,shape=shape,name='phi_wrap')
                phi_mask_placeholder = \
                        tf.placeholder(tf.float32,shape=shape[1:],name='phi_mask')
                keep_prob_placeholder = tf.placeholder(tf.float32,shape=(),name='keep_prob')
                ent_w = tf.placeholder(tf.float32,shape=(),name='ent_w')
                learning_rate_placeholder = tf.placeholder(tf.float32,shape=(),name='learning_rate')
                directions = self.pipeline.add_variable(name='directions',trainable=False,init_value=self.directions,dtype=tf.float32)
                adjoining_map = self.pipeline.add_variable(name='adjoining_map',trainable=False,init_value=self.adjoining_map,dtype=tf.int32)
                
                phase_unwrap_op, losses, K_dist = self._build_phase_unwrap(phi_wrap_placeholder,phi_mask_placeholder,keep_prob_placeholder,learning_rate_placeholder,ent_w,
                                                                          directions,adjoining_map)
                self.pipeline.store_ops([adjoining_map,directions,ent_w,phi_wrap_placeholder,phi_mask_placeholder, phase_unwrap_op, losses, K_dist, keep_prob_placeholder, learning_rate_placeholder],
                                        ["adjoining_map","directions","ent_w","phi_wrap","phi_mask","phase_unwrap","losses","K_dist","keep_prob","learning_rate"], 
                                        self.model_scope)
    def maybe_unwrap(self,phi_wrap):
        """Return a mask of shape (num_cum, num_ind) with True where
        unwrapping needed"""
        with self.graph.as_default():
            phi_wrap_placeholder,phi_mask_placeholder, keep_prob_placeholder = \
                    self.pipeline.grab_ops(["phi_wrap","phi_mask","keep_prob"], self.model_scope)
            #self.sess.run(tf.global_variables_initializer())
            self.pipeline.initialize_graph(self.sess)
            K_zero_logits = np.zeros(self.shape+(5,),dtype=np.float32)
            K_zero_logits[...,2] += 1000
#             K_zero_logits = np.zeros(self.shape,dtype=np.float32)
            
            _ = self.sess.run(self.zero_logits_op,feed_dict={self.K_zero_logits_placeholder: K_zero_logits})

            dp = 1.
            consistency = self.sess.run(self.consistency,
                    feed_dict={phi_wrap_placeholder:phi_wrap,
            phi_mask_placeholder:np.ones(self.shape[1:]),
                               keep_prob_placeholder:dp})
            return (np.sum(consistency>0.1,axis=0) > 0).astype(np.bool)
        
    def phase_unwrap(self, phi_wrap, phi_mask = None):
        """Run the simulation for current model"""
        if len(phi_wrap.shape) == 1:
            phi_wrap = phi_wrap[:,None,None]
        if phi_mask is None:
            phi_mask = self.maybe_unwrap(phi_wrap)
            print(phi_mask)
        #self._numpy_unwrap(phi_wrap, phi_mask)
        with self.graph.as_default():
            ent_w, phi_wrap_placeholder,phi_mask_placeholder, phase_unwrap_op, losses, K_dist,keep_prob_placeholder, learning_rate_placeholder = \
                    self.pipeline.grab_ops(["ent_w","phi_wrap","phi_mask","phase_unwrap","losses","K_dist","keep_prob","learning_rate"], self.model_scope)
            #self.sess.run(tf.global_variables_initializer())
            self.pipeline.initialize_graph(self.sess)
            loss_sum = np.inf
            last_residue = np.inf
            last_lse = np.inf
            last_ent = np.inf
            t0 = default_timer()-10.
            for epoch in range(1000):
                lr = 0.1
                dp = 0.5
                ew = 1e-5
                if last_lse < 0.5 and last_residue < 0.5:
                    ew = 1e-4
                    lr = 0.1
                    dp = 0.5
                if last_lse < 0.3 and last_residue < 0.3:
                    ew = 1e-3
                    lr = 0.06
                    dp = 0.5
                if last_lse < 0.15 and last_residue < 0.15:
                    ew = 5e-2
                    lr = 0.03
                    dp = 1.
                if last_lse < 0.05 and last_residue < 0.05:
                    ew = 1e-1
                    lr = 0.01
                    dp = 1.
                
                _, losses_val,K_dist_val = self.sess.run([phase_unwrap_op,losses,K_dist],
                        feed_dict={phi_wrap_placeholder:phi_wrap,
                phi_mask_placeholder:phi_mask,keep_prob_placeholder:dp,
                learning_rate_placeholder:lr,
                                  ent_w:ew})
                #print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_val),*losses_val))
                losses_val = [np.max(l) for l in losses_val]
                last_lse = np.max(losses_val[0])
                last_residue = np.max(losses_val[1])
                last_ent = np.max(losses_val[2])
                
                if np.sum(losses_val) < 0.05 or (epoch + 1) % 1000 == 0 or default_timer() - t0 > 10.:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f}".format(epoch,np.sum(losses_val),*losses_val))
                    t0 = default_timer()
                    
                if last_lse + last_residue+last_ent < 0.1:
                    #np.sum(losses_val) < 0.05:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f}".format(epoch,np.sum(losses_val),*losses_val))
                    
                    break
                         
            f_rec = np.zeros_like(phi_wrap)
            f_rec[self.path[0][0],:,:] = phi_wrap[self.path[0][0],:,:]
            #K_cum = np.cumsum((np.argmax(K_dist_val,axis=-1)-2)*2*np.pi,axis=1)
            K_cum = (np.argmax(K_dist_val,axis=-1)-2)*2*np.pi
            for i,p in enumerate(self.path):
                df = phi_wrap[p[1],:,:] - phi_wrap[p[0],:,:] + K_cum[p[1],:,:] - K_cum[p[0],:,:]
                f_rec[p[1],:,:] = f_rec[p[0],:,:] + df
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
            
                           
                

    def _build_phase_unwrap(self, phi_wrap_placeholder,phi_mask_placeholder,
                            keep_prob_placeholder,learning_rate_placeholder,ent_w,
                            directions,adjoining_map):
        with self.graph.as_default():
            with tf.name_scope("unwrapper"):
                def _wrap(a):
                    return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,tf.complex64))),tf.float32)

                triplets = self.pipeline.add_variable("triplets",init_value=self.triplets,
                        dtype=tf.int32,trainable=False)
                pairs = self.pipeline.add_variable("pairs",init_value=self.pairs,
                        dtype=tf.int32,trainable=False)
                
                ###
                # new
                ###
                
                # Each pair forms a segment (also triplets work)
                df = tf.gather(phi_wrap_placeholder,pairs[:,1]) - tf.gather(phi_wrap_placeholder,pairs[:,0])
                
                def trip_arc(i,j):
                    return _wrap(tf.gather(phi_wrap_placeholder,triplets[:,j]) - tf.gather(phi_wrap_placeholder,triplets[:,i]))
                df = trip_arc(0,1)
                
                
                
                ###

                K_logits = tf.get_variable("K",shape=self.shape + (5,),
                        dtype=tf.float32,initializer=tf.zeros_initializer)
                self.K_zero_logits_placeholder = tf.placeholder(shape=self.shape+(5,),dtype=tf.float32)
                self.zero_logits_op = tf.assign(K_logits,self.K_zero_logits_placeholder)
                K_dist = tf.nn.softmax(K_logits)        

                indices = tf.constant((np.arange(5)-2.)[None,None,None,:],dtype=tf.float32)
                K = tf.reduce_sum(K_dist*indices,axis=-1)*2*np.pi
                
#                 K = tf.get_variable("K",shape=self.shape,
#                         dtype=tf.float32,initializer=tf.zeros_initializer)
#                 self.K_zero_logits_placeholder = tf.placeholder(shape=self.shape,dtype=tf.float32)
#                 self.zero_logits_op = tf.assign(K,self.K_zero_logits_placeholder)
#                 diff = tf.square(K[...,None] - tf.constant([-2,-1,0,1,2],dtype=tf.float32))
#                 K_dist = tf.where(tf.equal(diff,tf.reduce_min(diff,axis=-1,keep_dims=True)),tf.ones_like(diff),tf.zeros_like(diff))
                

                f_noise = tf.get_variable("f_noise",shape=self.shape,dtype=tf.float32,initializer=tf.zeros_initializer)
                K_cum = K#tf.cumsum(K,axis=1)
            
            
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
                
                #
                f = phi_wrap_placeholder + K_cum + f_noise
                
                df = tf.gather(f,pairs[:,1]) - tf.gather(f,pairs[:,0])
                self.consistency = tf.sqrt(1.+tf.square(_wrap(tf.gather(phi_wrap_placeholder,pairs[:,1]) - tf.gather(phi_wrap_placeholder,pairs[:,0])) - df)) - 1.
                consistency = tf.nn.dropout(self.consistency,keep_prob_placeholder)
                
                
                Wf = _wrap(f)

                df01_ = tf.gather(Wf,triplets[:,1]) - tf.gather(Wf,triplets[:,0])
                df01 = _wrap(df01_) 
                df12_ = tf.gather(Wf,triplets[:,2]) - tf.gather(Wf,triplets[:,1])
                df12 = _wrap(df12_)
                df20_ = tf.gather(Wf,triplets[:,0]) - tf.gather(Wf,triplets[:,2])
                df20 = _wrap(df20_)
                
#                 loss_tv += tf.reduce_mean(tf.square(tf.gather(f,triplets[:,1]) - tf.gather(f,triplets[:,0]))\
#                                           + tf.square(tf.gather(f,triplets[:,0]) - tf.gather(f,triplets[:,2])))

                self.residue = tf.sqrt(1. + tf.square(df01 + df12 + df20))-1.
                residue = tf.nn.dropout(self.residue,keep_prob_placeholder)
                
                loss_residue = tf.reduce_mean(residue,axis=0)
                loss_lse = tf.reduce_mean(consistency,axis=0)
                loss_tv = tf.reduce_mean(tf.square(f_noise),axis=0)
                entropy = - tf.reduce_mean(tf.reduce_sum(K_dist*tf.log(K_dist+1e-10),axis=-1)*phi_mask_placeholder,axis=0)
                total_loss = loss_lse+loss_residue+loss_tv+ent_w*entropy
                total_loss *= phi_mask_placeholder
                
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
                train_op = opt.minimize(tf.reduce_mean(tf.reduce_mean(total_loss)))
#                 def _cond(i):
#                     return i < tf.constant(self.shape[2],dtype=tf.int32)
#                 def _body(i):
                    
#                     train_op = tf.cond(tf.equal(tf.reduce_sum(phi_mask_placeholder[...,i]),0), lambda: tf.no_op(), lambda: opt.minimize(tf.reduce_mean(total_loss[...,i])))
#                     with tf.control_dependencies([train_op]):
#                         next_i = i+1
#                     return next_i
#                 loop_vars = [tf.constant(0,dtype=tf.int32)]
#                 train_op = tf.while_loop(_cond,_body,loop_vars,parallel_iterations=32,back_prop=False)
                
                #train_op = opt.minimize(tf.reduce_mean(total_loss))
                losses = [loss_lse ,loss_residue,entropy,loss_tv]
                #with tf.control_dependencies([train_op]):
                return train_op, losses, K_dist

    def close(self):
        self.sess.close()
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
        dist = np.linalg.norm(np.concatenate([self.directions[self.triplets[:,1],:]\
                - self.directions[self.triplets[:,0],:], self.directions[self.triplets[:,2],:]\
                - self.directions[self.triplets[:,1],:],self.directions[self.triplets[:,0],:]\
                - self.directions[self.triplets[:,2],:]],axis=0),axis=1)
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

def generate_data_nonaliased_nonsquare_many(noise=0.,sample=100,n_cum=2,n_ind=3):
    """Generate fake (num_direction, cumulant, indepdenent)"""
    #max gradient at a
    assert sample > 0
    
    dx = int(np.ceil(2*np.pi/np.sqrt(sample)))

    a = np.pi/dx*np.random.normal(size=[1,n_cum,n_ind])
    
    #in dx want max_slope*dx < np.pi (nyquist limit)
    
    
    X = np.random.uniform(low=0,high=1,size=(sample,2))
    
    phi = a * X[:,0,None,None] + a * X[:,1,None,None]
    
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
    X,phi = generate_data_nonaliased_nonsquare_many(noise=0.05,sample=42,n_cum=1,n_ind=100)
    phi_wrap = np.angle(np.exp(1j*phi))
    real_mask = np.sum(np.abs(phi_wrap - phi) > np.pi/2.,axis=0)
#     graph = tf.Graph()
#     with tf.Session(graph = graph) as sess:
#         pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, sess = sess, graph = graph)
#         c_mask = pu.maybe_unwrap(phi_wrap)
        
    graph = tf.Graph()
    with tf.Session(graph = graph) as sess:
        pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, sess = sess, graph = graph)
        print(pu.adjoining_map)
        pu.plot_triplets()
        #pu._numpy_unwrap(phi_wrap, np.ones_like(phi_wrap))
        
        
        #phi_mask = np.ones([1,100],dtype=np.bool)
        f_rec = pu.phase_unwrap(phi_wrap,phi_mask=None)

    plot_phase(X,phi_wrap[:,0,0],label='phi_wrap',figname=None)
    plot_phase(X,f_rec[:,0,0],label='f_rec',figname=None)
    plot_phase(X,phi[:,0,0],label='true',figname=None)
    plot_phase(X,(f_rec-phi)[:,0,0],label='f_rec - true',figname=None)
    plot_phase(X,(f_rec-np.angle(np.exp(1j*f_rec)))[:,0,0]/(2*np.pi),label='jumps',figname=None)
    plot_phase(X,(phi-phi_wrap)[:,0,0]/(2*np.pi),label='true jumps',figname=None)

if __name__=='__main__':
    test_phase_unwrap()
