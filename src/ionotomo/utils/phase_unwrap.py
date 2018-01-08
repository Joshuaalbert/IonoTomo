import numpy as np
import tensorflow as tf
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings


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
        self.shape = shape
        self.path, self.triplets = self._create_triplets(self.directions,redundancy=self.redundancy)
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
                phi_var_placeholder = \
                        tf.placeholder(tf.float32,shape=shape,name='phi_var')
                keep_prob_placeholder = tf.placeholder(tf.float32,shape=(),name='keep_prob')
                learning_rate_placeholder = tf.placeholder(tf.float32,shape=(),name='learning_rate')
                
                phase_unwrap_op, losses, K_dist = self._build_phase_unwrap(phi_wrap_placeholder,phi_var_placeholder,keep_prob_placeholder,learning_rate_placeholder)
                self.pipeline.initialize_graph(self.sess)
                self.pipeline.store_ops(["phi_wrap","phi_var","phase_unwrap","losses","K_dist","keep_prob","learning_rate"],
                        [phi_wrap_placeholder,phi_var_placeholder, phase_unwrap_op, losses, K_dist, keep_prob_placeholder, learning_rate_placeholder], self.model_scope)

    def phase_unwrap(self, phi_wrap, phi_var = None):
        """Run the simulation for current model"""
        if len(phi_wrap.shape) == 1:
            phi_wrap = phi_wrap[:,None,None]
        if phi_var is None:
            phi_var = np.ones_like(phi_wrap)
        if len(phi_var.shape) == 1:
            phi_var = phi_var[:,None,None]
        with self.graph.as_default():
            phi_wrap_placeholder,phi_var_placeholder, phase_unwrap_op, losses, K_dist,keep_prob_placeholder, learning_rate_placeholder = \
                    self.pipeline.grab_ops(["phi_wrap","phi_var","phase_unwrap","losses","K_dist","keep_prob","learning_rate"], self.model_scope)
            sess.run(tf.global_variables_initializer())
            for epoch in range(25000):
                lr = 0.1
                dp = 0.2
                if epoch > 1000:
                    lr = 0.1
                    dp = 0.3
                if epoch > 5000:
                    lr = 0.05
                    dp = 0.3
                if epoch > 10000:
                    lr = 0.03
                    dp = 0.5
                if epoch > 15000:
                    lr = 0.01
                    dp = 0.5
                if epoch > 20000:
                    lr = 0.001
                    dp = 0.8
                _, losses_val,K_dist_val = self.sess.run([phase_unwrap_op,losses,K_dist],
                        feed_dict={phi_wrap_placeholder:phi_wrap,
                phi_var_placeholder:phi_var,keep_prob_placeholder:dp,
                learning_rate_placeholder:lr})
    
                if np.sum(losses_val) < 0.1 or (epoch + 1) % 1000 == 0:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_val),*losses_val))
                if np.sum(losses_val) < 0.1:
                    break
                         
            f_rec = np.zeros_like(phi_wrap)
            f_rec[self.path[0][0],:,:] = phi_wrap[self.path[0][0],:,:]
            K_cum = np.cumsum((np.argmax(K_dist_val,axis=-1)-2)*2*np.pi,axis=1)
            for i,p in enumerate(self.path):
                df = phi_wrap[p[1],:,:] - phi_wrap[p[0],:,:] + K_cum[p[1],:,:] - K_cum[p[0],:,:]
                f_rec[p[1],:,:] = f_rec[p[0],:,:] + df
        return f_rec


    def _create_triplets(self,X,redundancy=2):
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
                        break
        triplets = np.sort(triplets,axis=1)
        triplets = np.unique(triplets,axis=0)
        return path,triplets


    def _build_phase_unwrap(self, phi_wrap_placeholder,phi_var_placeholder,keep_prob_placeholder,learning_rate_placeholder):
        with self.graph.as_default():
            with tf.name_scope("unwrapper"):
                def _wrap(a):
                    return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,tf.complex64))),tf.float32)

                triplets = self.pipeline.add_variable("triplets",init_value=self.triplets,
                        dtype=tf.int32,trainable=False)
                pairs = self.pipeline.add_variable("pairs",init_value=self.pairs,
                        dtype=tf.int32,trainable=False)

                K_logits = tf.get_variable("K",shape=self.shape + (5,),
                        dtype=tf.float32,initializer=tf.zeros_initializer)
                K_dist = tf.nn.softmax(K_logits)        

                indices = tf.constant((np.arange(5)-2.)[None,None,None,:],dtype=tf.float32)
                K = tf.reduce_sum(K_dist*indices,axis=-1)*2*np.pi
                entropy = - tf.reduce_mean(tf.reduce_sum(K_dist*tf.log(K_dist),axis=-1))

                f_noise = tf.get_variable("f_noise",shape=self.shape,dtype=tf.float32,initializer=tf.zeros_initializer)
                K_cum = tf.cumsum(K,axis=1)
                f = phi_wrap_placeholder + K_cum + f_noise
                
                df = tf.gather(f,pairs[:,1]) - tf.gather(f,pairs[:,0])
                consistency = tf.sqrt(1.+tf.square(_wrap(tf.gather(phi_wrap_placeholder,pairs[:,1]) - tf.gather(phi_wrap_placeholder,pairs[:,0])) - df)) - 1.
                consistency = tf.nn.dropout(consistency,keep_prob_placeholder)
                loss_lse = tf.reduce_mean(consistency)
                loss_tv = tf.reduce_mean(tf.square(f_noise)/phi_var_placeholder)
                
                Wf = _wrap(f)

                df01 = tf.gather(Wf,triplets[:,1]) - tf.gather(Wf,triplets[:,0])
                df01 = _wrap(df01) 
                df12 = tf.gather(Wf,triplets[:,2]) - tf.gather(Wf,triplets[:,1])
                df12 = _wrap(df12)
                df20 = tf.gather(Wf,triplets[:,0]) - tf.gather(Wf,triplets[:,2])
                df20 = _wrap(df20)

                residue = tf.sqrt(1. + tf.square(df01 + df12 + df20))-1.
                residue = tf.nn.dropout(residue,keep_prob_placeholder)
                loss_residue = tf.reduce_mean(residue)

                
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
                train_op = opt.minimize(loss_lse+entropy+loss_residue+loss_tv)

                losses = [loss_lse ,loss_residue,entropy,loss_tv]
                return train_op, losses, K_dist

    def close(self):
        self.sess.close()
    def plot_triplets(self,figname=None):
        fig = plt.figure(figsize=(8,8))
        for i,j,k in self.triplets:
            plt.plot([self.directions[i,0],self.directions[j,0],self.directions[k,0],
                self.directions[i,0]],[self.directions[i,1],self.directions[j,1],
                    self.directions[k,1],self.directions[i,1]])
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
    #max gradient at b
    a = 20
    b = 1
    max_slope = np.abs(a/np.sqrt(np.exp(1))/b)
    
    #in dx want max_slope*dx = np.pi (nyquist limit)
    dx = np.pi/max_slope/2.
    
    #dx = sqrt(D^2/samples)
    assert sample > 0
    D = np.sqrt(dx**2*sample)
    
    X = np.random.uniform(low=-D/2.,high=D/2.,size=(sample,2))
    
    phi = a * np.exp(-(X[:,0]**2 + X[:,1]**2)/2./b**2)
    
    phi += a*noise*np.random.normal(size=phi.shape)

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
    X,phi = generate_data_nonaliased_nonsquare(0.03,sample=100)
    phi_wrap = np.angle(np.exp(1j*phi))[:,None,None]
    graph = tf.Graph()
    with tf.Session(graph = graph) as sess:
        pu = PhaseUnwrap(X, phi_wrap.shape, redundancy=2, sess = sess, graph = graph)
        f_rec = pu.phase_unwrap(phi_wrap)



    plot_phase(X,phi_wrap,label='phi_wrap',figname='phi_wrap.png')
    plot_phase(X,f_rec,label='f_rec',figname='phi_rec.png')
    plot_phase(X,phi,label='true',figname='phi_true.png')
    plot_phase(X,f_rec-phi,label='f_rec - true',figname='rec_true_diff.png')
    plot_phase(X,(f_rec-np.angle(np.exp(1j*f_rec)))/(2*np.pi),label='jumps',figname='jumps_rec.png')
    plot_phase(X,(phi-phi_wrap)/(2*np.pi),label='true jumps',figname='jumps_true.png')

if __name__=='__main__':
    test_phase_unwrap()
