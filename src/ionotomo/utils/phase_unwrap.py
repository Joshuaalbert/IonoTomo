import numpy as np
import tensorflow as tf
import pylab as plt
import cmocean
from scipy.spatial import cKDTree

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

def create_triplets(X,redundancy=2):
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

def plot_triplets(X,redundancy=1,figname=None):
    fig = plt.figure(figsize=(8,8))

    for i,j,k in create_triplets(X,redundancy=redundancy)[1]:
        plt.plot([X[i,0],X[j,0],X[k,0],X[i,0]],[X[i,1],X[j,1],X[k,1],X[i,1]])
    if figname is not None:
        plt.savefig('residue_triplets_3_redundant.png')
        plt.close("all")
    else:
        plt.show()

def phase_unwrap(X,phi_wrap,phi_wrap_var=None,redundancy=2,dropout=0.5):
    if len(phi_wrap.shape) == 1:
        phi_wrap = phi_wrap[None,None,:,None]
    Na,Nt,Nd,Nf = phi_wrap.shape
    path_, triplets_ = create_triplets(X,redundancy=redundancy)
    pairs = np.unique(np.sort(np.concatenate([triplets_[:,[0,1]],triplets_[:,[1,2]],triplets_[:,[2,0]]],axis=0),axis=1),axis=0)
    N = pairs.shape[0]

    g = tf.Graph()
    sess = tf.InteractiveSession(graph=g)
    with g.as_default():
        with tf.name_scope("unwrapper") as scope:
            g = tf.placeholder(tf.float32,shape=(Na,Nt,Nd,Nf),name='g')
            triplets = tf.placeholder(tf.int32,shape=(len(triplets_),3),name='triplets')
            path = tf.placeholder(tf.int32,shape=(len(path_),2),name='path')

            def _init(shape,dtype=tf.float64,partition_info=None):
                init = np.zeros(shape)
                #init[:,shape[1]>>1] = np.log(2)
                #init = tf.zeros(shape,dtype=dtype)
                #init[:,shape[1]>>1] = 1.
                return init

            K = tf.get_variable("K",shape=(Na,Nt,Nd,Nf,9),dtype=tf.float32,initializer=_init)
            K_softmax = tf.nn.softmax(K,dim=-1)        

            indices = tf.constant((np.arange(9)-4.).reshape((1,1,1,1,-1)),dtype=tf.float32)
    #         print(indices)
            K_int = tf.reduce_sum(K_softmax*indices,axis=-1)*2*np.pi
    #         print(K_int,triplets)

            #entropy
            entropy = - tf.reduce_mean(tf.reduce_sum(K_softmax*tf.log(K_softmax),axis=-1))

            def _wrap(a):
                return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,tf.complex64))),tf.float32)


            f_noise = tf.get_variable("f_noise",shape=(Na,Nt,Nd,Nf),dtype=tf.float32,initializer=_init)
            #f ~ N(f_obs,sigma_f^2)
            #f(K) = g_i + K 2pi
            # f(K) = int_p dg(x) + 2pi K(x)
            # K ~ N(0,C_K(x,x'))
            # K = K(theta) ~ P(K,  theta) = softmax(theta)
            # log P(K,theta) = sum softmax(theta)_i log(softmax(theta)_i)
            # Hamiltonian: 
            # H(K) =
            K_int_cum = tf.cumsum(K_int,axis=1)
            f = g + K_int_cum + f_noise
            #sigma_f = tf.get_variable("sigma_f",shape=(),dtype=tf.float32,initializer=tf.zeros_initializer)
            #prior for noise gaussian N(0,sigma_f^2)

            #df2 = tf.gather(f,path[:,1]) - tf.gather(f,path[:,0])
            #loss_path = tf.square(f[0] - g[0]) + tf.reduce_mean()
            
            dropout_ = tf.placeholder(tf.float32,shape=())
            
            phi_wrap_var_ = tf.placeholder(tf.float32,shape=phi_wrap.shape)

            df = tf.gather(f,pairs[:,1],axis=2) - tf.gather(f,pairs[:,0],axis=2)#tf.get_variable("df",shape=(N,),dtype=tf.float32,initializer=tf.zeros_initializer)
            consistency = tf.sqrt(1.+tf.square(_wrap(tf.gather(g,pairs[:,1],axis=2) - tf.gather(g,pairs[:,0],axis=2)) - df)) - 1.
            consistency = tf.nn.dropout(consistency,dropout_)
            loss_lse = tf.reduce_mean(consistency)
            #cov = tf.expand_dims(f_noise,-1)
            #loss_tv = tf.reduce_mean(tf.reduce_mean(tf.abs(cov*tf.transpose(cov,perm=[1,0])),axis=1),axis=0)
            loss_tv = tf.reduce_mean(tf.square(f_noise)/phi_wrap_var_)
            
#             smooth_residuals = tf.sqrt(1.+tf.square(tf.gather(f_noise,pairs[:,1],axis=1) - tf.gather(f_noise,pairs[:,0],axis=1))) - 1.
#             #smooth_residuals = tf.nn.dropout(smooth_residuals,dropout_)
#             loss_smooth = tf.reduce_mean(smooth_residuals)
#             #loss_tv += tf.reduce_mean(tf.square(tf.gather(K_int,pairs[:,1]) - tf.gather(K_int,pairs[:,0])))
            
            #loss_tv = tf.reduce_mean(tf.square(f_noise))
            #length_scale = np.mean(np.abs(X[pairs[:,1],:] - X[pairs[:,0],:]))
            #kernel = (0.1**2)*tf.cast(tf.exp(-pdist(tf.constant(X[None,:,:]))/2./(length_scale)**2),tf.float32)
            #loss_reg = tf.reduce_mean(tf.matmul(tf.expand_dims(K_int,0),tf.linalg.triangular_solve(kernel[0,:,:],tf.expand_dims(K_int,-1)))/2.)
            #tf.reduce_mean(tf.square(tf.gather(K_int,pairs[:,1]) - tf.gather(K_int,pairs[:,0])))

    #         mean,var = tf.nn.moments(df,axes=[0])
    #         loss_lse += var

            Wf = _wrap(f)

            df01 = tf.gather(Wf,triplets[:,1],axis=2) - tf.gather(Wf,triplets[:,0],axis=2)
            df01 = _wrap(df01) 
            df12 = tf.gather(Wf,triplets[:,2],axis=2) - tf.gather(Wf,triplets[:,1],axis=2)
            df12 = _wrap(df12)
            df20 = tf.gather(Wf,triplets[:,0],axis=2) - tf.gather(Wf,triplets[:,2],axis=2)
            df20 = _wrap(df20)

            residue = tf.sqrt(1. + tf.square(df01 + df12 + df20))-1.
            residue = tf.nn.dropout(residue,dropout_)
            loss_residue = tf.reduce_mean(residue)

            #K_int_mean = (tf.gather(K_int,triplets[:,0]) + tf.gather(K_int,triplets[:,1]) + tf.gather(K_int,triplets[:,2]))/3.
            #loss_reg = tf.reduce_mean(1./(1+0)*tf.abs(tf.gather(K_int,triplets[:,0]) - K_int_mean) + tf.abs(tf.gather(K_int,triplets[:,1]) - K_int_mean) + tf.abs(tf.gather(K_int,triplets[:,2]) - K_int_mean))
            #loss_reg = tf.reduce_mean(tf.sqrt(1.+tf.square(tf.gather(K_int,pairs[:,1]) - tf.gather(K_int,pairs[:,0]))))
            
            learning_rate = tf.placeholder(tf.float32,shape=())
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            entropy_weight = tf.placeholder(tf.float32,shape=())
            train_op = opt.minimize(loss_lse+entropy_weight*entropy+loss_residue+loss_tv)

            losses = [loss_lse ,loss_residue,entropy,loss_tv]

            sess.run(tf.global_variables_initializer())
            import time
            time_ = time.mktime(time.gmtime())
            loss_per_step_ = []
            
            for epoch in range(25000):
                ew = 0.0000001
                lr = 0.1
                dp = 0.2
                if epoch > 1000:
                    ew = 0.000001
                    lr = 0.1
                    dp = 0.3
                if epoch > 5000:
                    ew = 0.00001
                    lr = 0.05
                    dp = 0.3
                if epoch > 10000:
                    ew = 0.001
                    lr = 0.03
                    dp = 0.5
                if epoch > 15000:
                    ew = 0.01
                    lr = 0.01
                    dp = 0.5
                if epoch > 20000:
                    ew = 0.01
                    lr = 0.001
                    dp = 0.8
                
                if phi_wrap_var is None:
                    phi_wrap_var = np.ones_like(phi_wrap)
                
                
                    
                _,losses_,df_,K_int_,K_softmax_,f_noise_ = sess.run([train_op,losses,df,K_int,K_softmax,f_noise],
                                                           feed_dict={dropout_:dp,
                                                                      learning_rate:lr,
                                                                      entropy_weight: ew, 
                                                                      g : phi_wrap, 
                                                                      triplets: triplets_, 
                                                                      path:path_,
                                                                      phi_wrap_var_ : phi_wrap_var})
                loss_per_step_.append(np.sum(losses_))
                        
                if np.sum(losses_) < 0.1:
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_),*losses_))
                    break
                if time.mktime(time.gmtime()) - time_ > 5. or epoch==0:
                    time_ = time.mktime(time.gmtime())
                    print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_),*losses_))
                    if np.sum(losses_) < 0.1:
                        break
            print("Epoch : {} loss={:.4f} | LSE: {:.4f} | Residue: {:.4f} | Entropy: {:.4f} | TV: {:.4f} ".format(epoch,np.sum(losses_),*losses_))
            f_rec = np.zeros_like(phi_wrap)
            f_rec[:,:,path_[0][0],:] = phi_wrap[:,:,path_[0][0],:]
            K_int_sum_ = np.cumsum((np.argmax(K_softmax_,axis=4)-4)*2*np.pi,axis=1)
            #print(df_)
            for i,p in enumerate(path_):
                df_ = phi_wrap[:,:,p[1],:] - phi_wrap[:,:,p[0],:] + K_int_sum_[:,:,p[1],:] - K_int_sum_[:,:,p[0],:]
                f_rec[:,:,p[1],:] = f_rec[:,:,p[0],:] + df_
            plt.plot(loss_per_step_)
            plt.yscale('log')
            plt.show()
        return f_rec

def test_phase_unwrap():
    X,phi = generate_data_nonaliased_nonsquare(0.03,sample=100)
    path_, triplets_ = create_triplets(X,redundancy=2)
    dist = np.linalg.norm(np.concatenate([X[triplets_[:,1],:] - X[triplets_[:,0],:],
                           X[triplets_[:,2],:] - X[triplets_[:,1],:],
                           X[triplets_[:,0],:] - X[triplets_[:,2],:]],axis=0),axis=1)
    plt.hist(dist,bins=20)
    plt.show()

    phi_wrap = np.angle(np.exp(1j*phi))
    f_rec = phase_unwrap(X,phi_wrap,redundancy=2).flatten()



    plot_phase(X,phi_wrap,label='phi_wrap',figname='phi_wrap.png')
    plot_phase(X,f_rec,label='f_rec',figname='phi_rec.png')
    plot_phase(X,phi,label='true',figname='phi_true.png')
    plot_phase(X,f_rec-phi,label='f_rec - true',figname='rec_true_diff.png')
    plot_phase(X,(f_rec-np.angle(np.exp(1j*f_rec)))/(2*np.pi),label='jumps',figname='jumps_rec.png')
    plot_phase(X,(phi-phi_wrap)/(2*np.pi),label='true jumps',figname='jumps_true.png')

if __name__=='__main__':
    test_phase_unwrap()
