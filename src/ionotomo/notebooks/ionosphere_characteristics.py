
# coding: utf-8

# In[1]:

#%matplotlib
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
from ionotomo import *
from ionotomo.tomography.simulate import SimulateTec
import tensorflow as tf
import numpy as np
import gpflow as gpf
import pymc3 as pm
import os
import pylab as plt
import seaborn as sns

###
# Create radio array



load_preexisting = True
datapack_to_load = "../data/rvw_datapack_full_phase_dec27.hdf5"

if load_preexisting:
    datapack_facets = DataPack(filename=datapack_to_load)
    _,timestamps_flag = datapack_facets.get_times(-1)
    timestamps_flag = timestamps_flag[1:]
    freqs_flag = datapack_facets.get_freqs(-1)
    keep_freqs = freqs_flag[200:220]
    freqs_flag = freqs_flag[np.bitwise_not(np.isin(freqs_flag,keep_freqs))]
    datapack_facets.flag_times(timestamps_flag)
    #datapack_facets.flag_freqs(freqs_flag)
    #Flagged all but first time, channels 200-219, etc
else:
    ra = 126
    dec = 64
    timestamp = "2016-12-08T23:25:01.384"
    radio_array = generate_example_radio_array(config='lofar')
    p0 = ac.SkyCoord(ra=ra*au.deg,dec=dec*au.deg, frame='icrs')
    obstime = at.Time(timestamp,format='isot')
    location = radio_array.get_center()
    altaz = ac.AltAz(location = location, obstime = obstime)
    p = p0.transform_to(altaz)
    print(p)
    datapack_facets = generate_example_datapack(alt=p.alt.deg,az=p.az.deg,Ndir=42,Nfreqs=20,Ntime=1,radio_array=radio_array)

datapack_screen = phase_screen_datapack(15,datapack=datapack_facets)

times, timestamps = datapack_facets.get_times(-1)
antennas,antenna_labels = datapack_facets.get_antennas(-1)
freqs = datapack_facets.get_freqs(-1)

phase_track = datapack_facets.get_center_direction()
obstime = times[0]
location = datapack_facets.radio_array.get_center()

directions_facets,_ = datapack_facets.get_directions(-1)
Nd1 = directions_facets.shape[0]
directions_screen,_ = datapack_screen.get_directions(-1)
Nd2 = directions_screen.shape[0]

X_facets = np.array([directions_facets.ra.deg,directions_facets.dec.deg]).T
X_screen = np.array([directions_screen.ra.deg,directions_screen.dec.deg]).T

# uvw = UVW(location = location,obstime=obstime,phase = phase_track)

# X0 = directions_facets.transform_to(uvw)
# X0 = np.array([np.arctan2(X0.u.value,X0.w.value),np.arctan2(X0.v.value,X0.w.value)]).T

# X1 = directions_screen.transform_to(uvw)
# X1 = np.array([np.arctan2(X1.u.value,X1.w.value),np.arctan2(X1.v.value,X1.w.value)]).T

# x_scale = np.mean(np.std(X1,axis=0))
# X1 /= x_scale
# X0 /= x_scale




###
# Generate ionospheres following I(sigma, l)

def sample_ionosphere(sim,sigma,l):
    """Generate an ionosphere, I(sigma,l).
    sim : SimulatedTec object (non reentrant)
    sigma : float log_electron variance
    l : float length scale
    Returns a the model as ndarray
    """
    sim.generate_model(sigma, l)
    model = sim.model
    return model

###
# simulate and place in datapack_screen
def simulate_screen(sim,datapack,aj=0,s=1.01,ls=10.,draw_new=False):
    if draw_new:
        sim.generate_model(s,ls)
    
    tec = sim.simulate_tec()
    phase = tec[...,None]*-8.4479e9/freqs
    datapack.set_phase(phase,ant_idx=-1,time_idx=[aj],dir_idx=-1,freq_idx=-1)
    return tec
            
def log_posterior_true(tec,X1, tec_obs, X0,samples=1000):
    """
    Calculate the logp of the true underlying.
    tec : array (Nd1,)
    X1 : array (Nd1,2)
    tec_obs : array (Nd2,)
    X0 : array (Nd2, 2)
    """
    with pm.Model() as model:
        l = pm.Exponential('l',1.)
        sigma = pm.Exponential('sigma',1.)
        #c = pm.Normal('c',mu=0,sd=1)
        cov_func = pm.math.sqr(sigma)*pm.gp.cov.ExpQuad(1, ls=l)
        #mean_func = pm.gp.mean.Constant(c=c)
        gp = pm.gp.Marginal(cov_func=cov_func)
        eps = pm.HalfNormal('eps',sd=0.1)
        y0_ = gp.marginal_likelihood('y0',X0,tec_obs,eps)
        mp = pm.find_MAP()
        print(mp)
        trace = pm.sample(samples,start={'sigma':0.25,'l':0.25},chains=4)
        pm.traceplot(trace,combined=True)
        plt.show()
    print(pm.summary(trace))
    
    df = pm.trace_to_dataframe(trace, varnames=['sigma','l','eps'])
    sns.pairplot(df)
    plt.show()

    with model:
        y1_ = gp.conditional('y1',X1)#,given={'X':X0,'y':y0,'noise':0.1})
    logp = y1_.logp
        
    logp_val = np.zeros(len(trace))
    for i,point in enumerate(trace):
        point['y1'] = tec
        logp_val[i] = logp(point)
    
    return logp_val


# tec = simulate_screen(sim,datapack_screen,draw_new=True)
# d_mask = np.random.choice(Nd2,size=Nd1,replace=False)
# logp = log_posterior_true(tec[51,0,:],X1,tec[51,0,d_mask],X1[d_mask,:])

# logp = {}
# d_mask = np.random.choice(Nd2,size=Nd1,replace=False)
# for i in range(10):
#     tec = simulate_screen(sim,datapack_screen,draw_new=True)
#     logp[i] = []
#     for ai in range(1,62):
#         print(antenna_labels[ai])
#         tec_mean = np.mean(tec[ai,0,:])
#         tec_std = np.std(tec[ai,0,:])
#         tec_ = (tec[ai,0,:] - tec_mean) / tec_std
#         logp[i].append(np.mean(log_posterior_true(tec_,X1,tec_[d_mask],X1[d_mask,:])))
    


# In[2]:

import theano as th

def solve_vi(X,Y,initial=None,batch_size=100):
    X_t = th.shared(X)#pm.Minibatch(X,batch_size=batch_size,)
    Y_t = th.shared(Y)#pm.Minibatch(Y,batch_size=batch_size)
#    sigma_Y_t = th.shared(sigma_Y)#pm.Minibatch(sigma_Y,batch_size=batch_size)

    #initial=(0.3,0.5,2.)
    
    dx = np.max(X) - np.min(X)
    dy = np.max(Y) - np.min(Y)

    with pm.Model() as model:
        sigma_K = pm.HalfNormal('sigma_K',sd=dy/3.)
        l_space = pm.HalfNormal('l_space',sd=dx/3.,testval=1.)
        cov_func = sigma_K**2 * pm.gp.cov.ExpQuad(2,active_dims=[0,1], ls=l_space) 
        gp = pm.gp.Marginal(cov_func=cov_func)
        eps = pm.Uniform('eps',0.0,np.std(Y))
        y1 = gp.marginal_likelihood('y1',X_t,Y_t,eps)
        #y2 = gp.marginal_likelihood('y2',X[:100,:],Y[:100],eps*sigma_Y[:100])
        initial = initial or pm.find_MAP()
        approx = pm.fit(1000, start=initial,method='advi',callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
#         plt.plot(approx.hist)
#         plt.show()
        means = approx.bij.rmap(approx.mean.eval())
#         print(means)
#         sds = approx.bij.rmap(approx.std.eval())
#         print(sds)
        df = approx.sample(10000)
        p={k:pm.summary(df)['mean'][k] for k in pm.summary(df)['mean'].keys()}
#         pm.traceplot(df,lines=p)
#         plt.show()
    return p

from ionotomo.bayes.gpflow_contrib import GPR_v2
def solve_gpf(X,Y,initial=None,batch_size=100):
    dx = np.max(X[:,0]) - np.min(X[:,0])
    dy = np.max(Y) - np.min(Y)
    with gpf.defer_build():
        k_space = gpf.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.1])
        kern = k_space
        mean = gpf.mean_functions.Constant()
        m = GPR_v2(X, Y[:,None], kern, mean_function=mean,var=1.,trainable_var=True)
        m.kern.lengthscales.prior = gpf.priors.Uniform(0,dx)
        m.kern.variance.prior = gpf.priors.Uniform(0,dy)
        m.compile()
    
    o = gpf.train.ScipyOptimizer(method='BFGS')
    o.minimize(m,maxiter=100)
    ls= m.kern.lengthscales.value[0]
    v = m.kern.variance.value
    #print(m)
    return {"l_space":ls,"var":v, 'eps': m.likelihood.variance.value}

def _solve_gpf(arg):
    X,Y,initial = arg
    with tf.Session(graph=tf.Graph()):
        return solve_gpf(X,Y,initial)

from concurrent import futures
def parallel_solve_gpf(X,Y,initial=None,num_threads=1):
    """Assume batch dimension 0"""
    batch = Y.shape[0]
    with futures.ThreadPoolExecutor(max_workers=num_threads) as exe:
        args = []
        for i in range(batch):
            args.append((X,Y[i,...],initial))
        jobs = exe.map(_solve_gpf,args)
        results = list(jobs)
        return results


# In[3]:

def determine_simulated_characteristics(X_screen, freqs, s, l, num_threads):
    sim = SimulateTec(datapack_screen,spacing=1.,res_n=501)
    print("Generating {} km scale".format(l))
    sim.generate_model(s,l)
    print("Simulating {} km scale".format(l))
    tec = sim.simulate_tec()
    phase = tec[...,None]*-8.4479e9/freqs
    results = parallel_solve_gpf(X_screen,phase[1:,0,:,0],num_threads=num_threads)
    stats = {
                'l_space':[r['l_space'] for r in results],
                'var':[r['var'] for r in results],
                'eps':[r['eps'] for r in results]
               }
    return stats

def _determine_simulated_characteristics(arg):
    return determine_simulated_characteristics(*arg)

with futures.ProcessPoolExecutor(max_workers=4) as pexe:
    args = []
    for l in np.linspace(5.,50.,1):
        args.append(( X_screen, freqs, 1.008, l, 16))
    jobs = pexe.map(_determine_simulated_characteristics, args)
    results = {l: r for l,r in zip(np.linspace(5.,50.,1),list(jobs))}
    
    


from ionotomo import DatapackPlotter

# simulate_screen(sim,datapack_screen,s=1.01,ls=10.,draw_new=True)
# dp = DatapackPlotter(datapack = datapack_screen)
# dp.plot(observable='phase',show=True,labels_in_radec=True,plot_crosses=False)

