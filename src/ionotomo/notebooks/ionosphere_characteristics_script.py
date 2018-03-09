
# coding: utf-8

# In[ ]:

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

###
# Create radio array



load_preexisting = True
datapack_to_load = "../scripts/rvw_data_analysis/rvw_datapack_full_phase_dec27.hdf5"

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

uvw = UVW(location = location,obstime=obstime,phase = phase_track)

X0 = directions_facets.transform_to(uvw)
X0 = np.array([np.arctan2(X0.u.value,X0.w.value),np.arctan2(X0.v.value,X0.w.value)]).T

X1 = directions_screen.transform_to(uvw)
X1 = np.array([np.arctan2(X1.u.value,X1.w.value),np.arctan2(X1.v.value,X1.w.value)]).T

x_scale = np.mean(np.std(X1,axis=0))
X1 /= x_scale
X0 /= x_scale


#Simulator
sim = SimulateTec(datapack_screen,spacing=1.,res_n=401)

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
def simulate_screen(sim,datapack,s=1.01,ls=10.,draw_new=False):
    if draw_new:
        sim.generate_model(s,ls)
    
    tec = sim.simulate_tec()
    phase = tec[...,None]*-8.4479e9/freqs
    datapack.set_phase(phase,ant_idx=-1,time_idx=[0],dir_idx=-1,freq_idx=-1)
    return tec
            
def log_posterior_true(tec,X1, tec_obs, X0):
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
        c = pm.approx_hessian(model.test_point)
        step = pm.NUTS(scaling=c)
        trace = pm.sample(100,step=step,start=mp)
        print(trace.get_sampler_stats('depth'),trace.get_sampler_stats('tree_size'))
        pm.traceplot(trace,combined=True)
        plt.show()
    print(pm.summary(trace))

    with model:
        y1_ = gp.conditional('y1',X1)#,given={'X':X0,'y':y0,'noise':0.1})
    logp = y1_.logp
        
    logp_val = np.zeros(len(trace))
    for i,point in enumerate(trace):
        point['y1'] = tec
        logp_val[i] = logp(point)
    
    return logp_val

logp = {}
d_mask = np.random.choice(Nd2,size=Nd1,replace=False)
for i in range(10):
    tec = simulate_screen(sim,datapack_screen,draw_new=True)
    logp[i] = []
    for ai in range(1,62):
        print(antenna_labels[ai])
        tec_mean = np.mean(tec[ai,0,:])
        tec_std = np.std(tec[ai,0,:])
        tec_ = (tec[ai,0,:] - tec_mean) / tec_std
        logp[i].append(np.mean(log_posterior_true(tec_,X1,tec_[d_mask],X1[d_mask,:])))
    


