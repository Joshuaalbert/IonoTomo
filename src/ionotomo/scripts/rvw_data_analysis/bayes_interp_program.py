
from ionotomo import *
import numpy as np
import pylab as plt
import h5py

#import ionotomo.bayes.phase_screen_interp as bp
import ionotomo.utils.gaussian_process as gp

import cmocean
import os
import logging as log

#import tensorflow as tf
from dask.threaded import get


def make_xstar(X,N=50):
    """Make a screen NxN based on obs coords X.
    Returns shape (N*N, 2) array"""
    xmin = np.min(X[:,0])
    ymin = np.min(X[:,1])
    xmax = np.max(X[:,0])
    ymax = np.max(X[:,1])
    xvec = np.linspace(xmin,xmax,N)
    yvec = np.linspace(ymin,ymax,N)
    x,y = np.meshgrid(xvec,yvec,indexing='ij')
    Xstar = np.array([x.flatten(),y.flatten()]).T
    return Xstar

def interp_nearest(x,y,z,x_,y_):
    """given input arrays (x,y,z) return nearest neighbour to (x_, y_)
    Return same shape as x_"""
    dx = np.subtract.outer(x_,x)
    dy = np.subtract.outer(y_,y)
    r = dx**2
    dy *= dy
    r += dy
    np.sqrt(r,out=r)
    arg = np.argmin(r,axis=1)
    z_ = z[arg]
    return z_

    
def angular_space(dirs_uvw):
    """Go to angular space, small angles mean not much difference"""
    X = np.array([np.arctan2(dirs_uvw.u.value,dirs_uvw.w.value),
                  np.arctan2(dirs_uvw.v.value,dirs_uvw.w.value)]).T
    return X

def plot_data_posterior(x_obs, phase_obs, phase_obs_screen,uncert_obs_screen,phase_post,uncert_post,extent,plot_folder,antenna_label,timestamp):
    """Do the plotting of results"""
#    extent=(np.min(X[0][:,0]),np.max(X[0][:,0]), 
#                            np.min(X[0][:,1]),np.max(X[0][:,1]))
#    phase_obs_screen = y[0].reshape((res,res))+mean[0]
#    uncert_obs_screen = sigma_y[0].reshape((res,res))
#    phase_post = ystar.reshape((50,50))+mean[0]
#    uncert_post = np.sqrt(np.diag(cov)).reshape((50,50))
#    x_obs = X_angular[0]
#    phase_obs = phase[i,0,:,0]
#    antenna_label = antenna_labels[i]
#    timestamp = timestamps[0]

    vmin = np.min(phase_obs)
    vmax = np.max(phase_obs)

    fig = plt.figure(figsize=(2*6,2*6))

    ax = fig.add_subplot(2,2,1)
    #plot first time slot for example
    sc = ax.imshow(phase_obs_screen.T,origin='lower',
                    extent=extent,
                    cmap=cmocean.cm.phase,
                    vmin = vmin, vmax = vmax)#X[0][:,0],X[0][:,1],c=y[0],marker='+')
    plt.colorbar(sc)
    ax.set_title("Measured phases")
    ax = fig.add_subplot(2,2,2)
    sc = ax.imshow(uncert_obs_screen.T,origin='lower',
                    extent=extent,
                    cmap='bone')#X[0][:,0],X[0][:,1],c=y[0],marker='+')
    plt.colorbar(sc)
    plt.title("Measurement uncertainty")
    ax = fig.add_subplot(2,2,3)
    sc = ax.imshow(phase_post.T,origin='lower',
                    extent=extent,
                    cmap=cmocean.cm.phase,
                    vmin=vmin,vmax=vmax)
    plt.colorbar(sc)
    ax.set_title("Posterior mean")
    ax.scatter(x_obs[:,0],x_obs[:,1],
                   c=phase_obs,
                   cmap=cmocean.cm.phase,
                   edgecolors='black',
                   s = 100,
                   vmin=vmin,vmax=vmax)
    ax = fig.add_subplot(2,2,4)
    sc = ax.imshow(uncert_post.T,origin='lower',
                    extent=extent,
                   cmap='bone')
    plt.colorbar(sc)
    ax.set_title("Posterior uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder,"measured_and_posterior_{}_{}.png".format(antenna_label,timestamp)),format='png')

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    
    ax.hist(uncert_post.flatten(),bins=25)
    ax.set_title(r"uncert. dist: ${:.2f} \pm {:.2f}$".format(np.mean(uncert_post),np.std(uncert_post)))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder,"posterior_uncert_dist{}_{}.png".format(antenna_label,timestamp)),format='png')
    
    plt.close("all")



def main(output_folder,datapack_name,datapack_smooth_name,bayes_param_file,time_block_size = 120):
    """Main program to bayesian interpolate/smooth data.
    datapack_name : str
        The filename of input datapack
    datapack_smooth_name : str
        The filename of output smoothed datapack
    bayes_param_file : str
        The file that will contain the bayesian optimized regularization params    
    time_block_size : int
        The number of timestamps to use in statistic correlation determination.
        i.e. slow_gain resolution
    """

    output_folder = os.path.abspath(output_folder)
    diagnostic_folder = os.path.join(output_folder,'diagnostics')
    try:
        os.makedirs(diagnostic_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    
    #bayes_params = h5py.File(os.path.join(output_folder,bayes_param_file),'a')

    datapack = DataPack(filename=datapack_name)
    datapack_smooth = datapack.clone()

    times_all,timestamps_all = datapack.get_times(time_idx=-1)
    Na_all = len(times_all)

    time_block_idx = 0
    while time_block_idx*time_block_size < Na_all:
        start_time = time_block_idx * time_block_size
        stop_time = min(Na_all,(time_block_idx+1) * time_block_size)
        time_idx = range(start_time,stop_time)
        log.info("Processing time block {}: {} to {}".format(time_block_idx,timestamps_all[start_time],timestamps_all[stop_time]))
    
        #Will smooth at all antennas, all directions, all freq
        ant_idx = -1
        dir_idx = -1
        freq_idx = -1

        #derived from slow_gains
        std = np.sqrt(datapack.get_variance(ant_idx=ant_idx, time_idx=time_idx, dir_idx=dir_idx, freq_idx=freq_idx))

        #phase from dd and di solutions combined, we phase wrap here
        #TODO 2D+time phase unwrap
        phase = np.angle(np.exp(1j*datapack.get_phase(ant_idx=ant_idx, time_idx=time_idx, dir_idx=dir_idx, freq_idx=freq_idx)))

        directions, patch_names = datapack.get_directions(dir_idx=dir_idx)
        antennas, antenna_labels = datapack.get_antennas(ant_idx=ant_idx)
        times,timestamps = datapack.get_times(time_idx=time_idx)
        freqs = datapack.get_freqs(freq_idx=freq_idx)
        Na = len(antennas)
        Nt = len(times)
        Nd = len(directions)
        Nf = len(freqs)

        #to phase unwrap in time axis uncomment
        # from rathings.phase_unwrap import phase_unwrapp1d
        # phase = np.transpose(phase_unwrapp1d(np.transpose(phase,axes=(1,0,2,3)),axis=0),axes=(1,0,2,3))


        #define directions with Pointing (UVW fixed to first time but obstime actual)
        fixtime = times[0]
        fixfreq = freqs[Nf>>1]
        phase_center = datapack.get_center_direction()
        array_center = datapack.radio_array.get_center()
        uvw = [Pointing(location = array_center.earth_location,obstime = times[j],fixtime=fixtime, phase = phase_center) for j in range(Nt)]
        ants_uvw = [antennas.transform_to(uvw[j]) for j in range(Nt)]
        dirs_uvw = [directions.transform_to(uvw[j]) for j in range(Nt)]

        #Make coords in angular per time
        X_angular = np.array([angular_space(dirs_uvw[j]) for j in range(Nt)])
        

        def process_ant(i,antenna_labels,X_angular,Na,Nt,Nd,Nf,phase,std,time_block_idx,timestamps,diagnostic_folder,bayes_params_file):
            """Python call to process an antenna.
            This gets wrapped by tensorflow to automate parallelization
            i : int
                the antenna index to process
            Note: requires all global variables to be around.
            Note: This is stateless, so repeated calls produce the same output.
            This enables faster computation.
            """
            #make stateless
            np.random.seed(i)
            #import logging as log

            X_angular = list(X_angular)
            
            log.info("working on {}".format(antenna_labels[i]))
            res = 12
            sample_n = 144
            
            #we will do solve on "faceted solution coords"
            #Becaues it doesn't make sense to centralize the solutions at the centroid of facet
            #due to how calibration inherently is done for full facet
            X_nearest = [make_xstar(X_angular[j],res) for j in range(Nt)]
            
            mask = [np.random.choice(res*res,size=res*res - sample_n,replace=False) for j in range(Nt)]

            X = X_nearest

            ###
            # Real Part
            ###
            real = np.cos(phase[:,:,:])

            ###
            # Imag Part
            ###
            imag = np.sin(phase[:,:,:])

            ###
            # std
            ###
            std_real = np.abs(imag) * std[:,:,:]
            std_imag = np.abs(real) * std[:,:,:]

            mean_real = [np.mean(real[j,:,0]) for j in range(Nt)]
            mean_imag = [np.mean(imag[j,:,0]) for j in range(Nt)]

            y_real = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
                real[j,:,0]-mean_real[j], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]
            y_imag = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
                imag[j,:,0]-mean_imag[j], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]

            sigma_y_real = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
                                      std_real[j,:,0], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]
            sigma_y_imag = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
                                      std_imag[j,:,0], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]

            y_obs = np.angle(y_real[0] + mean_real[0] + 1j*(y_imag[0] + mean_imag[0])).reshape((res,res))
            std_obs = np.sqrt((y_real[0] + mean_real[0])**2 * sigma_y_real[0]**2 + (y_imag[0] + mean_imag[0])**2 * sigma_y_imag[0]**2).reshape((res,res))


            for j in range(Nt):
                y_real[j][mask[j]] = np.nan
                y_imag[j][mask[j]] = np.nan
                sigma_y_real[j][mask[j]] = np.nan
                sigma_y_imag[j][mask[j]] = np.nan

                        
            #sample the non-masked
            for j in range(Nt):
                X[j] = X[j][np.bitwise_not(np.isnan(y_real[j])),:]
                sigma_y_real[j] = sigma_y_real[j][np.bitwise_not(np.isnan(y_real[j]))]
                sigma_y_imag[j] = sigma_y_imag[j][np.bitwise_not(np.isnan(y_real[j]))]
                y_real[j] = y_real[j][np.bitwise_not(np.isnan(y_real[j]))]
                y_imag[j] = y_imag[j][np.bitwise_not(np.isnan(y_real[j]))]
               
            #Define GP kernel
            K1_real = gp.RationalQuadratic(2,l=0.02,sigma=0.52, alpha=2.)
            K1_real.set_hyperparams_bounds([0.005,0.10],name='l')
            K1_real.set_hyperparams_bounds([0.0005,4.],name='sigma')
            K1_real.set_hyperparams_bounds([0.05,100.],name='alpha')
            K2_real = gp.Diagonal(2,sigma=0.01)
            K2_real.set_hyperparams_bounds([0.00,0.20],name='sigma')
            K_real = K1_real+K2_real
            try:
                with h5py.File(bayes_params_file,'r') as bayes_params:
                    K_real.hyperparams = bayes_params['/{}/{}/real'.format(antenna_labels[i],time_block_idx)]
                    log.info("Loaded bayes params /{}/{}/real".format(antenna_labels[i],time_block_idx))
            except:
                log.info("Level 2 Solve...")
                K_real.hyperparams = gp.level2_multidataset_solve(X,y_real,sigma_y_real,K_real,n_random_start=0)
                with h5py.File(bayes_params_file,'a') as bayes_params:
                    bayes_params['/{}/{}/real'.format(antenna_labels[i],time_block_idx)] = K_real.hyperparams
                    bayes_params.flush()
            log.info(K_real)     

            #Define GP kernel
            K1_imag = gp.RationalQuadratic(2,l=0.02,sigma=0.52, alpha=2.)
            K1_imag.set_hyperparams_bounds([0.005,0.10],name='l')
            K1_imag.set_hyperparams_bounds([0.0005,4.],name='sigma')
            K1_imag.set_hyperparams_bounds([0.05,100.],name='alpha')
            K2_imag = gp.Diagonal(2,sigma=0.01)
            K2_imag.set_hyperparams_bounds([0.00,0.20],name='sigma')
            K_imag = K1_imag+K2_imag
            try:
                with h5py.File(bayes_params_file,'r') as bayes_params:
                    K_imag.hyperparams = bayes_params['/{}/{}/imag'.format(antenna_labels[i],time_block_idx)]
                    log.info("Loaded bayes params /{}/{}/imag".format(antenna_labels[i],time_block_idx))
            except:
                log.info("Level 2 Solve...")
                K_imag.hyperparams = K_real.hyperparams
                K_imag.hyperparams = gp.level2_multidataset_solve(X,y_imag,sigma_y_imag,K_imag,n_random_start=0)
                with h5py.File(bayes_params_file,'a') as bayes_params:
                    bayes_params['/{}/{}/imag'.format(antenna_labels[i],time_block_idx)] = K_imag.hyperparams
                    bayes_params.flush()
            log.info(K_imag)
            
            #plot first timestamp
            Xstar = make_xstar(X[0],N=50)

            ystar_real,cov_real,lml_real = gp.level1_solve(X_angular[0],real[0,:,0]-mean_real[0],std_real[0,:,0],Xstar,K_real)
            log.info("Hamiltonian (real): {}".format( -lml_real))

            ystar_imag,cov_imag,lml_imag = gp.level1_solve(X_angular[0],imag[0,:,0]-mean_imag[0],std_imag[0,:,0],Xstar,K_imag)
            log.info("Hamiltonian (imag): {}".format( -lml_imag))

            ystar = np.angle(ystar_real + mean_real[0] + 1j*(ystar_imag + mean_imag[0])).reshape((50,50))
            stdstar = np.sqrt((ystar_real + mean_real[0])**2 * np.diag(cov_real) + (ystar_imag + mean_imag[0])**2 * np.diag(cov_imag)).reshape((50,50))

            phase_smooth = np.zeros([1,Nt,Nd,Nf])
            variance_smooth = np.zeros([1,Nt,Nd,Nf])
            log.info("Smoothing time_block...")
            for j in range(Nt):
                for l in range(Nf):
                    mean_real = np.mean(real[j,:,l])
                    mean_imag = np.mean(imag[j,:,l])
                    Xstar=X_angular[j]
                    ystar_real,cov_real,lml_real = gp.level1_solve(X_angular[j],real[j,:,l]-mean_real,std_real[j,:,l],Xstar,K_real)
                    ystar_imag,cov_imag,lml_imag = gp.level1_solve(X_angular[j],imag[j,:,l]-mean_imag,std_imag[j,:,l],Xstar,K_imag)
                    phase_smooth[0,j,:,l] = np.angle(ystar_real + mean_real + 1j*(ystar_imag + mean_imag))
                    variance_smooth[0,j,:,l] = np.diag(cov_real) * (ystar_real + mean_real)**2 + np.diag(cov_imag) * (ystar_imag + mean_imag)**2
            return [phase_smooth, variance_smooth, X_angular[0], y_obs, std_obs, ystar, stdstar]

            


##            #center on zero for GP solve without basis
##            mean = [np.mean(phase[i,j,:,0]) for j in range(Nt)]
##            y = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
##                                phase[i,j,:,0]-mean[j], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]
##            
##            sigma_y = [interp_nearest(X_angular[j][:,0],X_angular[j][:,1],
##                                      std[i,j,:,0], X_nearest[j][:,0], X_nearest[j][:,1]) for j in range(Nt)]
##            for j in range(Nt):
##                y[j][mask[j]] = np.nan
##                sigma_y[j][mask[j]] = np.nan
##
##                        
##            #sample the non-masked
##            for j in range(Nt):
##                X[j] = X[j][np.bitwise_not(np.isnan(y[j])),:]
##                sigma_y[j] = sigma_y[j][np.bitwise_not(np.isnan(y[j]))]
##                y[j] = y[j][np.bitwise_not(np.isnan(y[j]))]
##               
##            #Define GP kernel
##            K1 = gp.RationalQuadratic(2,l=0.02,sigma=0.52, alpha=2.)
##            K1.set_hyperparams_bounds([0.005,0.10],name='l')
##            K1.set_hyperparams_bounds([0.005,4.],name='sigma')
##            K1.set_hyperparams_bounds([0.05,100.],name='alpha')
##            K2 = gp.Diagonal(2,sigma=0.01)
##            K2.set_hyperparams_bounds([0.00,0.20],name='sigma')
##            K = K1+K2
##            try:
##                K.hyperparams = bayes_params['/{}/{}/real'.format(antenna_labels[i],time_block_idx)]
##                log.info("Loaded bayes params /{}/{}/real".format(antenna_labels[i],time_block_idx))
##            except:
##                log.info("Level 2 Solve...")
##                K.hyperparams = gp.level2_multidataset_solve(X,y,sigma_y,K,n_random_start=1)
##                bayes_params['/{}/{}/real'.format(antenna_labels[i],time_block_idx)] = K.hyperparams
##                bayes_params.flush()
##            log.info(K)            
##            
##            #plot first timestamp
##            Xstar = make_xstar(X[0],N=50)
##            ystar,cov,lml = gp.level1_solve(X_angular[0],phase[i,0,:,0]-mean[0],std[i,0,:,0],Xstar,K)
##            log.info("Hamiltonian: {}".format( -lml))
            
            
                        
##            phase_smooth = np.zeros([1,Nt,Nd,Nf])
##            variance_smooth = np.zeros([1,Nt,Nd,Nf])
##            log.info("Smoothing time_block...")
##            for j in range(Nt):
##                for l in range(Nf):
##                    mean = np.mean(phase[i,j,:,l])
##                    Xstar=X_angular[j]
##                    ystar,cov,lml = gp.level1_solve(X_angular[j],phase[i,j,:,l]-mean,std[i,j,:,l],Xstar,K)
##                    phase_smooth[0,j,:,l] = ystar + mean
##                    variance_smooth[0,j,:,l] = np.diag(cov)
##            return [phase_smooth, variance_smooth]

#        log.info("Building TF graph")
#        g = tf.Graph()
#        sess = tf.InteractiveSession(graph=g,config=tf.ConfigProto(operation_timeout_in_ms=2000, inter_op_parallelism_threads=2, intra_op_parallelism_threads=1))
#        with g.as_default():
#            smooth_ = []
#            
#
#            for i in range(Na):
#                args = [tf.constant(phase[i,:,:,:]),
#                        tf.constant(std[i,:,:,:])]
#                smooth_.append(tf.py_func(lambda phase,std : process_ant(i,antenna_labels,X_angular,Na,Nt,Nd,Nf,phase,std,time_block_idx,timestamps,diagnostic_folder,os.path.join(output_folder,bayes_param_file))
#                    ,args,[tf.float64,tf.float64,tf.float64,tf.float64,tf.float64,tf.float64,tf.float64],stateful=False))
#        
#        log.info("Running graph")
#        res = sess.run(smooth_)
#
#        sess.close()

        log.info("Building dask graph")
        dsk = {}
        dsk['antenna_labels'] = antenna_labels
        dsk['X_angular'] = X_angular
        dsk['Na'] = Na
        dsk['Nt'] = Nt
        dsk['Nd'] = Nd
        dsk['Nf'] = Nf
        dsk['time_block_idx'] = time_block_idx
        dsk['timestamps'] = timestamps
        dsk['diagnostic_folder'] = diagnostic_folder
        dsk['bayes_param_file'] = os.path.join(output_folder,bayes_param_file)

        #res = []
        smooth_ = []
        for i in range(Na):
            dsk['phase'] = phase[i,:,:,:]
            dsk['std'] = std[i,:,:,:]
            #res.append(process_ant(i,antenna_labels,X_angular,Na,Nt,Nd,Nf,phase,std,time_block_idx,timestamps,diagnostic_folder,bayes_params))
            dsk[antenna_labels[i]] = (process_ant,i,'antenna_labels','X_angular','Na','Nt','Nd','Nf','phase','std','time_block_idx','timestamps','diagnostic_folder','bayes_param_file')
            smooth_.append(antenna_labels[i]) 
        log.info("Running graph")
        res = get(dsk,smooth_,num_workers=2)

        
        log.info("Storing results in datapack_smooth")
        for i in range(Na):
            phase_smooth, variance_smooth, x_obs, y_obs, std_obs, ystar, stdstar = res[i]
            extent = (np.min(x_obs[:,0]), np.max(x_obs[:,0]), np.min(x_obs[:,1]), np.max(x_obs[:,1]))
            plot_data_posterior(x_obs, phase[i,0,:,0], y_obs, std_obs, ystar, stdstar, extent, diagnostic_folder, antenna_labels[i], timestamps[0])
            datapack_smooth.set_phase(phase_smooth,ant_idx=[i],time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack_smooth.set_variance(variance_smooth,ant_idx=[i],time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
        log.info("Saving {}".format(datapack_smooth_name))
        datapack_smooth.save(datapack_smooth_name)

        time_block_idx += 1
    #bayes_params.close()

if __name__=='__main__':
    main("output_complex","rvw_datapack_full_phase.hdf5","rvw_datapack_full_phase_smooth_complex.hdf5","bayes_parameters_complex.hdf5",120)
