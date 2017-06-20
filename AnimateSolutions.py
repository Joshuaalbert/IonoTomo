
# coding: utf-8

# In[ ]:

import os
try:
    os.environ["QT_API"] = "pyqt"
    from mayavi import mlab
except:
    try:
        os.environ["QT_API"] = "pyside"
        from mayavi import mlab
    except:
        print("Unable to import mayavi")
from TricubicInterpolation import TriCubic
import numpy as np

def plotTCI(neTCI,rays=None,filename=None,show=False):
    xmin = neTCI.xvec[0]
    xmax = neTCI.xvec[-1]
    ymin = neTCI.yvec[0]
    ymax = neTCI.yvec[-1]
    zmin = neTCI.zvec[0]
    zmax = neTCI.zvec[-1]
    
    X,Y,Z = np.mgrid[xmin:xmax:len(neTCI.xvec)*1j,
                     ymin:ymax:len(neTCI.yvec)*1j,
                     zmin:zmax:len(neTCI.zvec)*1j]
    
    #reshape array
    data = neTCI.getShapedArray()
    import pylab as plt
    xy = np.mean(data,axis=2)
    yz = np.mean(data,axis=0)
    zx = np.mean(data,axis=1)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(xy,origin='lower',aspect='auto')
    ax2.imshow(yz,origin='lower',aspect='auto')
    ax3.imshow(zx,origin='lower',aspect='auto')
    plt.show()
    #print(np.mean(data),np.max(data),np.min(data))
    #mlab.close()
    #l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    #l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    #l._volume_property.shade = False
    mlab.contour3d(X,Y,Z,data,contours=10,opacity=0.2)
    mlab.colorbar()
    
    if rays is not None:
        for ray in rays.values():
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=0.75)
    if filename is not None:
        mlab.savefig('{}.png'.format(filename))#,magnification = 2)#size=(1920,1080))
    if show:
        mlab.show()
    mlab.close()

def animateSolutions(datafolder,template):
    '''animate the ne solutions'''
    import os
    import glob
    try:
        os.makedirs("{}/tmp".format(datafolder))
    except:
        pass
    files = glob.glob("{}/{}".format(datafolder,template))
    neTCIs = []
    for filename in files:
        print("plotting {}".format(filename))
        neTCIs.append(TriCubic(filename=filename))
    rays = np.load("{}/rays.npy".format(datafolder)).item(0)
    idx = 0
    for neTCI in neTCIs:
        plotTCI(neTCI,rays=rays[0]+rays[1]+rays[2],filename="{0}/tmp/fig{1:4d}.png".format(datafolder,idx),show=True)
        idx += 1
    #os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i {0}/tmp/fig%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {0}/solution_animation.mp4'.format(datafolder))
    
        
        
    
    
def plotWavefront(neTCI,rays,save=False,saveFile=None,animate=False):
    if saveFile is None:
        saveFile = "figs/wavefront.png"
    print("Saving to: {0}".format(saveFile))
    xmin = neTCI.xvec[0]
    xmax = neTCI.xvec[-1]
    ymin = neTCI.yvec[0]
    ymax = neTCI.yvec[-1]
    zmin = neTCI.zvec[0]
    zmax = neTCI.zvec[-1]
    
    X,Y,Z = np.mgrid[xmin:xmax:len(neTCI.xvec)*1j,
                     ymin:ymax:len(neTCI.yvec)*1j,
                     zmin:zmax:len(neTCI.zvec)*1j]
    
    #reshape array
    data = neTCI.getShapedArray()
    #print(np.mean(data),np.max(data),np.min(data))
    #mlab.close()
    #l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    #l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    #l._volume_property.shade = False
    mlab.contour3d(X,Y,Z,data,contours=10,opacity=0.2)
    mlab.colorbar()
    
    def getWave(rays,idx):
        xs = np.zeros(len(rays))
        ys = np.zeros(len(rays))
        zs = np.zeros(len(rays))
        ridx = 0
        while ridx < len(rays):
            xs[ridx] = rays[ridx]['x'][idx]
            ys[ridx] = rays[ridx]['y'][idx]
            zs[ridx] = rays[ridx]['z'][idx]
            ridx += 1
        return xs,ys,zs
    
    if rays is not None:
        for datumIdx in range(len(rays)):
            ray = rays[datumIdx]
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=0.75)
        if animate:
            plt = mlab.points3d(*getWave(rays,0),color=(1,0,0),scale_mode='vector', scale_factor=10.)
            #mlab.move(-200,0,0)
            view = mlab.view()
            @mlab.animate(delay=100)
            def anim():
                nt = len(rays[0]["s"])
                f = mlab.gcf()
                save = False
                while True:
                    i = 0
                    while i < nt:
                        #print("updating scene")
                        xs,ys,zs = getWave(rays,i)
                        plt.mlab_source.set(x=xs,y=ys,z=zs)
                        #mlab.view(*view)
                        if save:
                            #mlab.view(*view)
                            mlab.savefig('figs/wavefronts/wavefront_{0:04d}.png'.format(i))#,magnification = 2)#size=(1920,1080))
                        #f.scene.render()
                        i += 1
                        yield
                    save = False
            anim()
    
    if save and animate:
        import os
        os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i figs/wavefronts/wavefront_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p figs/wavefronts/wavefront.mp4')
    else:
        if save:
            mlab.savefig(saveFile,figure=mlab.gcf())
        else:
            mlab.show()
            
            
def plotModel(neTCI,save=False):
    '''Plot the model contained in a tricubic interpolator (a convienient container for one)'''
    plotWavefront(neTCI,None,save=save)
    
def animateResults(files):
    from TricubicInterpolation import TriCubic
    import os
    images = []
    index = 0
    for file in files:
        
        abspath = os.path.abspath(file)
        print("Plotting: {0}".format(abspath))
        if os.path.isfile(abspath):
            dir = os.path.dirname(abspath)
            froot = os.path.split(abspath)[1].split('.')[0]
        else:
            continue
        data = np.load(abspath)
        xvec = data['xvec']
        yvec = data['yvec']
        zvec = data['zvec']
        M = data['M']
        Kmu = data['Kmu'].item(0)
        rho = data['rho']
        Krho = data['Krho'].item(0)
        if 'rays' in data.keys():
            rays = data['rays'].item(0)
        else:
            rays = None
        TCI = TriCubic(xvec,yvec,zvec,M)
        TCI.clearCache()
        images.append("{0}/frame-{1:04d}.png".format(dir,index))
        plotWavefront(TCI,rays,save=True,saveFile=images[-1],animate=False)
        index += 1
    os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i {0}/frame-%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {1}/solution_animation.mp4'.format(dir,dir))
    print("Saved to {1}/solution_animation.mp4".format(dir))
    
if __name__ == '__main__':
    #plotTCI(neTCI,rays=None,filename=None,show=False)
    animateSolutions("output/simulatedInversion_2","initial_neModel.npy")
    

