
# coding: utf-8

# In[ ]:

####
## All the plotting functions required
###

from mayavi import mlab
import numpy as np

def plotWavefront(neTCI,rays,save=False,animate=False):
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
    print(np.mean(data),np.max(data),np.min(data))
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    l._volume_property.shade = False
    mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
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
        for datumIdx in rays.keys():
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
    mlab.show()
    if save and rays is not None:
        return
        import os
        os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i figs/wavefronts/wavefront_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p figs/wavefronts/wavefront.mp4')

def plotModel(neTCI,save=False):
    '''Plot the model contained in a tricubic interpolator (a convienient container for one)'''
    plotWavefront(neTCI,None,save=save)
    

