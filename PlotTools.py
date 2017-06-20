
# coding: utf-8

# In[1]:

import pylab as plt
import numpy as np
import os

def makeAnimation(datafolder,prefix='fig',fps=3):
    if os.system('ffmpeg.exe -framerate {} -i {}/{}-%04d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 {}/animation.mp4'.format(fps,datafolder,prefix,datafolder)):
        print("{}/animation.mp4 exists already".format(datafolder))


def animateTCISlices(TCI,outputFolder,numSeconds=10.):
    try:
        os.makedirs(outputFolder)
    except:
        pass
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    M = TCI.getShapedArray()
    if np.sum(M<0) > 0:
        print("Using linear scaling")
        logSpacing = False
    else:
        print("Using log scaling")
        logSpacing = True
        M[M==0] = np.min(M[M>0])
    levels = [np.min(M),np.max(M)]
    for q in np.linspace(1,99,15*5+2):
        if logSpacing:
            l = 10**np.percentile(np.log10(M),q)
            if l not in levels and len(levels) < 15:
                levels.append(l)
        else:
            l = np.percentile(M,q)
            if l not in levels  and len(levels) < 15:
                levels.append(l) 
    levels = np.sort(levels)
    #N = max(1,int((len(levels)-2)/13))
    #levels = [levels[0]] + levels[1:-1][::N] + [levels[-1]]
    print("plotting levels : {}".format(levels))
    #M[M<levels[0]] = np.nan
    #M[M>levels[-1]] = np.nan
    vmin = np.min(M)
    vmax = np.max(M)
    Y_1,X_1 = np.meshgrid(TCI.yvec,TCI.xvec,indexing='ij')
    Z_2,Y_2 = np.meshgrid(TCI.zvec,TCI.yvec,indexing='ij')
    Z_3,X_3 = np.meshgrid(TCI.zvec,TCI.xvec,indexing='ij')
    i = 0
    while i < TCI.nz:
        xy = M[:,:,i].transpose()#x by y
        j1 = int(i/float(TCI.nz)*TCI.nx)
        #j1 = TCI.nx >> 1
        yz = M[j1,:,:].transpose()#y by z
        j2 = (TCI.ny - 1) - int(i/float(TCI.nz)*TCI.ny)
        #j2 = TCI.ny >> 1
        xz = M[:,j2,:].transpose()#x by z
        #yz 
#         zmask = slice(0,TCI.nz)
#         ymask = slice(0,TCI.ny)
#         im = ax1.scatter(np.ones(Y_2.shape)[zmask,ymask].flatten()*TCI.xvec[0],Y_2[zmask,ymask].flatten(),
#                          Z_2[zmask,ymask].flatten(),c=yz[zmask,ymask].flatten(),vmin=vmin,vmax=vmax,marker='.')
#         #xz
#         zmask = slice(0,TCI.nz)
#         xmask = slice(0,TCI.nx)
#         im = ax1.scatter(X_3[zmask,xmask].flatten(),np.ones(X_3.shape)[zmask,xmask].flatten()*TCI.yvec[TCI.ny - 1],
#                          Z_3[zmask,xmask].flatten(),c=xz[zmask,xmask].flatten(),vmin=vmin,vmax=vmax,marker='.')    
#         im = ax1.scatter(X_1.flatten(),Y_1.flatten(),np.ones(X_1.size)*TCI.zvec[i],
#                          c=xy.flatten(),vmin=vmin,vmax=vmax,marker='.')
#         ax1.set_xlabel('X km')
#         ax1.set_ylabel('Y km')
#         ax1.set_zlabel('Z km')
        
#         ax1.set_xlim([TCI.xvec[0],TCI.xvec[-1]])
#         ax1.set_ylim([TCI.yvec[0],TCI.yvec[-1]])
#         ax1.set_zlim([TCI.zvec[0],TCI.zvec[-1]])
        
        im = ax2.imshow(xy,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.xvec[0],TCI.xvec[-1],TCI.yvec[0],TCI.yvec[-1]],cmap=plt.cm.bone)
        CS = ax2.contour(xy, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.xvec[0],TCI.xvec[-1],TCI.yvec[0],TCI.yvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        ax2.set_title("Height: {:.2g} km".format(TCI.zvec[i]))
        ax2.set_xlabel('X km')
        ax2.set_ylabel('Y km')
        
        im = ax3.imshow(yz,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.yvec[0],TCI.yvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.bone)
        CS = ax3.contour(yz, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.yvec[0],TCI.yvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        #ax3.set_title("Solution")
        ax3.set_title("X_slice: {:.2g} km".format(TCI.xvec[j1]))
        ax3.set_ylabel('Z km')
        ax3.set_xlabel('Y km')
        
        im = ax4.imshow(xz,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.xvec[0],TCI.xvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.bone)
        CS = ax4.contour(xz, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.xvec[0],TCI.xvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        ax4.set_title("Y_slice: {:.2g} km".format(TCI.yvec[j2]))
        ax4.set_xlabel('X km')
        ax4.set_ylabel('Z km')
        plt.savefig("{}/fig-{:04d}.png".format(outputFolder,i))  
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        i += 1
    makeAnimation(outputFolder,prefix='fig',fps=int(TCI.nz/float(numSeconds)))
    
def animateDatapack(datapack,outputfolder, antIdx=-1,timeIdx=-1,dirIdx=-1):
    from RealData import plotDataPack
    try:
        os.makedirs(outputfolder)
    except:
        pass
    antennas,antennaLabels = datapack.get_antennas(antIdx = antIdx)
    patches, patchNames = datapack.get_directions(dirIdx = dirIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    datapack.setReferenceAntenna(antennaLabels[0])
    #plotDataPack(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx,figname='{}/dobs'.format(outputfolder))
    dobs = datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx)
    vmin = np.percentile(dobs,1)
    vmax = np.percentile(dobs,99)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    j = 0
    idx = 0
    while j < Nt:
        fig = "{}/fig-{:04d}".format(outputfolder,idx)
        plotDataPack(datapack,antIdx=antIdx,timeIdx=[j,j+1], dirIdx=dirIdx,figname=fig,vmin=vmin,vmax=vmax)
        idx += 1
        j += 2
    makeAnimation(outputFolder,prefix="fig".format(outputfolder),fps=int(5.))
            
def test_animateTCISlices():
    from TricubicInterpolation import TriCubic
    TCI = TriCubic(filename="output/test/neModelTurbulent.hdf5").copy()
    import os
    try: 
        os.makedirs("output/test/fig")
    except:
        pass
    animateTCISlices(TCI,"output/test/fig")
    
def test_animateDatapack():
    from RealData import DataPack
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    animateDatapack(datapack,"output/test/animateDatapack", antIdx=-1,timeIdx=-1,dirIdx=-1)
if __name__ == '__main__':
    #test_animateTCISlices()
    #test_animateDatapack()
    makeAnimation("output/test/animateDatapack",prefix="fig",fps=int(5.))
    


# In[ ]:



