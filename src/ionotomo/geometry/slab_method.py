"""An efficient computation of the intersection of rays and blocks"""


import numpy as np

import dask.array as da
from dask import delayed 
from dask.threaded import get

class Ray(object):
    def __init__(self,r0,n):
        self.r0 = np.array(r0)
        self.n = np.array(n)
        self.n /= np.linalg.norm(self.n)
        self.inv_n = 1./self.n
    def __call__(self,t):
        return self.r0 + t*self.n

def slab_method_ray_box(ray,x_min,y_min,z_min,x_max,y_max,z_max):
    tx1 = (x_min - ray.r0[0])*ray.inv_n[0]
    tx2 = (x_max - ray.r0[0])*ray.inv_n[0]
     
    if np.isnan(tx1):
        tx1 = 0.
    if np.isnan(tx2):
        tx2 = 0.
    
    tmin_x = min(tx1,tx2)
    tmax_x = max(tx1,tx2)

    ty1 = (y_min - ray.r0[1])*ray.inv_n[1]
    ty2 = (y_max - ray.r0[1])*ray.inv_n[1]
     
    if np.isnan(ty1):
        ty1 = 0.
    if np.isnan(ty2):
        ty2 = 0.
    
    tmin_y = min(ty1,ty2)
    tmax_y = max(ty1,ty2)


    tz1 = (z_min - ray.r0[2])*ray.inv_n[2]
    tz2 = (z_max - ray.r0[2])*ray.inv_n[2]
     
    if np.isnan(tz1):
        tz1 = 0.
    if np.isnan(tz2):
        tz2 = 0.
    
    tmin_z = min(tz1,tz2)
    tmax_z = max(tz1,tz2)
    tmax = max(tmin_x,tmin_y,tmin_z)
    tmin = min(tmax_x,tmax_y,tmax_z)
    if (tmax < tmin and tmax > 0):
        return np.linalg.norm(ray.n*(tmax - tmin)),ray.r0+ray.n*(tmax + tmin)/2.
    else:
        return 0.,[0.,0.,0.]

def slab_method_3d_ray(ray,xvec,yvec,zvec):
    """Rays is a list of ray objects, xvec, yvec, zvec are centers of voxels.
    Create an array of shape [nr,nx,ny,nz] that is the length of ray ijk in voxel"""
    #abssicas are xvec - dx/2. + one more
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    dz = zvec[1] - zvec[0]
    nx = len(xvec)
    ny = len(yvec)
    nz = len(zvec)
    ox = np.ones(nx)
    oy = np.ones(ny)
    oz = np.ones(nz)
    
    tx1 = (xvec-dx/2. - ray.r0[0])*ray.inv_n[0]
    tx2 = (xvec+dx/2. - ray.r0[0])*ray.inv_n[0]
    tx1[np.isnan(tx1)] = 0.
    tx2[np.isnan(tx2)] = 0.
    tmin_x = np.min([tx1,tx2],axis=0)
    tmax_x = np.max([tx1,tx2],axis=0)

    ty1 = (yvec-dy/2. - ray.r0[1])*ray.inv_n[1]
    ty2 = (yvec+dy/2. - ray.r0[1])*ray.inv_n[1]
    ty1[np.isnan(ty1)] = 0.
    ty2[np.isnan(ty2)] = 0.
    tmin_y = np.min([ty1,ty2],axis=0)
    tmax_y = np.max([ty1,ty2],axis=0)
    
    tz1 = (zvec-dz/2. - ray.r0[2])*ray.inv_n[2]
    tz2 = (zvec+dz/2. - ray.r0[2])*ray.inv_n[2]
    tz1[np.isnan(tz1)] = 0.
    tz2[np.isnan(tz2)] = 0.
    tmin_z = np.min([tz1,tz2],axis=0)
    tmax_z = np.max([tz1,tz2],axis=0)

    max_x = np.einsum("i,j,k->ijk",tmax_x,oy,oz)
    max_y = np.einsum("i,j,k->jik",tmax_y,ox,oz)
    max_z = np.einsum("i,j,k->kji",tmax_z,oy,ox)
    min_x = np.einsum("i,j,k->ijk",tmin_x,oy,oz)
    min_y = np.einsum("i,j,k->jik",tmin_y,ox,oz)
    min_z = np.einsum("i,j,k->kji",tmin_z,oy,ox)

    tmax_ = np.max([min_x,min_y,min_z],axis=0)
    tmin_ = np.min([max_x,max_y,max_z],axis=0)

    intersection = np.bitwise_or(np.bitwise_and(tmax_ < tmin_, tmax_>0),tmax_< 0)
    seg = np.einsum("ijk,l->ijkl",(tmax_ - tmin_),ray.n)
    seg *= seg
    seg = np.sum(seg,axis=-1)
    np.sqrt(seg,out=seg)

    out = np.where(intersection,tmax_-tmin_,0.)
    return out

def slab_method_3d(rays, xvec, yvec, zvec,out=None):
    """Rays is a list of ray objects, xvec, yvec, zvec are centers of voxels.
    Create an array of shape [nr,nx,ny,nz] that is the length of ray ijk in voxel"""
    #abssicas are xvec - dx/2. + one more
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    dz = zvec[1] - zvec[0]
    nx = len(xvec)
    ny = len(yvec)
    nz = len(zvec)
    nr = len(rays)
    ox = np.ones(nx)
    oy = np.ones(ny)
    oz = np.ones(nz)
    if out is not None:
        assert out.shape[0] == nr and out.shape[1] == nx and out.shape[2] == ny and out.shape[3] == nz
    else:
        out = np.zeros([nr,nx,ny,nz],dtype=float)
    
    max_xy = np.zeros([nx,ny],dtype=float)
    min_xy = np.zeros([nx,ny],dtype=float)
    intersection_xy = np.zeros([nx,ny],dtype=float)

    max_xz = np.zeros([nx,nz],dtype=float)
    min_xz = np.zeros([nx,nz],dtype=float)
    intersection_xz = np.zeros([nx,nz],dtype=float)

    max_yz = np.zeros([ny,nz],dtype=float)
    min_yz = np.zeros([ny,nz],dtype=float)
    intersection_yz = np.zeros([ny,nz],dtype=float)
    ray_idx = 0
    for ray in rays:
        out[ray_idx,...] = slab_method_3d_ray_dask(ray, xvec, yvec, zvec).compute()
        ray_idx += 1
    return out

def slab_method_3d_dask(rays, xvec, yvec, zvec,out=None,num_threads = None):
    """Rays is a list of ray objects, xvec, yvec, zvec are centers of voxels.
    Create an array of shape [nr,nx,ny,nz] that is the length of ray ijk in voxel"""
    #abssicas are xvec - dx/2. + one more
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    dz = zvec[1] - zvec[0]
    nx = len(xvec)
    ny = len(yvec)
    nz = len(zvec)
    nr = len(rays)
    ox = np.ones(nx)
    oy = np.ones(ny)
    oz = np.ones(nz)
    if out is not None:
        assert out.shape[0] == nr and out.shape[1] == nx and out.shape[2] == ny and out.shape[3] == nz
    else:
        out = np.zeros([nr,nx,ny,nz],dtype=float)
    
    max_xy = np.zeros([nx,ny],dtype=float)
    min_xy = np.zeros([nx,ny],dtype=float)
    intersection_xy = np.zeros([nx,ny],dtype=float)

    max_xz = np.zeros([nx,nz],dtype=float)
    min_xz = np.zeros([nx,nz],dtype=float)
    intersection_xz = np.zeros([nx,nz],dtype=float)

    max_yz = np.zeros([ny,nz],dtype=float)
    min_yz = np.zeros([ny,nz],dtype=float)
    intersection_yz = np.zeros([ny,nz],dtype=float)
    ray_idx = 0
    out = da.stack([da.from_delayed(delayed(slab_method_3d_ray)(ray,xvec,yvec,zvec),shape=(nx,ny,nz),dtype=float) for ray in rays],axis=0)
    return out.compute(get=get,num_workers=num_threads)

if __name__ =='__main__':
    r0 = np.array([0,0,0])
    n = np.array([0.05,0.05,0.05])
    ray = Ray(r0,n)
    xvec = np.linspace(0,1,100)
    yvec = np.linspace(0,1,100)
    zvec = np.linspace(0,10,1000)
    from timeit import timeit
    t1 = timeit()
    #res = slab_method_3d_dask([ray]*10,xvec,yvec,zvec,num_threads=8)       
    #for n in range(1,16):
    #    print(timeit(lambda : slab_method_3d_dask([ray]*20*15,xvec,yvec,zvec,num_threads=8),number = 1))
    from time import clock
    t1 = clock()
    print([slab_method_ray_box(ray,xvec[0]-1/99.,yvec[0]-1/99.,zvec[0]-1/999.,xvec[0] + 1./99.,yvec[0]+1./99.,zvec[0]+10./999.) for i in range(100)])
    print(clock() - t1)


        
                
        
