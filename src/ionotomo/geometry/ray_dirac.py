import numpy as np
from ionotomo.geometry.tri_cubic import bisection
from ionotomo.geometry.slab_method import slab_method_ray_box, Ray

def get_ray_dirac(rays,tci):
    '''Return dirac delta where of shape tci.M.shape with 1 where ray passes through.
    rays is chunked (usually by direction) N1,N2,4,N in shape and tci is a TriCubic object'''
    #easy way is to make sampling very fine and use bisection
    N1,N2,_,Ns = rays.shape
    x,y,z = tci.xvec,tci.yvec,tci.zvec
    dx,dy,dz = x[1]-x[0],y[1]-y[0],z[1]-z[0]
    dirac_ray = np.zeros(tci.M.shape,dtype=float)
    #because origin is always in set use origin for non interp
    seg_midpoints = np.zeros([3,N1,N2,len(x),len(y),len(z)],dtype=float) 
    dirac = np.zeros([N1,N2,len(x),len(y),len(z)],dtype=float)
    for i in range(N1):
        for j in range(N2):
            ray_x = rays[i,j,0,:]
            ray_y = rays[i,j,1,:]
            ray_z = rays[i,j,2,:]
            ray = Ray(rays[i,j,0:3,0],rays[i,j,0:3,-1]-rays[i,j,0:3,0])
            dirac_ray *= 0.
            for s in range(Ns):
                xi_c,yi_c,zi_c = bisection(x,ray_x[s]),bisection(y,ray_y[s]),bisection(z,ray_z[s])
                for xi in range(max(0,xi_c-1),min(len(x),xi_c+2)):
                    for yi in range(max(0,yi_c-1),min(len(y),yi_c+2)):
                        for zi in range(max(0,zi_c-1),min(len(z),zi_c+2)):
                            ds,midpoint = slab_method_ray_box(ray,x[xi]-dx/2.,y[yi]-dy/2.,z[zi]-dz/2.,x[xi]+dx/2,y[yi]+dy/2.,z[zi]+dz/2.)
                            #print(ds,midpoint)
                            seg_midpoints[:,i,j,xi,yi,zi] = midpoint
                            dirac_ray[xi,yi,zi] = ds
                           #print(dirac_ray[xi,yi,zi])
            dirac[i,j,...] += dirac_ray
    return dirac,seg_midpoints
