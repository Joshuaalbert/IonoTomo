
# coding: utf-8

# In[3]:

'''Based on the paper doi=10.1.1.89.7835
the tricubic interpolation of a regular possibly non uniform grid can be seen as a computation of 21 cubic splines.
A cubic spline is a special case of cubic interpolation, and in general these 21 cubic splines perform many redundant
calulations. Here we formulate the full tricubic interpolation where the value of a scalar function defined on a 3d
grid can be reconstructed to allow full C1, and thus langrangian structures to persist.'''

import numpy as np

class TriCubic(object):
    def __init__(self,xvec,yvec,zvec,M,copy = False, useCache = True,default=0.,xUniform=True,yUniform=True,zUniform = True):
        self.default = default
        self.nx = np.size(xvec)
        self.ny = np.size(yvec)
        self.nz = np.size(zvec)
        self.xvec = xvec.ravel(order='C')
        self.yvec = yvec.ravel(order='C')
        self.zvec = zvec.ravel(order='C')
        #determine uniformity
        dx = self.xvec[1:] - self.xvec[:-1]
        self.dx = np.mean(dx)
        self.xUniform = xUniform
        dy = self.yvec[1:] - self.yvec[:-1]
        self.dy = np.mean(dy)
        self.yUniform = yUniform    
        dz = self.zvec[1:] - self.zvec[:-1]
        self.dz = np.mean(dz)
        self.zUniform = zUniform
            
        self.m = M.ravel(order='C')
        self.setBinv()
        #self.checkIndexing(M)
        self.useCache = useCache
        if self.useCache:
            self.cache = {}
        else:
            self.cache = None
        self.cijk0 = None
        self.iPowers,self.jPowers,self.kPowers = np.meshgrid([0,1,2,3],[0,1,2,3],[0,1,2,3],indexing='ij')
        #print(self.iPowers,self.jPowers,self.kPowers)
    def copy(self):
        return TriCubic(self.xvec.copy(),self.yvec.copy(),self.zvec.copy(),self.m.copy())
    def index(self,i,j,k):
        '''Correct indexing of 3-tensor in ravelled vector'''
        return k + self.nz*(j + self.ny*i)
        
    def checkIndexing(self,M,N=100):
        '''Check that ordering of elements is correct'''
        N = min(N,np.size(self.m))
        idx = 0
        while idx < N:
            i = np.random.randint(self.nx)
            j = np.random.randint(self.ny)
            k = np.random.randint(self.nz)
            assert self.m[self.index(i,j,k)] == M[i,j,k], "Ordering of indexing is wrong"
            idx += 1
        return True
    
    def interp3(self,x,y,z,doDouble=False):
        '''Do only on single points'''
        #get cell starts for all dimensions
        if self.xUniform:
            xi = int((x - self.xvec[0])/self.dx)
        else:
            xi = np.argmin(np.abs(np.floor(x - self.xvec)))
        if self.yUniform:
            yi = int((y - self.yvec[0])/self.dy)
        else:
            yi = np.argmin(np.abs(np.floor(y - self.yvec)))
        if self.zUniform:
            zi = int((z - self.zvec[0])/self.dz)
        else:
            zi = np.argmin(np.abs(np.floor(z - self.zvec)))
        try:
            # require one extra room for edge case because we need a 3x3x3 cube to get C1 interpolant (could try non central diff)
            assert xi > 0 and xi < self.nx-2, "x {0} out of range: [{1} ... {2]}]".format(x,self.xvec[0],self.xvec[-1])
            assert yi > 0 and yi < self.ny-2, "y {0} out of range: [{1} ... {2]}]".format(y,self.yvec[0],self.yvec[-1])
            assert zi > 0 and zi < self.nz-2, "z {0} out of range: [{1} ... {2]}]".format(z,self.zvec[0],self.zvec[-1])
        except:
            #else we could return a default value (easy to add later)
            if xi <= 0:
                xi = 0
            if xi >= self.nx - 2:
                xi = self.nx - 2
            if yi <= 0:
                yi = 0
            if yi >= self.ny - 2:
                yi = self.ny - 2
            if zi <= 0:
                zi = 0
            if zi >= self.nz - 2:
                zi = self.nz - 2
            self.f = self.m[self.index(xi,yi,zi)]
            self.fx = 0.
            self.fy = 0.
            self.fz = 0.
            return self.f,self.fx,self.fy,self.fz
        #get interpolant from current, cache, or build
        ijk0 = self.index(xi,yi,zi)#bottom corner of cube
        if self.cijk0 == ijk0:
            A_ijk = self.cA_ijk#the interpolant of the cube
        else:
            build = True
            if self.useCache:
                if ijk0 in self.cache.keys():
                    A_ijk = self.cache[ijk0]
                    build = False
            if build:
                b = self.get_bVec(xi,yi,zi)
                A_ijk = (self.Binv.dot(b)).reshape([4,4,4])
                if self.useCache:
                    self.cache[ijk0] = A_ijk
            self.cijk0 = ijk0
            self.cA_ijk = A_ijk
        #use interpolant to calculate f,fx,fy,fz
        u = (x - self.xvec[xi])#/self.dx
        v = (y - self.yvec[yi])#/self.dy
        w = (z - self.zvec[zi])#/self.dz
        U = u**(self.iPowers)
        V = v**(self.jPowers)
        W = w**(self.kPowers)
        inter = A_ijk*U*V*W
        f = np.sum(inter)
        #fx = i * a_ijk * x^(i-1) y^j z^k
        # = i * a_ijk * x^i y^j z^k / x
        # x = 0 implies only i = 1
        # fx = a_1jk y^j z^k
        if u == 0.:
            fx = np.sum((A_ijk*V*W)[1,:,:])
        else:
            fx = inter*self.iPowers
            fx /= u
            fx = np.sum(fx)
        if v == 0.:
            fy = np.sum((A_ijk*U*W)[:,1,:])
        else:
            fy = inter*self.jPowers
            fy /= v
            fy = np.sum(fy)
        if w == 0.:
            fz = np.sum((A_ijk*U*V)[:,:,1])
        else:
            fz = inter*self.kPowers
            fz /= w
            fz = np.sum(fz)
        #fxy = ij A_ijk x^(i-1) y^(j-1) z^k
        # x = 0 implies i = 1, y = 0 implies j = 1
        #fxy = i j A_ijk x^(i-1) y^(j-1) z^k
        #
        if doDouble:
            fxy,fxz,fyz = 0.,0.,0.
            i = 0
            while i <= 3:
                j = 0
                while j <= 3:
                    k = 0
                    while k <= 3:
                        #ijk = k + 4*(j + 4*i)
                        a = A_ijk[i,j,k]
                        if i>0 and j>0:
                            fxy += i*j*a * u**(i-1) * v**(j-1) * w**k
                        if i>0 and k>0:
                            fxz += i*k*a * u**(i-1) * v**j * w**(k-1)
                        if j>0 and k>0:
                            fyz += j*k*a * u**i * v**(j-1) * w**(k-1)
                        k += 1
                    j += 1
                i += 1
            return f,fx,fy,fz,fxy,fxz,fyz
        
        return f,fx,fy,fz
        """
        f,fx,fy,fz = 0.,0.,0.,0.
        i = 0
        while i <= 3:
            j = 0
            while j <= 3:
                k = 0
                while k <= 3:
                    ijk = k + 4*(j + 4*i)
                    a = A_ijk[ijk]
                    f += a * u**i * v**j * w**k
                    if i != 0:
                        fx += i*a * u**(i-1) * v**j * w**k
                    if j != 0:
                        fy += j*a * u**i * v**(j-1) * w**k
                    if k != 0:
                        fz += k*a * u**i * v**j * w**(k-1)
                    k += 1
                j += 1
            i += 1
        return f,fx,fy,fz"""
        
    def interp1(self,x,y,z,doDouble=False):
        '''Do only on single points'''
        #get cell starts for all dimensions
        if self.xUniform:
            xi = int((x - self.xvec[0])/self.dx)
        else:
            xi = np.argmin(np.abs(np.floor(x - self.xvec)))
        if self.yUniform:
            yi = int((y - self.yvec[0])/self.dy)
        else:
            yi = np.argmin(np.abs(np.floor(y - self.yvec)))
        if self.zUniform:
            zi = int((z - self.zvec[0])/self.dz)
        else:
            zi = np.argmin(np.abs(np.floor(z - self.zvec)))
        try:
            # require one extra room for edge case because we need a 3x3x3 cube to get C1 interpolant (could try non central diff)
            assert xi > 0 and xi < self.nx-2, "x {0} out of range: [{1} ... {2]}]".format(x,self.xvec[0],self.xvec[-1])
            assert yi > 0 and yi < self.ny-2, "y {0} out of range: [{1} ... {2]}]".format(y,self.yvec[0],self.yvec[-1])
            assert zi > 0 and zi < self.nz-2, "z {0} out of range: [{1} ... {2]}]".format(z,self.zvec[0],self.zvec[-1])
        except:
            #else we could return a default value (easy to add later)
            if xi <= 0:
                xi = 0
            if xi >= self.nx - 2:
                xi = self.nx - 2
            if yi <= 0:
                yi = 0
            if yi >= self.ny - 2:
                yi = self.ny - 2
            if zi <= 0:
                zi = 0
            if zi >= self.nz - 2:
                zi = self.nz - 2
            f = self.m[self.index(xi,yi,zi)]
            return f
        #get interpolant from current, cache, or build
        ijk0 = self.index(xi,yi,zi)#bottom corner of cube
        if self.cijk0 == ijk0:
            A_ijk = self.cA_ijk#the interpolant of the cube
        else:
            build = True
            if self.useCache:
                if ijk0 in self.cache.keys():
                    A_ijk = self.cache[ijk0]
                    build = False
            if build:
                b = self.get_bVec(xi,yi,zi)
                A_ijk = (self.Binv.dot(b)).reshape([4,4,4])
                if self.useCache:
                    self.cache[ijk0] = A_ijk
            self.cijk0 = ijk0
            self.cA_ijk = A_ijk
        #use interpolant to calculate f,fx,fy,fz
        u = (x - self.xvec[xi])#/self.dx
        v = (y - self.yvec[yi])#/self.dy
        w = (z - self.zvec[zi])#/self.dz
        U = u**(self.iPowers)
        V = v**(self.jPowers)
        W = w**(self.kPowers)
        inter = A_ijk*U*V*W
        f = np.sum(inter)
        
        return f
    
            
    def get_bVec(self,i,j,k):
        '''Get the corner vec defined by f, fx,fy,fz,fxy,fxz,fyz,fxyz'''
        im = i - 1
        iz = i
        ip = i + 1
        iP = i + 2
        jm = j - 1
        jz = j
        jp = j + 1
        jP = j + 2
        km = k - 1
        kz = k
        kp = k + 1
        kP = k + 2
        mmm = self.index(im,jm,km)
        mmz = self.index(im,jm,kz)
        mmp = self.index(im,jm,kp)
        mmP = self.index(im,jm,kP)
        mzm = self.index(im,jz,km)
        mzz = self.index(im,jz,kz)
        mzp = self.index(im,jz,kp)
        mzP = self.index(im,jz,kP)
        mpm = self.index(im,jp,km)
        mpz = self.index(im,jp,kz)
        mpp = self.index(im,jp,kp)
        mpP = self.index(im,jp,kP)
        mPm = self.index(im,jP,km)
        mPz = self.index(im,jP,kz)
        mPp = self.index(im,jP,kp)
        mPP = self.index(im,jP,kP)
        zmm = self.index(iz,jm,km)
        zmz = self.index(iz,jm,kz)
        zmp = self.index(iz,jm,kp)
        zmP = self.index(iz,jm,kP)
        zzm = self.index(iz,jz,km)
        zzz = self.index(iz,jz,kz)
        zzp = self.index(iz,jz,kp)
        zzP = self.index(iz,jz,kP)
        zpm = self.index(iz,jp,km)
        zpz = self.index(iz,jp,kz)
        zpp = self.index(iz,jp,kp)
        zpP = self.index(iz,jp,kP)
        zPm = self.index(iz,jP,km)
        zPz = self.index(iz,jP,kz)
        zPp = self.index(iz,jP,kp)
        zPP = self.index(iz,jP,kP)
        pmm = self.index(ip,jm,km)
        pmz = self.index(ip,jm,kz)
        pmp = self.index(ip,jm,kp)
        pmP = self.index(ip,jm,kP)
        pzm = self.index(ip,jz,km)
        pzz = self.index(ip,jz,kz)
        pzp = self.index(ip,jz,kp)
        pzP = self.index(ip,jz,kP)
        ppm = self.index(ip,jp,km)
        ppz = self.index(ip,jp,kz)
        ppp = self.index(ip,jp,kp)
        ppP = self.index(ip,jp,kP)
        pPm = self.index(ip,jP,km)
        pPz = self.index(ip,jP,kz)
        pPp = self.index(ip,jP,kp)
        pPP = self.index(ip,jP,kP)
        Pmm = self.index(iP,jm,km)
        Pmz = self.index(iP,jm,kz)
        Pmp = self.index(iP,jm,kp)
        PmP = self.index(iP,jm,kP)
        Pzm = self.index(iP,jz,km)
        Pzz = self.index(iP,jz,kz)
        Pzp = self.index(iP,jz,kp)
        PzP = self.index(iP,jz,kP)
        Ppm = self.index(iP,jp,km)
        Ppz = self.index(iP,jp,kz)
        Ppp = self.index(iP,jp,kp)
        PpP = self.index(iP,jp,kP)
        PPm = self.index(iP,jP,km)
        PPz = self.index(iP,jP,kz)
        PPp = self.index(iP,jP,kp)
        PPP = self.index(iP,jP,kP)
        x0 = -self.m[pzz]
        x1 = self.m[mzz] + x0
        x2 = 1/(self.xvec[im] - self.xvec[ip])
        x3 = -self.m[zpz]
        x4 = 1/(self.yvec[jm] - self.yvec[jp])
        x5 = -self.m[zzp]
        x6 = 1/(self.zvec[km] - self.zvec[kp])
        x7 = -self.m[pmz]
        x8 = -self.m[mpz] + self.m[ppz]
        x9 = x2*x4
        x10 = -self.m[pzm]
        x11 = -self.m[mzp] + self.m[pzp]
        x12 = x2*x6
        x13 = -self.m[zmp]
        x14 = -self.m[zpm]
        x15 = x4*x6
        x16 = -self.m[ppm]
        x17 = self.m[mpm] + x16
        x18 = -self.m[mpp]
        x19 = -self.m[pmp]
        x20 = -self.m[pzp]
        x21 = self.m[mzp] + x20
        x22 = -self.m[zpp]
        x23 = -self.m[zzz]
        x24 = 1/(self.zvec[kP] - self.zvec[kz])
        x25 = self.m[ppp] + x18
        x26 = -self.m[pzP]
        x27 = -self.m[mzz] + self.m[pzz]
        x28 = x2*x24
        x29 = -self.m[zmz]
        x30 = -self.m[zpP]
        x31 = x24*x4
        x32 = -self.m[ppz]
        x33 = self.m[mpz] + x32
        x34 = 1/(self.yvec[jP] - self.yvec[jz])
        x35 = -self.m[pPz]
        x36 = x2*x34
        x37 = -self.m[zPp]
        x38 = -self.m[zzm]
        x39 = x34*x6
        x40 = -self.m[ppp]
        x41 = -self.m[pPp]
        x42 = -self.m[ppP]
        x43 = -self.m[zPz]
        x44 = -self.m[zzP]
        x45 = x24*x34
        x46 = self.m[Pzz] + x23
        x47 = 1/(self.xvec[iP] - self.xvec[iz])
        x48 = -self.m[Ppz] + self.m[zpz]
        x49 = x4*x47
        x50 = -self.m[Pzp] + self.m[zzp]
        x51 = x47*x6
        x52 = self.m[Ppp] + x22
        x53 = self.m[Pzp] + x5
        x54 = -self.m[Ppp] + self.m[zpp]
        x55 = -self.m[Pzz] + self.m[zzz]
        x56 = x24*x47
        x57 = self.m[Ppz] + x3
        x58 = x34*x47
        bvec = np.array([self.m[zzz],
         x1*x2,
         x4*(self.m[zmz] + x3),
         x6*(self.m[zzm] + x5),
         x9*(self.m[mmz] + x7 + x8),
         x12*(self.m[mzm] + x10 + x11),
         x15*(self.m[zmm] + self.m[zpp] + x13 + x14),
         x15*(-self.m[mmm] + self.m[mmp] + self.m[pmm] + self.m[ppp] + x17 + x18 + x19)/(-self.xvec[im] + self.xvec[ip]),
         self.m[zzp],
         x2*x21,
         x4*(self.m[zmp] + x22),
         x24*(self.m[zzP] + x23),
         x9*(self.m[mmp] + x19 + x25),
         x28*(self.m[mzP] + x26 + x27),
         x31*(self.m[zmP] + self.m[zpz] + x29 + x30),
         x24*x9*(self.m[mmP] - self.m[mmz] - self.m[mpP] - self.m[pmP] + self.m[pmz] + self.m[ppP] + x33),
         self.m[zpz],
         x2*x33,
         x34*(self.m[zPz] + x23),
         x6*(self.m[zpm] + x22),
         x36*(self.m[mPz] + x27 + x35),
         x12*(x17 + x25),
         x39*(self.m[zPm] + self.m[zzp] + x37 + x38),
         x36*x6*(self.m[mPm] - self.m[mPp] - self.m[mzm] - self.m[pPm] + self.m[pPp] + self.m[pzm] + x21),
         self.m[zpp],
         x2*(self.m[mpp] + x40),
         x34*(self.m[zPp] + x5),
         x24*(self.m[zpP] + x3),
         x36*(self.m[mPp] + x11 + x41),
         x28*(self.m[mpP] + x42 + x8),
         x45*(self.m[zPP] + self.m[zzz] + x43 + x44),
         x24*x36*(self.m[mPP] - self.m[mPz] - self.m[mzP] - self.m[pPP] + self.m[pPz] + self.m[pzP] + x1),
         self.m[pzz],
         x46*x47,
         x4*(self.m[pmz] + x32),
         x6*(self.m[pzm] + x20),
         x49*(self.m[Pmz] + x29 + x48),
         x51*(self.m[Pzm] + x38 + x50),
         x15*(self.m[pmm] + self.m[ppp] + x16 + x19),
         x49*x6*(self.m[Pmm] - self.m[Pmp] - self.m[Ppm] - self.m[zmm] + self.m[zmp] + self.m[zpm] + x52),
         self.m[pzp],
         x47*x53,
         x4*(self.m[pmp] + x40),
         x24*(self.m[pzP] + x0),
         x49*(self.m[Pmp] + x13 + x54),
         x56*(self.m[PzP] + x44 + x55),
         x31*(self.m[pmP] + self.m[ppz] + x42 + x7),
         x24*x49*(self.m[PmP] - self.m[Pmz] - self.m[PpP] - self.m[zmP] + self.m[zmz] + self.m[zpP] + x57),
         self.m[ppz],
         x47*x57,
         x34*(self.m[pPz] + x0),
         x6*(self.m[ppm] + x40),
         x58*(self.m[PPz] + x43 + x55),
         x51*(self.m[Ppm] + x14 + x54),
         x39*(self.m[pPm] + self.m[pzp] + x10 + x41),
         x58*x6*(self.m[PPm] - self.m[PPp] - self.m[Pzm] - self.m[zPm] + self.m[zPp] + self.m[zzm] + x53),
         self.m[ppp],
         x47*x52,
         x34*(self.m[pPp] + x20),
         x24*(self.m[ppP] + x32),
         x58*(self.m[PPp] + x37 + x50),
         x56*(self.m[PpP] + x30 + x48),
         x45*(self.m[pPP] + self.m[pzz] + x26 + x35),
         x24*x58*(self.m[PPP] - self.m[PPz] - self.m[PzP] - self.m[zPP] + self.m[zPz] + self.m[zzP] + x46)])
        return bvec
    
    def setBinv(self):
        self.Binv = np.array([
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-3,0,0,-2,0,0,0,0,3,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [2,0,0,1,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,-3,0,0,0,-2,0,0,0,3,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,2,0,0,0,1,0,0,0,-2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,-3,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [9,0,6,6,0,0,4,0,-9,0,-6,3,0,0,2,0,-9,0,3,-6,0,0,2,0,9,0,-3,-3,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-6,0,-4,-3,0,0,-2,0,6,0,4,-3,0,0,-2,0,6,0,-2,3,0,0,-1,0,-6,0,2,3,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-6,0,-3,-4,0,0,-2,0,6,0,3,-2,0,0,-1,0,6,0,-3,4,0,0,-2,0,-6,0,3,2,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [4,0,2,2,0,0,1,0,-4,0,-2,2,0,0,1,0,-4,0,2,-2,0,0,1,0,4,0,-2,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,-3,0,0,0,-2,0,0,0,3,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,2,0,0,0,1,0,0,0,-2,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,-3,0,0,-2,0,0,0,0,3,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,2,0,0,1,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,-3,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,9,0,0,6,6,0,4,0,-9,0,0,-6,3,0,2,0,-9,0,0,3,-6,0,2,0,9,0,0,-3,-3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,-6,0,0,-4,-3,0,-2,0,6,0,0,4,-3,0,-2,0,6,0,0,-2,3,0,-1,0,-6,0,0,2,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,-6,0,0,-3,-4,0,-2,0,6,0,0,3,-2,0,-1,0,6,0,0,-3,4,0,-2,0,-6,0,0,3,2,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,4,0,0,2,2,0,1,0,-4,0,0,-2,2,0,1,0,-4,0,0,2,-2,0,1,0,4,0,0,-2,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-3,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [9,6,0,6,0,4,0,0,-9,-6,0,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-9,3,0,-6,0,2,0,0,9,-3,0,-3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-6,-4,0,-3,0,-2,0,0,6,4,0,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,-2,0,3,0,-1,0,0,-6,2,0,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,-3,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,9,0,6,0,6,4,0,0,-9,0,-6,0,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-9,0,3,0,-6,2,0,0,9,0,-3,0,-3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,-6,0,-4,0,-3,-2,0,0,6,0,4,0,-3,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,-2,0,3,-1,0,0,-6,0,2,0,3,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [9,6,6,0,4,0,0,0,0,0,0,0,0,0,0,0,-9,-6,3,0,2,0,0,0,0,0,0,0,0,0,0,0,-9,3,-6,0,2,0,0,0,0,0,0,0,0,0,0,0,9,-3,-3,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,9,0,6,6,4,0,0,0,0,0,0,0,0,0,0,0,-9,0,-6,3,2,0,0,0,0,0,0,0,0,0,0,0,-9,0,3,-6,2,0,0,0,0,0,0,0,0,0,0,0,9,0,-3,-3,1,0,0,0,0,0,0,0,0],
                [-27,-18,-18,-18,-12,-12,-12,-8,27,18,18,-9,12,-6,-6,-4,27,18,-9,18,-6,12,-6,-4,-27,-18,9,9,6,6,-3,-2,27,-9,18,18,-6,-6,12,-4,-27,9,-18,9,6,-3,6,-2,-27,9,9,-18,-3,6,6,-2,27,-9,-9,-9,3,3,3,-1],
                [18,12,12,9,8,6,6,4,-18,-12,-12,9,-8,6,6,4,-18,-12,6,-9,4,-6,3,2,18,12,-6,-9,-4,-6,3,2,-18,6,-12,-9,4,3,-6,2,18,-6,12,-9,-4,3,-6,2,18,-6,-6,9,2,-3,-3,1,-18,6,6,9,-2,-3,-3,1],
                [-6,-4,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,6,4,-3,0,-2,0,0,0,0,0,0,0,0,0,0,0,6,-2,3,0,-1,0,0,0,0,0,0,0,0,0,0,0,-6,2,3,0,-1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,-6,0,-4,-3,-2,0,0,0,0,0,0,0,0,0,0,0,6,0,4,-3,-2,0,0,0,0,0,0,0,0,0,0,0,6,0,-2,3,-1,0,0,0,0,0,0,0,0,0,0,0,-6,0,2,3,-1,0,0,0,0,0,0,0,0],
                [18,12,9,12,6,8,6,4,-18,-12,-9,6,-6,4,3,2,-18,-12,9,-12,6,-8,6,4,18,12,-9,-6,-6,-4,3,2,-18,6,-9,-12,3,4,-6,2,18,-6,9,-6,-3,2,-3,1,18,-6,-9,12,3,-4,-6,2,-18,6,9,6,-3,-2,-3,1],
                [-12,-8,-6,-6,-4,-4,-3,-2,12,8,6,-6,4,-4,-3,-2,12,8,-6,6,-4,4,-3,-2,-12,-8,6,6,4,4,-3,-2,12,-4,6,6,-2,-2,3,-1,-12,4,-6,6,2,-2,3,-1,-12,4,6,-6,-2,2,3,-1,12,-4,-6,-6,2,2,3,-1],
                [2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-6,-3,0,-4,0,-2,0,0,6,3,0,-2,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,-3,0,4,0,-2,0,0,-6,3,0,2,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [4,2,0,2,0,1,0,0,-4,-2,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-4,2,0,-2,0,1,0,0,4,-2,0,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,-6,0,-3,0,-4,-2,0,0,6,0,3,0,-2,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,-3,0,4,-2,0,0,-6,0,3,0,2,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,4,0,2,0,2,1,0,0,-4,0,-2,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-4,0,2,0,-2,1,0,0,4,0,-2,0,-2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [-6,-3,-4,0,-2,0,0,0,0,0,0,0,0,0,0,0,6,3,-2,0,-1,0,0,0,0,0,0,0,0,0,0,0,6,-3,4,0,-2,0,0,0,0,0,0,0,0,0,0,0,-6,3,2,0,-1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,-6,0,-3,-4,-2,0,0,0,0,0,0,0,0,0,0,0,6,0,3,-2,-1,0,0,0,0,0,0,0,0,0,0,0,6,0,-3,4,-2,0,0,0,0,0,0,0,0,0,0,0,-6,0,3,2,-1,0,0,0,0,0,0,0,0],
                [18,9,12,12,6,6,8,4,-18,-9,-12,6,-6,3,4,2,-18,-9,6,-12,3,-6,4,2,18,9,-6,-6,-3,-3,2,1,-18,9,-12,-12,6,6,-8,4,18,-9,12,-6,-6,3,-4,2,18,-9,-6,12,3,-6,-4,2,-18,9,6,6,-3,-3,-2,1],
                [-12,-6,-8,-6,-4,-3,-4,-2,12,6,8,-6,4,-3,-4,-2,12,6,-4,6,-2,3,-2,-1,-12,-6,4,6,2,3,-2,-1,12,-6,8,6,-4,-3,4,-2,-12,6,-8,6,4,-3,4,-2,-12,6,4,-6,-2,3,2,-1,12,-6,-4,-6,2,3,2,-1],
                [4,2,2,0,1,0,0,0,0,0,0,0,0,0,0,0,-4,-2,2,0,1,0,0,0,0,0,0,0,0,0,0,0,-4,2,-2,0,1,0,0,0,0,0,0,0,0,0,0,0,4,-2,-2,0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,4,0,2,2,1,0,0,0,0,0,0,0,0,0,0,0,-4,0,-2,2,1,0,0,0,0,0,0,0,0,0,0,0,-4,0,2,-2,1,0,0,0,0,0,0,0,0,0,0,0,4,0,-2,-2,1,0,0,0,0,0,0,0,0],
                [-12,-6,-6,-8,-3,-4,-4,-2,12,6,6,-4,3,-2,-2,-1,12,6,-6,8,-3,4,-4,-2,-12,-6,6,4,3,2,-2,-1,12,-6,6,8,-3,-4,4,-2,-12,6,-6,4,3,-2,2,-1,-12,6,6,-8,-3,4,4,-2,12,-6,-6,-4,3,2,2,-1],
                [8,4,4,4,2,2,2,1,-8,-4,-4,4,-2,2,2,1,-8,-4,4,-4,2,-2,2,1,8,4,-4,-4,-2,-2,2,1,-8,4,-4,-4,2,2,-2,1,8,-4,4,-4,-2,2,-2,1,8,-4,-4,4,2,-2,-2,1,-8,4,4,4,-2,-2,-2,1]],dtype=np.double)

def generateBinv():
    from sympy import symbols, Rational, Matrix
    from sympy.vector import Vector

    x,y,z = symbols('x y z')
    wholeFunc = Rational(0)
    alphaVec = []
    i = 0
    while i <= 3:
        j = 0
        while j <= 3:
            k = 0
            while k <= 3:
                a = symbols('a_{0}{1}{2}'.format(i,j,k))
                alphaVec.append(a)
                wholeFunc += a*x**Rational(i)*y**Rational(j)*z**Rational(k)
                k += 1
            j += 1
        i += 1

    print(alphaVec)

    #order per corner f,fx,fy,fz,fxy,fxz,fyz,fxyz
    cornerVec = Matrix([wholeFunc,
                 wholeFunc.diff(x),
                 wholeFunc.diff(y),
                 wholeFunc.diff(z),
                 wholeFunc.diff(x,y),
                 wholeFunc.diff(x,z),
                 wholeFunc.diff(y,z),
                 wholeFunc.diff(x,y,z)])
    #build B suchthat b = B.a

    def getBRows(vec,alpha):
        B = []
        for el in vec:
            row = []
            for a in alpha:
                k = Rational(el.diff(a))
                row.append(k)
            B.append(row)
        print(len(vec))
        return Matrix(B)

    #x=0,y=0,z=0
    row0 = getBRows(cornerVec.subs({x:0,y:0,z:0}),alphaVec)
    row1 = getBRows(cornerVec.subs({x:0,y:0,z:1}),alphaVec)
    row2 = getBRows(cornerVec.subs({x:0,y:1,z:0}),alphaVec)
    row3 = getBRows(cornerVec.subs({x:0,y:1,z:1}),alphaVec)
    row4 = getBRows(cornerVec.subs({x:1,y:0,z:0}),alphaVec)
    row5 = getBRows(cornerVec.subs({x:1,y:0,z:1}),alphaVec)
    row6 = getBRows(cornerVec.subs({x:1,y:1,z:0}),alphaVec)
    row7 = getBRows(cornerVec.subs({x:1,y:1,z:1}),alphaVec)
    Binv = Matrix([row0,row1,row2,row3,row4,row5,row6,row7]).inv()
    string = "["
    for i in range(64):
        string += '['+('{:},'*63).format(*Binv[i,:64])+'{:}'.format(Binv[i,-1])+'],\n'
    string += ']'
    print(string)
    
def optimizeBvecFormation():
    from sympy import symbols, Matrix
    from sympy import cse
    
    def index1(i):
        str = ''
        if i == -1:
            str += 'm'
        if i == 0:
            str += 'z'
        if i == 1:
            str += 'p'
        if i == 2:
            str += 'P'
        return str
    
    def index2(i):
        str = ''
        if i == -1:
            str += 'm'
        if i == 0:
            str += 'z'
        if i == 1:
            str += 'p'
        if i == 2:
            str += 'P'
        return str
    
    def index(i,j,k):
        str = ''
        if i == -1:
            str += 'm'
        if i == 0:
            str += 'z'
        if i == 1:
            str += 'p'
        if i == 2:
            str += 'P'
        if j ==-1:
            str += 'm'
        if j == 0:
            str += 'z'
        if j == 1:
            str += 'p'
        if j == 2:
            str += 'P'
        if k ==-1:
            str+='m'
        if k == 0:
            str += 'z'
        if k == 1:
            str += 'p'
        if k == 2:
            str += 'P'
        return str

    vec = []

    i = 0
    while i <= 1:
        j = 0
        while j <= 1:
            k = 0
            while k <= 1:
                #f
                vec.append(symbols('f_{0}'.format(index(i,j,k))))
                
                x12 = symbols('x_{0}'.format(index1(i+1))) - symbols('x_{0}'.format(index1(i-1)))
                y12 = symbols('y_{0}'.format(index1(j+1))) - symbols('y_{0}'.format(index1(j-1)))
                z12 = symbols('z_{0}'.format(index1(k+1))) - symbols('z_{0}'.format(index1(k-1)))
                #fx,fy,fz
                vec.append((symbols('f_{0}'.format(index(i+1,j,k))) - symbols('f_{0}'.format(index(i-1,j,k))) )/x12)
                vec.append((symbols('f_{0}'.format(index(i,j+1,k))) - symbols('f_{0}'.format(index(i,j-1,k))) )/y12)
                vec.append((symbols('f_{0}'.format(index(i,j,k+1))) - symbols('f_{0}'.format(index(i,j,k-1))) )/z12)

                #fxy,fxz,fyz
                vec.append((((symbols('f_{0}'.format(index(i+1,j+1,k))) - symbols('f_{0}'.format(index(i-1,j+1,k))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k))) - symbols('f_{0}'.format(index(i-1,j-1,k))) )/x12))/y12)
                vec.append((((symbols('f_{0}'.format(index(i+1,j,k+1))) - symbols('f_{0}'.format(index(i-1,j,k+1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j,k-1))) - symbols('f_{0}'.format(index(i-1,j,k-1))) )/x12))/z12)
                vec.append((((symbols('f_{0}'.format(index(i,j+1,k+1))) - symbols('f_{0}'.format(index(i,j-1,k+1))) )/y12)-((symbols('f_{0}'.format(index(i,j+1,k-1))) - symbols('f_{0}'.format(index(i,j-1,k-1))) )/y12))/z12)
                
                #fxyz
                vec.append((((((symbols('f_{0}'.format(index(i+1,j+1,k+1))) - symbols('f_{0}'.format(index(i-1,j+1,k+1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k+1))) - symbols('f_{0}'.format(index(i-1,j-1,k+1))) )/x12))/y12)-((((symbols('f_{0}'.format(index(i+1,j+1,k-1))) - symbols('f_{0}'.format(index(i-1,j+1,k-1))) )/x12)-((symbols('f_{0}'.format(index(i+1,j-1,k-1))) - symbols('f_{0}'.format(index(i-1,j-1,k-1))) )/x12))/y12))/z12)         

                k += 1
            j += 1
        i += 1
    vec = Matrix(vec)

    cseFunc = cse(vec,optimizations='basic')
    #generate the indices
    lines = ['im = i - 1','iz = i','ip = i + 1','iP = i + 2',
            'jm = j - 1','jz = j','jp = j + 1','jP = j + 2',
            'km = k - 1','kz = k','kp = k + 1','kP = k + 2']
    i = -1
    while i <= 2:
        j = -1
        while j <= 2:
            k = -1
            while k <= 2:
                var = index(i,j,k)
                line = "{0} = self.index(i{1},j{2},k{3})".format(index(i,j,k),index2(i),index2(j),index2(k))
                lines.append(line)
                k += 1
            j += 1
        i += 1
    def replaceIndices(f):
        i = -1
        while i <= 2:
            j = -1
            while j <= 2:
                k = -1
                while k <= 2:
                    var = index(i,j,k)
                    f = f.replace(var,'[{0}]'.format(var))
                    k += 1
                j += 1
            i += 1
        return f
    def replaceIndices2(f):
        f = f.replace('x_m','self.xvec[im]')
        f = f.replace('x_z','self.xvec[iz]')
        f = f.replace('x_p','self.xvec[ip]')
        f = f.replace('x_P','self.xvec[iP]')
        f = f.replace('y_m','self.yvec[jm]')
        f = f.replace('y_z','self.yvec[jz]')
        f = f.replace('y_p','self.yvec[jp]')
        f = f.replace('y_P','self.yvec[jP]')
        f = f.replace('z_m','self.zvec[km]')
        f = f.replace('z_z','self.zvec[kz]')
        f = f.replace('z_p','self.zvec[kp]')
        f = f.replace('z_P','self.zvec[kP]')
        return f
        
    for expr in cseFunc[0]:
        f = str(expr[1])
        f = replaceIndices(f)
        f = replaceIndices2(f)
        f = f.replace('f_','self.m')
        line = '{0} = {1}'.format(expr[0], f)
        lines.append(line)
        
    bvec = str(cseFunc[1][0].transpose())
    bvec = bvec.replace('Matrix([','bvec = np.array(')
    bvec = bvec.replace('])',')')
    bvec = bvec.replace(',',',\n')
    bvec = replaceIndices(bvec)
    bvec = replaceIndices2(bvec)
    bvec = bvec.replace('f_','self.m')
    lines.append(bvec)
    code = ''
    for line in lines:
        code += line+'\n'
    print(code)

def testResult():
    xvec = np.linspace(0,1,40)
    yvec = np.linspace(-1,5,70)
    zvec = np.linspace(0,1,50)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    M = X**2*Y*Z
    print (M.shape)
    tci = TriCubic(xvec,yvec,zvec,M)
    tci.checkIndexing(M)
    print(M[20,20,20])
    print(2*(xvec[20]+0.0001)*(yvec[20]+0.0001)*(zvec[20]+0.0001))
    print((xvec[20]+0.0001)**2*(zvec[20]+0.0001))
    print((xvec[20]+0.0001)**2*(yvec[20]+0.0001))
    print(2*(xvec[20]+0.0001)*(zvec[20]+0.0001))
    print(2*(xvec[20]+0.0001)*(yvec[20]+0.0001))
    print((xvec[20]+0.0001)**2)
    get_ipython().magic(u'timeit -n 100 tci.interp3(xvec[20]+0.0001,yvec[20]+0.0001,zvec[20]+0.0001,doDouble=True)')
    
if __name__=='__main__':
    #generateBinv()
    #optimizeBvecFormation()
    #testResult()
    xvec = np.linspace(0,1,40)
    yvec = np.linspace(-1,5,70)
    zvec = np.linspace(0,1,50)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    M = X**2*Y*Z
    print (M.shape)
    tci = TriCubic(xvec,yvec,zvec,M)
    tci.checkIndexing(M)
    print(M[20,20,20])
    print(2*(xvec[20]+0.0001)*(yvec[20]+0.0001)*(zvec[20]+0.0001))
    print((xvec[20]+0.0001)**2*(zvec[20]+0.0001))
    print((xvec[20]+0.0001)**2*(yvec[20]+0.0001))
    print(2*(xvec[20]+0.0001)*(zvec[20]+0.0001))
    print(2*(xvec[20]+0.0001)*(yvec[20]+0.0001))
    print((xvec[20]+0.0001)**2)
    get_ipython().magic(u'timeit -n 100 tci.interp3(xvec[20]+0.0001,yvec[20]+0.0001,zvec[20]+0.0001,doDouble=False)')

