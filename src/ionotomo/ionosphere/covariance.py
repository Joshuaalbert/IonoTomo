'''Handle covariance operations'''
import numpy as np
from scipy.special import gamma
from scipy.ndimage import convolve
from scipy.optimize import fmin_l_bfgs_b, minimize
from ionotomo.utils.gaussian_process import *
from ionotomo.inversion.line_search import vertex
class Covariance(object):
    '''Use for repeated use of covariance.
    `tci` : `tomoiono.geometry.tri_cubic.TriCubic`
        TricubicCubic interpolator that contains the geometry of the volume
    `sigma` : `float`
    variance of diagonal terms (sigma_1)
    `corr` : `float`
        correlation length of covariance (sigma_2)
    `nu` : `float`
        smoothness parameter 1./2. results in exponential, 3./2. to 7./2. more 
        smooth realistic ionosphere, as nu -> inf it approaches 
        square-exponential covariance (too smooth to be real)'''
    def __init__(self,K=None, dx = None, dy = None, dz = None, tci = None):
        if K is None:
            self.K = MaternPSep(3,0,l=20.,sigma=1.,p=0)*MaternPSep(3,1,l=20.,sigma=1.,p=0)*MaternPSep(3,2,l=20.,sigma=1.,p=0) 
        else:
            assert isinstance(K,KernelND)
            assert K.ndims == 3
            self.K = K
        self.dx = dx
        self.dy = dy
        self.dz = dz
        if tci is not None:
            self.dx = tci.xvec[1] - tci.xvec[0]
            self.dy = tci.yvec[1] - tci.yvec[0]
            self.dz = tci.zvec[1] - tci.zvec[0]
        if self.dx is not None and self.dy is not None and self.dz is not None:
            self.create_c_stencil()
        else:
            self.c_stencil = None

    def __call__(self,X,Y=None):
        '''Return the covariance between all pairs in X or between X and Y if Y is not None.
        X : (M1, self.ndim)
        Y : (M2, self.ndim)
        Return : (M1, M2)'''
        return self.K(X,Y=Y)

    def create_c_stencil(self):
        m = 5
        xvec = np.linspace(-self.dx*(m>>1),self.dx*(m>>1),m)
        yvec = np.linspace(-self.dy*(m>>1),self.dy*(m>>1),m)
        zvec = np.linspace(-self.dz*(m>>1),self.dz*(m>>1),m)
        X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
        x_eval = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        c = self(np.zeros([1,3]),x_eval).reshape(X.shape)
        while np.min(c)/np.max(c) > 0.05:
            m += 2
            xvec = np.linspace(-self.dx*(m>>1),self.dx*(m>>1),m)
            yvec = np.linspace(-self.dy*(m>>1),self.dy*(m>>1),m)
            zvec = np.linspace(-self.dz*(m>>1),self.dz*(m>>1),m)
            X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
            x_eval = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
            c = self(np.zeros([1,3]),x_eval).reshape(X.shape)
        print("Selecting stencil size {}".format(m))
        self.c_stencil = c

    def create_inverse_stencil(self):
        m = self.c_stencil.shape[0]
        from keras.models import Model
        from keras.layers import Input, Conv3D
        from keras.optimizers import Adam
        delta = np.zeros([m,m,m])
        delta[m>>1,m>>1,m>>1] = 1.
        #add channel and batch_num
        delta = np.expand_dims(np.expand_dims(delta,0),0)
        #delta *= 0
        #delta += 1

        data = np.random.normal(size=[10,1,m,m,m])

        inputs = Input(shape=delta.shape[1:])
        conv_ = Conv3D(1,self.c_stencil.shape,
                data_format='channels_first',
                padding='same',use_bias=False,
                trainable=False,
                weights=[np.expand_dims(np.expand_dims(self.c_stencil,-1),-1)])
        conv_inv_ = Conv3D(1,self.c_stencil.shape,
                data_format='channels_first',
                padding='same',use_bias=False,trainable=True)
        o1 = conv_(conv_inv_(conv_(conv_inv_(inputs))))
        o2 = conv_inv_(conv_(conv_inv_(conv_(inputs))))
        model = Model(inputs=inputs,outputs=[o1,o2])
        model.compile(optimizer=Adam(lr=1e-3),loss=['mse']*2)
        conv_.set_weights([np.expand_dims(np.expand_dims(self.c_stencil,-1),-1)])
        model.fit(delta,[delta,delta],batch_size=1,epochs=10000,verbose=1)
        #print(model.layers[1].get_weights()[0][:,:,:,0,0])
        print(model.predict(delta)[0][0,0,m>>1,m>>1,m>>1])


    def create_inverse_stencil_(self):
        m = self.c_stencil.shape[0]
        K = 5
        from sympy import symbols, Array,Matrix
        p = K >> 1
        X = []
        for x in range(K):
            Y = []
            for y in range(K):
                Z = []
                for z in range(K):
                    s = symbols("s{}{}{}".format(x,y,z))
                    Z.append(s)
                Y.append(Z)
            X.append(Y)
        M = X
        
        for x in range(0,p+1):
            for y in range(0,p+1):
                for z in range(0,p+1):
                    c = (p,p,p)
                    for x_ in [-1,1]:
                        for y_ in [-1,1]:
                            for z_ in [-1,1]:
                                xi = p + x_*x
                                yi = p + y_*y
                                zi = p + z_*z
                                M[xi][yi][zi] = M[p-x][p-y][p-z]
        M = Array(M)
        #print(M)
        self.inverse_stencil = S0
        u = []
        for e in M:
            if e not in u:
                u.append(e)

        print("Degrees of freedom : {}".format(len(u)))
        #pad with K-1 zeros
        pad = 2*(K-1)
        C = np.zeros([m+pad,m+pad,m+pad])
        C[K-1:K-1+m,K-1:K-1+m,K-1:K-1+m] = self.c_stencil
        N = C.shape[0]
        p = N >> 1
        eqs = []
        B = []
        w = []
        for i in range(N-K):
            for j in range(i,N-K):
                for k in range(j,N-K):
                    constraint = 0
                    for a in range(K):
                        for b in range(K):
                            for c in range(K):
                                constraint += C[i+a,j+b,k+c]*M[a,b,c]
                    eq = []
                    for e in u:
                        eq.append(constraint.diff(e))

                    eqs.append(eq)
                    if i==p and j == p and k == p:
                        B.append(1.)
                        w.append(0.)
                    else:
                        B.append(0.)
                        w.append(np.sqrt((a-K/2.)**2 + (b-K/2.)**2 + (c-K/2.)**2))
        A = np.array(eqs,dtype=float)
        B = np.array(B,dtype=float)
        w = np.array(w,dtype=float)
        print("A : {}".format(A.shape))
        x = np.linalg.pinv(A).dot(B)
        print(x)
        from scipy.optimize import minimize

#        w = np.exp(-w/(np.max(w)/2.))
#        w[(w<0.75)*(w>0.25)] = 0
        
        w = np.zeros(B.shape)
        w[B==1] = 1.
        w[B==0] = 1./np.sum(B==0)

        res = minimize(lambda x: np.sum(((A.dot(x) - B))**2),x)
        print(res)
        x = res.x
        M_ = M.subs({u[i]:x[i] for i in range(len(u))})
        M = np.zeros([K,K,K],dtype=float)
        for a in range(K):
            for b in range(K):
                for c in range(K):
                    M[a,b,c] = M_[a,b,c]
        M1 = self.create_inverse_stencil_()
        print(M)
        print(M1)
        print(M-M1)
        delta = np.zeros([N,N,N])
        for i in range(N-K):
            for j in range(N-K):
                for k in range(N-K):
                    constraint = 0
                    for a in range(K):
                        for b in range(K):
                            for c in range(K):
                                delta[i,j,k] += C[i+a,j+b,k+c]*M1[a,b,c]


        #delta = convolve(self.c_stencil,M,mode='constant')
        print(delta)
        peak = np.max(delta)
        delta[delta == peak] = 0.
        print("Peak {} max sidelobe {}".format(peak,np.max(np.abs(delta))))
        print("Peak / max sidelobe {}".format(peak/np.max(np.abs(delta))))


                    
    def create_inverse_stencil_(self):
        m = self.c_stencil.shape[0]
        n = 5
#        from keras.models import Model
#        from keras.layers import Input, Conv3D
#        from keras.optimizers import Adam
#        input_set = []
#        output_set = []
#        for i in range(100):
#            inp = np.random.normal(size=[4*m,4*m,4*m,1])
#            inp[inp < 0.35] = 0.
#            input_set.append(self.smooth(inp[:,:,:,0])[:,:,:,np.newaxis])
#            output_set.append(inp)
#        input_set.append(self.c_stencil[:,:,:,np.newaxis])
#        
#        delta = np.zeros([m,m,m,1])
#        p = m>>1
#        delta[p,p,p,0] = 1.
#        output_set.append(delta)
#        input_set = np.stack(input_set,axis=0)
#        output_set = np.stack(output_set,axis=0)

#        inputs = Input(shape=(4*m,4*m,4*m,1))
#        l = Conv3D(1,(n,n,n),data_format='channels_last',padding='same',use_bias=False)
#        o1 = l(inputs)
#        model = Model(inputs=inputs,outputs=o1)
#        model.compile(optimizer=Adam(lr=1e-3),loss=['mse'])
#        model.fit(input_set,output_set,batch_size=100,epochs=10000,verbose=1)
#        S0 = model.layers[1].get_weights()[0][:,:,:,0,0]
#        self.inverse_stencil = S0
#        return
        def evaluate_stencil(s,c,n,m):
            #s is (n>>1,n>>1,n>>1)
            S = np.zeros([n,n,n])
            p = n>>1
            s = s.reshape([p+1,p+1,p+1])
            S[:p+1,:p+1,:p+1] = s
            S[:p+1,:p+1,p:] = s[:,:,::-1]
            S[:p+1,p:,:p+1] = s[:,::-1,:]
            S[:p+1,p:,p:] = s[:,::-1,::-1]
            S[p:,:p+1,:p+1] = s[::-1,:,:]
            S[p:,:p+1,p:] = s[::-1,:,::-1]
            S[p:,p:,:p+1] = s[::-1,::-1,:]
            S[p:,p:,p:] = s[::-1,::-1,::-1]
            p = m>>1
            res = convolve(self.c_stencil,S,mode='constant',cval=0.)
            res[p,p,p] -= 1.
            diff = res[:p+1,:p+1,:p+1]
            l2 =  np.sum(diff**2)
            #print(l2)
            return l2 
        S0 = np.zeros([n,n,n])
        p = n>>1
        s0 = S0[:p+1,:p+1,:p+1].flatten()
        res = fmin_l_bfgs_b(evaluate_stencil,s0,approx_grad=True,args=(self.c_stencil,n,m),maxfun=1e5,m=int(1e3))
        #res = minimize(evaluate_stencil,s0,args=(self.c_stencil,n,m),method='BFGS')
        assert res[2]['warnflag'] == 0,'Did not converge, try smaller stencils'
        s = res[0].reshape([p+1,p+1,p+1])
        S = np.zeros([n,n,n])
        S[:p+1,:p+1,:p+1] = s
        S[:p+1,:p+1,p:] = s[:,:,::-1]
        S[:p+1,p:,:p+1] = s[:,::-1,:]
        S[:p+1,p:,p:] = s[:,::-1,::-1]
        S[p:,:p+1,:p+1] = s[::-1,:,:]
        S[p:,:p+1,p:] = s[::-1,:,::-1]
        S[p:,p:,:p+1] = s[::-1,::-1,:]
        S[p:,p:,p:] = s[::-1,::-1,::-1]
        delta = convolve(self.c_stencil,S,mode='constant')
        peak = np.max(delta)
        delta[delta == peak] = 0.
        print("Peak {} max sidelobe {}".format(peak,np.max(np.abs(delta))))
        print("Peak / max sidelobe {}".format(peak/np.max(np.abs(delta))))
        return S
    def clean(self,phi,inplace = False):
        '''Use clean algorithm to deconvolve'''
        nx,ny,nz = phi.shape
        def index_inv(h):
            '''Invert flattened index to the indices'''
            z = h % nz
            h -= z
            h /= nz
            y = h % ny
            h -= y
            h /= ny
            x = h
            return int(x),int(y),int(z)
        if inplace:
            phihat = phi
        else:
            phihat = phi.copy()
        model = np.zeros(phi.shape,dtype=float)
        gain = 0.1
        stencil = self.c_stencil
        m = stencil.shape[0] >> 1
        peak = stencil[m,m,m]
        stencil /= peak
        m += 1
        phihat[:m,:,:] = 0
        phihat[:,:m,:] = 0
        phihat[:,:,:m] = 0
        phihat[-m:,:,:] = 0
        phihat[:,-m:,:] = 0
        phihat[:,:,-m:] = 0
        m -= 1
        for i in range(10):
            args = np.argsort(np.abs(phihat.flatten()))[::-1]
            print("Residual: ",np.std( phihat))
            iter = 0    
            #while iter < 1000:
            #for x,y,z in zip(X,Y,Z):
            for arg in args:
                #arg = args[iter]#np.argmax(np.abs(phihat))
                x,y,z = index_inv(arg)
                f = phihat[x,y,z]*gain
                #print("Model at ",x,y,z,"flux ",f)
                phihat[x-m:x+m+1,y-m:y+m+1,z-m:z+m+1] -= stencil*f
                
                #print("Max Residual: ",np.max(np.abs(phihat)))
                model[x,y,z] += f/peak
                iter += 1
        return model

    def contract_(self,phi):
        '''Do Cm^{-1}.phi using numerical stencil method'''
        return convolve(phi,self.inverse_stencil,mode='nearest')
    def contract(self,phi):
        '''Use succesive approximation to contract'''
        return self.clean(phi)
        var = np.max(self.c_stencil)
        #            print(x,y,z)
        Q = 0.01*np.ones(phi.shape)/var
        phihat = self.contract_(phi)
        #phihat = phi/var
        def eval(phihat):
            d = phi - self.smooth(phihat)
            grad = d/var
            res = np.mean(np.abs(d)**2)
            return res, grad
        iter  = 0
        ep = 0.5
        while iter < 20:
            f,grad = eval(phihat)
            print("Current",f)
            if iter < 30:
                search = (f,ep)
                count = 0
                tries = 0
                line = []
                val = []
                while count < 3:
                    f_,_ = eval(phihat + ep*grad)
                    print(ep,f_,f - f_)
                    if f_ < f:
                        line.append((ep))
                        val.append(f_)
                        count += 1
                    tries += 1
                    if tries > 15:
                        return phihat
                    ep /= 2.
                ep *= 16.
                vx,vy = vertex(*line[-3:],*val[-3:])
                vx = vx
                print("Vertex:",vx,vy)
                if np.abs(vy) > np.min(val) or vx < 0. or np.isinf(vx):
                    vx = line[np.argmin(val)]
                    ep = 0.5
                #vx = 0.06
            phihat += vx*grad
            iter += 1
        return phihat

    def smooth(self,phi):
        '''Do Cm^.phi using numerical stencil method'''
        return convolve(phi,self.c_stencil,mode='nearest')

        
