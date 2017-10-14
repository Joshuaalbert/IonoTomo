import numpy as np
import pylab as plt
from scipy.special import erf
from scipy.integrate import simps
from scipy.linalg import cho_solve
#from ChoSolver import choSolve, choBackSubstitution
        
def styblinsky(x):
    return (x[0]**4 - 16*x[0]**2 + 5*x[0] + x[1]**4 - 16*x[1]**2 + 5*x[1])/2.

def rosenbrock(x):
    a = 1
    b = 100
    return (a-x[0])**2 + b*(x[1] - x[0]**2)**2

def complexInjunction(x):
    Nm = len(x)
    a = np.arange(Nm)
    A = np.outer(np.cos(np.arange(Nm)),np.sin(1j*np.arange(Nm))-Nm)
    y = np.exp(1j*A.dot(x))
    return -np.abs((np.min(y)/np.max(y)).real) 
    
def mean(x):
    #return styblinsky(x)
    return np.log10(1+rosenbrock(x))# + rosenbrock((x-1))
    return np.sqrt((x[0]-0.5)**2 + (x[1])**2)


def M52(XX,theta):
    theta0 = theta[0]
    nu = theta[1]
    lengthScales = theta[2:]
    N = XX.shape[0]
    r2 = np.zeros([N,N],dtype=np.double)
    K = np.zeros([N,N],dtype=np.double)
    i = 0
    while i < len(lengthScales):
        r2 += (XX[:,i,:,i]/lengthScales[i])**2
        i += 1
    K += r2*(5./3.)
    np.sqrt(5*r2,out=r2)
    K += 1+r2
    np.exp(-r2,out=r2)
    K *= r2
    K *= theta0
    return K

def expK(XX,theta):
    theta0 = theta[0]
    nu = theta[1]
    lengthScales = theta[2:]
    N = XX.shape[0]
    K = np.zeros([N,N],dtype=np.double)
    i = 0
    while i < len(lengthScales):
        K -= (XX[:,i,:,i]/lengthScales[i])**2
        i += 1
    K /= 2.
    np.exp(K,out=K)
    K *= theta0
    #K += nu**2*np.eye(N)
    return K

def expK_derivative(XX,theta):
    theta0 = theta[0]
    nu = theta[1]
    lengthScales = theta[2:]
    N = XX.shape[0]
    Kdiff = np.zeros([N,N,len(theta)],dtype=np.double)
    K = np.zeros([N,N],dtype=np.double)
    #0 -> exp(-r^2)
    #1 -> 2*eye(N)*nu
    #2: ->-2r*eye(-r^2)*-2*(x1[i]-x2[i])^2/(lengthScale[i])^3
    i = 0
    while i < len(lengthScales):
        Kdiff[:,:,0] -= (XX[:,i,:,i]/lengthScales[i])**2
        Kdiff[:,:,2+i] += 4*XX[:,i,:,i]**2/lengthScales[i]**3
        i += 1
    #*r
    #np.rollaxis(K[:,:,2:],2,0) *= np.sqrt(-Kdiff[:,:,0])
    K /= 2.
    np.exp(K,out=K)
    K *= theta0
    K += nu**2*np.eye(N)
    return K

class Prior(object):
    def __init__(self, **kwargs):
        for key in kwargs.keys():
            setattr(self,key,kwargs[key])
    def domain(self):
        '''Get domain of prior'''
        return None
    def sample(self,N=1):
        '''get a sample from the distribution'''
        return None
    def pdf(self,x):
        '''get the pdf at x'''
        return None

class UniformPrior(Prior):
    def __init__(self,xmin,xmax):
        d = {"xmin":float(min(xmin,xmax)),"xmax":float(max(xmin,xmax)),"width":float(max(xmin,xmax) - min(xmin,xmax))}
        super(UniformPrior,self).__init__(**d)
    def sample(self,N=1):
        return np.random.uniform(low=self.xmin,high=self.xmax,size=N)
    def pdf(self,x):
        out = np.ones_like(x)
        out /= self.width
        out[x>self.xmax] *= 0.
        out[x<self.xmin] *= 0.
        return out
    
class NormalPrior(Prior):
    def __init__(self,mean,std):
        d = {"mean":float(mean),"std":float(std)}
        super(NormalPrior,self).__init__(**d)
    def sample(self,N=1):
        return self.mean + self.std*np.random.normal(size=N)
    def pdf(self,x):
        return np.exp(-(x - self.mean)**2/self.std**2/2.)/np.sqrt(2*np.pi)/self.std

class LogNormalPrior(Prior):
    def __init__(self,mean,std):
        d = {"mean":float(mean),"std":float(std)}
        super(LogNormalPrior,self).__init__(**d)
    def sample(self,N=1):
        return np.random.lognormal(mean=self.mean, sigma=self.std, size=N)
    def pdf(self,x):
        return np.exp(-(np.log(x) - self.mean)**2/self.std**2/2.)/np.sqrt(2*np.pi)/self.std/x

class ClassPrior(Prior):
    def __init__(self,numClasses,weights=None):
        if weights is None:
            weights = np.ones(numClasses,dtype=np.double)/numClasses
        d = {"numClasses":float(numClasses),"weights":float(weights)}
        super(ClassPrior,self).__init__(**d)
    def sample(self,N=1):
        samples = np.zeros(N,dtype=np.int64)
        i = 0
        while i < N:
            c = -1
            while c == -1:    
                c_ = np.random.randint(self.numClasses)
                if np.random.uniform() < self.weights[c_]:
                    c = c_
            samples[i] = c
            i += 1            
        return samples
    def pdf(self,x):
        return self.weights[np.int64(x)]
    
class DiscretePrior(Prior):
    def __init__(self,values,prior=None):
        if prior is None:
            prior = UniformPrior(np.min(values),np.max(values))
        d = {"values":values,"prior":prior}
        super(DiscretePrior,self).__init__(**d)
    def sample(self,N=1):
        samples = np.zeros(N,dtype=np.int64)
        i = 0
        while i < N:
            c = -1
            while c == -1:    
                c_ = np.random.randint(len(self.values))
                if np.random.uniform() < self.prior.pdf(self.values[c_]):
                    c = c_
            samples[i] = self.values[c]
            i += 1            
        return samples
    def pdf(self,x):
        return self.prior.pdf(x)
    
if __name__ == '__main__':
    def sampleX(xPriors,N):
        X = np.zeros([N,len(xPriors)],dtype=np.double)
        for i in range(len(xPriors)):
            X[:,i] = xPriors[i].sample(N)
        return X
    
    def computeAquisition(Xstar,X,y,thetaPriors,iteration=1):
        Xstar = np.atleast_2d(Xstar)
        shape = []
        indices = []
        for thetaPrior in thetaPriors:
            ar = thetaPrior.values
            shape.append(len(ar))
            indices.append(np.arange(len(ar)))
        n = len(thetaPriors)
        postTheta = np.zeros(shape,dtype=np.double)
        COMP = np.zeros(shape,dtype=np.double)
        DF = np.zeros(shape,dtype=np.double)
        LML = np.zeros(shape,dtype=np.double)
        Xboth = np.vstack([X,Xstar])
        XXboth = np.subtract.outer(Xboth,Xboth)
        arg = np.argsort(y)
        xbest = X[arg[0],:]
        fbest = y[arg[0]]
        aq_full = np.zeros([Xstar.shape[0]]+shape,dtype=np.double)
        for idx in product(*indices):
            theta = np.zeros(len(indices),dtype=np.double)
            for i in range(len(idx)):
                theta[i] = thetaPriors[i].values[idx[i]]
            nu = theta[1]
            #Kboth = expK(XXboth,theta)
            Kboth = M52(XXboth,theta)
            K00 = Kboth[0:X.shape[0],0:X.shape[0]] 
            K00 += nu**2*np.eye(X.shape[0])
            K01 = Kboth[0:X.shape[0],X.shape[0]:]
            K10 = K01.T
            K11 = Kboth[X.shape[0]:,X.shape[0]:]
            L = np.linalg.cholesky(K00)
            alpha = cho_solve((L,False),y)#choSolve(L,y,False)
            #mu[j] = sum_i alpha[i]K01[i,j]
            mu = K10.dot(alpha)
            #cov = K11 - K10.(K00+sigma)(^-1).K01
            V = choBackSubstitution(L,K01,True,False)
            std = np.sqrt(np.diag(K11 - V.T.dot(V)))
            gamma = (fbest - mu)/std
            #POI
            cum = (1 + erf(gamma/np.sqrt(2)))/2.
            #return
            #EI
            aq = std*(gamma*cum + np.exp(-gamma**2/2)/np.sqrt(2*np.pi))
            #aq = (1./(iteration+1))*std - mu
            datafit = -y.dot(alpha)/2.
            complexity = np.sum(np.log(np.diag(L)))
            marLik = np.exp(datafit - complexity  - np.log(2*np.pi)*n/2.)
            COMP[idx] = complexity
            DF[idx] = datafit
            LML[idx] = np.log(marLik)
            prior = 1.
            for t,tp in zip(theta,thetaPriors):
                prior  *= tp.pdf(t)    
            postTheta[idx] = marLik * prior 
            aq_full[ [slice(0,Xstar.shape[0])]+list(idx)] = aq*postTheta[idx]
        prob = np.copy(postTheta)
        for axis in range(len(thetaPriors)):
            aq_full = simps(aq_full,thetaPriors[len(thetaPriors)-axis-1].values,axis=len(thetaPriors)-axis)
            prob = simps(prob,thetaPriors[len(thetaPriors)-axis-1].values,axis=len(thetaPriors)-axis-1)
        aq_full /= prob
        postTheta /= prob
        return aq_full,postTheta

    def maximizeAquisition(xPriors,X,y,thetaPriors=None,iteration=0):
        '''Using gradient (or steepest if desired) maximize the Expected Improvment aquisition
        while integration over aquisition hyper parameters.
        '''
        if thetaPriors is None:
            #Set up thetaPriors
            res = 10
            #theta0 ~ max(y) - min(y), uniform, log spacing 4 mag
            m2 = np.max(y) - np.min(y)
            m1 = m2/1e4
            theta0Prior = DiscretePrior(10**np.linspace(np.log10(m1),np.log10(m2),res),
                                   prior=UniformPrior(m1,m2))
            # nu ~ obs noise. similarly but scaled down by 10%
            m2 = (np.max(y) - np.min(y))/10.
            m1 = (m2/1e4)/10.
            nuPrior = DiscretePrior(10**np.linspace(np.log10(m1),np.log10(m2),res),
                                   prior=UniformPrior(m1,m2))
            thetaPriors = [theta0Prior,nuPrior]
            for i in range(len(xPriors)):
                #handles uniform x priors right now
                m2 = (xPriors[i].xmax - xPriors[i].xmin)*10.
                m1 = (xPriors[i].xmax - xPriors[i].xmin)/10.
                lsPrior = DiscretePrior(10**np.linspace(np.log10(m1),np.log10(m2),res),
                                   prior=UniformPrior(m1,m2))
                thetaPriors.append(lsPrior)
        for thetaPrior in thetaPriors:
            assert isinstance(thetaPrior,DiscretePrior), "one theta prior is not discrete"
        from itertools import product
        #First sample points to initialize maximization
        #create aquisition at x
        Xstar = sampleX(xPriors,max(2,len(thetaPriors))**max(2,len(xPriors)))
        Xstar = sampleX(xPriors,10**max(2,len(xPriors)))
        arg = np.argsort(y)
        xbest = X[arg[0],:]
        fbest = y[arg[0]]
        aq_all = []
        Xstar_all = []
        N = len(y)
        aq_init,postTheta = computeAquisition(Xstar,X,y,thetaPriors,iteration)
        aq_all.append(aq_init)
        Xstar_all.append(Xstar)
        arg = np.argsort(aq_init)
        Xsimp = Xstar[arg[-len(xPriors)-1:],:]
        aq_simp = aq_init[arg[-len(xPriors)-1:]]
        #min to max
        alpha,gamma,rho,sigma = 1.,2.,0.5,0.5
        iter = 0
        NonCovergent = True
        while NonCovergent:
            if iter >= 5:
                break
            iter += 1
            #order for min (flip aq sign)
            arg = np.argsort(-aq_simp)
            aq_simp = aq_simp[arg]
            Xsimp = Xsimp[arg,:]
            #print(Xsimp,aq_simp)
            #centorid except last
            x0 = np.mean(Xsimp[:-1,:],axis=0)
            #reflection
            xr = x0 + alpha*(x0 - Xsimp[-1,:])
            aq_r,postTheta = computeAquisition(xr,X,y,thetaPriors,iteration)
            #print(xr,aq_r)
            aq_all.append(aq_r)
            Xstar_all.append(xr)
            if -aq_simp[0] <= -aq_r and -aq_r < -aq_simp[-2]:
                Xsimp[-1,:] = xr
                aq_simp[-1] = aq_r
                continue
            #expansion
            if -aq_r < -aq_simp[0]:
                xe = x0 + gamma*(xr - x0)
                aq_e,postTheta = computeAquisition(xe,X,y,thetaPriors,iteration)
                aq_all.append(aq_e)
                Xstar_all.append(xe)
                if -aq_e < -aq_r:
                    Xsimp[-1,:] = xe
                    aq_simp[-1] = aq_e
                    continue
                else:
                    Xsimp[-1,:] = xr
                    aq_simp[-1] = aq_r
                    continue
            #contractions
            xc = x0 + rho*(Xsimp[-1,:] - x0)
            aq_c,postTheta = computeAquisition(xc,X,y,thetaPriors,iteration)
            aq_all.append(aq_c)
            Xstar_all.append(xc)
            if -aq_c < -aq_simp[-1]:
                Xsimp[-1,:] = xc
                aq_simp[-1] = aq_c
                continue
            #shrink
            for i in range(Xsimp.shape[0]):
                Xsimp[i,:] = Xsimp[0,:] + sigma*(Xsimp[i,:] - Xsimp[0,:])        
        xbest_nm = Xsimp[0,:]
        #print(xbest_nm)
        aq_all = np.hstack(aq_all)
        Xstar = np.vstack(Xstar_all)
        arg = np.argsort(aq_all)
        xbest = Xstar[arg[-1],:]
        if True:    
            vmin = np.min(aq_all)
            vmax = np.max(aq_all)
            plt.figure()
            sc=plt.scatter(Xstar[:,0],Xstar[:,1],c=aq_all,
                           vmin=vmin,vmax=vmax,alpha=0.6)
            plt.scatter(xbest[0],xbest[1],c='red',alpha=0.6)
            plt.scatter(xbest_nm[0],xbest_nm[1],c='red',marker='*',alpha=0.6)
            plt.colorbar(sc)
            plt.show() 
            fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
            ax1.plot(thetaPriors[0].values,
                    simps(simps(simps(postTheta,thetaPriors[3].values,axis=3),
                          thetaPriors[2].values,axis=2),
                           thetaPriors[1].values,axis=1))
            ax1.set_xlabel("theta0")
            ax2.plot(thetaPriors[1].values,
                    simps(simps(simps(postTheta,thetaPriors[3].values,axis=3),
                          thetaPriors[2].values,axis=2),
                           thetaPriors[0].values,axis=0))
            ax2.set_xlabel("nu")
            ax3.plot(thetaPriors[2].values,
                    simps(simps(simps(postTheta,thetaPriors[3].values,axis=3),
                          thetaPriors[1].values,axis=1),
                           thetaPriors[0].values,axis=0))
            ax3.set_xlabel("ls0")
            ax4.plot(thetaPriors[3].values,
                    simps(simps(simps(postTheta,thetaPriors[2].values,axis=2),
                          thetaPriors[1].values,axis=1),
                           thetaPriors[0].values,axis=0))
            ax4.set_xlabel("ls1")
            plt.show()         
        return xbest
        
    #Set up data
    np.random.seed(12344)
    nu = 0.01
    xPriors = [UniformPrior(-1,1.5),
               UniformPrior(-1,1.5)]
    thetaPriors = [DiscretePrior(10**np.linspace(np.log10(0.1),np.log10(5),10),prior=UniformPrior(0,5)),
                   DiscretePrior(10**np.linspace(np.log10(0.001),np.log10(0.5),10),prior=LogNormalPrior(np.log(0.1),np.log(0.5/0.01))),
                   DiscretePrior(np.linspace(0.5,6,10),prior=LogNormalPrior(np.log(1),np.log(6/0.5))),
                   DiscretePrior(np.linspace(0.5,6,10),prior=LogNormalPrior(np.log(1),np.log(6/0.5)))]
    
    X,Y = np.meshgrid(np.linspace(xPriors[0].xmin,xPriors[0].xmax,100),
                         np.linspace(xPriors[1].xmin,xPriors[1].xmax,100),
                     indexing='ij')
    A = []
    for x,y in zip(X.flatten(),Y.flatten()):
        A.append(mean(np.array([x,y])))
    Niter = 10
    minidx = np.zeros([4,Niter],dtype=np.double)
    for r in range(4):
        score = []
        #plt.figure()
        c1 = plt.contour(X,Y,np.array(A).reshape(X.shape),20)
        plt.clabel(c1,inline=1,fontsize=10)
        plt.title("True")
        plt.xlabel("x")
        plt.ylabel("y")
        arg = np.argsort(A)
        plt.scatter(X.flatten()[arg[0]],Y.flatten()[arg[0]],zorder=20,c='red',marker='*',alpha=1)
        #sample corners and center
        xCorners = []
        for xPrior in xPriors:
            xCorners.append([xPrior.xmin,xPrior.xmax])
        from itertools import product
        Xdata = []
        y = []
        for x in product(*xCorners):
            Xdata.append(np.array(x))
            y.append(mean(Xdata[-1]) + nu*np.random.normal())
        Xdata.append(np.mean(np.array(xCorners),axis=1))
        y.append(mean(Xdata[-1]) + nu*np.random.normal())
        Xdata = np.array(Xdata)
        y = np.array(y) 
        sc=plt.scatter(Xdata[:,0],Xdata[:,1],c=y,vmin=np.min(y),vmax=np.max(y),alpha=0.6)
        arg = np.argsort(y)
        plt.scatter(Xdata[arg[0],0],Xdata[arg[0],1],c='red',vmin=np.min(y),vmax=np.max(y),alpha=1)
        plt.colorbar(sc)
        plt.show() 
        #do iterations to find min
        arg = np.argsort(y) 
        fbest = y[arg[0]]
        xprev = Xdata[arg[0]]
        i = 0
        while i < Niter:
            #do gradient decent to find max of full aquisition
            xnext = maximizeAquisition(xPriors,Xdata,y,thetaPriors=None,iteration=i)
            xprev = xnext
            #print(y)
            f = mean(xnext) + nu*np.random.normal()
            Xdata = np.vstack([Xdata,xnext])
            y = np.hstack([y,f])
            fbest = np.min(y)
            score.append(f)
            print(xnext,f,fbest)
            i += 1
        c1 = plt.contour(X,Y,np.array(A).reshape(X.shape),20)
        plt.clabel(c1,inline=1,fontsize=10)
        plt.title("True")
        plt.xlabel("x")
        plt.ylabel("y")
        arg = np.argsort(A)
        plt.scatter(X.flatten()[arg[0]],Y.flatten()[arg[0]],zorder=20,c='red',marker='*',alpha=1)
        sc=plt.scatter(Xdata[:,0],Xdata[:,1],c=y,vmin=np.min(y),vmax=np.max(y),alpha=0.6)
        arg = np.argsort(y)
        plt.scatter(Xdata[arg[0],0],Xdata[arg[0],1],c='red',vmin=np.min(y),vmax=np.max(y),alpha=1)
        plt.colorbar(sc)
        plt.show() 
        plt.plot(score)
        plt.ylabel('score (lower better)')
        plt.xlabel("iteration")
        plt.show()
        minidx[r,:] = score
    plt.plot(np.mean(minidx,axis=0))
    plt.plot(np.mean(minidx,axis=0)+np.std(minidx,axis=0),ls='--')
    plt.plot(np.mean(minidx,axis=0)-np.std(minidx,axis=0),ls='--')
    plt.show()
        
    
 
