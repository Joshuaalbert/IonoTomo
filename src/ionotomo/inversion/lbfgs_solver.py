from ionotomo.inversion.solver import Solver
from ionotomo.ionosphere.covariance import Covariance
class LBFGSSolver(Solver):
    def __init__(self,datapack,output_folder,diagnostic_folder,**kwarg):
        super(LBFGSSolver,self).__init__(datapack,output_folder,diagnostic_folder,**kwargs)
    def setup(self,**kwargs):
        super(LBFGSSolver,self).setup(**kwargs)
        K_ne,m_tci,rays,CdCt = self.K_ne, self.m_tci, self.rays, self.CdCt

        Nkernel = max(1,int(float(L_ne)/size_cell))
        #a priori infomation
        sigma_m = np.log(10.) 
        cov_obj = Covariance(m_tci,sigma_m,L_ne,2./3.)
        F0 = ne_tci.copy()
        F0.M[:,:,:] = 1.#initially update everywhere fully
    def go():
        #stopping conditions:
        #
        pass   
