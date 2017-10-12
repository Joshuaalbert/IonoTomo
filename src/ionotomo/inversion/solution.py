from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.astro.frames.pointing_frame import Pointing
import h5py
import numpy as np
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at
from astropy.coordinates.representation import CartesianRepresentation

class Solution(TriCubic):
    '''Contains the TCI as well as frame'''
    def __init__(self,xvec=None,yvec=None,zvec=None,M=None,
            tci=None,pointing_frame=None,filename=None):
        if filename is not None:
            self.load(filename)
            return
        if tci is not None:
            super(Solution,self).__init__(tci.xvec,tci.yvec,tci.zvec,tci.M)
        elif xvec is not None and yvec is not None and zvec is not None and M is not None:
            super(Solution,self).__init__(xvec,yvec,zvec,M)
        self.pointing_frame = pointing_frame
    def copy(self):
        """Return copy of self"""
        tci = super(Solution,self).copy()
        return Solution(tci=tci,pointing_frame=self.pointing_frame)
    def save(self,filename):
        super(Solution,self).save(filename)
        if self.pointing_frame is not None:
            f = h5py.File(filename,'a')
            f['/TCI'].attrs['obstime'] = self.pointing_frame.obstime.gps
            f['/TCI'].attrs['fixtime'] = self.pointing_frame.fixtime.gps
            f['/TCI'].attrs['location'] = self.pointing_frame.location.to(au.km).value
            f['/TCI'].attrs['phase'] = [self.pointing_frame.phase.ra.deg,self.pointing_frame.phase.dec.deg]
            f.close()
    def load(self,filename):
        super(Solution,self).load(filename)
        f = h5py.File(filename,'r')
        try:
            obstime = at.Time(f['TCI'].attrs['obstime'],format='gps',scale='tai')
            fixtime = at.Time(f['TCI'].attrs['fixtime'],format='gps',scale='tai')
            location = ac.ITRS().realize_frame(CartesianRepresentation(*(f['TCI'].attrs['location']*au.km)))
            phase = ac.SkyCoord(*(f['TCI'].attrs['phase']*au.deg),frame='icrs')
            pointing_frame = Pointing(location=location,obstime=obstime,phase=phase,fixtime=fixtime)
        except:
            pointing_frame = None
        f.close()
        self.__init__(pointing_frame=pointing_frame)

def transfer_solutions(solution_from,solution_to):
    '''interpolate the solutions from one solution into the frame of another'''
    x,y,z = solution_to.get_model_coordinates()
    coords = ac.SkyCoord(x*au.km,y*au.km,z*au.km,frame=solution_to.pointing_frame).transform_to(solution_from.pointing_frame).cartesian.xyz.to(au.km).value.T
    x,y,z = coords[:,0],coords[:,1],coords[:,2]
    m_to = solution_from.extrapolate(x,y,z)
    solution_to.M = m_to.reshape(solution_to.nx, solution_to.ny, solution_to.nz)
    return solution_to


