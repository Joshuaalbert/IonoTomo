
# coding: utf-8

# In[ ]:

from RealData import DataPack
from FermatClass import Fermat

def simulateDtec(neTCI,dataPack):
    '''Fill out the dtec values in an initialized ``DataPack`` object `dataPack`.
    ionosphere model is a ``TriCubicInterpolator`` object with electron density values of ionosphere
    '''
    antennas,antennaLabels = dataPack.get_antennas(antIdx = -1)
    directions, patchNames = dataPack.get_directions(dirIdx=-1)
    antennas, antLabels = dataPack.get_antennas(antIdx=-1)
    times,timestamps = dataPack.get_times(timeIdx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    fermat = Fermat(neTCI=neTCI,frequency = dataPack.radioArray.frequency,type='s',straightLineApprox=True)
    i = 0
    while i < Na:
        j = 0
        while j < Nt:
            k = 0
            while k < Nd:
                origin = antennas[i]
                direction = d
    

