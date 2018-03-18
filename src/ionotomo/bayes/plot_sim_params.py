import numpy as np
import os
import pylab as plt
import astropy.units as au
import astropy.coordinates as ac
plt.style.use('ggplot')

data = np.load('iono_char_save.npz')['arr_0'].item(0)
iono_ls = np.array(list(data.keys()))
length_scales = []
var_scales = []
for k in data:
    length_scales.append(data[k]['l_space'])
    var_scales.append(data[k]['var'])
length_scales = np.stack(length_scales,0)
var_scales = np.stack(var_scales,0)

from ionotomo import RadioArray, ENU
radio_array = RadioArray(array_file=RadioArray.lofar_array)
antennas = radio_array.get_antenna_locs()
array_center = ac.ITRS(np.mean(antennas.data))
enu = ENU(location = array_center)
ants_enu = antennas.transform_to(enu)
ref_dist = np.sqrt((antennas.x.to(au.km).value-antennas.x.to(au.km).value[0])**2+(antennas.y.to(au.km).value-antennas.y.to(au.km).value[0])**2+(antennas.z.to(au.km).value-antennas.z.to(au.km).value[0])**2)

east_color = (ants_enu.east.to(au.km).value - np.min(ants_enu.east.to(au.km).value)) / (np.max(ants_enu.east.to(au.km).value) - np.min(ants_enu.east.to(au.km).value))
north_color = (ants_enu.north.to(au.km).value - np.min(ants_enu.north.to(au.km).value)) / (np.max(ants_enu.north.to(au.km).value) - np.min(ants_enu.north.to(au.km).value))


print(length_scales.shape, var_scales.shape)
plt.plot(iono_ls, length_scales)
plt.show()
[plt.scatter(iono_ls,np.log10(var_scales[:,i]),alpha=1.,color=(ec**2,nc**2,0.)) for i,(ec,nc) in enumerate(zip(east_color[1:],north_color[1:]))]
plt.show()
