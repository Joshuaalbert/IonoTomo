import numpy as np
import pylab as plt
from ionotomo import *

def test_solution():
    datapack = generate_example_datapack()
    phase = datapack.get_center_direction()
    times,timestamps = datapack.get_times(time_idx=-1)
    Nt = len(times)
    obstime = times[0]
    fixtime = times[Nt >> 1]
    tci = create_initial_model(datapack)
    pointing = Pointing(location = datapack.radio_array.get_center().earth_location,obstime = obstime, fixtime=fixtime, phase = phase)
    solution = Solution(tci=tci,pointing_frame = pointing)
    solution.save("test_solution.hdf5")
    solution2 = Solution(filename="test_solution.hdf5")
    assert np.all(solution.M == solution2.M)
    assert solution.pointing_frame.obstime.gps == solution2.pointing_frame.obstime.gps
    assert solution.pointing_frame.fixtime.gps == solution2.pointing_frame.fixtime.gps
    import astropy.units as au
    assert np.allclose(solution.pointing_frame.location.to(au.km).value, solution2.pointing_frame.location.to(au.km).value)
    assert np.all(solution.pointing_frame.phase.cartesian.xyz.value == solution2.pointing_frame.phase.cartesian.xyz.value)


