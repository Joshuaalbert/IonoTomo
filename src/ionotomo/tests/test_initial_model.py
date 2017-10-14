import numpy as np
import pylab as plt
from ionotomo import *

def test_initial_model():
    datapack = generate_example_datapack()
    ne_tci = create_initial_model(datapack)
    pert_tci = create_turbulent_model(datapack)


