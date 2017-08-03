from ionotomo.astro.real_data import generate_example_datapack, phase_screen_datapack
from ionotomo.astro.radio_array import generate_example_radio_array
from ionotomo.astro.antenna_facet_selection import select_antennas
from ionotomo.inversion.initial_model import *
import logging as log
import os
from time import clock

def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)

    try:
        os.makedirs(output_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    tci_folder = os.path.join(output_folder,"ionospheres")
    datapack_folder = os.path.join(output_folder,"datapacks")
    try:
        os.makedirs(tci_folder)
    except:
        pass
    try:
        os.makedirs(datapack_folder)
    except:
        pass
    log.info("Generating a number of ionospheres")
    #using lofar configuration generate for a number of random pointings and times
    radio_array = generate_example_radio_array(config='lofar')
    info = open("{}/info".format(output_folder),'w')
    info.write("#time alt az corr datapack_filename tci_filename\n")
    for time in ["2017-08-02T00:00:00.000","2017-08-02T06:00:00.000","2017-08-02T12:00:00.000","2017-08-02T18:00:00.000"]:
        for alt in [30.,50.,70.,90.]:
            for az in [0.,90.,180.,270.]:
                datapack = phase_screen_datapack(10,alt=alt,az=az,radio_array=radio_array,time=time)
                datapack = select_antennas(10,datapack)
                datapack_filename=os.path.join(datapack_folder,"datapack_{}_{:.2f}_{:.2f}.hdf5".format(time,alt,az))
                datapack.save(datapack_filename)
                seed = int(clock())
                for corr in [10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]:
                    pert_tci = create_turbulent_model(datapack,corr=corr,seed=seed)
                    tci_filename=os.path.join(tci_folder,"ionosphere_{}_{:.2f}_{:.2f}_{:.2f}.hdf5".format(time,alt,az,corr))
                    log.info("{} {} {} {} {} {}\n".format(time,alt,az,corr,datapack_filename,tci_filename))
                    info.write("{} {} {} {} {} {}\n".format(time,alt,az,corr,datapack_filename,tci_filename))
                    pert_tci.save(tci_filename)
    info.close()

if __name__=='__main__':
    run("output")
