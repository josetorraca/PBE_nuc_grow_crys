import sys
import numpy as np
sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/')
import data_access_object
import mdl_vessel_carbonate
import h5py

def test_dae_saving_to_mongo():
    # ob = mdl_vessel_carbonate.SimulationOutput(np.array([1.1,2.21,3.0]), np.array([1.1,2.21,3.0]),
    # 21.0, 21.0, 121.0, 21.0, 21.0, 121.0, 21.0, 21.0, 121.0,
    # 21.0, 21.0, 121.0, 21.0, 21.0)

    tspan = np.linspace(0.0, 10.0, 10)
    span_ob = [mdl_vessel_carbonate.SimulationOutput(np.array([1.1,2.21,3.0]), np.array([1.1,2.21,3.0]),
    i, 21.0, 121.0, 21.0, 21.0, 121.0, 21.0, 21.0, 121.0,
    21.0, 21.0, 121.0, 21.0, 21.0) for i in tspan]
    mdl = mdl_vessel_carbonate.MyModel(10, 5,
        mdl_vessel_carbonate.create_physicoChemicalParameters()
    )
    mdl.out_span = span_ob
    data_access_object.insert_simulation_run(tspan, mdl, 'rkgill', 'simulation tesing')




def test_saving_as_pytables():
    # ob = mdl_vessel_carbonate.SimulationOutput(np.array([1.1,2.21,3.0]), np.array([1.1,2.21,3.0]),
    # 21.0, 21.0, 121.0, 21.0, 21.0, 121.0, 21.0, 21.0, 121.0,
    # 21.0, 21.0, 121.0, 21.0, 21.0)

    tspan = np.linspace(0.0, 10.0, 10)
    span_ob = [mdl_vessel_carbonate.SimulationOutput(np.array([1.1,2.21,3.0]), np.array([1.1,2.21,3.0]),
    i, 21.0, 121.0, 21.0, 21.0, 121.0, 21.0, 21.0, 121.0,
    21.0, 21.0, 121.0, 21.0, 21.0) for i in tspan]
    mdl = mdl_vessel_carbonate.MyModel(10, 5,
        mdl_vessel_carbonate.create_physicoChemicalParameters()
    )
    mdl.out_span = span_ob


if __name__ == "__main__":
    test_saving_as_pytables()
