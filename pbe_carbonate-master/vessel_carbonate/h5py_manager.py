from datetime import datetime
import sys

import h5py
import numpy as np
from marshmallow import Schema, fields

import mdl_vessel_carbonate
# import data_access_object
from utils_cm_toolbox import h5py_simple

sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/')


def set_configuration_details(tspan, integration_string, mdl):
    d = {
        'tspan': tspan.tolist(),
        'nt': len(tspan),
        'Npts': mdl.ind.Npts0,
        'integration': integration_string,
        'kb': mdl.kb,
        'kg': mdl.kg,
        'kaggr': mdl.agg_regular_val,
    }
    return d

def create_object_serialized(mdl, descr_msg, tspan, integration_string):
    sim_output_span = mdl.out_span
    now = datetime.now()
    oschema = OutputSchema()
    regular_obj = [oschema.dump(sim_output).data for sim_output in sim_output_span]
    r = {
        'created_at': str(now),
        'description': descr_msg,
        'payload' : regular_obj,
        'config': set_configuration_details(tspan, integration_string, mdl)
    }
    return r


class OutputSchema(Schema):
    x = fields.List(fields.Float())
    N = fields.List(fields.Float())
    mCa = fields.Float()
    mC = fields.Float()
    mNa = fields.Float()
    mCl = fields.Float()
    V = fields.Float()
    S = fields.Float()
    B = fields.Float()
    Ksp = fields.Float()
    massCrystal = fields.Float()
    mCaCl2_added = fields.Float()
    IAP = fields.Float()
    pH = fields.Float()
    I = fields.Float()
    sigma = fields.Float()

def test_data():
    tspan = np.linspace(0.0, 10.0, 10)
    span_ob = [mdl_vessel_carbonate.SimulationOutput(np.array([1.1,2.21,3.0]), np.array([1.1,2.21,3.0]),
    i, 21.0, 121.0, 21.0, 21.0, 121.0, 21.0, 21.0, 121.0,
    21.0, 21.0, 121.0, 21.0, 21.0) for i in tspan]
    mdl = mdl_vessel_carbonate.MyModel(10, 5,
        mdl_vessel_carbonate.create_physicoChemicalParameters()
    )
    mdl.out_span = span_ob
    ob_serialized = create_object_serialized(
        mdl, 'dsuiaudsiajdjisaijd', tspan, 'rkgill'
    )
    return ob_serialized

def save_data_to_h5_file(fname, descr_msg, mdl, tspan, integration_string):
    # ob = test_data()
    ob = create_object_serialized(mdl, descr_msg, tspan, integration_string)
    f = h5py.File(fname, 'w')
    h5py_simple.save_dict_to_file(f, ob)

def testing_only():
    ob = test_data()
    f = h5py.File('sidjis.h5', 'w')
    h5py_simple.save_dict_to_file(f, ob)



if __name__ == "__main__":
    testing_only()
