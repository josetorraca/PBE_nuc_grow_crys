from datetime import datetime
from pymongo import MongoClient
from marshmallow import Schema, fields

client = MongoClient('mongodb://localhost:27017/')
db = client['psd']
collection = db['simulations']

def insert_data(data):
    """
    Insert new data or document in collection
    :param data:
    :return:
    """
    document = collection.insert_one(data)
    return document.inserted_id

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

def insert_simulation_run(tspan, mdl, integration_string, descr_msg):
    r = create_object_serialized(mdl, descr_msg, tspan, integration_string)
    inserted_id = insert_data(r)
    return inserted_id

def create_object_serialized(mdl, descr_msg, tspan, integration_string):
    sim_output_span = mdl.out_span
    now = datetime.now()
    oschema = OutputSchema()
    regular_obj = [oschema.dump(sim_output).data for sim_output in sim_output_span]
    r = {
        'time': now,
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
