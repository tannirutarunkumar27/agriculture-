"""Fixed engineering transform, independent of private training statistics."""
import hashlib
import math
import numpy as np

# Fixed scale constants, not crop suitability bounds or data-fitted statistics.
SCALES={'soil_awc':100.,'soil_clay':1.,'soil_pH':14.,'nitrogen_residual':100.,
        'N_amount':100.,'P_amount':100.,'irrigation_amount':100.,'sowing_sin':1.,'sowing_cos':1.}
CATEGORIES=('previous_crop','crop_season','pest_control')
SPECIES=('barley','chickpea','fababean','lentil','lupin','pea','wheat_aestivum','wheat_turgidum')
BINS=8


def schema():
    names=[n for k in SCALES for n in (k,k+'__missing')]
    names += [n for k in CATEGORIES for n in (k+'__missing',*[k+'__hash_'+str(i) for i in range(BINS)])]
    names += ['contains_'+s for s in SPECIES]+['is_intercrop']
    return {'version':'fixed-public-transform-v1','numeric_scales':dict(SCALES),
            'category_hash':'SHA256 field/value, eight fixed bins; collisions possible',
            'feature_names':names,'numeric_policy':'tanh(value/scale), missing value=0 plus indicator',
            'fit_on_client_data':False,'agronomic_rule':False}


def missing(value):
    return value is None or isinstance(value,(float,np.floating)) and math.isnan(value)


def transform_records(records):
    output=[]
    for row in records:
        values=[]
        for name,scale in SCALES.items():
            value=row.get(name)
            if missing(value): values.extend((0.,1.));continue
            if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value):
                raise ValueError('Finite numerical value required: '+name)
            values.extend((math.tanh(float(value)/scale),0.))
        for name in CATEGORIES:
            value=row.get(name);absent=missing(value)
            encoded=[0.]*BINS
            if not absent:
                if not isinstance(value,str):raise ValueError('String category required: '+name)
                bucket=int.from_bytes(hashlib.sha256((name+'\0'+value).encode('utf-8')).digest(),'big')%BINS
                encoded[bucket]=1.
            values.extend((float(absent),*encoded))
        for name in ['contains_'+s for s in SPECIES]+['is_intercrop']:
            value=row.get(name)
            if value not in (0,1) or missing(value):raise ValueError('Fixed taxonomy flag required: '+name)
            values.append(float(value))
        output.append(values)
    return np.asarray(output,dtype=np.float32).reshape(len(output),len(schema()['feature_names']))
