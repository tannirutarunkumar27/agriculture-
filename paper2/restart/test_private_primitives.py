import unittest
import numpy as np
import torch
from paper2.restart.public_preprocessing import schema,transform_records,SPECIES
from paper2.restart.noisy_aggregation import aggregate_simulation


def record(**changes):
    row={'contains_'+s:0 for s in SPECIES};row['is_intercrop']=0
    row.update(soil_awc=70.,soil_clay=.2,soil_pH=7.,nitrogen_residual=None,N_amount=0.,
               P_amount=10.,irrigation_amount=None,sowing_sin=0.,sowing_cos=1.,
               previous_crop='wheat',crop_season='spring',pest_control=None)
    row.update(changes);return row


class PublicTransformTests(unittest.TestCase):
    def test_other_clients_cannot_change_transform(self):
        row=record();single=transform_records([row])[0]
        batch=transform_records([record(soil_awc=1e20,previous_crop='new'),row])
        np.testing.assert_array_equal(single,batch[1])
    def test_fixed_schema_unknown_category(self):
        a=transform_records([record(previous_crop='never-seen')]);b=transform_records([record()])
        self.assertEqual(a.shape,b.shape);self.assertEqual(a.shape[1],54)
        self.assertTrue(np.isfinite(a).all())
        self.assertFalse(schema()['fit_on_client_data'])
    def test_missing_and_extreme_inputs(self):
        a=transform_records([record(soil_awc=None,N_amount=1e300)])
        names=schema()['feature_names']
        self.assertEqual(a[0,names.index('soil_awc__missing')],1.)
        self.assertEqual(a[0,names.index('N_amount')],1.)
        self.assertTrue((abs(a)<=1).all())
    def test_bad_inputs(self):
        for row in (record(soil_pH=float('inf')),record(previous_crop=3),record(is_intercrop=2)):
            with self.assertRaises(ValueError):transform_records([row])


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.base={'w':torch.zeros(2,dtype=torch.float64)}
        self.clients={'a':{'w':torch.tensor([3.,4.],dtype=torch.float64)},'b':{'w':torch.zeros(2,dtype=torch.float64)}}
    def call(self,**changes):
        values=dict(reference=self.base,clients=self.clients,public_slots=('a','b'),clip_norm=2.,
                    std_over_sensitivity=1.,noise_rng=np.random.default_rng(42),simulation_only=True)
        values.update(changes);return aggregate_simulation(**values)
    def test_known_clipped_mean_and_noise_scale(self):
        state,event,meta=self.call()
        expected=np.array([.6,.8])+np.random.default_rng(42).normal(0,2,size=2)
        np.testing.assert_allclose(state['w'].numpy(),expected,rtol=1e-14)
        self.assertEqual(event,{'sensitivity':2.,'noise_std':2.,'count':1})
        self.assertFalse(meta['application_privacy_certified'])
        self.assertEqual(self.clients['a']['w'][0],3.)
    def test_slot_and_schema_rejection(self):
        for clients in ({'a':self.clients['a']},{'a':{'bad':torch.zeros(2)},'b':self.clients['b']},
                        {'a':{'w':torch.ones(3,dtype=torch.float64)},'b':self.clients['b']}):
            with self.assertRaises(ValueError):self.call(clients=clients)
    def test_production_and_zero_noise_rejected(self):
        with self.assertRaises(ValueError):self.call(simulation_only=False)
        with self.assertRaises(ValueError):self.call(std_over_sensitivity=0.)
    def test_all_parameters_share_one_clip_bound(self):
        base={'a':torch.zeros(1,dtype=torch.float64),'b':torch.zeros(1,dtype=torch.float64)}
        clients={'x':{'a':torch.tensor([3.],dtype=torch.float64),'b':torch.tensor([4.],dtype=torch.float64)}}
        state,event,_=self.call(reference=base,clients=clients,public_slots=('x',))
        noise=np.random.default_rng(42).normal(0,4,size=2)
        np.testing.assert_allclose(np.array([state['a'].item(),state['b'].item()])-noise,[1.2,1.6],rtol=1e-14)
