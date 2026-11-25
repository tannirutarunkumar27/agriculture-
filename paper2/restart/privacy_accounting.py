"""Full-participation Gaussian accounting and clipping; no private training yet.

Implements the Gaussian zCDP bound and composition from Bun/Steinke (2016).
Noise std is measured relative to query sensitivity, including replacement's
factor of two. This module does not generate noise or certify an application.
"""
import math
import numbers
import numpy as np


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid '+name)
    return float(value)


def clip_vector(vector, norm_bound):
    bound=finite(norm_bound,'clip norm',True)
    x=np.asarray(vector)
    if x.dtype!=np.float64 or x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty float64 vector required')
    maximum=float(np.max(np.abs(x)))
    if maximum==0:return x.copy()
    scaled=x/maximum
    norm=float(np.linalg.norm(scaled))
    if maximum <= bound/norm:return x.copy()
    # Stable even where direct sum-of-squares would overflow.
    return scaled/norm*(bound*(1-1e-15))


def replacement_sensitivity(clip_norm, public_slots):
    clip_norm=finite(clip_norm,'clip norm',True)
    if type(public_slots) is not int or public_slots<1:
        raise ValueError('Fixed positive public client-slot count required')
    sensitivity=clip_norm/public_slots*2
    if not math.isfinite(sensitivity) or sensitivity<=0:
        raise ValueError('Sensitivity outside numerical range')
    return sensitivity


def gaussian_budget(events, delta):
    delta=finite(delta,'delta',True)
    if delta>=1:raise ValueError('delta must be in (0,1)')
    terms=[];releases=0
    for event in events:
        if set(event)!={'sensitivity','noise_std','count'}:
            raise ValueError('Gaussian event schema only; no subsampling assumption')
        sensitivity=finite(event['sensitivity'],'sensitivity')
        noise=finite(event['noise_std'],'Gaussian standard deviation',True)
        count=event['count']
        if type(count) is not int or count<0:raise ValueError('Nonnegative integer release count required')
        ratio=sensitivity/noise
        try:rho=.5*ratio**2*count
        except OverflowError as exc:raise ValueError('Privacy cost overflow') from exc
        if not math.isfinite(rho):raise ValueError('Privacy cost overflow')
        if sensitivity>0 and count>0 and rho==0:
            raise ValueError('Privacy cost underflow; cannot report zero cost')
        terms.append(rho);releases+=count
    try:rho=math.fsum(terms)
    except OverflowError as exc:raise ValueError('Privacy cost overflow') from exc
    epsilon=rho+2*math.sqrt(rho)*math.sqrt(-math.log(delta))
    if not math.isfinite(epsilon):raise ValueError('Privacy bound overflow')
    return {'rho':rho,'epsilon_upper_bound':epsilon,'delta':delta,'releases':releases,
            'accountant':'Gaussian zCDP composition, no sampling amplification',
            'application_privacy_certified':False}
