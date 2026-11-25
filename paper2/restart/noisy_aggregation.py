"""Clipped equal-slot Gaussian aggregation for explicitly marked simulation.

NumPy RNG and floating-point simulation are not certified DP release machinery.
Raw updates are visible to a trusted server; secure aggregation is absent.
"""
import numpy as np
import torch
from paper2.restart.privacy_accounting import clip_vector,replacement_sensitivity,finite


def aggregate_simulation(reference,clients,public_slots,clip_norm,std_over_sensitivity,noise_rng,*,simulation_only):
    if simulation_only is not True:
        raise ValueError('Production privacy release unavailable; explicit simulation required')
    if not isinstance(noise_rng,np.random.Generator):raise ValueError('Explicit simulation RNG required')
    if not isinstance(public_slots,tuple) or not public_slots or any(not isinstance(s,str) or not s for s in public_slots) or len(set(public_slots))!=len(public_slots):
        raise ValueError('Unique fixed public slots required')
    if set(clients)!=set(public_slots):raise ValueError('All fixed public slots must contribute')
    if not reference:raise ValueError('Nonempty reference state required')
    keys=sorted(reference)
    for name in keys:
        tensor=reference[name]
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or tensor.dtype not in (torch.float32,torch.float64) or not tensor.numel() or not torch.isfinite(tensor).all():
            raise ValueError('Finite floating CPU model required')
    base=np.concatenate([reference[k].detach().numpy().astype(np.float64).ravel() for k in keys])
    clipped=[]
    for slot in public_slots:
        update=clients[slot]
        if set(update)!=set(keys):raise ValueError('Parameter keys mismatch')
        for key in keys:
            value=update[key]
            if not isinstance(value,torch.Tensor) or value.shape!=reference[key].shape or value.dtype!=reference[key].dtype or value.device.type!='cpu' or not torch.isfinite(value).all():
                raise ValueError('Client parameter schema mismatch')
        vector=np.concatenate([update[k].detach().numpy().astype(np.float64).ravel() for k in keys])
        with np.errstate(over='raise',invalid='raise'):
            delta=vector-base
        clipped.append(clip_vector(delta,clip_norm))
    sensitivity=replacement_sensitivity(clip_norm,len(public_slots))
    multiplier=finite(std_over_sensitivity,'noise multiplier',True)
    noise_std=sensitivity*multiplier
    if not np.isfinite(noise_std) or noise_std<=0:raise ValueError('Noise std outside numerical range')
    # Divide each clipped vector before summing to limit intermediate overflow.
    mean=np.sum(np.stack([v/len(public_slots) for v in clipped]),axis=0)
    vector=base+mean+noise_rng.normal(0.,noise_std,size=base.shape)
    if not np.isfinite(vector).all():raise FloatingPointError('Nonfinite noisy aggregate')
    output={};offset=0
    for key in keys:
        count=reference[key].numel()
        output[key]=torch.from_numpy(vector[offset:offset+count].copy()).reshape(reference[key].shape).to(reference[key].dtype)
        if not torch.isfinite(output[key]).all():raise FloatingPointError('Model conversion overflow')
        offset+=count
    return output,{'sensitivity':sensitivity,'noise_std':noise_std,'count':1}, {
        'mode':'SIMULATION_ONLY','public_slots':len(public_slots),'parameter_dimensions':len(base),
        'weighting':'equal fixed slots','application_privacy_certified':False,
        'secure_aggregation':False,'noise_generator':'NumPy simulation RNG; not release-certified'}
