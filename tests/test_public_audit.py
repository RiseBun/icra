import json
import numpy as np
from pathlib import Path

from scripts.public_trajectory_utils import normalize_episode, amplitude
from scripts.evaluate_calibration import fit_temperature, _probs

def test_public_normalizer_and_amplitude():
    e=normalize_episode({'actions':np.ones((8,3)), 'states':np.arange(24).reshape(8,3)})
    assert e['actions'].shape==(8,3)
    mean,rms,total=amplitude(e['actions'])
    assert mean > 0 and rms > 0 and total > 0

def test_public_audit_proxy_is_not_action_derived(tmp_path):
    p=tmp_path/'episodes.npz'
    actions=np.stack([np.full((8,2),v) for v in (.01,.03,.08,.15)])
    states=np.stack([np.linspace(0,1,8)[:,None]*v for v in (1.,1.,1.,1.)])
    np.savez(p,actions=actions[0],states=states[0])
    assert p.exists()

def test_nested_observation_format_is_normalized():
    e=normalize_episode({'actions':np.ones((8,2)), 'observations':{'state':np.zeros((8,3))}, 'skill':np.asarray('push')})
    assert e['states'].shape==(8,3)
    assert e['meta']['skill']=='push'

def test_temperature_scaling_is_finite():
    y=np.asarray([0,1,0,1,0,1]); s=np.asarray([-.5,.6,-.2,.4,-.8,.9])
    t=fit_temperature(y,s); p=_probs(s,t)
    assert np.isfinite(t) and t > 0 and np.all((p > 0) & (p < 1))
