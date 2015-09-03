import numpy as np
from scipy.interpolate import interp1d
import quantities as pq

import sciunit, sciunit.scores
from neuronunit.capabilities.channel import * 

class IVCurveTest(sciunit.Test):
    """Test the agreement between channel IV curves produced by models and those observed in experiments"""
    
    def __init__(self, observation, name='IV Curve Test', **params):
        self.validate_observation(observation)
        for key,value in observation.items():
            setattr(self, key, value)
        super(IVCurveTest,self).__init__(observation, name=name, **params)
    
    required_capabilities = (ProducesIVCurve,)
    score_type = sciunit.scores.BooleanScore
    meta = True # i.e. don't use derive tests directly from this class.  
    
    def validate_observation(self, observation):
        assert type(observation) is dict
        for item in ['v', 'i']:
            assert item in observation
            assert type(observation[item]) in [list,tuple] \
                or isinstance(observation[item],np.ndarray)
        if hasattr(observation['v'],'units'):
            observation['v'] = observation['v'].rescale(pq.V) # Rescale to volts.  
        if hasattr(observation['i'],'units'):
            observation['i'] = observation['i'].rescale(pq.pA) # Rescale to picoamps.      
    
    def generate_prediction(self, model):
        raise Exception('This is a meta-class for tests; use tests derived from this class instead')
        
    def interp_IV_curves(self, v_obs, i_obs, v_pred, i_pred):
        """
        Interpolate IV curve along a larger and evenly spaced range of holding potentials. 
        Ensures that test is restricted to ranges present in both predicted and observed data.  
        v_pred: The array of holding potentials in model.  
        i_pred: The array of dict of resulting currents (single values) from model.  
        v_obs: The array of holding potentials from the experiments.  
        i_obs: The array of or dict of resulting currents (single values) from the experiment.  
        """
        
        start = max(v_obs[0],v_pred[0]) # Min voltage in mV
        stop = min(v_obs[-1],v_pred[-1]) # Max voltage in mV
        v_new = np.linspace(start, stop, 100) # 1 mV interpolation
        if type(i_obs)==dict:
            i_obs = np.array([i_obs[key] for key in v_obs]) # Convert dict to numpy array.  
        f_obs = interp1d(v_obs, i_obs, kind='cubic')
        if type(i_pred)==dict:
            i_pred = np.array([i_pred[key] for key in v_pred]) # Convert dict to numpy array.  
        f_pred = interp1d(v_pred, i_pred, kind='cubic')
        i_obs_interp = f_obs(v_new) # Interpolated current from data.  
        i_pred_interp = f_pred(v_new) # Interpolated current from model.  
        return {'v':v_new, 'i_pred':i_pred_interp, 'i_obs':i_obs_interp}
    
    def compute_score(self, observation, prediction):
        # Sum of the difference between the curves.
        o = observation
        p = prediction
        interped = self.interp_IV_curves(o['v'], o['i'], p['v'], p['i'])
        sse = np.sum((interped['i_obs'] - interped['i_pred'])**2) # The sum of the squared differences
        score = sciunit.scores.BooleanScore(sse<100)
        score.description = ("The sum-squared difference in the observed and predicted current values "
                             "over the range of the tested holding potentials.")
        score.value = '%.4g pA^2' % sse
        # Observationa and prediction are automatically stored in score, but since that is before
        # interpolation, we store the interpolated versions as well.  
        score.related_data = interped
        score.plot = self.last_model.plot_iv_curve
        return score


class IVCurveSSTest(IVCurveTest):
    """Test IV curves using steady-state curent"""
    
    def generate_prediction(self, model):
        return model.produce_iv_curve_ss(**self.params)
    
    
class IVCurvePeakTest(IVCurveTest):
    """Test IV curves using steady-state curent"""
    
    def generate_prediction(self, model):
        return model.produce_iv_curve_peak(**self.params)