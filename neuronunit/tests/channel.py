import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import quantities as pq

import sciunit
from sciunit.scores import BooleanScore,FloatScore
from sciunit.converters import AtMostToBoolean
from neuronunit.capabilities.channel import * 

class IVCurveTest(sciunit.Test):
    """Test the agreement between channel IV curves produced by models and those 
    observed in experiments"""
    
    def __init__(self, observation, name='IV Curve Test', scale=False, 
                 **params):
        self.validate_observation(observation)
        for key,value in observation.items():
            setattr(self, key, value)
        self.scale = scale # Whether to scale the predicted IV curve to 
                           # minimize distance from the observed IV curve
        self.converter = AtMostToBoolean(pq.Quantity(1.0,'pA**2'))
        super(IVCurveTest,self).__init__(observation, name=name, **params)
    
    required_capabilities = (ProducesIVCurve,)
    score_type = BooleanScore
    meta = True # base class for deriving useful test classes 

    def validate_observation(self, observation):
        assert type(observation) is dict
        for item in ['v', 'i']:
            assert item in observation
            assert type(observation[item]) in [list,tuple] \
                or isinstance(observation[item],np.ndarray)
        if hasattr(observation['v'],'units'):
            observation['v'] = observation['v'].rescale(pq.V) # Rescale to V 
        if hasattr(observation['i'],'units'):
            observation['i'] = observation['i'].rescale(pq.pA) # Rescale to pA     
    
    def generate_prediction(self, model, verbose=False):
        raise Exception(("This is a meta-class for tests; use tests derived "
                         "from this class instead"))
        
    def interp_IV_curves(self, v_obs, i_obs, v_pred, i_pred):
        """
        Interpolate IV curve along a larger and evenly spaced range of holding 
        potentials. Ensures that test is restricted to ranges present in both 
        predicted and observed data.
        v_pred: The array of holding potentials in model.  
        i_pred: The array or dict of resulting currents (single values) 
                from model .
        v_obs: The array of holding potentials from the experiments.  
        i_obs: The array or dict of resulting currents (single values) 
               from the experiment .
        """
        
        if type(i_obs)==dict:
            # Convert dict to numpy array. 
            units = list(i_obs.values())[0].units
            i_obs = np.array([i_obs[float(key)] for key in v_obs]) * units
        if type(i_pred)==dict:
            # Convert dict to numpy array.
            units = list(i_pred.values())[0].units
            i_pred = np.array([i_pred[float(key)] for key in v_pred]) * units
        
        # Removing units temporarily due to 
        # https://github.com/python-quantities/python-quantities/issues/105
        v_obs = v_obs.rescale(pq.mV).magnitude
        i_obs = i_obs.rescale(pq.pA).magnitude
        v_pred = v_pred.rescale(pq.mV).magnitude
        i_pred = i_pred.rescale(pq.pA).magnitude
        
        f_obs = interp1d(v_obs, i_obs, kind='cubic')
        f_pred = interp1d(v_pred, i_pred, kind='cubic')
        start = max(v_obs[0],v_pred[0]) # Min voltage in mV
        stop = min(v_obs[-1],v_pred[-1]) # Max voltage in mV
        
        v_new = np.linspace(start, stop, 100) # 1 mV interpolation
        i_obs_interp = f_obs(v_new)*pq.pA # Interpolated current from data  
        i_pred_interp = f_pred(v_new)*pq.pA # Interpolated current from model
        
        return {'v':v_new*pq.mV, 'i_pred':i_pred_interp, 'i_obs':i_obs_interp}
    
    def compute_score(self, observation, prediction, verbose=False):
        # Sum of the difference between the curves.
        o = observation
        p = prediction
        interped = self.interp_IV_curves(o['v'], o['i'], p['v'], p['i'])
        
        if self.scale:
            def f(sf):
                score = FloatScore.compute_ssd(interped['i_obs'],
                                               (10**sf)*interped['i_pred'])
                return score.score.magnitude
            result = minimize(f,0.0)
            scale_factor = 10**result.x
            interped['i_pred'] *= scale_factor
        else:
            scale_factor = 1
        
        score = FloatScore.compute_ssd(interped['i_obs'],interped['i_pred'])
        score.related_data['scale_factor'] = scale_factor
        self.interped = interped
        return score

    def bind_score(self,score,model,observation,prediction):
        score.description = ("The sum-squared difference in the observed and "
                             "predicted current values over the range of the "
                             "tested holding potentials.")
        # Observation and prediction are automatically stored in score, 
        # but since that is before interpolation, we store the interpolated 
        # versions as well.  
        score.related_data.update(self.interped)
        score.plot = self.last_model.plot_iv_curve
        return score


class IVCurveSSTest(IVCurveTest):
    """Test IV curves using steady-state curent"""
    
    def generate_prediction(self, model, verbose=False):
        return model.produce_iv_curve_ss(**self.params)
    
    
class IVCurvePeakTest(IVCurveTest):
    """Test IV curves using steady-state curent"""
    
    def generate_prediction(self, model, verbose=False):
        return model.produce_iv_curve_peak(**self.params)