from neuronunit.tests.base import VmTest
import pickle
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache


class AllenTest(VmTest):
    def __init__(self,
                 observation={'mean': None, 'std': None},
                 name='generic_allen',
                 prediction={'mean': None, 'std': None},
                 **params):
        #super(self,VmTest).__init__()
        super(AllenTest, self).__init__(observation, name, **params)

    #required_capabilities = (scap.Runnable, ncap.ProducesMembranePotential,)
    name = ''

    # units = pass #
    # ephysprop_name = ''
    aliases = ''
    def generate_prediction(self,model = None):
        from neuronunit.optimisation.optimization_management import allen_wave_predictions
        if model is None:    
            return self.prediction
        else:
            dtc = model.model_to_dtc()
            dtc,ephys0 = allen_wave_predictions(dtc,thirty=True)
            dtc,ephys1 = allen_wave_predictions(dtc,thirty=False)
            if name in ephys0.keys():
                feature = ephys0['name']
                self.prediction = {}
                self.prediction['mean'] = feature
                self.prediction['std'] = feature
            if name in ephys1.keys():
                feature = ephys1['name']
                self.prediction = {}
                self.prediction['mean'] = feature
                self.prediction['std'] = feature
            return self.prediction
        #ephys1.update()
        #if not len(self.prediction.keys()):

    

    def compute_params(self):
        self.params['t_max'] = (self.params['delay'] +
                               self.params['duration'] +
                               self.params['padding'])


    def set_observation(self,observation):
        self.observation = {}
        self.observation['mean'] = observation
        self.observation['std'] = observation


    def set_prediction(self,prediction):
        self.prediction = {}
        self.prediction['mean'] = prediction
        self.prediction['std'] = prediction
