import numpy as np
import quantities as pq

import sciunit
from sciunit.scores import BooleanScore,FloatScore
from neuronunit.capabilities.channel import * 
from .__init__ import RheobaseTest

class TFRTypeTest(RheobaseTest):
    """Base class for tests involving the membrane potential of a model."""

    name = "Firing Rate Type test"

    description = (("A test of the instantaneous firing rate dynamics, i.e. "
                    "type 1 or type 2"))

    units = None

    score_type = scores.BooleanScore

    def validate_observation(self, observation):
        assert observation in [1,2], (("Observation must be either 1 or 2, "
            "corresponding to type 1 or 2 threshold firing rate dynamics."))
        
    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by ProducesActionPotentials capability.
        
        units = self.observation['value'].units
        model.rerun = True
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        supra = np.array([x for x in lookup if lookup[x]>0])*units
        if verbose:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % (float(sub.max().round(2)))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s" \
                      % supra.min().round(2))
            else:
                print("No suprathreshold current was tested.")
                
        prediction = None
        if len(sub) and len(supra):
            supra = np.array([x for x in lookup if lookup[x]>0]) # No units
            thresh_i = supra.min()
            n_spikes_at_thresh = lookup[thresh_i]
            if n_spikes_at_thresh == 1:
                prediction = 1 # Type 1 dynamics.
            elif n_spikes_at_thresh > 1:
                prediction = 2 # Type 2 dynamics.
        
        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type(prediction == observation)
        return score