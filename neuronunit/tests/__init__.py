"""NeuronUnit Test classes."""

from .passive import *
from .waveform import *
# from .dynamics import *
from .fi import *
# from .elephant_tests import *
# from .druckman2013 import *
from sciunit import scores, errors

from sciunit.errors import CapabilityError, InvalidScoreError
import sciunit
class FakeTest(sciunit.Test):

    #from sciunit.errors import CapabilityError, InvalidScoreError

    #score_type = scores.RatioScore
    score_type = sciunit.scores.ZScore

    def generate_prediction(self, model):
        self.key_param = self.name.split('_')[1]
        self.prediction = model.attrs[self.key_param]
        return self.prediction

    def compute_score(self, observation, prediction):
        mean = observation[0]
        std = observation[1]
        z = (prediction - mean)/std
        #self.prediction = prediction
        #print(scores.ZScore(z))
        return scores.ZScore(z)
