"""Dynamic neuronunit tests, e.g. investigating dynamical systems properties"""


from elephant.statistics import isi
from elephant.statistics import cv
from elephant.statistics import lv
from neuronunit.capabilities.channel import *
from .base import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
from .waveform import InjectedCurrentAPWidthTest
from .fi import RheobaseTestP, RheobaseTest


class TFRTypeTest(RheobaseTest):
    """Tests whether a model has particular threshold firing rate dynamics,
    i.e. type 1 or type 2."""

    name = "Firing Rate Type test"

    description = (("A test of the instantaneous firing rate dynamics, i.e. "
                    "type 1 or type 2"))

    score_type = scores.BooleanScore

    def __init__(self, *args, **kwargs):
        super(TFRTypeTest,self).__init__(*args,**kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    def validate_observation(self, observation):
        super(TFRTypeTest,self).validate_observation(observation,
                                                     united_keys=['rheobase'],
                                                     nonunited_keys=['type'])
        assert ('type' in observation) and (observation['type'] in [1,2]), \
            ("observation['type'] must be either 1 or 2, corresponding to "
             "type 1 or type 2 threshold firing rate dynamics.")

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""

        model.rerun = True
        if 'rheobase' in self.observation:
            guess = self.observation['rheobase']
        else:
            guess = 100.0*pq.pA
        lookup = self.threshold_FI(model, self.units, guess=guess)
        sub = np.array([x for x in lookup if lookup[x]==0])*self.units
        supra = np.array([x for x in lookup if lookup[x]>0])*self.units
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % float(sub.max().round(2)))
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

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type(prediction == observation)
        return score


class BurstinessTest(InjectedCurrentAPWidthTest):
    """Tests whether a model exhibits the observed burstiness"""

    def __init__(self, observation={'cv_mean':None,'cv_std':None},
                 name=None,
                 params=None):
        pass

    name = "Burstiness test"

    description = (("A test of AP bursting at the provided current"))

    score_type = scores.RatioScore

    units = pq.Dimensionless

    cv_threshold = 1.0

    def validate_observation(self, observation):
        super(TFRTypeTest,self).validate_observation(observation,
                                                     nonunited_keys=['cv'])

    def generate_prediction(self, model):
        model.inject_square_current(observation['current'])
        spike_train = model.get_spike_train()
        if len(spike_train) >= 3:
            cv = cv2(spike_train)
            isis = isi(spike_train)
            cv_old = np.std(isis) / np.mean(isis)
            print(cv,cv_old)
        else:
            cv = None
        return {'cv':cv}

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation,prediction,key='cv')
        return score


class FiringRateTest(RheobaseTest):
    """Tests whether a model exhibits the observed burstiness"""

    def __init__(self, observation={'cv_mean':None,'cv_std':None},
                 name=None,
                 params=None):
        pass

    name = "Local Variation test"
    description = (("For neurons and muscle cells with slower non firing dynamics like CElegans neurons check to see how much \
    varition is in the continuous membrane potential."))
    score_type = scores.RatioScore
    units = pq.Dimensionless
    local_variation = 0.0 # 1.0

    def __init__(self, *args, **kwargs):
        super(LocalVariationTest,self).__init__(*args,**kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    '''
    def validate_observation(self, observation):
        super(LocalVariationTest,self).validate_observation(observation,
                                                     united_keys=['rheobase'],
                                                     nonunited_keys=['type'])

        assert ('type' in observation) and (observation['type'] in [1,2]), \
            ("observation['type'] must be either 1 or 2, corresponding to "
             "type 1 or type 2 threshold firing rate dynamics.")
    '''

    def generate_prediction(self, model = None, waveform =None):
        """Implementation of sciunit.Test.generate_prediction."""

        if waveform is not None:
            prediction = lv(waveform)
        return prediction

        '''
        model.rerun = True
        if 'rheobase' in self.observation:
            guess = self.observation['rheobase']
        else:
            guess = 100.0*pq.pA
        lookup = self.threshold_FI(model, self.units, guess=guess)
        sub = np.array([x for x in lookup if lookup[x]==0])*self.units
        supra = np.array([x for x in lookup if lookup[x]>0])*self.units
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % float(sub.max().round(2)))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s" \
                      % supra.min().round(2))
            else:
                print("No suprathreshold current was tested.")

        prediction = None
        if len(sub) and len(supra):
            sub = np.array([x for x in lookup if lookup[x]<0]) # No units
            below_thresh_i = sub.max()
            _ = self.threshold_FI(model, self.units, guess = below_thresh_i)
        '''



    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation,prediction,key='cv')
        return score


class CVTest(VmTest):
    """Tests whether a model exhibits the observed burstiness"""

    def __init__(self, observation={'cv_mean':None,'cv_std':None},
                 name=None,
                 params=None):
        pass

    name = "Local Variation test"
    description = (("For neurons and muscle cells with slower non firing dynamics like CElegans neurons check to see how much \
    varition is in the continuous membrane potential."))
    score_type = scores.RatioScore
    units = pq.Dimensionless
    #local_variation = 0.0 # 1.0

    def __init__(self, *args, **kwargs):
        super(CVTest,self).__init__(*args,**kwargs)
        #super(TFRTypeTest,self).__init__(*args,**kwargs)

        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    '''
    def validate_observation(self, observation):
        super(CVTest,self).validate_observation(observation,
                                                     united_keys=['rheobase'],
                                                     nonunited_keys=['type'])

        assert ('type' in observation) and (observation['type'] in [1,2]), \
            ("observation['type'] must be either 1 or 2, corresponding to "
             "type 1 or type 2 threshold firing rate dynamics.")
    '''

    def generate_prediction(self, model = None):
        prediction = abs(cv(model.get_membrane_potential()))
        return prediction




    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation,prediction,key='cv')
        return score

class LocalVariationTest(VmTest):
    """Tests whether a model exhibits the observed burstiness"""

    def __init__(self, observation={'cv_mean':None,'cv_std':None},
                 name=None,
                 params=None):
        pass

    name = "Local Variation test"
    description = (("For neurons and muscle cells with slower non firing dynamics like CElegans neurons check to see how much \
    varition is in the continuous membrane potential."))
    score_type = scores.RatioScore
    units = pq.Dimensionless
    local_variation = 0.0 # 1.0

    def __init__(self, *args, **kwargs):
        super(LocalVariationTest,self).__init__(*args,**kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    '''
    def validate_observation(self, observation):
        super(LocalVariationTest,self).validate_observation(observation,
                                                     united_keys=['rheobase'],
                                                     nonunited_keys=['type'])

        assert ('type' in observation) and (observation['type'] in [1,2]), \
            ("observation['type'] must be either 1 or 2, corresponding to "
             "type 1 or type 2 threshold firing rate dynamics.")
    '''


    def generate_prediction(self, model = None):
        prediction = lv(model.get_membrane_potential())
        return prediction

        return prediction




    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #    (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation,prediction,key='lv')
        return score
