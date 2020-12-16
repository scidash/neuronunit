"""Dynamic neuronunit tests, e.g. investigating dynamical systems properties"""


from elephant.statistics import isi
from elephant.statistics import cv
from elephant.statistics import lv
from elephant.spike_train_generation import threshold_detection

from neuronunit.capabilities.channel import *
from .base import np, pq, ncap, VmTest, scores, AMPL, DELAY, DURATION
from .waveform import InjectedCurrentAPWidthTest
from .fi import RheobaseTest


class TFRTypeTest(RheobaseTest):
    """Test whether a model has particular threshold firing rate dynamics,
    i.e. type 1 or type 2."""

    def __init__(self, *args, **kwargs):
        super(TFRTypeTest, self).__init__(*args, **kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    name = "Firing Rate Type test"

    description = (("A test of the instantaneous firing rate dynamics, i.e. "
                    "type 1 or type 2"))

    score_type = scores.BooleanScore

    def validate_observation(self, observation):
        super(TFRTypeTest, self).validate_observation(observation,
                                                      united_keys=['rheobase'],
                                                      nonunited_keys=['type'])
        assert ('type' in observation) and (observation['type'] in [1, 2]), \
            ("observation['type'] must be either 1 or 2, corresponding to "
             "type 1 or type 2 threshold firing rate dynamics.")

    def generate_prediction(self, model):
        """Implement sciunit.Test.generate_prediction."""

        model.rerun = True
        if 'rheobase' in self.observation:
            guess = self.observation['rheobase']
        else:
            guess = 100.0*pq.pA
        lookup = self.threshold_FI(model, self.units, guess=guess)
        sub = np.array([x for x in lookup if lookup[x] == 0])*self.units
        supra = np.array([x for x in lookup if lookup[x] > 0])*self.units
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s"
                      % float(sub.max().round(2)))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s"
                      % supra.min().round(2))
            else:
                print("No suprathreshold current was tested.")

        prediction = None
        if len(sub) and len(supra):
            supra = np.array([x for x in lookup if lookup[x] > 0])  # No units
            thresh_i = supra.min()
            n_spikes_at_thresh = lookup[thresh_i]
            if n_spikes_at_thresh == 1:
                prediction = 1  # Type 1 dynamics.
            elif n_spikes_at_thresh > 1:
                prediction = 2  # Type 2 dynamics.

        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type(prediction == observation)
        return score


class BurstinessTest(InjectedCurrentAPWidthTest):
    """Test whether a model exhibits the observed burstiness"""

    name = "Burstiness test"

    description = (("A test of AP bursting at the provided current"))

    score_type = scores.RatioScore

    units = pq.dimensionless

    nonunited_observation_keys = ['cv']

    cv_threshold = 1.0

    def generate_prediction(self, model):
        model.inject_square_current(self.run_params['current'])
        spike_train = model.get_spike_train()
        if len(spike_train) >= 3:
            value = cv(spike_train)*pq.dimensionless
        else:
            value = None
        return {'cv': value}

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation, prediction, key='cv')
        return score


class ISICVTest(VmTest):
    """Test whether a model exhibits the observed burstiness"""

    name = "ISI Coefficient of Variation Test"
    description = (("For neurons and muscle cells check the Coefficient of "
                    "Variation on a list of Interval Between Spikes given a "
                    "spike train recording."))
    score_type = scores.RatioScore
    units = pq.dimensionless
    united_observation_keys = []
    nonunited_observation_keys = ['cv']

    def generate_prediction(self, model=None):
        st = model.get_spike_train()
        if len(st) >= 3:
            value = abs(cv(st))*pq.dimensionless
        else:
            value = None
        prediction = {'cv': value}
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""

        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            if self.verbose:
                print(observation, prediction)
            score = self.score_type.compute(observation, prediction, key='cv')
        return score


class ISITest(VmTest):
    """Test whether a model exhibits the observed Inter Spike Intervals"""

    def __init__(self, observation={'isi_mean': None, 'isi_std': None},
                 name=None,
                 params=None):
        pass

    name = "Inter Spike Interval Tests"
    description = (("For neurons and muscle cells check the mean Interval "
                    "Between Spikes given a spike train recording."))
    score_type = scores.ZScore
    units = pq.ms

    def __init__(self, *args, **kwargs):
        super(ISITest, self).__init__(*args, **kwargs)

        if self.name == self.__class__.name:
            self.name = "Inter Spike Interval Test %d test" % \
                        self.observation['type']

    def generate_prediction(self, model=None):
        st = model.get_spike_train()
        isis = isi(st)
        value = float(np.mean(isis))*1000.0*pq.ms
        prediction = {'value': alue}
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation, prediction)
        return score



import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

class InjectedCurrent:
    """Metaclass to mixin with InjectedCurrent tests."""
    def __init__(self,amp):
        self.amp = amp
        default_params = dict(VmTest.default_params)
        default_params.update({'amplitude': 100*pq.pA})
        default_params.update({'delay': 100*pq.ms})
        default_params.update({'duration': 1000*pq.ms})
        self.default_params = default_params
    required_capabilities = (ncap.ReceivesSquareCurrent,)


    def get_params(self):
        self.verbose = False
        self.params = {}
        self.params['injected_square_current'] = self.default_params
        self.params['injected_square_current']['amplitude'] = \
            self.amp
        return self.params


class AdaptionIndexTest(VmTest):

    def generate_prediction(self,model):

        model.rheobase * 1.5

        #I = np.arange(-150,200,20.0)  # a range of current inputs
        m = model.coef_*(pq.Hz/pq.pA)
        self.prediction = {'value':m}

        return self.prediction
    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation, prediction)
        return score

def get_firing_rate(model, input_current):

    # inject a test current into the neuron and call it's run() function.
    # get the spike times using spike_tools.get_spike_times
    # from the spike times, calculate the firing rate f
    #AMPL = input_current
    IC = InjectedCurrent(amp = input_current*pq.pA)
    params = IC.get_params()
    model.inject_square_current(params)
    vm = model.get_membrane_potential()
    spikes = threshold_detection(vm,threshold=0*pq.mV)
    if len(spikes):
        isi_easy = isi(spikes)
        rate = 1.0/np.mean(isi_easy)

        if rate == np.nan or np.isnan(rate):
            rate = 0
        rate = rate*pq.Hz
    else:
        rate = 0*pq.Hz
    return rate



class FITest(VmTest):
    name = "FITest"
    '''
    file:///home/user/Downloads/CellTypes_Ephys_Overview.pdf
    For all long square responses, the average firing rate and the stimulus amplitude were combined to estimate
    the curve of frequency response of the cell versus stimulus intensity (“f-I curve”). The suprathreshold part of this
    curve was fit to a straight line, and the slope of this line was recorded as a cell-wide feature (Figure 6C).
    '''
    def generate_prediction(self,model,plot=False):

        #I = np.arange(-150,200,20.0)  # a range of current inputs
        I = np.arange(-20,200,10.0)  # a range of current inputs

        fr = []
        # loop over current values
        supra_thresh_I = []
        for I_amp in I:
            firing_rate = get_firing_rate(model, I_amp)
            if firing_rate>0:
                supra_thresh_I.append(I_amp)
                fr.append(firing_rate)
        model = LinearRegression()
        x = np.array(supra_thresh_I).reshape(-1, 1)
        y = np.array(fr)
        if len(x):
            model.fit(x, y)
            m = model.coef_*(pq.Hz/pq.pA)
            #print(m)
            if type(m) is type(list()):
                m = m[0]
            self.prediction = {'value':m}
            if plot:
                c = model.intercept_
                y = supra_thresh_I*float(m) +c
                plt.figure()  # new figure
                plt.plot(supra_thresh_I, fr)
                plt.plot(supra_thresh_I, y)
                plt.xlabel('Amplitude of Injecting step current (pA)')
                plt.ylabel('Firing rate (Hz)')
                plt.grid()
                plt.show()
        else:
            self.prediction = None
        return self.prediction
    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            #print(observation, prediction)
            score = self.score_type.compute(observation, prediction)
        return score

class FiringRateTest(RheobaseTest):
    """Test whether a model exhibits the observed burstiness"""

    def __init__(self, *args, **kwargs):
        super(LocalVariationTest, self).__init__(*args, **kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    name = "Firing Rate Test"
    description = (("Spikes Per Second."))
    score_type = scores.RatioScore
    units = pq.dimensionless

    def generate_prediction(self, model=None):
        """Implements sciunit.Test.generate_prediction."""

        ass = model.get_membrane_potential()
        spike_count = model.get_spike_count()

        window = ass.t_stop-ass.t_start
        prediction = {'value': float(spike_count/window)}
        # prediction should be number of spikes divided by time.
        return prediction

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation, prediction,
                                            key='sps_mean')
        return score


class LocalVariationTest(VmTest):
    """Tests whether a model exhibits the observed burstiness"""

    def __init__(self, observation={'cv_mean': None, 'cv_std': None},
                 name=None,
                 params=None):
        pass

    required_capabilities = (ncap.ReceivesSquareCurrent,
                             ncap.ProducesSpikes)

    name = "Local Variation test"
    description = (("For neurons and muscle cells with slower non firing "
                    "dynamics like CElegans neurons check to see how much "
                    "variation is in the continuous membrane potential."))
    score_type = scores.RatioScore
    units = pq.dimensionless
    local_variation = 0.0  # 1.0

    def __init__(self, *args, **kwargs):
        super(LocalVariationTest, self).__init__(*args, **kwargs)
        if self.name == self.__class__.name:
            self.name = "Firing Rate Type %d test" % self.observation['type']

    def generate_prediction(self, model=None):
        prediction = lv(model.get_membrane_potential())
        return prediction

        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = self.score_type.compute(observation, prediction, key='lv')
        return score
