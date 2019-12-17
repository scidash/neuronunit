"""Passive neuronunit tests, requiring no active conductances or spiking."""

from .base import np, pq, ncap, VmTest, scores
from scipy.optimize import curve_fit
from sciunit.tests import ProtocolToFeaturesTest
import gc
from neuronunit import neuroelectro

DURATION = 500.0*pq.ms
DELAY = 200.0*pq.m
try:
    import asciiplotlib as apl
    fig.plot([1,0], [0,1])

    ascii_plot = True
    import gc

except:
    ascii_plot = False
class TestPulseTest(ProtocolToFeaturesTest):
    """A base class for tests that use a square test pulse."""

    def __init__(self, *args, **kwargs):
        super(TestPulseTest, self).__init__(*args, **kwargs)

    default_params = dict(VmTest.default_params)
    default_params['amplitude'] = -10.0 * pq.pA

    required_capabilities = (ncap.ReceivesSquareCurrent,)

    name = ''

    score_type = scores.ZScore
    @classmethod
    def get_default_injected_square_current(cls):
        current = {key: cls.default_params[key]
                   for key in ['duration', 'delay', 'amplitude']}
        return current

    def get_injected_square_current(self):
        current = {key: self.default_params[key]
                   for key in ['duration', 'delay', 'amplitude']}
        return current

    @classmethod
    def neuroelectro_summary_observation(cls, neuron, cached=False):
        reference_data = neuroelectro.NeuroElectroSummary(
            neuron=neuron,  # Neuron type lookup using the NeuroLex ID.
            ephysprop={'name': cls.ephysprop_name},  # Ephys property name in
                                                     # NeuroElectro ontology.
            cached=cached
            )

        # Get and verify summary data from neuroelectro.org.
        reference_data.get_values(quiet=not cls.verbose)

        if hasattr(reference_data, 'mean'):
            observation = {'mean': reference_data.mean*cls.units,
                           'std': reference_data.std*cls.units,
                           'n': reference_data.n}
        else:
            observation = None

        return observation

    @classmethod
    def neuroelectro_pooled_observation(cls, neuron, cached=False, quiet=True):
        reference_data = neuroelectro.NeuroElectroPooledSummary(
            neuron=neuron,  # Neuron type lookup using the NeuroLex ID.
            # Ephys property name in NeuroElectro ontology.
            ephysprop={'name': cls.ephysprop_name},
            cached=cached
            )
        # Get and verify summary data from neuroelectro.org.
        reference_data.get_values(quiet=quiet)
        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
        return observation


    def compute_params(self):
        super(TestPulseTest, self).compute_params()
        self.params['injected_square_current'] = \
            self.get_injected_square_current()

    def condition_model(self, model):
        if str('tmax') not in self.params.keys():
            self.params['tmax'] = 1000.0*pq.ms
        t_stop = self.params['tmax']
        model.get_backend().set_stop_time(t_stop)

    def setup_protocol(self, model):
        """Implement sciunit.tests.ProtocolToFeatureTest.setup_protocol."""
        """Not a great design for parallel code as model can't be shared"""
        self.condition_model(model)
        model.inject_square_current(self.params['injected_square_current'])

        # self.condition_model(model)
        # model.inject_square_current(self.params['injected_square_current'])
        #return vm

    def get_result(self, model):
        vm = model.get_membrane_potential()
        return vm

    def extract_features(self, model, vm):
        i = self.params['injected_square_current']
        if np.any(np.isnan(vm)) or np.any(np.isinf(vm)):
            return None

        return (i, vm)

    @classmethod
    def get_segment(cls, vm, start, finish):
        start = int((start/vm.sampling_period).simplified)
        finish = int((finish/vm.sampling_period).simplified)
        return vm[start:finish]

    @classmethod
    def get_rin(cls, vm, i):
        start, stop = -11*pq.ms, -1*pq.ms
        before = cls.get_segment(vm, start+i['delay'], stop+i['delay'])
        after = cls.get_segment(vm, start+i['delay']+i['duration'],
                                stop+i['delay']+i['duration'])
        r_in = (after.mean()-before.mean())/i['amplitude']
        return r_in.simplified

    @classmethod
    def get_tau(cls, vm, i):
        # 10 ms before pulse start or halfway between sweep start
        # and pulse start, whichever is longer
        start = max(i['delay'] - 10*pq.ms, i['delay']/2)
        stop = i['duration']+i['delay'] - 1*pq.ms  # 1 ms before pulse end
        region = cls.get_segment(vm, start, stop)
        if len(set(r[0] for r in region.magnitude))>1 and np.std(region.magnitude)>0.0:
            try:
                amplitude, tau, y0 = cls.exponential_fit(region, i['delay'])
            except:
                import pdb
                pdb.set_trace()
        else:
            tau = None
        return tau

    @classmethod
    def exponential_fit(cls, segment, offset):
        t = segment.times.rescale('ms')
        start = t[0]
        offset = offset-start
        t = t-start
        t = t.magnitude
        vm = segment.rescale('mV').magnitude
        offset = (offset * segment.sampling_rate).simplified
        assert offset.dimensionality == pq.dimensionless
        offset = int(offset)
        guesses = [vm.min(),  # amplitude (mV)
                   10,  # time constant (ms)
                   vm.max()]  # y0 (mV)
        vm_fit = vm.copy()

        def func(x, a, b, c):
            """Produce an exponential function.

            Given function parameters a, b, and c, returns the exponential
            decay function for those parameters.
            """
            vm_fit[:offset] = c
            shaped = len(np.shape(vm_fit))
            if shaped > 1:
                vm_fit[offset:, 0] = a * np.exp(-t[offset:]/b) + c
            elif shaped == 1:
                vm_fit[offset:] = a * np.exp(-t[offset:]/b) + c

            return vm_fit.squeeze()
        # Estimate starting values for better convergence
        popt, pcov = curve_fit(func, t, vm.squeeze(), p0=guesses)
        amplitude = popt[0]*pq.mV
        tau = popt[1]*pq.ms
        y0 = popt[2]*pq.mV
        return amplitude, tau, y0

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            score = None
            return score  # scores.InsufficientDataScore(None)

        else:
            score = super(TestPulseTest, self).\
                        compute_score(observation, prediction)
        return score


class InputResistanceTest(TestPulseTest):
    """
    def __init__(self, **kwargs):
        super(InputResistanceTest, self).__init__(**kwargs)
        self.param = {}
        self.params['tmax'] = 1000.0*pq.ms
        if str('params') in kwargs:
            self.params = kwargs['params']
        else:
            self.params = None
    """
    """Test the input resistance of a cell."""

    name = "Input resistance test"

    description = ("A test of the input resistance of a cell.")

    units = pq.UnitQuantity('megaohm', pq.ohm*1e6, symbol='Mohm')  # Megaohms

    ephysprop_name = 'Input Resistance'

    def extract_features(self, model, result):
        features = super(InputResistanceTest, self).\
                            extract_features(model, result)
        if features is not None:
            i, vm = features
            r_in = self.__class__.get_rin(vm, i)
            print("r_in = r_in.simplified")
            print("Put prediction in a form that compute_score() can use.")
            features = {'value': r_in}
        return features
    """
    def compute_score(self, observation, prediction):
        Implement sciunit.Test.score_prediction.
        if prediction is None:
            return None  # scores.InsufficientDataScore(None)
        score = None
        if 'n' in prediction.keys():
            if prediction['n'] == 0:  # if prediction is None:
                score = scores.InsufficientDataScore(None)
        else:
            prediction['value'] = prediction['value'].simplified
            observation['value'] = observation['value'].simplified
            score = super(InputResistanceTest, self).compute_score(observation,
                                                                prediction)

        return score
    """

class TimeConstantTest(TestPulseTest):
    """Test the input resistance of a cell."""

    name = "Time constant test"

    description = ("A test of membrane time constant of a cell.")

    units = pq.ms

    ephysprop_name = 'Membrane Time Constant'
    """
    def __init__(self):
        super(TimeConstantTest, self).__init__()
        self.param = {}
        self.params['tmax'] = 1000.0*pq.ms
        if str('params') in kwargs:
            self.params = kwargs['params']
        else:
            self.params = None
    """

    def extract_features(self, model, result):
        features = super(TimeConstantTest, self).\
                            extract_features(model, result)
        if features is not None:
            i, vm = features
            tau = self.__class__.get_tau(vm, i)
            tau = tau.simplified
            # Put prediction in a form that compute_score() can use.
            features = {'value': tau}
        return features

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            return None  # scores.InsufficientDataScore(None)

        if 'n' in prediction.keys():
            if prediction['n'] == 0:  # if prediction is None:
                score = scores.InsufficientDataScore(None)
            else:
                score = super(TimeConstantTest, self).compute_score(observation,
                                                                prediction)
        else:
            # prediction['value'] = prediction['value']
            score = super(TimeConstantTest, self).compute_score(observation,
                                                                prediction)

        return score


class CapacitanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Capacitance test"

    description = ("A test of the membrane capacitance of a cell.")

    units = pq.pF

    ephysprop_name = 'Cell Capacitance'
    """
    def __init__(self):
        super(CapacitanceTest, self).__init__()
        self.param = {}
        self.params['tmax'] = 1000.0*pq.ms
        if str('params') in kwargs:
            self.params = kwargs['params']
        else:
            self.params = None
    """
    def extract_features(self, model, result):
        features = super(CapacitanceTest, self).extract_features(model, result)
        if features is not None:
            i, vm = features
            r_in = self.__class__.get_rin(vm, i)
            tau = self.__class__.get_tau(vm, i)
            if tau is not None:
                c = (tau/r_in).simplified
            # Put prediction in a form that compute_score() can use.
            else:
                c = None
            features = {'value': c}
        return features

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            return None  # scores.InsufficientDataScore(None)

        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
            else:
                score = super(CapacitanceTest, self).compute_score(observation,prediction)
        else:
            score = super(CapacitanceTest, self).compute_score(observation,prediction)
        return score


class RestingPotentialTest(TestPulseTest):
    """Tests the resting potential under zero current injection."""

    default_params = dict(TestPulseTest.default_params)

    default_params['amplitude'] = 0.0 * pq.pA

    name = "Resting potential test"

    description = ("A test of the resting potential of a cell "
                   "where injected current is set to zero.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Resting membrane potential'
    """
    def __init__(self, *args, **kwargs):
        print(args,kwargs)
        super(RestingPotentialTest, self).__init__()
        self.param = {}
        #self.params['tmax'] = 1000.0*pq.ms
        #if str('params') in kwargs:
        #    self.params = kwargs['params']
        #else:
        #    self.params = None
    """
    def extract_features(self, model, result):
        features = super(RestingPotentialTest, self).\
                            extract_features(model, result)
        if features is not None:
            median = model.get_median_vm()  # Use median for robustness.
            std = model.get_std_vm()
            features = {'mean': median, 'std': std}
        return features

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            return None  # scores.InsufficientDataScore(None)
        else:
            #prediction['value'] = prediction['value'].simplified
            #observation['value'] = observation['value'].simplified

            score = super(RestingPotentialTest, self).\
                        compute_score(observation, prediction)
        if self.verbose:
            print(score)
            print(observation, prediction)
        return score
