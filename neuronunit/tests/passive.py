#0.001 * pq.nA
"""PAassive neuronunit tests, requiring no active conductances or spiking."""

from .base import np, pq, ncap, VmTest, scores
from scipy.optimize import curve_fit
import gc
DURATION = 500.0*pq.ms
DELAY = 200.0*pq.m
try:
    import asciiplotlib as apl
    fig.plot([1,0], [0,1])

    ascii_plot = True
    import gc

except:
    ascii_plot = False
class TestPulseTest(VmTest):
    """A base class for tests that use a square test pulse."""

    def __init__(self, *args, **kwargs):
        super(TestPulseTest, self).__init__(*args, **kwargs)
        self.param = {}
        self.params['tmax'] = 1000.0*pq.ms
        if str('params') in kwargs:
            self.params = kwargs['params']
        else:
            self.params = None
        self.verbose = None
    default_params = dict(VmTest.default_params)
    default_params['amplitude'] = -10.0 * pq.pA
    default_params['tmax'] = 1000.0*pq.ms
    required_capabilities = (ncap.ReceivesSquareCurrent,)
    name = ''

    score_type = scores.ZScore
    def condition_model(self, model):
        if str('tmax') not in self.params.keys():
            self.params['tmax'] = 1000.0*pq.ms
        t_stop = self.params['tmax']
        model.get_backend().set_stop_time(t_stop)

    def setup_protocol(self, model):
        """Not a great design for parallel code as model can't be shared"""
        self.condition_model(model)
        model.inject_square_current(self.params['injected_square_current'])
        #return vm
    def get_result(self, model):
        #self.condition_model(model)
        #model.inject_square_current(self.params['injected_square_current'])
        vm = model.get_membrane_potential()
        return vm

    def extract_features(self, model,result):

        model.inject_square_current(self.params['injected_square_current'])
        vm = model.get_membrane_potential()

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
        after_ = cls.get_segment(vm, start+i['delay']+i['duration'],
                                stop+i['delay']+i['duration'])
        r_in = (after_.mean()-before.mean())/i['amplitude']

        vs =[float(v) for v in vm.magnitude]

        ts =[float(t) for t in vm.times]
        tMax = np.max(vm.times).simplified
        N = int(tMax/vm.sampling_period)
        Iext = np.zeros(N)

        delay_ind = int((i['delay']/tMax)*N)
        duration_ind = int((i['duration']/tMax)*N)

        Iext[0:delay_ind-1] = i['amplitude']
        Iext[delay_ind:delay_ind+duration_ind-1] = i['amplitude']#*1000.0
        Iext[delay_ind+duration_ind::] = i['amplitude']
        if ascii_plot:
            fig = apl.figure()
            fig.plot(ts, vs, label=str('voltage from negative injection: '), width=100, height=20)
            fig.show()
            fig.plot(ts, Iext, label=str('negative injection times 1,000: '), width=100, height=20)
            fig.show()
            gc.collect()
        return r_in.simplified

    @classmethod
    def get_tau(cls, vm, i):
        # 10 ms before pulse start or halfway between sweep start
        # and pulse start, whichever is longer
        start = max(i['delay'] - 10*pq.ms, i['delay']/2)
        stop = i['duration']+i['delay'] - 1*pq.ms  # 1 ms before pulse end
        region = cls.get_segment(vm, start, stop)
        #import pdb
        #pdb.set_trace()
        #if :
        if len(set(r[0] for r in region.magnitude))>1 and np.std(region.magnitude)>0.0:
            amplitude, tau, y0 = cls.exponential_fit(region, i['delay'])
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
        #import pdb
        #pdb.set_trace()

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
    """Test the input resistance of a cell."""

    name = "Input resistance test"

    description = ("A test of the input resistance of a cell.")

    #*1e6
    units = pq.UnitQuantity('megaohm', pq.ohm*1e6, symbol='Mohm')  # Megaohms
    ephysprop_name = 'Input Resistance'

    def extract_features(self, model, result):
        features = super(InputResistanceTest, self).\
                            extract_features(model, result)
        if features is not None:
            i, vm = features
            r_in = self.__class__.get_rin(vm, i)
            #r_in = r_in.simplified
            features = {'value': r_in}
        return features

    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
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


class TimeConstantTest(TestPulseTest):
    """Test the input resistance of a cell."""

    name = "Time constant test"

    description = ("A test of membrane time constant of a cell.")

    units = pq.ms

    ephysprop_name = 'Membrane Time Constant'

    def extract_features(self, model, result):
        features = super(TimeConstantTest, self).\
                            extract_features(model, result)
        if features is not None:
            i, vm = features
            tau = self.__class__.get_tau(vm, i)
            if tau is not None:
                tau = tau.simplified
            else:
                tau = None

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
            score = super(TimeConstantTest, self).compute_score(observation,
                                                                prediction)
        return score


class CapacitanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Capacitance test"

    description = ("A test of the membrane capacitance of a cell.")

    units = pq.pF

    ephysprop_name = 'Cell Capacitance'

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
    #
    def compute_score(self, observation, prediction):
        """Implement sciunit.Test.score_prediction."""
        if prediction is None:
            return None  # scores.InsufficientDataScore(None)

        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
            else:
                score = super(CapacitanceTest, self).compute_score(observation,
                                                               prediction)
        else:
            score = super(CapacitanceTest, self).compute_score(observation,
                                                               prediction)
        return score


class RestingPotentialTest(TestPulseTest):
    """Tests the resting potential under zero current injection."""

    default_params = dict(TestPulseTest.default_params)
    default_params['amplitude'] = -10.0 * pq.pA

    name = "Resting potential test"

    description = ("A test of the resting potential of a cell "
                   "where injected current is set to zero.")

    #score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Resting membrane potential'

    def extract_features(self, model, result):
        #features = super(RestingPotentialTest, self).\
        #                    extract_features(model, result)
        self.params['injected_square_current']['amplitude'] = -10*0.001 * pq.pA
        model.inject_square_current(self.params['injected_square_current'])

        #if features is not None:
        median = -np.abs(model.get_median_vm())  # Use median for robustness.
        std = -np.abs(model.get_std_vm())

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
