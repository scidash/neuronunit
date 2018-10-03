"""Passive neuronunit tests, e.g. requiring no active conductances or spiking"""

from .base import np, pq, sciunit, cap, VmTest, scores, AMPL, DELAY, DURATION
from scipy.optimize import curve_fit

DURATION = 500.0*pq.ms
DELAY = 200.0*pq.ms

class TestPulseTest(VmTest):
    """A base class for tests that use a square test pulse"""

    required_capabilities = (cap.ReceivesSquareCurrent,)

    name = ''

    score_type = scores.ZScore

    #params = {'injected_square_current':
    #            {'amplitude':-10.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}
    #print(params)

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        t_stop = self.params['injected_square_current']['delay'] + \
                   self.params['injected_square_current']['duration'] + \
                   100.0 * pq.ms
        model.get_backend().set_stop_time(t_stop)
        #print('injected current seen: ',self.params['injected_square_current'])
        model.inject_square_current(self.params['injected_square_current'])
        vm = model.get_membrane_potential()
        i = self.params['injected_square_current']
        return (i,vm)

    @classmethod
    def get_segment(cls, vm, start, finish):
        start = int((start/vm.sampling_period).simplified)
        finish = int((finish/vm.sampling_period).simplified)
        return vm[start:finish]

    @classmethod
    def get_rin(cls, vm, i):
        start, stop = -11*pq.ms, -1*pq.ms
        before = cls.get_segment(vm,start+i['delay'],
                                     stop+i['delay'])
        after = cls.get_segment(vm,start+i['delay']+i['duration'],
                                    stop+i['delay']+i['duration'])
        r_in = (after.mean()-before.mean())/i['amplitude']
        return r_in.simplified

    @classmethod
    def get_tau(cls, vm, i):
        # 10 ms before pulse start or halfway between sweep start
        # and pulse start, whichever is longer
        start = max(i['delay']-10*pq.ms,i['delay']/2)
        stop = i['duration']+i['delay']-1*pq.ms # 1 ms before pulse end
        region = cls.get_segment(vm,start,stop)
        amplitude,tau,y0 = cls.exponential_fit(region, i['delay'])
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
        guesses = [vm.min(), # amplitude (mV)
                   10, # time constant (ms)
                   vm.max()] # y0 (mV)
        vm_fit = vm.copy()

        def func(x, a, b, c):
            '''
            This function is simply the shape of exponential decay which must be differenced, its basically an ideal template
            An exp decay equation derived from experiments.
            For the model to compare against.
            '''
            vm_fit[:offset] = c
            vm_fit[offset:,0] = a * np.exp(-t[offset:]/b) + c
            return vm_fit.squeeze()

        popt, pcov = curve_fit(func, t, vm.squeeze(), p0=guesses) # Estimate starting values for better convergence
        amplitude = popt[0]*pq.mV
        tau = popt[1]*pq.ms
        y0 = popt[2]*pq.mV
        return amplitude,tau,y0


class InputResistanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Input resistance test"

    description = ("A test of the input resistance of a cell.")

    units = pq.UnitQuantity('megaohm', pq.ohm*1e6, symbol='Mohm')  # Megaohms

    ephysprop_name = 'Input Resistance'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(InputResistanceTest,self).\
                            generate_prediction(model)
        for param in ['delay', 'duration', 'amplitude']:
            i[param] = self.params['injected_square_current'][param]

        # i['amplitude'] = -10*pq.pA*1000
        # self.params['injected_square_current'][param] = -10*pq.pA*1000

        r_in = self.__class__.get_rin(vm, i)
        r_in = r_in.simplified
        # Put prediction in a form that compute_score() can use.
        prediction = {'value':r_in}

        return prediction


    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #	 (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
            #score = scores.ErrorScore(None)

        else:
            score = super(InputResistanceTest,self).\
                        compute_score(observation, prediction)
            #print('input resistance score: {0}'.format(score))
            #self.bind_score(score,None,observation,prediction)
        return score


class TimeConstantTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Time constant test"

    description = ("A test of membrane time constant of a cell.")

    units = pq.ms

    ephysprop_name = 'Membrane Time Constant'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(TimeConstantTest,self).generate_prediction(model)
        tau = self.__class__.get_tau(vm, i)
        tau = tau.simplified
        # Put prediction in a form that compute_score() can use.
        prediction = {'value':tau}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""

        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
        else:
            prediction['value']=prediction['value']
            score = super(TimeConstantTest,self).compute_score(observation,
                                                          prediction)

        return score


class CapacitanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Capacitance test"

    description = ("A test of the membrane capacitance of a cell.")

    units = pq.pF

    ephysprop_name = 'Cell Capacitance'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(CapacitanceTest,self).generate_prediction(model)
        r_in = self.__class__.get_rin(vm, i)
        tau = self.__class__.get_tau(vm, i)
        c = (tau/r_in).simplified
        # Put prediction in a form that compute_score() can use.
        prediction = {'value':c}

        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""

        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
        else:
            score = super(CapacitanceTest,self).compute_score(observation,
                                                          prediction)
        return score


class RestingPotentialTest(VmTest):
    """Tests the resting potential under zero current injection."""

    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':0.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Resting potential test"

    description = ("A test of the resting potential of a cell "
                   "where injected current is set to zero.")

    score_type = scores.ZScore


    units = pq.mV

    ephysprop_name = 'Resting membrane potential'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""

        model.rerun = True

        model.inject_square_current(self.params['injected_square_current'])

        median = model.get_median_vm() # Use median for robustness.
        std = model.get_std_vm()
        prediction = {'mean':median, 'std':std}

        mp=model.get_membrane_potential()
        import math
        for i in mp:
            if math.isnan(i):
                return None
        prediction = {'mean':median, 'std':std}
        self.prediction = prediction
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #	 (self.name,str(observation),str(prediction)))
        if prediction is None:
            score = scores.InsufficientDataScore(None)
            #score = scores.ErrorScore(None)

        else:
            score = super(RestingPotentialTest,self).\
                        compute_score(observation, prediction)
            #self.bind_score(score,None,observation,prediction)
        return score
