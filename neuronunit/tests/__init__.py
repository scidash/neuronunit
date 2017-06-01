"""NeuronUnit Test classes"""

import inspect
from types import MethodType

import quantities as pq
from quantities.quantity import Quantity
import numpy as np
import matplotlib as mpl
mpl.use('agg',warn=False)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sciunit
import sciunit.scores as scores

import neuronunit.capabilities as cap
#import neuronunit.capabilities.spike_functions as sf
from neuronunit import neuroelectro
#from .channel import *
from scoop import futures

AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 1000.0*pq.ms

class VmTest(sciunit.Test):
    """Base class for tests involving the membrane potential of a model."""

    def __init__(self,
                 observation={'mean':None,'std':None},
                 name=None,
                 **params):

        super(VmTest,self).__init__(observation,name,**params)
        cap = []
        for cls in self.__class__.__bases__:
            cap += cls.required_capabilities
        self.required_capabilities += tuple(cap)
        self._extra()
    
    required_capabilities = (cap.ProducesMembranePotential,)

    name = ''

    units = pq.Dimensionless

    ephysprop_name = ''

    # Observation values with units.
    united_observation_keys = ['value','mean','std']

    def _extra(self):
        pass

    def validate_observation(self, observation,
                             united_keys=['value','mean'], nonunited_keys=[]):
        try:
            assert type(observation) is dict
            assert any([key in observation for key in united_keys]) \
                or len(nonunited_keys)
            for key in united_keys:
                if key in observation:
                    assert type(observation[key]) is Quantity
            for key in nonunited_keys:
                if key in observation:
                    assert type(observation[key]) is not Quantity \
                        or observation[key].units == pq.Dimensionless
        except Exception as e:
            key_str = 'and/or a '.join(['%s key' % key for key in united_keys])
            msg = ("Observation must be a dictionary with a %s and each key "
                   "must have units from the quantities package." % key_str)
            raise sciunit.ObservationError(msg)
        for key in united_keys:
            if key in observation:
                provided = observation[key].simplified.units
                required = self.units.simplified.units
                if provided != required: # Units don't match spec.
                    msg = ("Units of %s are required for %s but units of %s "
                           "were provided" % (required.dimensionality.__str__(),
                                              key,
                                              provided.dimensionality.__str__())
                           )
                    raise sciunit.ObservationError(msg)

    def bind_score(self, score, model, observation, prediction):
        score.related_data['vm'] = model.get_membrane_potential()
        score.related_data['model_name'] = '%s_%s' % (model.name,self.name)

        def plot_vm(self, ax=None, ylim=(None,None)):
            """A plot method the score can use for convenience."""
            if ax is None:
                ax = plt.gca()
            vm = score.related_data['vm'].rescale('mV')
            ax.plot(vm.times,vm)
            y_min = float(vm.min()-5.0*pq.mV) if ylim[0] is None else ylim[0]
            y_max = float(vm.max()+5.0*pq.mV) if ylim[1] is None else ylim[1]
            ax.set_xlim(vm.times.min(),vm.times.max())
            ax.set_ylim(y_min,y_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vm (mV)')
        score.plot_vm = MethodType(plot_vm, score) # Bind to the score.
        score.unpicklable.append('plot_vm')
        return score

    @classmethod
    def neuroelectro_summary_observation(cls, neuron):
        reference_data = neuroelectro.NeuroElectroSummary(
            neuron = neuron, # Neuron type lookup using the NeuroLex ID.
            ephysprop = {'name': cls.ephysprop_name} # Ephys property name in
                                                     # NeuroElectro ontology.
            )
        reference_data.get_values(quiet=not cls.verbose) # Get and verify summary data
                                    # from neuroelectro.org.


        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
        return observation

    @classmethod
    def neuroelectro_pooled_observation(cls, neuron, quiet=True):
        reference_data = neuroelectro.NeuroElectroPooledSummary(
            neuron = neuron, # Neuron type lookup using the NeuroLex ID.
            ephysprop = {'name': cls.ephysprop_name} # Ephys property name in
                                                     # NeuroElectro ontology.
            )
        reference_data.get_values(quiet=quiet) # Get and verify summary data
                                    # from neuroelectro.org.
        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
        return observation

    def sanity_check(self,rheobase,model):
        '''
        check if the membrane potential and its derivative, constitute continuous differentiable
        functions
        If they don't output boolean false, such that the corresponding model can be discarded
        inputs: a rheobase value and a model.
        outputs: a boolean flag.
        '''
        self.params['injected_square_current']['delay'] = DELAY
        self.params['injected_square_current']['duration'] = DURATION
        self.params['injected_square_current']['amplitude'] = rheobase
        model.inject_square_current(self.params['injected_square_current'])
        import numpy as np
        import copy
        import math
        def nan_test(mp):
            for i in mp:
                if type(i)==np.float64:
                    if math.isnan(i):
                        return False
                    if (i == float('inf')) or (i == float('-inf')):
                        return False
                    if math.isnan(i):
                        return False

            x = np.array(mp).std()
            if x == 0:
                return False
        model.re_init(model.attrs)
        #mp = np.array(copy.copy(model.results['vm']))
        mp = model.results['vm']
        import math
        for i in mp:
            if math.isnan(i):
                return False
        boolean = True
        boolean = nan_test(mp)
        if boolean == False:
            return False
        self.params['injected_square_current']['amplitude'] = -10.0
        model.inject_square_current(self.params['injected_square_current'])
        model.re_init(model.attrs)
        mp = np.array(copy.copy(model.results['vm']))
        boolean = nan_test(mp)
        if boolean == False:
            return False

        import neuronunit.capabilities as cap

        sws=cap.spike_functions.get_spike_waveforms(model.get_membrane_potential())
        #sws = model.get_spike_waveforms()
        #print(sws)

        #sws=spike_functions.get_spike_waveform(mp)
        for i,s in enumerate(sws):
            s = np.array(s)
            dvdt = np.diff(s)
            import math
            for j in dvdt:
                if math.isnan(j):
                    return False
        return True





class TestPulseTest(VmTest):
    """A base class for tests that use a square test pulse"""

    required_capabilities = (cap.ReceivesSquareCurrent,)

    name = ''

    score_type = scores.ZScore

    params = {'injected_square_current':
                {'amplitude':-10.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
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
            vm_fit[:offset] = c
            vm_fit[offset:] = a * np.exp(-t[offset:]/b) + c
            return vm_fit
        
        popt, pcov = curve_fit(func, t, vm, p0=guesses) # Estimate starting values for better convergence
        #plt.plot(t,vm)
        #plt.plot(t,func(t,*popt))
        #print(popt)
        amplitude = popt[0]*pq.mV
        tau = popt[1]*pq.ms
        y0 = popt[2]*pq.mV
        return amplitude,tau,y0


class InputResistanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""
    score_type = scores.ZScore
    name = "Input resistance test"

    name = "Input resistance test"

    description = ("A test of the input resistance of a cell.")

    units = pq.ohm*1e6

    ephysprop_name = 'Input Resistance'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(InputResistanceTest,self).\
                            generate_prediction(model)
        r_in = self.__class__.get_rin(vm, i)
        r_in = r_in.simplified
        # Put prediction in a form that compute_score() can use.
        prediction = {'value':r_in}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""

        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
        else:
            prediction['value']=prediction['value']
            score = super(InputResistanceTest,self).compute_score(observation,
                                                          prediction)
        return score



class TimeConstantTest(TestPulseTest):
    """Tests the input resistance of a cell."""
    score_type = scores.ZScore
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
    score_type = scores.ZScore
    name = "Capacitance test"

    description = ("A test of the membrane capacitance of a cell.")

    units = pq.F*1e-12

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


class APWidthTest(VmTest):
    """Tests the full widths of action potentials at their half-maximum."""

    required_capabilities = (cap.ProducesActionPotentials,)

    name = "AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height.")

    score_type = scores.ZScore

    units = pq.ms

    ephysprop_name = 'Spike Half-Width'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        widths = model.get_AP_widths()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean':np.mean(widths) if len(widths) else None,
                      'std':np.std(widths) if len(widths) else None,
                      'n':len(widths)}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APWidthTest,self).compute_score(observation,
                                                          prediction)
        return score


class InjectedCurrentAPWidthTest(APWidthTest):
    """
    Tests the full widths of APs at their half-maximum
    under current injection.
    """
    score_type = scores.ZScore
    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Injected current AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        return super(InjectedCurrentAPWidthTest,self).generate_prediction(model)


class APAmplitudeTest(VmTest):
    """Tests the heights (peak amplitude) of action potentials."""

    required_capabilities = (cap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Amplitude'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        heights = model.get_AP_amplitudes() - model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean':np.mean(heights) if len(heights) else None,
                      'std':np.std(heights) if len(heights) else None,
                      'n':len(heights)}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APAmplitudeTest,self).compute_score(observation,
                                                              prediction)
        return score

    @classmethod
    def neuroelectro_summary_observation(cls, neuron):
        reference_data = neuroelectro.NeuroElectroSummary(
            neuron = neuron, # Neuron type lookup using the NeuroLex ID.
            ephysprop = {'name': cls.ephysprop_name} # Ephys property name in
                                                     # NeuroElectro ontology.
            )
        reference_data.get_values() # Get and verify summary data
                                    # from neuroelectro.org.
        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
        return observation


class InjectedCurrentAPAmplitudeTest(APAmplitudeTest):
    """
    Tests the heights (peak amplitude) of action potentials
    under current injection.
    """

    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Injected current AP amplitude test"

    description = ("A test of the heights (peak amplitudes) of "
                   "action potentials when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        return super(InjectedCurrentAPAmplitudeTest,self).\
                generate_prediction(model)


class APThresholdTest(VmTest):
    """Tests the full widths of action potentials at their half-maximum."""

    required_capabilities = (cap.ProducesActionPotentials,)

    name = "AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Threshold'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        threshes = model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        prediction = {'mean':np.mean(threshes) if len(threshes) else None,
                      'std':np.std(threshes) if len(threshes) else None,
                      'n':len(threshes)}
        return prediction

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        if prediction['n'] == 0:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(APThresholdTest,self).compute_score(observation,
                                                              prediction)
        return score


class InjectedCurrentAPThresholdTest(APThresholdTest):
    """
    Tests the thresholds of action potentials
    under current injection.
    """
    score_type = scores.ZScore
    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Injected current AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced under current injection.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        return super(InjectedCurrentAPThresholdTest,self).\
                generate_prediction(model)


class RheobaseTest(VmTest):
    """
    Tests the full widths of APs at their half-maximum
    under current injection.
    """
    def _extra(self):
        self.prediction = None
        self.high = 300*pq.pA
        self.small = 0*pq.pA
        self.rheobase_vm = None
    
    required_capabilities = (cap.ReceivesSquareCurrent,
                             cap.ProducesSpikes)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Rheobase test"

    description = ("A test of the rheobase, i.e. the minimum injected current "
                   "needed to evoke at least one spike.")

    units = pq.pA
    score_type = scores.RatioScore


    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        prediction = {'value': None}
        model.rerun = True
        units = self.observation['value'].units
        import time
        begin_rh=time.time()
        lookup = self.threshold_FI(model, units)
        sub = np.array([x for x in lookup if lookup[x]==0])*units
        supra = np.array([x for x in lookup if lookup[x]>0])*units
        self.verbose=True
        if self.verbose:
            if len(sub):
                print("Highest subthreshold current is %s" \
                      % (float(sub.max().round(2))*units))
            else:
                print("No subthreshold current was tested.")
            if len(supra):
                print("Lowest suprathreshold current is %s" \
                      % supra.min().round(2))
            else:
                print("No suprathreshold current was tested.")

        if len(sub) and len(supra):
            rheobase = supra.min()
        else:
            rheobase = None
        prediction['value'] = rheobase

        self.prediction = prediction
        return self.prediction


    def threshold_FI(self, model, units, guess=None):
        lookup = {} # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.params.copy()['injected_square_current']
                #This does not do what you would expect.
                #due to syntax I don't understand.
                #updating the dictionary keys with new values doesn't work.

                uc = {'amplitude':ampl}
                current.update(uc)
                model.inject_square_current(current)
                n_spikes = model.get_spike_count()
                if self.verbose:
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes
                spike_counts = np.array([n for x,n in lookup.items() if n>0])
                if n_spikes and n_spikes <= spike_counts.min():
                    self.rheobase_vm = model.get_membrane_potential()

        max_iters = 10

        #evaluate once with a current injection at 0pA
        high=self.high
        small=self.small

        f(high)
        i = 0

        while True:
            #sub means below threshold, or no spikes
            sub = np.array([x for x in lookup if lookup[x]==0])*units
            #supra means above threshold, but possibly too high above threshold.

            supra = np.array([x for x in lookup if lookup[x]>0])*units
            #The actual part of the Rheobase test that is
            #computation intensive and therefore
            #a target for parellelization.

            if i >= max_iters:
                break
            #Its this part that should be like an evaluate function that is passed to futures map.
            if len(sub) and len(supra):
                f((supra.min() + sub.max())/2)

            elif len(sub):
                f(max(small,sub.max()*2))

            elif len(supra):
                f(min(-small,supra.min()*2))
            i += 1

        return lookup

    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #	 (self.name,str(observation),str(prediction)))
        if prediction['value'] is None:
            score = scores.InsufficientDataScore(None)
        else:
            score = super(RheobaseTest,self).\
                        compute_score(observation, prediction)
            #self.bind_score(score,None,observation,prediction)
        return score

    def bind_score(self, score, model, observation, prediction):
        super(RheobaseTest,self).bind_score(score, model, 
                                            observation, prediction)
        if self.rheobase_vm is not None:
            score.related_data['vm'] = self.rheobase_vm

# class RheobaseTestHacked(VmTest):
#     """
#     A hacked version of test Rheobase.
#     Tests the full widths of APs at their half-maximum
#     under current injection.
#     """
#     def _extra(self):    
#         self.prediction=None
    
#     required_capabilities = (cap.ReceivesSquareCurrent,
#                              cap.ProducesSpikes)

#     params = {'injected_square_current':
#                 {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

#     name = "Rheobase test"

#     description = ("A test of the rheobase, i.e. the minimum injected current "
#                    "needed to evoke at least one spike.")

#     units = pq.pA
#     score_type = scores.RatioScore

#     def generate_prediction(self, model):
#         return self.prediction

#     def compute_score(self, observation, prediction):
#         """Implementation of sciunit.Test.score_prediction."""
#         #print("%s: Observation = %s, Prediction = %s" % \
#         #	 (self.name,str(observation),str(prediction)))

#         if self.prediction is not None:
#             if self.prediction['value'] is None:

#                 score = scores.InsufficientDataScore(None)
#             else:
#                 score = super(RheobaseTest,self).\
#                             compute_score(observation, self.prediction)
#                 #self.bind_score(score,None,observation,prediction)
#             return score


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

    def validate_observation(self, observation):
        try:
            assert type(observation['mean']) is Quantity
            assert type(observation['std']) is Quantity
        except Exception as e:
            raise sciunit.ObservationError(("Observation must be of the form "
                                    "{'mean':float*mV,'std':float*mV}"))

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""


        assert model!=None
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


    '''
    def compute_score(self, observation, prediction):

        """Implementation of sciunit.Test.score_prediction."""
        #print("%s: Observation = %s, Prediction = %s" % \
        #	 (self.name,str(observation),str(prediction))
        #else:
        score = super(RestingPotentialTest,self).\
                    compute_score(observation, prediction)
        return score
    '''
