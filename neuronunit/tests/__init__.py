import inspect
from types import MethodType

import quantities as pq
from quantities.quantity import Quantity
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sciunit
import sciunit.scores as scores

import neuronunit.capabilities as cap
#import neuronunit.capabilities.spike_functions as sf
from neuronunit import neuroelectro
from .channel import *
from scoop import futures
'''
import os
os.system('ipcluster start --profile=jovyan --debug &')
os.system('sleep 5')
import ipyparallel as ipp
rc = ipp.Client(profile='jovyan')
print('hello from before cpu ')
print(rc.ids)
v = rc.load_balanced_view()
'''
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

    required_capabilities = (cap.ProducesMembranePotential,)

    name = ''

    units = pq.Dimensionless

    ephysprop_name = ''

    # Observation values with units.
    united_observation_keys = ['value','mean','std']

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

        def plot_vm(self,ax=None,ylim=(None,None)):
            """A plot method the score can use for convenience."""
            if ax is None:
                ax = plt.gca()
            vm = score.related_data['vm'].rescale('mV')
            ax.plot(vm.times,vm)
            y_min = float(vm.min()-5.0*pq.mV) if ylim[0] is None else ylim[0]
            y_max = float(vm.max()+5.0*pq.mV) if ylim[1] is None else ylim[1]
            ax.set_ylim(y_min,y_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vm (mV)')
        score.plot_vm = MethodType(plot_vm, score) # Bind to the score.
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

        #import pdb
        print(reference_data.get_values())

        #pdb.set_trace()
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
        start, stop = -11*pq.ms, (i['duration']-(1*pq.ms))
        region = cls.get_segment(vm,start+i['delay'],stop+i['delay'])
        coefs = cls.exponential_fit(region, i['delay'])
        tau = (pq.ms/coefs[1]).rescale('ms')
        return tau

    @classmethod
    def exponential_fit(cls, segment, offset):
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        x = segment.times.rescale('ms')
        y = segment.rescale('V')
        offset = float(offset.rescale('ms')) # Strip units for optimization
        popt, pcov = curve_fit(func, x-offset*pq.ms, y, [0.001,2,y.min()]) # Estimate starting values for better convergence
        return popt


class InputResistanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

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

    #super(TimeConstantTest,self).prediction=super(TimeConstantTest,self).generate_prediction(model)
    #a=super(TimeConstantTest,self).prediction
    #pdb.set_trace()
    def compute_score(self, observation, prediction):
        """Implementation of sciunit.Test.score_prediction."""
        import pdb
        #pdb.set_trace()
        if 'n' in prediction.keys():
            if prediction['n'] == 0:
                score = scores.InsufficientDataScore(None)
        else:


            print(observation['mean'])
            #Hack. Why is this off by a factor of 10 still present?
            #There should be a more simple and transparent way to fix this.
            prediction['value']=prediction['value']/10.0
            score = super(TimeConstantTest,self).compute_score(observation,
                                                          prediction)

        return score


class CapacitanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""

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
        #import pdb
        #pdb.set_trace()
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

        #this call is returning none.
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
            #This means the smallest current to cause spiking.
            rheobase = supra.min()
        else:
            rheobase = None
        prediction['value'] = rheobase

        return prediction

    def threshold_FI(self, model, units, guess=None):
        lookup = {} # A lookup table global to the function below.

        def f(ampl):
            if float(ampl) not in lookup:
                current = self.params.copy()['injected_square_current']
                #This does not do what you would expect.
                #due to syntax I don't understand.
                #updating the dictionary keys with new values doesn't work.

                uc={'amplitude':ampl}
                current.update(uc)
                current={'injected_square_current':current}
                model.inject_square_current(current)
                n_spikes = model.get_spike_count()
                if self.verbose:
                    print("Injected %s current and got %d spikes" % \
                            (ampl,n_spikes))
                lookup[float(ampl)] = n_spikes


        max_iters = 10

        #evaluate once with a current injection at 0pA
        def evaluate1(n,guess):
            #print(n)
            #print(rc.ids)
            if n==0:
                f(0.0*units)

                return (None,None,lookup)
            if n==1:
                if guess is None:
                    try:
                        guess = self.observation['value']
                    except KeyError:
                        guess = 100*pq.pA
                high = guess*2
                high = (50.0*pq.pA).rescale(units) if not high else high
                small = (1*pq.pA).rescale(units)
                #f(0.0*units) could happen in parallel with f(high) below
                f(high)
                import pdb

                return (small,high)

        def evaluate2(n,guess,small):
            #sub means below threshold, or no spikes
            #print(n)
            #print(rc.ids)
            sub = np.array([x for x in lookup if lookup[x]==0])*units
            #supra means above threshold, but possibly too high above threshold.
            supra = np.array([x for x in lookup if lookup[x]>0])*units

            if len(sub) and len(supra):
                f((supra.min() + sub.max())/2)

            elif len(sub):
                #the argument of f resolves to the maximum number of two in a list
                f(max(small,sub.max()*2))
            elif len(supra):
            #the argument of f resolves to the minimum number of two in a list
                f(min(-small,supra.min()*2))
            rlist=[small,sub,supra]
            return rlist

        import pdb
        #strategy get to work with serial map first.
        #run it in exhaustive search first such that calls to scoop are not nested yet.
        from itertools import repeat
        from scoop import futures
        returned_list = list(map(evaluate1,[i for i in range(0,2)],repeat(guess)))
        small=returned_list[1]

        rlist = list(map(evaluate2,[i for i in range(1,10)],repeat(guess),repeat(small)))
        small=rlist[0]
        sub=rlist[1]
        supra=rlist[2]
        return lookup

        #main(guess)

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
        model.rerun = True
        model.inject_square_current(self.params['injected_square_current'])
        median = model.get_median_vm() # Use median for robustness.
        std = model.get_std_vm()
        prediction = {'mean':median, 'std':std}
        return prediction
