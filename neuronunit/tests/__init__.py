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
import neuronunit.capabilities.spike_functions as sf
from neuronunit import neuroelectro
from .channel import *
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
            x = np.array(mp).std()
            if x <= 0:
                return False

            for i in mp:
                if type(i)==np.float64:
                    if math.isnan(i):
                        return False
                    if (i == float('inf')) or (i == float('-inf')):
                        return False
                    if math.isnan(i):
                        return False

        #update run params is necessary to over write previous recording
        #vectors
        #Its also necessary to destroy and recreate the model in the HOC memory space
        #As models that persist in memory, retained model charge from current injections,
        #from past simulations.
        model.update_run_params(model.params)
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
        model.update_run_params(model.params)
        mp = np.array(copy.copy(model.results['vm']))
        boolean = nan_test(mp)
        if boolean == False:
            return False

        import neuronunit.capabilities as cap

        sws=cap.spike_functions.get_spike_waveforms(model.get_membrane_potential())
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
        #tau = tau /100000.0
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
            vm_fit[offset:,0] = a * np.exp(-t[offset:]/b) + c
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


class RheobaseTestOriginal(VmTest):
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

'''
def rheobase_checking(vmpop, rh_value=None):

    from itertools import repeat
    import pdb
    def bulk_process(vm,rh_value):
        #package arguments and call the parallel searcher
        if type(vm) is not type(None):
            rh_param = (False,rh_value)
            vm = searcher(rh_param,vm)
            return vm

    if type(vmpop) is not type(list):
        return bulk_process(vmpop,0)

    elif type(vmpop) is type(list):
        vmtemp = []
        if type(rh_value) is type(None):
            vmtemp = bulk_process(copy.copy(vmpop),0)
            #vmtemp = list(self.map(bulk_process,vmpop,repeat(0)))
        elif type(rh_value) is not type(None):
            vmtemp = bulk_process(vmpop,rh_value)
            #vmtemp = list(self.map(bulk_process,vmpop,rh_value))
        return vmtemp
'''

class VirtualModel:
    '''
    This is a pickable dummy clone
    version of the NEURON simulation model
    It does not contain an actual model, but it can be used to
    wrap the real model.
    This Object class serves as a data type for storing rheobase search
    attributes and other useful parameters,
    with the distinction that unlike the NEURON model this class
    can be transported across HOSTS/CPUs
    '''
    def __init__(self):
        self.lookup={}
        self.trans_dict=None
        self.rheobase=None
        self.previous=0
        self.run_number=0
        self.attrs=None
        self.steps=None
        self.name=None
        self.s_html=None
        self.results=None
        self.error=None
        self.td = None
        self.score = None


class RheobaseTest(VmTest):
     """
     A hacked version of test Rheobase.
     Tests the full widths of APs at their half-maximum
     under current injection.
     """
     def _extra(self):
         self.prediction=None

     required_capabilities = (cap.ReceivesSquareCurrent,
                              cap.ProducesSpikes)

     params = {'injected_square_current':
                 {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

     name = "Rheobase test"

     description = ("A test of the rheobase, i.e. the minimum injected current "
                    "needed to evoke at least one spike.")

     units = pq.pA
     score_type = scores.RatioScore

     def model2map(param_dict):#This method must be pickle-able for scoop to work.
         vm=VirtualModel()
         vm.attrs={}
         for k,v in param_dict.items():
             vm.attrs[k]=v
         return vm

     def check_fix_range(vms):
         '''
         Inputs: lookup, A dictionary of previous current injection values
         used to search rheobase
         Outputs: A boolean to indicate if the correct rheobase current was found
         and a dictionary containing the range of values used.
         If rheobase was actually found then rather returning a boolean and a dictionary,
         instead logical True, and the rheobase current is returned.
         given a dictionary of rheobase search values, use that
         dictionary as input for a subsequent search.
         '''
         import pdb
         sub=[]
         supra=[]
         steps=[]
         vms.rheobase=0.0
         for k,v in vms.lookup.items():
             if v==1:
                 #A logical flag is returned to indicate that rheobase was found.
                 vms.rheobase=float(k)
                 print(type(vms.rheobase))
                 vms.steps=0.0
                 return (True,vms)
             elif v==0:
                 sub.append(k)
             elif v>0:
                 supra.append(k)

         sub=np.array(sub)
         supra=np.array(supra)

         if len(sub)!=0 and len(supra)!=0:
             #this assertion would only be wrong if there was a bug
             print(str(bool(sub.max()>supra.min())))
             assert not sub.max()>supra.min()
         if len(sub) and len(supra):
             everything=np.concatenate((sub,supra))

             center = np.linspace(sub.max(),supra.min(),7.0)
             centerl = list(center)
             for i,j in enumerate(centerl):
                 if i in list(everything):
                     np.delete(center,i)
                     del centerl[i]
             #delete the index
             #np.delete(center,np.where(everything is in center))
             #make sure that element 4 in a seven element vector
             #is exactly half way between sub.max() and supra.min()
             center[int(len(center)/2)+1]=(sub.max()+supra.min())/2.0
             steps = [ i*pq.pA for i in center ]

         elif len(sub):
             steps2 = np.linspace(sub.max(),2*sub.max(),7.0)
             np.delete(steps2,np.array(sub))
             steps = [ i*pq.pA for i in steps2 ]

         elif len(supra):
             steps2 = np.linspace(-2*(supra.min()),supra.min(),7.0)
             np.delete(steps2,np.array(supra))
             steps = [ i*pq.pA for i in steps2 ]

         vms.steps=steps
         vms.rheobase=None
         return (False,vms)

     def check_current(ampl,vm):
         '''
         Inputs are an amplitude to test and a virtual model
         output is an virtual model with an updated dictionary.
         '''
         import copy
         import scoop
         #print('the scoop worker id: {0}'.format(scoop.utils.getWorkerQte(scoop.utils.getHosts())))


         if float(ampl) not in vm.lookup or len(vm.lookup)==0:

             current = params.copy()['injected_square_current']

             uc = {'amplitude':ampl}
             current.update(uc)
             current = {'injected_square_current':current}
             vm.run_number += 1
             model.update_run_params(vm.attrs)
             model.inject_square_current(current)
             vm.previous=ampl
             n_spikes = model.get_spike_count()
             vm.lookup[float(ampl)] = n_spikes
             if n_spikes == 1:
                 model.rheobase_memory=float(ampl)
                 vm.rheobase=float(ampl)
                 print(type(vm.rheobase))
                 print('current {0} spikes {1}'.format(vm.rheobase,n_spikes))
                 return vm

             return vm
         if float(ampl) in vm.lookup:
             return vm



     def searcher(rh_param,vms):
         '''
         inputs f a function to evaluate. rh_param a tuple with element 1 boolean, element 2 float or list
         and a  virtual model object.
         '''
         if rh_param[0]==True:
             return rh_param[1]
         lookuplist=[]
         cnt=0
         boolean=False
         model.update_run_params(vms.attrs)

         from itertools import repeat
         while boolean == False and cnt < 12:
             if len(model.params)==0:
                 assert len(vms.attrs)!=0
                 assert type(vms.attrs) is not type(None)
                 model.update_run_params(vms.attrs)
             if type(rh_param[1]) is float:
                 #if its a single value educated guess
                 if model.rheobase_memory == None:
                     model.rheobase_memory = rh_param[1]
                 vms = check_current(model.rheobase_memory,vms)
                 model.update_run_params(vms.attrs)

                 boolean,vms = check_fix_range(vms)
                 if boolean:
                     return vms
                 else:
                     #else search returned none type, effectively false
                     rh_param = (None,None)

             elif len(vms.lookup)==0 and type(rh_param[1]) is list:
                 #If the educated guess failed, or if the first attempt is parallel vector of samples
                 assert vms is not None
                 returned_list = list(map(check_current,rh_param[1],repeat(vms)))
                 for v in returned_list:
                     vms.lookup.update(v.lookup)
                 boolean,vms=check_fix_range(vms)
                 assert vms!=None
                 if boolean:
                     return vms

             else:
                 #Finally if a parallel vector of samples failed zoom into the
                 #smallest relevant interval and re-sample at a higher resolution
                 returned_list=[]
                 if type(vms.steps) is type(None):
                     steps = np.linspace(50,150,7.0)
                     steps_current = [ i*pq.pA for i in steps ]
                     vms.steps = steps_current
                     assert type(vms.steps) is not type(None)
                 returned_list = list(map(check_current,vms.steps,repeat(vms)))
                 for v in returned_list:
                     vms.lookup.update(v.lookup)
                 boolean,vms=check_fix_range(vms)
                 if boolean:
                     return vms
             cnt+=1
         return vms

     def generate_prediction(self, model):

         vms = searcher(rh_param,vms)
         self.prediction = vms.rheobase
         return self.prediction

     def compute_score(self, observation, prediction):
         """Implementation of sciunit.Test.score_prediction."""
         #print("%s: Observation = %s, Prediction = %s" % \
         #	 (self.name,str(observation),str(prediction)))

         if self.prediction is not None:
             if self.prediction['value'] is None:

                 score = scores.InsufficientDataScore(None)
             else:
                 score = super(RheobaseTest,self).\
                             compute_score(observation, self.prediction)
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
