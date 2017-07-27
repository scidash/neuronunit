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

#import get_tau_module

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
    #@classmethod
    def nan_inf_test(mp):
        '''
        Check if a HOC recording vector of membrane potential contains nans or infinities.
        Also check if it does not perturb due to stimulating current injection
        '''
        import math
        mp = np.array(mp)
        for i in mp:
            if math.isnan(i) or i==float('inf') or i==float('-inf'):
                return False
        return True

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

    #def nan_inf_test(self,params):




class TestPulseTest(VmTest):
    """
    A base class for tests that use a square test pulse
    Needs elaboration because DELAY and DURATION are inappropriately standard
    as compared to other tests, however they should not be.

    """

    required_capabilities = (cap.ReceivesSquareCurrent,)

    name = ''

    score_type = scores.ZScore



    params = {'injected_square_current':
                {'amplitude':-10.0*pq.pA, 'delay':30*pq.ms, 'duration':100*pq.ms}}

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
            '''
            This function is simply the shape of exponential decay which must be differenced, its basically an ideal template
            An exp decay equation derived from experiments.
            For the model to compare against.
            '''
            vm_fit[:offset] = c
            vm_fit[offset:,0] = a * np.exp(-t[offset:]/b) + c
            return vm_fit.squeeze()

        #popt, pcov = curve_fit(func, t, vm, p0=guesses) # Estimate starting values for better convergence
        popt, pcov = curve_fit(func, t, vm.squeeze(), p0=guesses) # Estimate starting values for better convergence
          #plt.plot(t,vm)
        amplitude = popt[0]*pq.mV
        tau = popt[1]*pq.ms
        y0 = popt[2]*pq.mV
        return amplitude,tau,y0


class InputResistanceTest(TestPulseTest):
    """Tests the input resistance of a cell."""
    def _extra(self):
        self.prediction = None

    name = "Input resistance test"

    description = ("A test of the input resistance of a cell.")

    units = pq.ohm*1e6

    ephysprop_name = 'Input Resistance'

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(InputResistanceTest,self).\
                            generate_prediction(model)
        i['duration'] = 100 * pq.ms

        r_in = self.__class__.get_rin(vm, i)
        r_in = r_in.simplified
        # Put prediction in a form that compute_score() can use.
        self.prediction = {'value':r_in}
        return self.prediction




class TimeConstantTest(TestPulseTest):
    """Tests the input resistance of a cell."""

    name = "Time constant test"

    description = ("A test of membrane time constant of a cell.")

    units = pq.ms

    ephysprop_name = 'Membrane Time Constant'
    def _extra(self):
        self.prediction = None

    def __init__(self):
        DURATION = 100*pq.ms
        amplitude = -10*pq.pA
        DELAY = 30*pq.ms
        self.params['injected_square_current']['delay'] = DELAY
        self.params['injected_square_current']['duration'] = DURATION
        self.params['injected_square_current']['amplitude'] = amplitude

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        i,vm = super(TimeConstantTest,self).generate_prediction(model)
        tau = self.__class__.get_tau(vm, i)
        tau = tau.simplified
        # Put prediction in a form that compute_score() can use.
        self.prediction = {'value':tau}

        return self.prediction

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
    def _extra(self):
        self.prediction = None

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
        self.prediction = {'value':c}
        return self.prediction

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
    def _extra(self):
        self.prediction = None
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
        #import copy
        #tvec = copy.copy(model.results['t'])
        #dt = (tvec[1]-tvec[0])*pq.ms
        widths = model.get_AP_widths()
        # Put prediction in a form that compute_score() can use.
        self.prediction = {'mean':np.mean(widths) if len(widths) else None,
                      'std':np.std(widths) if len(widths) else None,
                      'n':len(widths)}
        return self.prediction

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
    def _extra(self):
        self.prediction = None
    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Injected current AP width test"

    description = ("A test of the widths of action potentials "
                   "at half of their maximum height when current "
                   "is injected into cell.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        self.prediction = super(InjectedCurrentAPWidthTest,self).generate_prediction(model)
        return self.prediction


class APAmplitudeTest(VmTest):
    """Tests the heights (peak amplitude) of action potentials."""

    required_capabilities = (cap.ProducesActionPotentials,)

    name = "AP amplitude test"

    description = ("A test of the amplitude (peak minus threshold) of "
                   "action potentials.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Amplitude'
    def _extra(self):
        self.prediction = None
    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""
        # Method implementation guaranteed by
        # ProducesActionPotentials capability.
        model.rerun = True
        heights = model.get_AP_amplitudes() - model.get_AP_thresholds()
        # Put prediction in a form that compute_score() can use.
        self.prediction = {'mean':np.mean(heights) if len(heights) else None,
                      'std':np.std(heights) if len(heights) else None,
                      'n':len(heights)}
        return self.prediction

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
    def _extra(self):
        self.prediction = None

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        prediction = super(InjectedCurrentAPAmplitudeTest,self).\
                generate_prediction(model)
        self.prediction = prediction
        return prediction

class APThresholdTest(VmTest):
    """Tests the full widths of action potentials at their half-maximum."""

    required_capabilities = (cap.ProducesActionPotentials,)

    name = "AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced.")

    score_type = scores.ZScore

    units = pq.mV

    ephysprop_name = 'Spike Threshold'
    def _extra(self):
        self.prediction = None

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
        self.prediction = prediction
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
    def _extra(self):
        self.prediction = None

    required_capabilities = (cap.ReceivesSquareCurrent,)

    params = {'injected_square_current':
                {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Injected current AP threshold test"

    description = ("A test of the membrane potential threshold at which "
                   "action potentials are produced under current injection.")

    def generate_prediction(self, model):
        model.inject_square_current(self.params['injected_square_current'])
        prediction = super(InjectedCurrentAPThresholdTest,self).\
                generate_prediction(model)
        self.prediction = prediction
        return prediction



class RheobaseTestNew(VmTest):
    def check_rheobase(virtual_model):
        '''
        inputs a population of genes/alleles, the population size MU, and an optional argument of a rheobase value guess
        outputs a population of genes/alleles, a population of individual object shells, ie a pickleable container for gene attributes.
        Rationale, not every gene value will result in a model for which rheobase is found, in which case that gene is discarded, however to
        compensate for losses in gene population size, more gene samples must be tested for a successful return from a rheobase search.
        If the tests return are successful these new sampled individuals are appended to the population, and then their attributes are mapped onto
        corresponding virtual model objects.
        '''
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
            import copy
            import numpy as np
            import quantities as pq
            sub=[]
            supra=[]
            steps=[]
            vms.rheobase=0.0
            for k,v in vms.lookup.items():
                if v==1:
                    #A logical flag is returned to indicate that rheobase was found.
                    vms.rheobase=float(k)
                    vms.steps = 0.0
                    vms.boolean = True
                    return vms
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
                everything = np.concatenate((sub,supra))

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

            vms.steps = steps
            vms.rheobase = None
            return copy.copy(vms)


        def check_current(ampl,vm):
            '''
            Inputs are an amplitude to test and a virtual model
            output is an virtual model with an updated dictionary.
            '''

            global model
            import quantities as pq
            import get_neab

            from neuronunit.models import backends
            from neuronunit.models.reduced import ReducedModel
            #ar = rc[:].apply_async(os.getpid)
            #pids = ar.get_dict()
            #rc[:]['pid_map'] = pids
            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            #os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
            model = ReducedModel(new_file_path,name=str(vm.attrs),backend='NEURON')
            model.load_model()
            model.update_run_params(vm.attrs)

            DELAY = 100.0*pq.ms
            DURATION = 1000.0*pq.ms
            params = {'injected_square_current':
                      {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}


            if float(ampl) not in vm.lookup or len(vm.lookup)==0:

                current = params.copy()['injected_square_current']

                uc = {'amplitude':ampl}
                current.update(uc)
                current = {'injected_square_current':current}
                vm.run_number += 1
                model.update_run_params(vm.attrs)
                model.inject_square_current(current)
                vm.previous = ampl
                n_spikes = model.get_spike_count()
                vm.lookup[float(ampl)] = n_spikes
                if n_spikes == 1:
                    vm.rheobase = float(ampl)
                    print('current {0} spikes {1}'.format(vm.rheobase,n_spikes))
                    vm.name = str('rheobase {0} parameters {1}'.format(str(current),str(model.params)))
                    return vm

                return vm
            if float(ampl) in vm.lookup:
                return vm

        from itertools import repeat
        import numpy as np
        import copy
        import pdb
        import get_neab

        def final_check(vms, pop):
            '''
            Not none can be drawn from.
            '''
            not_none = [ pop[k] for k,vm in enumerate(vms) if type(vm.rheobase) is not type(None) ]
            for k,vm in enumerate(vms):
                j = 0
                ind = pop[k]
                while type(vm.rheobase) is type(None):
                    for key in range(0,len(pop[0])):
                        ind[key] = np.mean([i[key] for i in pop])
                    vm = None
                    vm = update_vm_pop(ind)
                    vm = init_vm(vm)
                    vm = find_rheobase(vm)
                    print('trying value {0}'.format(vm.rheobase))
                    if type(vm.rheobase) is not type(None):
                        print('rheobase value is updating {0}'.format(vm.rheobase))
                        break
                assert type(vm.rheobase) is not type(None)
            return (vms, pop)

        def init_vm(vm):
            import quantities as pq
            import numpy as np
            vm.boolean = False
            steps = list(np.linspace(-50,200,7.0))
            steps_current = [ i*pq.pA for i in steps ]
            vm.steps = steps_current
            return vm

        def find_rheobase(vm):
            from neuronunit.models import backends
            from neuronunit.models.reduced import ReducedModel
            import get_neab
            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            #os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
            model = ReducedModel(new_file_path,name=str(vm.attrs),backend='NEURON')
            #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str(vm.attrs),backend='NEURON')
            model.load_model()
            model.update_run_params(vm.attrs)
            cnt = 0
            while vm.boolean == False:# and cnt <21:
                for step in vm.steps:
                    vm = check_current(step, vm)#,repeat(vms))
                    vm = check_fix_range(vm)
                    cnt+=1
            return vm
        def generate_prediction(self, model):
            virtual_model = utilities.VirtualModel()
            virtual_model.attrs = model.params
            virtual_model.name = vm.attrs
            virtualmodel = init_vm(virtual_model)
            virtualmodel = find_rheobase(virtual_model)
            print('rheobase method{0}'.format(virtual_model.rheobase))
            return virtual_model.rheobase

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

    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
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


     DELAY = 100.0*pq.ms
     DURATION = 1000.0*pq.ms
     params = {'injected_square_current':
                 {'amplitude':100.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

     name = "Rheobase test"

     description = ("A test of the rheobase, i.e. the minimum injected current "
                    "needed to evoke at least one spike.")

     units = pq.pA
     score_type = scores.RatioScore


     def generate_prediction(self, model):

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

    DELAY = 100.0*pq.ms
    DURATION = 1000.0*pq.ms
    params = {'injected_square_current':
                {'amplitude':0.0*pq.pA, 'delay':DELAY, 'duration':DURATION}}

    name = "Resting potential test"

    description = ("A test of the resting potential of a cell "
                   "where injected current is set to zero.")

    score_type = scores.ZScore


    units = pq.mV

    ephysprop_name = 'Resting membrane potential'
    def _extra(self):
        self.prediction = None


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
