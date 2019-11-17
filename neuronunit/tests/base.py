"""Base classes and attributes for many neuronunit tests.

No classes here meant for direct use in testing.
"""

from types import MethodType

import quantities as pq
import numpy as np
import sciunit.scores as scores
from sciunit.errors import ObservationError
import neuronunit.capabilities as ncap
import neuronunit.capabilities as cap

from neuronunit import neuroelectro
import pickle
'''
wave_dict = pickle.load(open('waves.p','rb'))
keys = list(wave_dict.keys())
injected_current = {}
injected_square_current = wave_dict[keys[0]]
injected_square_current['duration'] = wave_dict[keys[0]]['durations']


ls1 = pickle.load(open('models/backends/generic_current_injection.p','rb'))
ls = ls1[0]['stimulus']
DT = sampling_period = 1.0/ls1[0]['sampling_rate']#*pq.s
on_indexs = np.where(ls==np.max(ls))

ALLEN_STIM = ls
ALLEN_ONSET = start = np.min(on_indexs)*DT
ALLEN_STOP = stop = np.max(on_indexs)*DT
ALLEN_FINISH = len(ls)*DT
'''
AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms#wave_dict[keys[0]]['delay']*pq.s#100.0*pq.ms
DURATION = 1000.0*pq.ms#wave_dict[keys[0]]['delay']*pq.s#1000.0*pq.ms

#import pdb; pdb.set_trace()

PASSIVE_AMPL = -10.0*pq.pA
PASSIVE_DELAY = 100.0*pq.ms
PASSIVE_DURATION = 300.0*pq.ms


import sciunit
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

    # Observation values without units.
    nonunited_observation_keys = []

    def _extra(self):
        pass

    def validate_observation(self, observation,
                             united_keys=None,
                             nonunited_keys=None):
        if united_keys is None:
            united_keys = self.united_observation_keys
        if nonunited_keys is None:
            nonunited_keys = self.nonunited_observation_keys
        try:
            assert type(observation) is dict
            assert any([key in observation for key in united_keys]) \
                or len(nonunited_keys)
            for key in united_keys:
                if key in observation:
                    assert type(observation[key]) is pq.quantity.Quantity
            for key in nonunited_keys:
                if key in observation:
                    assert type(observation[key]) is not pq.quantity.Quantity \
                        or observation[key].units == pq.Dimensionless
        except Exception as e:
            key_str = 'and/or a '.join(['%s key' % key for key in united_keys])
            msg = ("Observation must be a dictionary with a %s and each key "
                   "must have units from the quantities package." % key_str)
            raise ObservationError(msg)
        for key in united_keys:
            if key in observation:
                provided = observation[key].simplified.units
                if not isinstance(self.units,pq.Dimensionless):
                    required = self.units.simplified.units
                else:
                    required = self.units
                if provided != required: # Units don't match spec.
                    msg = ("Units of %s are required for %s but units of %s "
                           "were provided" % (required.dimensionality.__str__(),
                                              key,
                                              provided.dimensionality.__str__())
                           )
                    raise ObservationError(msg)
        if 'std' not in observation:
            if all([x in observation for x in ['sem','n']]):
                observation['std'] = observation['sem'] * np.sqrt(observation['n'])
            elif 'mean' in observation:
                raise ObservationError(("Observation must have an 'std' key "
                                                "or both 'sem' and 'n' keys."))
        return observation

    def bind_score(self, score, model, observation, prediction):
        model.inject_square_current(self.params['injected_square_current'])
        score.related_data['vm'] = model.get_membrane_potential()
        score.related_data['model_name'] = '%s_%s' % (model.name,self.name)

        def plot_vm(self, ax=None, ylim=(None,None)):
            """A plot method the score can use for convenience."""
            import matplotlib.pyplot as plt
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
    def neuroelectro_summary_observation(cls, neuron, cached=False):
        reference_data = neuroelectro.NeuroElectroSummary(
            neuron = neuron, # Neuron type lookup using the NeuroLex ID.
            ephysprop = {'name': cls.ephysprop_name}, # Ephys property name in
                                                      # NeuroElectro ontology.
            cached = cached
            )
        reference_data.get_values(quiet=not cls.verbose) # Get and verify summary data
                                    # from neuroelectro.org.
        #print(reference_data)
        #import pdb; pdb.set_trace()
        if hasattr(reference_data,'mean'):
            observation = {'mean': reference_data.mean*cls.units,
                           'std': reference_data.std*cls.units,
                           'n': reference_data.n}
        else:
            observation = None

        return observation

    @classmethod
    def neuroelectro_pooled_observation(cls, neuron, cached=False, quiet=True):
        reference_data = neuroelectro.NeuroElectroPooledSummary(
            neuron = neuron, # Neuron type lookup using the NeuroLex ID.
            ephysprop = {'name': cls.ephysprop_name}, # Ephys property name in
                                                      # NeuroElectro ontology.
            cached = cached
            )
        #print(reference_data)
        #import pdb; pdb.set_trace()
        reference_data.get_values(quiet=quiet) # Get and verify summary data
                                    # from neuroelectro.org.
        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
        return observation

    def sanity_check(self,rheobase,model):
        #model.inject_square_current(self.params['injected_square_current'])
        self.params['injected_square_current']['delay'] = DELAY
        self.params['injected_square_current']['duration'] = DURATION
        self.params['injected_square_current']['amplitude'] = rheobase
        model.inject_square_current(self.params['injected_square_current'])

        vm  = model.results['vm']

        if np.any(np.isnan(vm)) or np.any(np.isinf(vm)):
            return False

        sws = cap.spike_functions.get_spike_waveforms(model.get_membrane_potential())

        for i,s in enumerate(sws):
            s = np.array(s)
            dvdt = np.diff(s)
            for j in dvdt:
                if np.isnan(j):
                    return False
        return True

    @property
    def state(self):
        state = super(VmTest,self).state
        return self._state(state=state, exclude=['unpicklable','verbose'])
