"""Base classes and attributes for many neuronunit tests.  No classes here meant
for direct use in testing"""

from types import MethodType

import quantities as pq
import numpy as np

import sciunit
import sciunit.scores as scores
import neuronunit.capabilities as cap
from neuronunit import neuroelectro


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
                    assert type(observation[key]) is pq.quantity.Quantity
            for key in nonunited_keys:
                if key in observation:
                    assert type(observation[key]) is not pq.quantity.Quantity \
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
        #model.inject_square_current(self.params['injected_square_current'])
        self.params['injected_square_current']['delay'] = DELAY
        self.params['injected_square_current']['duration'] = DURATION
        self.params['injected_square_current']['amplitude'] = rheobase
        model.inject_square_current(self.params['injected_square_current'])

        mp = model.results['vm']
        import math
        for i in mp:
            if math.isnan(i):
                return False

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

    @property
    def state(self):
        state = super(VmTest,self).state
        return self._state(state=state, exclude=['unpicklable','verbose'])