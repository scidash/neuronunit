"""Base classes and attributes for many neuronunit tests.

No classes here meant for direct use in testing.
"""

from types import MethodType

import quantities as pq
import numpy as np

import sciunit
import sciunit.scores as scores
from sciunit.errors import ObservationError
import neuronunit.capabilities as cap
from neuronunit import neuroelectro


AMPL = 0.0*pq.pA
DELAY = 100.0*pq.ms
DURATION = 300.0*pq.ms


class VmTest(sciunit.Test):
    """Base class for tests involving the membrane potential of a model."""

    def __init__(self,
                 observation={'mean': None, 'std': None},
                 name=None,
                 **params):
        super(VmTest, self).__init__(observation, name, **params)
        cap = []
        for cls in self.__class__.__bases__:
            cap += cls.required_capabilities
        self.required_capabilities += tuple(cap)
        self._extra()

    required_capabilities = (cap.Runnable, cap.ProducesMembranePotential,)

    name = ''

    units = pq.Dimensionless

    ephysprop_name = ''

    observation_schema = [{'mean': {'units': True, 'required': True},
                           'std': {'units': True, 'min': 0, 'required': True},
                           'n': {'type': 'integer', 'min': 1}},
                          {'mean': {'units': True, 'required': True},
                           'sem': {'units': True, 'min': 0, 'required': True},
                           'n': {'type': 'integer', 'min': 1,
                                 'required': True}}]

    def _extra(self):
        pass

    def validate_observation(self, observation):
        super(VmTest, self).validate_observation(observation)
        # Catch another case that is trickier
        if 'std' not in observation:
            observation['std'] = observation['sem'] * np.sqrt(observation['n'])
        return observation

    def bind_score(self, score, model, observation, prediction):
        score.related_data['vm'] = model.get_membrane_potential()
        score.related_data['model_name'] = '%s_%s' % (model.name, self.name)

        def plot_vm(self, ax=None, ylim=(None, None)):
            """A plot method the score can use for convenience."""
            import matplotlib.pyplot as plt
            if ax is None:
                ax = plt.gca()
            vm = score.related_data['vm'].rescale('mV')
            ax.plot(vm.times, vm)
            y_min = float(vm.min()-5.0*pq.mV) if ylim[0] is None else ylim[0]
            y_max = float(vm.max()+5.0*pq.mV) if ylim[1] is None else ylim[1]
            ax.set_xlim(vm.times.min(), vm.times.max())
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Vm (mV)')
        score.plot_vm = MethodType(plot_vm, score)  # Bind to the score.
        score.unpicklable.append('plot_vm')

    @classmethod
    def neuroelectro_summary_observation(cls, neuron, cached=False):
        reference_data = neuroelectro.NeuroElectroSummary(
            neuron=neuron,  # Neuron type lookup using the NeuroLex ID.
            ephysprop={'name': cls.ephysprop_name},  # Ephys property name in
                                                     # NeuroElectro ontology.
            cached=cached
            )
        # Get and verify summary data from neuroelectro.org.
        reference_data.get_values(quiet=not cls.verbose)  #

        observation = {'mean': reference_data.mean*cls.units,
                       'std': reference_data.std*cls.units,
                       'n': reference_data.n}
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

    def sanity_check(self, rheobase, model):
        self.params['injected_square_current']['delay'] = DELAY
        self.params['injected_square_current']['duration'] = DURATION
        self.params['injected_square_current']['amplitude'] = rheobase
        model.inject_square_current(self.params['injected_square_current'])

        mp = model.results['vm']
        import math
        for i in mp:
            if math.isnan(i):
                return False

        sws = cap.spike_functions.get_spike_waveforms(
                                        model.get_membrane_potential())

        for i, s in enumerate(sws):
            s = np.array(s)
            dvdt = np.diff(s)
            import math
            for j in dvdt:
                if math.isnan(j):
                    return False
        return True

    @property
    def state(self):
        state = super(VmTest, self).state
        return self._state(state=state, exclude=['unpicklable', 'verbose'])


class FakeTest(sciunit.Test):
    """Fake test class.
    Just computes agreement between an observation key and a model attribute.
    e.g.
    observation = {'a':[0.8,0.3], 'b':[0.5,0.1], 'vr':[-70*pq.mV,5*pq.mV]}
    fake_test_a = FakeTest("test_a",observation=observation)
    fake_test_b = FakeTest("test_b",observation=observation)
    fake_test_vr = FakeTest("test_vr",observation=observation)
    """

    def generate_prediction(self, model):
        self.key_param = self.name.split('_')[1]
        return model.attrs[self.key_param]

    def compute_score(self, observation, prediction):
        mean = observation[self.key_param][0]
        std = observation[self.key_param][1]
        z = (prediction - mean)/std
        return scores.ZScore(z)
