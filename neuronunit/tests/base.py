"""Base classes and attributes for many neuronunit tests.

No classes here meant for direct use in testing.
"""

from types import MethodType

import numpy as np
import quantities as pq

import sciunit
from sciunit.tests import ProtocolToFeaturesTest
import sciunit.scores as scores
import neuronunit.capabilities as ncap
import sciunit.capabilities as scap
from neuronunit import neuroelectro


class VmTest(ProtocolToFeaturesTest):
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

    required_capabilities = (scap.Runnable, ncap.ProducesMembranePotential,)

    name = ''

    units = pq.Dimensionless

    ephysprop_name = ''

    observation_schema = [("Mean, Standard Deviation, N",
                           {'mean': {'units': True, 'required': True},
                            'std': {'units': True, 'min': 0, 'required': True},
                            'n': {'type': 'integer', 'min': 1}}),
                          ("Mean, Standard Error, N",
                           {'mean': {'units': True, 'required': True},
                            'sem': {'units': True, 'min': 0, 'required': True},
                            'n': {'type': 'integer', 'min': 1,
                                  'required': True}})]

    default_params = {'amplitude': 0.0*pq.pA,
                      'delay': 100.0*pq.ms,
                      'duration': 300.0*pq.ms,
                      'dt': 0.025*pq.ms,
                      'padding': 200*pq.ms}

    params_schema = {'dt': {'type': 'time', 'min': 0, 'required': False},
                     'tmax': {'type': 'time', 'min': 0, 'required': False},
                     'delay': {'type': 'time', 'min': 0, 'required': False},
                     'duration': {'type': 'time', 'min': 0, 'required': False},
                     'amplitude': {'type': 'current', 'required': False},
                     'padding': {'type': 'time', 'min': 0, 'required': False}}

    def _extra(self):
        pass

    def compute_params(self):
        self.params['tmax'] = (self.params['delay'] +
                               self.params['duration'] +
                               self.params['padding'])

    def validate_observation(self, observation):
        super(VmTest, self).validate_observation(observation)
        # Catch another case that is trickier
        if 'std' not in observation:
            observation['std'] = observation['sem'] * np.sqrt(observation['n'])
        return observation

    def condition_model(self, model):
        model.set_run_params(t_stop=self.params['tmax'])

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

    def sanity_check(self, rheobase, model):
        self.params['injected_square_current']['delay'] = self.params['delay']
        self.params['injected_square_current']['duration'] = \
            self.params['duration']
        self.params['injected_square_current']['amplitude'] = rheobase
        model.inject_square_current(self.params['injected_square_current'])

        mp = model.results['vm']
        if np.any(np.isnan(mp)) or np.any(np.isinf(mp)):
            return False

        sws = ncap.spike_functions.get_spike_waveforms(
                                    model.get_membrane_potential())

        for i, s in enumerate(sws):
            s = np.array(s)
            dvdt = np.diff(s)
            for j in dvdt:
                if np.isnan(j):
                    return False
        return True

    @classmethod
    def get_default_injected_square_current(cls):
        current = {key: cls.default_params[key]
                   for key in ['duration', 'delay', 'amplitude']}
        return current

    def get_injected_square_current(self):
        current = {key: self.default_params[key]
                   for key in ['duration', 'delay', 'amplitude']}
        return current

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
