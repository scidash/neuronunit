from sciunit.models import RunnableModel
from sciunit.models.backends import Backend
from sciunit.scores import RelativeDifferenceScore
from neuronunit.tests import RestingPotentialTest
from neuronunit.capabilities import ProducesMembranePotential, ReceivesCurrent
from neo.core import AnalogSignal
import numpy as np
import quantities as pq
# To add to neuronunit later
class StaticBackend(Backend):
    def _backend_run(self):
        pass
    def set_stop_time(self, t_stop):
        pass
# To add to neurounit later
class RandomVmModel(RunnableModel, ProducesMembranePotential, ReceivesCurrent):
    def get_membrane_potential(self):
        # Random membrane potential signal
        vm = (np.random.randn(10000)-60)*pq.mV
        vm = AnalogSignal(vm, sampling_period=0.1*pq.ms)
        return vm
    def inject_square_current(self, current):
        pass
# Random production of a Vm, using a static backend
model = RandomVmModel(name="Random", backend=StaticBackend)
# RestingPotentialTest for some neuron
test_class = RestingPotentialTest
observation = test_class.neuroelectro_summary_observation({'nlex_id': 'nifext_50'})
test = test_class(observation=observation)
# Change to RelativeDifferenceScore
test.score_type = RelativeDifferenceScore
# Judge the model and return the RelativeDifferenceScore
score = test.judge(model)
print(dir(score))
print(score.raw)
