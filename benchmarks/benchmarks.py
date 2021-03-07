# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from neuronunit.unit_test.opt_ephys_properties import testOptimizationEphysCase
from neuronunit.unit_test.scores_unit_test import testOptimizationAllenMultiSpike
from neuronunit.unit_test.rheobase_model_test import testModelRheobase


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def speed_check():
        testModelRheobase.setUp()
        testModelRheobase.test_opt_1()

class MemSuite:
    def mem_list(self):
        return [0] * 256
