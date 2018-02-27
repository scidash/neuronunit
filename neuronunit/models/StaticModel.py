
import sciunit
import pickle
from neo.core import AnalogSignal
class StaticModel(sciunit.Model,
    cap.HasMembranePotential):
    def __init__(self,vm):
        if instance(vm,str):
            with open(vm,'r'mus) as f:
                vm = pickle.load(f)
        if not isinstance(vm,AnalogSignal):
            raise TypeError('vm ')

        self.vm = vm
    def get_membrane_potential():
        return self.vm
