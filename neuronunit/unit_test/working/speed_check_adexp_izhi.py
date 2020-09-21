from neuronunit.optimisation.data_transport_container import DataTC
#dtc = DataTC(backend="ADEXP")
from neuronunit.optimisation.optimization_management import dtc_to_rheo
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
import numpy as np
from neuronunit.optimisation.optimization_management import timer
params={k:np.mean(v) for k,v in MODEL_PARAMS["ADEXP"].items()}
#print(params)
dtc = DataTC(backend="ADEXP",attrs=params)
#print(dtc)

print("ADEXP")

@timer
def rh(dtc):
    dtc = dtc_to_rheo(dtc)
    print(dtc.rheobase)
    return dtc
# dry run
dtc = rh(dtc)
print('\n\n')
dtc = rh(dtc)

print("IZHI")


params={k:np.mean(v) for k,v in MODEL_PARAMS["IZHI"].items()}
#print(params)
dtc = DataTC(backend="IZHI",attrs=params)
#print(dtc)
@timer
def rh(dtc):
    dtc = dtc_to_rheo(dtc)
    print(dtc.rheobase)
    return dtc
# dry run
dtc = rh(dtc)
print('\n\n')
dtc = rh(dtc)