import matplotlib.pyplot as plt
#plt.plot([1,0],[1,0])
#plt.show()
from neuronunit.optimisation.data_transport_container import DataTC
#dtc = DataTC(backend="ADEXP")
from neuronunit.optimisation.optimization_management import dtc_to_rheo, inject_and_plot_model
from neuronunit.optimisation.model_parameters import MODEL_PARAMS, type2007
import numpy as np
from neuronunit.optimisation.optimization_management import timer

#@timer
def rh(dtc):
    dtc = dtc_to_rheo(dtc)
    print(dtc.rheobase)
    return dtc

params={k:np.mean(v) for k,v in MODEL_PARAMS["IZHI"].items()}

#print(params)
dtc = DataTC(backend="IZHI",attrs=params)
#print(dtc)

print("IZHI dry run")
dtc = rh(dtc)
print("IZHI actual run")

print('\n\n')
dtc = rh(dtc)

# dry run



# dry run
print("Adexp dry run")

dtc = rh(dtc)
print('\n\n')
print("Adexp actual run")

dtc = rh(dtc)
print("ADEXP follows")
params={k:np.mean(v) for k,v in MODEL_PARAMS["ADEXP"].items()}
#print(params)
dtc = DataTC(backend="ADEXP",attrs=params)
#print(dtc)

print("ADEXP")

'''

params={k:np.mean(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}
#params['celltype'] = 1
#params = {k:v for k,v in zip((params.keys()),type2007['RS'])}

dtc = DataTC(backend="NEURONHH",attrs=params)

args = inject_and_plot_model(dtc,plotly=False)
#print(args)
args[1].savefig('isthisright_hh.png')


params={k:np.mean(v) for k,v in MODEL_PARAMS["IZHI"].items()}
params['celltype'] = 1
params = {k:v for k,v in zip((params.keys()),type2007['RS'])}

@timer
def rh(dtc):
    dtc = dtc_to_rheo(dtc)
    print(dtc.rheobase)
    return dtc
# dry run
dtc = rh(dtc)
print('\n\n')
dtc = rh(dtc)
print("GLIF")
'''

#print(dtc)
'''
@timer
def rh(dtc):
    dtc = dtc_to_rheo(dtc)
    print(dtc.rheobase)
    return dtc
# dry run
dtc = rh(dtc)
print('\n\n')
dtc = rh(dtc)
'''
'''
params0={k:np.max(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}
params1={k:np.min(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}
paramsm={k:np.mean(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}

print(params0)

dtc = DataTC(backend="NEURONHH",attrs=params0)
dtc.attrs = params0
model0 = dtc.dtc_to_model()
model0.attrs = params0
args = inject_and_plot_model(dtc,plotly=False)
#print(args)
args[1].savefig('isthisrightNEURONHH.png')

dtc = DataTC(backend="NEURONHH",attrs=params1)
dtc.attrs = params1

params0={k:np.max(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}
params1={k:np.min(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}
paramsm={k:np.mean(v) for k,v in MODEL_PARAMS["NEURONHH"].items()}

print(params0)

dtc = DataTC(backend="NEURONHH",attrs=params0)
dtc.attrs = params0
model0 = dtc.dtc_to_model()
model0.attrs = params0
args = inject_and_plot_model(dtc,plotly=False)
#print(args)
args[1].savefig('isthisrightHH.png')

dtc = DataTC(backend="NEURONHH",attrs=params1)
dtc.attrs = params1
'''

#model1 = dtc.dtc_to_model()
#model1.attrs = params1

#args = inject_and_plot_model(dtc,plotly=False)
#print(args)
#args[1].savefig('isthisright1.png')
'''
dtc = DataTC(backend="GLIF",attrs=paramsm)

dtc.attrs = paramsm
dtc.attrs['spike_cut_length'] =0.001

#modelm.attrs = paramsm

args = inject_and_plot_model(dtc,plotly=False)
#print(args)
args[1].savefig('isthisrightm.png')



import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms

params = {}
params['injected_square_current'] = {}
params['injected_square_current']['amplitude'] = -10*pq.pA
params['injected_square_current']['delay'] = PASSIVE_DELAY
params['injected_square_current']['duration'] = PASSIVE_DURATION
model = dtc.dtc_to_model()
model.attrs['spike_cut_length'] =0.1
model._backend.inject_square_current(params)

vm = model0._backend.inject_square_current(params)
plt.clf()
plt.plot(vm.times,vm)
plt.savefig("glif_passive.png")
'''
#model1._backend.inject_square_current(params)
#model._backend.check_defaults(params)

#print(paramsm,params0,params1)
#print(MODEL_PARAMS["GLIF"])
