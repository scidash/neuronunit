import matplotlib as mpl
mpl.use('Agg')
# coding: utf-8



from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo
import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC
from neuronunit.optimisation import model_parameters
model_parameters.MODEL_PARAMS.keys()
backends =  ["RAW", "HH", "ADEXP", "BHH"]
for b in backends:
    attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[b].items()}
    pre_model = DataTC()
    pre_model.attrs = attrs
    pre_model.backend = b
    inject_and_plot_model(attrs,b)


from neuronunit.optimisation import model_parameters
model_parameters.MODEL_PARAMS.keys()
backend = "RAW"
raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[backend].items()}

inject_and_plot_model(raw_attrs,backend)


pre_model.rheobase
exit()
