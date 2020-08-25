
# coding: utf-8

# In[1]:


from neuronunit.optimisation.optimization_management import inject_and_plot_model, dtc_to_rheo

import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC


# In[2]:


from neuronunit.optimisation import model_parameters
model_parameters.MODEL_PARAMS.keys()
backends = "RAW", "ADEXP", "HH", "BHH"
for b in backends:
    raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[b].items()}
    pre_model = DataTC()
    pre_model.attrs = raw_attrs
    pre_model.backend = b
    dtc = dtc_to_rheo(pre_model)
    print(dtc.rheobase)
    inject_and_plot_model(raw_attrs,b)


# In[ ]:


from neuronunit.optimisation import model_parameters
model_parameters.MODEL_PARAMS.keys()
backend = "RAW"
raw_attrs = {k:np.mean(v) for k,v in model_parameters.MODEL_PARAMS[backend].items()}


# In[ ]:


inject_and_plot_model(raw_attrs,backend)


# In[ ]:


pre_model.rheobase

