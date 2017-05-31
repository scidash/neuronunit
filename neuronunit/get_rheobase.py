import numpy as np
import quantities as pq
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.extract_cell_features import get_square_stim_characteristics,\
                                                 get_sweep_from_nwb
from allensdk.core import nwb_data_set
import quantities as pq
#from neuronunit import tests as nu_tests, neuroelectro
neuron = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell
tests = []
import pdb
dataset_id = 354190013
ct = CellTypesApi()
print(dir(ct))
print(dir(ct.get_ephys_features()))


cmd = ct.get_ephys_features(dataset_id)
experiment_params = ct.get_ephys_sweeps(dataset_id)
sweep_ids=cmd['rheobase_sweep_id'] #Retrieva all of the sweeps corresponding to finding rheobase.

found_exp=[]
for sp in experiment_params:
   for i in sweep_ids:
      if sp['id']==i:
          found_exp.append(i) 
          print('sweepid',i)
          #break
print('sweepid',found_exp)
 
#ct.save_ephys_data(dataset_id,'local_cache')




#observation = aibs.get_observation(dataset_id,'rheobase')

