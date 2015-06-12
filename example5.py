# Standard library.  
import os

# Installed packages.  
import sciunit
from neuronunit import neuroelectro,tests,capabilities
import neuronunit.neuroconstruct.models as nc_models
from pythonnC.utils.putils import OSB_MODELS

nlex_ids = {}
brain_area = 'cerebellum'
neuron_type = 'cerebellar_granule_cell'
path = os.path.join(OSB_MODELS,brain_area,neuron_type)
neurolex_id = 'nifext_128' # Cerebellar Granule Cell
# Specify reference data for this test.  
reference_data = neuroelectro.NeuroElectroSummary(
    neuron = {'nlex_id':neurolex_id}, # Neuron type.  
    ephysprop = {'name':'Resting Membrane Potential'}) # Electrophysiological property name. 
# Get and verify summary data for the combination above from neuroelectro.org. 
if reference_data.get_values() is None:
    raise ValueError("Unable to get the reference data from NeuroElectro.org.")
    
test = tests.RestingPotentialTest(
    observation = {'mean':reference_data.mean,
                      'std':reference_data.std})
suite = sciunit.TestSuite('Resting Potential',test)

if not os.path.isdir(path):
    raise IOError('No such path: %s' % path)
model_names = os.listdir(path)
models = []
for model_name in model_names:
    print(model_name)
    if model_name in ['GranCellRothmanIf']:
        continue
    model_info = (brain_area,neuron_type,model_name)
    model = nc_models.OSBModel(*model_info)
    models.append(model)

# (1) Check capabilities,
# (2) take the test, 
# (3) generate a score and validate it,
# (4) bind the score to model/test combination. 

score_matrix = suite.judge(models,stop_on_error=True)    

# Summarize the result.  
print("=========\r"*3)
#score.summarize()
print("=========\r"*3)

# Get model output used for this test (optional).
#vm = score.related_data

