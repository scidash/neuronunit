import os

import sciunit
from neuronunit import neuroelectro,tests,capabilities
from neuronunit.neuroconstruct import models

# Specify reference data for this test.  
reference_data = neuroelectro.NeuroElectroSummary(
	neuron = {'name':'Cerebellum Purkinje Cell'}, # Neuron type.  
	ephysprop = {'name':'Resting Membrane Potential'}) # Electrophysiological property name.  

# Get and verify summary data for the combination above from neuroelectro.org. 
reference_data.get_values()  

# Initialize the test with summary statistics from the reference data
# and arguments for the model (model).    
from sciunit.comparators import ZComparator
test = tests.RestingPotentialTest(
	reference_data = {'mean':reference_data.mean,
					  'std':reference_data.std},
	model_args = {})

# Initialize (parameterize) the model with some initialization parameters.
model = models.OSBModel(
	"cerebellum", # Brain area.  
	"cerebellar_purkinje_cell", # Neuron type.  
	"PurkinjeCell", # Model name.
	)

# (1) Check capabilities,
# (2) take the test, 
# (3) generate a score and validate it,
# (4) bind the score to model/test combination. 
score = test.judge(model)

# Summarize the result.  
score.summarize()

# Get model output used for this test (optional).
vm = score.related_data['vm']

