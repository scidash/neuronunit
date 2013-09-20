import os

import sciunit
from neurounit import neuroelectro,tests,capabilities
from neurounit.neuroconstruct import models

"""
# A fake model:  
class DRGCell(models.FakeNeuroConstructModel):
	def __init__(self,**kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
	def set_current(self,current):
		self.i = current
	def get_membrane_potential(self):
		return self.vm
"""

# A real model.  
class CA1PyramidalCell(models.NeuroConstructModel,
					   capabilities.ReceivesCurrent):
	"""CA1 Pyramidal Cell model from /neuroConstruct/osb/hippocampus/
	CA1_pyramidal_neuron/CA1PyramidalCell"""
	def __init__(self,**kwargs):
		# Put any other initialization here.
		project_path = os.path.join(models.putils.OSB_MODELS,
									"hippocampus",
									"CA1_pyramidal_neuron",
									"CA1PyramidalCell",
									"neuroConstruct")
		models.NeuroConstructModel.__init__(self,project_path,**kwargs)
		#super(CA1PyramidalCell,self).__init__(**kwargs)
	def set_current(self,current):
		# This isn't actually used here, but it could be used in the 
		# appropriate model to inject current into the cell.  
		self.i = current
	def get_firing_rate(self):
		# This isn't actually used here, but it could be used in the 
		# appropriate model to inject current into the cell.  
		return len(self.spikes)

# A real model.  
class PurkinjeCell(models.NeuroConstructModel,
				   capabilities.ReceivesCurrent):
	"""Purkinje Cell model from /neuroConstruct/osb/cerebellum/
	cerebellar_purkinje_cell/PurkinjeCell"""
	def __init__(self,**kwargs):
		# Put any other initialization here.
		project_path = os.path.join(models.putils.OSB_MODELS,
									"cerebellum",
									"cerebellar_purkinje_cell",
									"PurkinjeCell",
									"neuroConstruct")
		models.NeuroConstructModel.__init__(self,project_path,**kwargs)
		#super(CA1PyramidalCell,self).__init__(**kwargs)
	def set_current(self,current):
		# This isn't actually used here, but it could be used in the 
		# appropriate model to inject current into the cell.  
		self.i = current
	
# Get reference data from neuroelectro.org.  
reference_data = neuroelectro.NeuroElectroSummary(
	neuron = {'id':72}, # DRG neuron (18: Purkinje cell).   
	ephysprop = {'id':23}) # Spike width. 

# Get and verify summary data for the combination above. 
reference_data.get_values()  

# Initialize (parameterize) the model with some initialization parameters.
model = PurkinjeCell(population_name="OrigChansCellGroup_0")   

#model = CA1PyramidalCell(population_name="CG_CML_0")   
#model = DRGCell(diameter=17.0,branches=10)   

# Initialize the test with summary statistics from the reference data
# and arguments for the model (model).    
from sciunit.comparators import ZComparator
test = tests.SpikeWidthTestDynamic(
	reference_data = {'mean':reference_data.mean,
					  'std':reference_data.std},
	model_args = {'current':40.0},
	comparator = ZComparator)

# (1) Check capabilities,
# (2) take the test, 
# (3) generate a score and validate it,
# (4) bind the score to model/test combination. 
result = sciunit.judge(test,model)

# Summarize the result.  
result.summarize()

