from numpy.random import randn
from sciunit import check_capabilities,judge,TestResult
from neurounit.neuroelectro import NeuroElectroSummary
from neurounit.tests import SpikeWidthTestDynamic

#from neuroConstruct.models import DRGCell # Doesn't actually exist, so...
#from neuroConstruct.models import NeuroConstructModel # Doesn't actually implement run(), so...
# Just make an ad hoc fake model:  
from neurounit.neuroconstruct import FakeNeuroConstructModel
class DRGCell(FakeNeuroConstructModel):
	def __init__(self,**kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
	def set_current(self,current):
		self.i = current
	def get_membrane_potential(self):
		return self.vm
	
# Get reference data from neuroelectro.org.  
reference_data = NeuroElectroSummary(neuron={'id':72}, # DRG neuron.   
									 ephysprop={'id':23}) # Spike width. 

# Get and verify summary data for the combination above. 
reference_data.get_values()  

# Initialize (parameterize) the model with some initialization parameters.
candidate = DRGCell(diameter=17.0,branches=10)   

# Initialize the test with summary statistics from the reference data
# and arguments for the candidate (model).    
test = SpikeWidthTestDynamic(reference_data = {'mean':reference_data.mean,
											   'std':reference_data.std},
							 candidate_args = {'current':40.0})

# (1) Check capabilities,
# (2) take the test, 
# (3) generate a score and validate it,
# (4) bind the score to candidate/test combination. 
result = judge(test,candidate)

# Summarize the result.  
result.summarize()

