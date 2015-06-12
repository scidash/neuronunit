NEUROML2_PATH = "/Users/rgerkin/NeuroML2" # Path to NeuroML2 compliant OSB models.  

# Standard library.  
import sys
import os
import glob

QUIET = False
for arg in sys.argv[1:]:
	try:
		key,value = arg.split(":")
	except ValueError as e:
		print("Command line argument %s had error %s" % (arg,e.strerror))
	else:
		if key == "quiet":
			if int(value):            
				QUIET = 1
			else:
				QUIET = 0

# Installed packages.  
import neuroml.loaders as loaders
import osb
import sciunit
from neuronunit import neuroelectro,tests,capabilities
from neuronunit.neuroconstruct import models

def qprint(string):
	if not QUIET:
		print(string)

ephys_property = "Resting Membrane Potential"  
nml2_model_list = os.listdir(NEUROML2_PATH)

# Get projects with medium curation level. 
osb_projects = osb.get_projects('Medium')  

for model_name in nml2_model_list:
	print(model_name)
	path = os.path.join(NEUROML2_PATH,model_name)
	nml_files = glob.glob(path+"/*.nml")
	if not len(nml_files):
		qprint("No .nml files found for model %s" % model_name)
	for nml_file in nml_files:
		nml = loaders.NeuroMLLoader.load(nml_file)
		for cell in nml.cells:
			nlex_id = cell.neuro_lex_id
			if nlex_id is None:
				project = osb.get_project_with_identifier(model_name,osb_projects)
				if project:
					nlex_id_list = project.NEUROLEX_IDS_CELLS
					if len(nlex_id_list):
						if ';' not in nlex_id_list:
							nlex_id = nlex_id_list
						else:
							qprint("Multiple neurolex ids found; skipping...")
							continue
					else:
						qprint("No neurolex id found; skipping...")
						continue
			qprint("Model %s had neurolex id %s" % (model_name,nlex_id))

			# Specify reference data for this test.  
			reference_data = neuroelectro.NeuroElectroSummary(
				neuron = {'nlex_id':nlex_id}, 
				# Neuron type.  
				ephysprop = {'name':ephys_property}
				# Electrophysiological property name.
				)   

				# Get and verify summary data for the combination above 
				# from neuroelectro.org. 
			if not reference_data.get_values():
				string = "No NeuroElectro API data found for the neuron with"
				string += "neurolex id %s and ephys property %s" % (nlex_id,ephys_property)
				qprint(string)
				continue

			# Initialize the test with summary statistics from the 
			# reference data and arguments for the model (model).    
			test = tests.RestingPotentialTest(
					observation = {'mean':reference_data.mean,
								   'std':reference_data.std})

			# Initialize (parameterize) the model with some initialization parameters.
			model = models.NeuroML2Model(model_name)

			# (1) Check capabilities,
			# (2) take the test, 
			# (3) generate a score and validate it,
			# (4) bind the score to model/test combination. 
			score = test.judge(model)

			# Summarize the result.  
			score.summarize()

			# Get model output used for this test (optional).
			vm = score.related_data

