from neuronunit.optimisation import get_neab
from neuronunit import tests as _, neuroelectro
from neuronunit.tests import fi, passive, waveform
import pickle
from neuronunit.tests import fi, passive, waveform
def get_neuron_criteria(cell_id,file_name = None):#,observation = None):
    # Use neuroelectro experimental obsevations to find test
    # criterion that will be used to inform scientific unit testing.
    # some times observations are not sourced from neuroelectro,
    # but they are imputated or borrowed from other TestSuite
    # if that happens make test objections using observations external
    # to this method, and provided as a method argument.
    tests = {}
    observations = None
    test_classes = None
    test_classes = [fi.RheobaseTest,
                     passive.InputResistanceTest,
                     passive.TimeConstantTest,
                     passive.CapacitanceTest,
                     passive.RestingPotentialTest,
                     waveform.InjectedCurrentAPWidthTest,
                     waveform.InjectedCurrentAPAmplitudeTest,
                     waveform.InjectedCurrentAPThresholdTest]#,
    observations = {}
    for index, t in enumerate(test_classes):
        try:
            obs = t.neuroelectro_summary_observation(cell_id)

            if obs is not None:
                if 'mean' in obs.keys():
                    tests[t.name] = t(obs)
                    observations[t.ephysprop_name] = obs
        except:
            pass
    #hooks = {tests[0]:{'f':update_amplitude}} #This is a trick to dynamically insert the method
    #update amplitude at the location in sciunit thats its passed to, without any loss of generality.
    #suite = sciunit.TestSuite(tests,name="vm_suite")

    if file_name is not None:
        file_name = file_name +'.p'
        with open(file_name, 'wb') as f:
            pickle.dump(tests, f)

    return tests,observations

def get_all_cells():
	purkinje ={"id": 18, "name": "Cerebellum Purkinje cell", "neuron_db_id": 271, "nlex_id": "sao471801888"}
	#fi_basket = {"id": 65, "name": "Dentate gyrus basket cell", "neuron_db_id": None, "nlex_id": "nlx_cell_100201"}
	pvis_cortex = {"id": 111, "name": "Neocortex pyramidal cell layer 5-6", "neuron_db_id": 265, "nlex_id": "nifext_50"}
	#This olfactory mitral cell does not have datum about rheobase, current injection values.
	olf_mitral = {"id": 129, "name": "Olfactory bulb (main) mitral cell", "neuron_db_id": 267, "nlex_id": "nlx_anat_100201"}
	ca1_pyr = {"id": 85, "name": "Hippocampus CA1 pyramidal cell", "neuron_db_id": 258, "nlex_id": "sao830368389"}
	cell_list = [ olf_mitral, ca1_pyr, purkinje,  pvis_cortex]
	cell_constraints = {}
	for cell in cell_list:

	    tests,observations = get_neuron_criteria(cell)
	    cell_constraints[cell["name"]] = tests



	with open('russell_tests.p','wb') as f:
	    pickle.dump(cell_constraints,f)


                
