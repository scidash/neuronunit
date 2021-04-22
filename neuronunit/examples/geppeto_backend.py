import quantities as pq
from neuronunit.tests.passive import InputResistanceTest
from neuronunit.models.reduced import ReducedModel
test = InputResistanceTest(observation={'mean': 200.0*pq.MOhm, 
                                        'std': 50.0*pq.MOhm})
model_url = ("https://raw.githubusercontent.com/scidash/neuronunit"
             "/dev/neuronunit/models/NeuroML2/LEMS_2007One.xml")
model = ReducedModel(model_url, backend='Geppetto')
test.setup_protocol(model)
print(model.lems_file_path)