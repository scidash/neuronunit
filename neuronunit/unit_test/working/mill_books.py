import papermill as pm
import sklearn
import pickle
tests = pickle.load(open("processed_multicellular_constraints.p","rb"))
test_types = ['Neocortex pyramidal cell layer 5-6', \
    'Hippocampus CA1 pyramidal cell' \
    ,'Cerebellum Purkinje cell'
    ]
models = ["RAW", "HH"]
samples = [25,50,75,100]
for test_type in test_types:
    for model in models:
        for s in samples:
            pm.execute_notebook(
           'paramaterized.ipynb',
           'milled_nbooks_{0}_{1}.ipynb'.format(model,test_type),
            parameters = dict(model_type=str(model), test_type=test_type, NGEN =s, MU =s)
       )
