import contextlib, io, warnings
warnings.filterwarnings('ignore')
s = io.StringIO()

import papermill as pm
#import sklearn
import pickle
tests = pickle.load(open("processed_multicellular_constraints.p","rb"))
test_types = ['Neocortex pyramidal cell layer 5-6', \
    'Hippocampus CA1 pyramidal cell' \
    ,'Cerebellum Purkinje cell'
    ]
models = [str("RAW"), str("HH")]

samples_sd = [10,20,30,100]

with contextlib.redirect_stdout(s):
    for test_type in test_types:
        for model in models:
            for MU in samples_sd:
                pm.execute_notebook(
                    'paramaterized.ipynb',
                    'milled_nbooks_{0}_{1}.ipynb'.format(model,test_type),
                    parameters = dict(model_type=str(model), test_type=test_type, NGEN =105, MU =MU)
            )
