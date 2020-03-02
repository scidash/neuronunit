import papermill as pm
import sklearn
backends = ["RAW", "HH"]

#samples = [75,100]
samples_sd = [75,100,125,150]
for s in samples_sd:
    pm.execute_notebook(
        'simulated_data_paramaterized.ipynb',
        'milled_nbooks_{0}_{1}.ipynb'.format(s,s),
        parameters = dict(backends=backends,NGEN =s, MU =s)
    )
