import papermill as pm
import sklearn
backends = [str("RAW"), str("HH")]

#samples = [75,100]
samples_sd = [75,100,125,150]
for b in backends:
    for s in samples_sd:
        pm.execute_notebook(
            'simulated_data_paramaterized_2.ipynb',
            'milled_nbooks_{0}_{1}.ipynb'.format(s,s),
            parameters = dict(backend=b,NGEN =s, MU =s)
        )
