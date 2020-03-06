import papermill as pm
backends = [str("RAW"), str("HH")]

#samples = [75,100]
samples_sd = [50,75,100,125,150]
for b in backends:
    for s in samples_sd:
        pm.execute_notebook(
            'simulated_data_paramaterized_2.ipynb',
            'milled_nbooks_sim_data_MU_{0}_backend_{1}.ipynb'.format(s,b),
            parameters = dict(backend=b,NGEN =150, MU =s)
        )
