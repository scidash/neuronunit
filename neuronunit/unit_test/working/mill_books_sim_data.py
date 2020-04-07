import contextlib, io, warnings
warnings.filterwarnings('ignore')
s = io.StringIO()

import papermill as pm
backends = [str("RAW"), str("HH")]

pop_size = [20,40,60,80,100]
with contextlib.redirect_stdout(s):
   for b in backends:
      for s in pop_size:
         pm.execute_notebook(
         'simulated_data_paramaterized_3.ipynb',
         'milled_nbooks_{0}_{1}_{2}.ipynb'.format(s,s,b),
         parameters = dict(backend=b,NGEN =100, MU =s)
      )
