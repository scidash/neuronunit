import contextlib, io, warnings
warnings.filterwarnings('ignore')
s = io.StringIO()

import papermill as pm
#import sklearn
#backends = [str("RAW"), str("HH")]
backends = str("RAW")#, str("HH")]

#samples = [75,100]
samples_sd = [5,10,12]
#for b in backends:
b = str("RAW")
with contextlib.redirect_stdout(s):
   for s in samples_sd:
       pm.execute_notebook(
       'simulated_data_paramaterized_3.ipynb',
       'milled_nbooks_{0}_{1}.ipynb'.format(s,s),
       parameters = dict(backend=b,NGEN =10, MU =s)
    )
