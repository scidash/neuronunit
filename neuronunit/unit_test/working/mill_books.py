import papermill as pm

pm.execute_notebook(
   'paramaterized.ipynb',
   'milled_nbooks.ipynb',
   parameters = dict(model_type="RAW", test_type="Neocortex pyramidal cell layer 5-6")
)
