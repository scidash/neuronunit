import numpy as np


model_params={}

model_params['vr'] = np.linspace(-75.0,-50.0,9)


model_params['a'] = np.linspace(0.0,0.945,9)
model_params['b'] = np.linspace(-5*10E-10,-0.25*10E-9,9)
model_params['vpeak'] =np.linspace(30.0,40.0,9)


model_params['k'] = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,9)
model_params['C'] = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,9)
model_params['c'] = np.linspace(-55,-60,9)
model_params['d'] = np.linspace(0.050,0.2,9)
model_params['v0'] = np.linspace(-75.0,-45.0,9)
model_params['vt'] =  np.linspace(-50.0,-30.0,9)

'''
+model_params['vr'] = np.linspace(-95.0,-30.0,9)

+model_params['a'] = np.linspace(0.0,0.945,9)
+model_params['b'] = np.linspace(-3.5*10E-10,-0.5*10E-9,9)
+model_params['vpeak'] =np.linspace(0.0,80.0,9)

+model_params['k'] = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,9)
+model_params['C'] = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,9)
+model_params['c'] = np.linspace(-55,-60,9)
+model_params['d'] = np.linspace(0.050,0.2,9)
+model_params['v0'] = np.linspace(-85.0,-15.0,9)
+model_params['vt'] =  np.linspace(-70.0,0.0,9)
'''
