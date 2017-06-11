import numpy as np
import quantities as pq


model_params={}
#4*4*4*4==4^4==256

model_params['vr'] = np.linspace(-75.0,-50.0,3)
#model_params['a'] = np.linspace(0.015,0.045,10)
#model_params['a'] = (0.015 + 0.045) / 2.0

model_params['a'] = np.linspace(0.0,0.945,10)
model_params['b'] = np.linspace(-3.5*10E-10,-0.5*10E-9,3)
model_params['vpeak'] =np.linspace(30.0,40.0,2)

model_params['k'] = np.linspace(7.0E-4-+7.0E-5,7.0E-4+70E-5,10)
model_params['C'] = np.linspace(1.00000005E-4-1.00000005E-5,1.00000005E-4+1.00000005E-5,10)
model_params['c'] = np.linspace(-55,-60,10)
model_params['d'] = np.linspace(0.050,0.2,10)
model_params['v0'] = np.linspace(-75.0,-45.0,10)
model_params['vt'] =  np.linspace(-50.0,-30.0,10)

steps2 = np.linspace(50,190,4.0)
steps = [ i*pq.pA for i in steps2 ]

guess_attrs=[]
#guess_attrs.append(model_params['a'])
guess_attrs.append(np.mean( [ i for i in model_params['a'] ]))
guess_attrs.append(np.mean( [ i for i in model_params['b'] ]))
guess_attrs.append(np.mean( [ i for i in model_params['vr'] ]))
guess_attrs.append(np.mean( [ i for i in model_params['vpeak'] ]))
