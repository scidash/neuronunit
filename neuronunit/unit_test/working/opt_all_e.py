#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import hide_imports
import copy
#plt.plot([0,1],[1,0])
#plt.show()
mpl.use('agg')
import copy

def permutations(use_test,backend,MU=80,NGEN=100):
    use_test = hide_imports.TSD(use_test)
    use_test.use_rheobase_score = True
    edges = hide_imports.model_parameters.MODEL_PARAMS[backend]
    ga_out = use_test.optimize(edges,backend=backend,protocol={'allen': False, 'elephant': True}, MU=MU,NGEN=NGEN)
    dtc = ga_out['pf'][0].dtc
    vm,plt = hide_imports.inject_and_plot_model(dtc)
    plt.savefig(str(backend)+str(MU)+str(NGEN)+"_for_example.png")
    return dtc, ga_out['DO'], vm

test_frame = hide_imports.test_frame
test_frame.pop('Olfactory bulb (main) mitral cell',None)
OMObjects = []
backends = ["RAW","HH"]#"ADEXP","BHH"]
test_frame = hide_imports.test_frame
#t = test_frame['Neocortex pyramidal cell layer 5-6']


MU = NGEN = 85 
backends = ["RAW","HH"]#,"ADEXP","BHH"]
for t in test_frame.values():
    b = backends[0]
    (dtc,DO,vm) = permutations(copy.copy(t),b,MU,NGEN)


for t in test_frame.values():
    b = backends[1]
    (dtc,DO,vm) = permutations(copy.copy(t),b)


