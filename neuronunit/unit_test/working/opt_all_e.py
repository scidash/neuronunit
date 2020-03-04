#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import hide_imports
import copy
#plt.plot([0,1],[1,0])
#plt.show()

plt.clf()
import copy

def permutations(use_test,backend,MU=80,NGEN=100):
    use_test = hide_imports.TSD(use_test)
    use_test.use_rheobase_score = True
    edges = hide_imports.model_parameters.MODEL_PARAMS[backend]
    ga_out = use_test.optimize(edges,backend=backend,protocol={'allen': False, 'elephant': True}, MU=MU,NGEN=NGEN)
    dtc = ga_out['pf'][0].dtc
    vm,plt = hide_imports.inject_and_plot_model(dtc)
    plt.savefig(str(backend)+str(MU)+str(NGEN)+str('_.png'))
    return dtc, ga_out['DO'], vm

OMObjects = []
backends = ["RAW","HH"]#"ADEXP","BHH"]
test_frame = hide_imports.test_frame
#t = test_frame['Neocortex pyramidal cell layer 5-6']


#backends = ["RAW","HH","ADEXP","BHH"]
MU = NGEN = 85 

for t in test_frame.values():
    b = backends[0]
    (dtc,DO,vm) = permutations(copy.copy(t),b,MU,NGEN)
    display(dtc.SM)
    display(dtc.obs_preds)


for t in test_frame.values():
    b = backends[1]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()



for t in test_frame.values():
    #for b in backends:
    b = backends[2]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()



for t in test_frame.values():
    #for b in backends:
    b = backends[3]
    (dtc,DO,vm) = permutations(copy.copy(t),b)
    display(dtc.SM)
    display(dtc.obs_preds)
    plt.plot(vm.times,vm.magnitude)
    plt.show()




(dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],"ADEXP")
display(dtc.SM)
display(dtc.obs_preds)




backends = ["RAW","HH","BHH"]  



#for b in backends:
(dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],"RAW")


# In[ ]:


backends = iter(["RAW","HH","BHH"])


from IPython.display import HTML, display

for b in backends:
    (dtc,DO) = permutations(test_frame['Neocortex pyramidal cell layer 5-6'],b)
    display(dtc.SM)
    display(dtc.obs_preds)

