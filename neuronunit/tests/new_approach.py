from __future__ import print_function
import get_neab

import pickle
import bluepyopt as bpop
import matplotlib.pyplot as plt
import numpy as np
import gbevaluator
#import stdputil
import get_neab


import array
import random
import json

import numpy as np

from math import sqrt
import time
#cp_filename = 'checkpoints/checkpoint.pkl'

evaluator = gbevaluator.BBEvaluator()

'''
import os
os.system('ipcluster start --profile=jovyan --debug &')
os.system('sleep 5')
import ipyparallel as ipp
rc = ipp.Client(profile='jovyan')
print('hello from before cpu ')
print(rc.ids)
#quit()
v = rc.load_balanced_view()
'''
map_function=map
opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=10,
                                          eta=10, mutpb=0.3, cxpb=0.7, map_function=map_function
                                          )





#NDIM = 4

#param=['vr','a','b']
#rov=[]
#parameter.lower_bound=

#rov0 = np.linspace(-65,-55,2)
#rov1 = np.linspace(0.015,0.045,2)
#rov2 = np.linspace(-0.0010,-0.0035,2)
'''
rov0 = np.linspace(-65,-55,1000)
rov1 = np.linspace(0.015,0.045,1000)
rov2 = np.linspace(-0.0010,-0.0035,1000)
rov.append(rov0)
rov.append(rov1)
rov.append(rov2)
seed_in=1
BOUND_LOW=[ np.min(i) for i in rov ]
BOUND_UP=[ np.max(i) for i in rov ]
NDIM = len(rov)
LOCAL_RESULTS=[]
'''


def run_model():
    """Run model"""
    cp_filename = 'dummy_file.pkl'
    _, _, _, _ = opt.run(
        max_ngen=200, cp_filename=cp_filename, cp_frequency=100)



def main():
    """Main"""
    run_model()

if __name__ == '__main__':
    main()
