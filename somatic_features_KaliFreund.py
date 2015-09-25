import os

from quantities import mV, nA
import sciunit
from hippounit import models
from hippounit import tests
from hippounit import capabilities

import matplotlib.pyplot as plt
import json
from hippounit import plottools
import collections


with open('./stimfeat/PC_newfeat_No14112401_15012303-m990803_stimfeat.json') as f:
    config = json.load(f, object_pairs_hook=collections.OrderedDict)

observation = config['features']

show_plot=True
test = tests.SomaticFeaturesTest(observation, force_run=False, show_plot=show_plot)

model = models.KaliFreund()

score = test.judge(model)

score.summarize()

if show_plot: plt.show()
