
import os

from quantities import mV, nA
import sciunit
from hippounit import models
from hippounit import tests
from hippounit import capabilities

import matplotlib.pyplot as plt


target_Ith = 0.6*nA
Ith_SD = 0.3*nA
target_Veq = -40.1*mV
Veq_SD = 3.4*mV

observation = {'mean_Ith':target_Ith,'Ith_std':Ith_SD, 'mean_Veq': target_Veq, 'Veq_std': Veq_SD}

show_plot=False
test = tests.DepolarizationBlockTest(observation, force_run=False, show_plot=show_plot)

model = models.KaliFreund()
score = test.judge(model)
score.summarize()

if show_plot: plt.show()
