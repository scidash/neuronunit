Introduction
============

This is a unittest for testing the somatic behavior and the signal integration in radial oblique dendrites of hippocampal CA1 pyramidal cell models (hoc models built using the NEURON software).

DepolarizationBlockTest:Tests if the model enters depolarization block under current injection of increasing amplitudes to the soma.

Features tested:
Ith: the threshold current to reach the depolarization block,  the amplitude of the current injection at which the cell exhibits the maximum number of spikes
Veq: the average equilibrium value during the depolarization block, average of the membrane potential over the last 100 ms of a current pulse 50 pA above Ith.
(Bianchi et al. (2012) J Comput Neurosci, 33: 207-225)

ObliqueIntegrationTest: Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs.

Features tested:
voltage threshold: the upper limit of the threshold for dendritic spike initiation (an expected somatic depolarization at wich step like dV/dt increase occur)
proximal threshold
distal threshold
degree of nonlinearity at threshold
suprathreshold degree of nonlinearity
peak derivative of (somatic voltage) at threshold
peak amplitude of somatic EPSP
time to peak of somatic EPSP
degree of nonlinearity in the case of asynchronous inputs
(A. Losonczy and J. C. Magee (2006) Neuron, 50: 291-307)

Score used: p value from T-test. If the p value < 0.05, the model mean differs from the experimental mean

SomaticFeaturesTest: Tests some somatic features using the BBP eFel



Requirements
============

Python
neuron
eFel
sciunit
scipy
NumPy
matplotlib
quantities
collections
multiprocessing
functools
math
json
cPickle
gzip
copy_reg
types


Quick Start
===========

To do the depolarization block test on the the Kali-Freund model for example, run depolarization_block_KaliFreund.py.
The figures are saved in a subdirectory; the path of it, and the obtained scores are printed.
The vectors of the somatic and dendritic voltage values are saved in files during the simulation. If force_run=False, and the files already exist, the test uses the saved files for further calculations and for creating the figures. If force_run=True, or the files don't exist, the test runs the whole simulation.

To test a new model, add the model's directory to the hippounit/models/hoc_models subdirectory. The variable time step have to be switched off in the model (cvode_active(0)).
Create a Class for your model in the hoc_models/__init__.py file:

Change these in one of the examples for DepolarizationBlockTest:

name of the class
name in def __init__()
modelpath
in set_cclamp(self, amp): change the name of the soma in self.stim = h.IClamp(h.soma(0.5)), if needed
in run_cclamp(self, amp): change the name of the soma in rec_v.record(h.soma(0.5)._ref_v), if needed

Change the name of the model in depolarization_block.py at model = models.KaliFreund()

Change these in one of the examples for ObliqueIntegrationTest:

name of the class
name in def __init__()
modelpath

self.dend_loc: This list contains the dendritic locations for synaptic stimulation.
self.dend_loc=[[number of the dendrite, proximal location on dendrite],[same dendrite, distal location],[other dendrite, proximal location],[other dendrite, distal location],...]  (the tested dendrites should be close to the soma (<120 um from the soma on the trunk), proximal locations: 5-50 um from the trunk, distal locations: 60-126 um from the trunk)

in set_ampa_nmda(self, dend_loc0) change the NMDA receptor model in self.nmda = h.NMDA_JS(xloc, sec=h.dendrite[ndend])

in def set_num_weight(self, number=1, AMPA_weight) you can change the AMPA/NMDA ratio at self.nmda_nc.weight[0] =AMPA_weight/0.2

in run_syn(self) change the name of the dendrite list at rec_v_dend.record(h.dendrite[self.ndend](self.xloc)._ref_v), if needed
in run_syn(self) change the name of the soma at rec_v.record(h.soma(0.5)._ref_v), if needed

The synaptic weights are set automatically.

Change the name of the model in somatic_features.py at model = models.KaliFreund()

Change these in one of the examples for SomaticFeaturesTest:

name of the class
name in def __init__()
self.soma in def __init__()
modelpath
Change the name of the model in oblique_integration.py at model = models.KaliFreund()



This unittest is based on:
============

NeuronUnit: A SciUnit repository for neuroscience-related tests, capabilities and so on.

![NeuronUnit Logo](https://raw.githubusercontent.com/scidash/assets/master/logos/neuronunit.png)

https://github.com/rgerkin/papers/blob/master/neuronunit_frontiers/Paper.pdf
