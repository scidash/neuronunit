"""Model classes for NeuronUnit"""

import warnings
from .static import StaticModel, ExternalModel, RandomVmModel
try:
    from .lems import LEMSModel
except:
    print("neuroml not installed")
from .channel import ChannelModel
from .reduced import ReducedModel
from . import backends  # Required to register backends
