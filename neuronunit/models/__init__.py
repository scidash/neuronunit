"""Model classes for NeuronUnit"""

import warnings
from .static import StaticModel, ExternalModel, RandomVmModel
try:
    from .lems import LEMSModel
    from .channel import ChannelModel
except:
    print("neuroml not installed")
from .reduced import ReducedModel
from . import backends  # Required to register backends
