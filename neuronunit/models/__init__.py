"""Model classes for NeuronUnit"""

import warnings
from .static import StaticModel, ExternalModel
from .lems import LEMSModel
from .channel import ChannelModel
from .reduced import ReducedModel
from . import backends  # Required to register backends
