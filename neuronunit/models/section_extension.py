
# These classes exist for compatibility with the old neuronunit.neuron module.
import sciunit
#
from neuronunit.models.backends import NEURONBackend
class HasSegment(sciunit.Capability):
    """Model has a membrane segment of NEURON simulator"""

    def setSegment(self, section, location = 0.5):
        """Sets the target NEURON segment object
        section: NEURON Section object
        location: 0.0-1.0 value that refers to the location
        along the section length. Defaults to 0.5
        """

        self.section = section
        self.location = location

    def getSegment(self):
        """Returns the segment at the active section location"""

        return self.section(self.location)

class SingleCellModel(NEURONBackend):
    def __init__(self, \
                 neuronVar, \
                 section, \
                 loc = 0.5, \
                 name=None):
        super(SingleCellModel,self).__init__()#name=name, hVar=hVar)
        hs = HasSegment()
        hs.setSegment(section, loc)
        self.reset_neuron(neuronVar)
        self.section = section
        self.loc = loc
        self.name = name
        #section = soma
        #super(SingleCellModel,self).reset_neuron(self, neuronVar)
        # Store voltage and time values
        self.tVector = self.h.Vector()
        self.vVector = self.h.Vector()
        self.vVector.record(hs.getSegment()._ref_v)
        self.tVector.record(self.h._ref_t)

        return
