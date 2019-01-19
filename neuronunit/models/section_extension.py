"""These classes are for compatibility w/ the old neuronunit.neuron module."""
import sciunit
from neuronunit.models.backends import NEURONBackend


class HasSegment(sciunit.Capability):
    """Model has a membrane segment of NEURON simulator."""

    def setSegment(self, section, location=0.5):
        """Set the target NEURON segment object.

        section: NEURON Section object
        location: 0.0-1.0 value that refers to the location
        along the section length. Defaults to 0.5
        """
        self.section = section
        self.location = location

    def getSegment(self):
        """Return the segment at the active section location."""
        return self.section(self.location)


class SingleCellModel(sciunit.Model):
    def __init__(self,
                 neuronVar,
                 section,
                 loc=0.5,
                 name=None):
        self._backend = NEURONBackend()
        super(SingleCellModel, self).__init__()
        hs = HasSegment()
        hs.setSegment(section, loc)
        self.reset_neuron(neuronVar)
        self.section = section
        self.loc = loc
        self.name = name
        self.tVector = self.h.Vector()
        self.vVector = self.h.Vector()
        self.vVector.record(hs.getSegment()._ref_v)
        self.tVector.record(self.h._ref_t)

        return
