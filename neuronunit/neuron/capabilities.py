import sciunit

class HasSegment(sciunit.Capability):
    """Model has a membrane segment of NEURON simulator"""
    
    def setSegment(self, section, location = 0.5):
        """Sets the target NEURON segment object"""
        """section: NEURON Section object"""
        """location: 0.0-1.0 value that refers to the location along the section length. Defaults to 0.5"""
            
        self.section = section
        self.location = location

    def getSegment(self):
        """Returns the segment at the active section location"""
        
        return self.section(self.location)