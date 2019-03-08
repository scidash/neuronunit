"""NeuronUnit model class for NEURON HOC defined cell models"""

import os
import sciunit
import neuronunit.capabilities.morphology as cap
import quantities as pq

class SwcCellModel(sciunit.Model, cap.ProducesSWC):
    """A model for cells defined using SWC files. Requires a path to the SWC file."""

    def __init__(self, swc_path, name=None):
        """
        hoc_path: Path to SWC file.

        name: Optional model name.
        """

        self.swc_path = os.path.abspath(swc_path)

        if name is None:
            name = os.path.basename(self.swc_path).replace('.swc','')

        super(SwcCellModel,self).__init__(name=name)

    def produce_swc(self):
        return os.path.abspath(self.swc_path)
