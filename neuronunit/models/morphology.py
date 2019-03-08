"""NeuronUnit model class for NEURON HOC defined cell models"""

import os
import sciunit
import neuronunit.capabilities.morphology as cap
import quantities as pq

from hoc2swc import hoc2swc

class HocCellModel(sciunit.Model, cap.ProducesSWC):
    """A model for cells defined using NEURON HOC files. Requires just a path to the HOC file."""

    hoc_swc_cache = {}

    def __init__(self, hoc_path, name=None):
        """
        hoc_path: Path to HOC file.

        name: Optional model name.
        """

        self.hoc_path = os.path.abspath(hoc_path)
        self.swc_path = self.hoc_path + ".swc"

        if name is None:
            name = os.path.basename(self.hoc_path).replace('.hoc','')

        super(HocCellModel,self).__init__(name=name)

    def produce_swc(self):

        # Check if SWC has been generated and cache it if not
        if self.hoc_path not in self.hoc_swc_cache:
            # Generate and SWC file from the HOC file
            hoc2swc(self.hoc_path, self.swc_path)

            # Cache swc file path
            self.hoc_swc_cache[self.hoc_path] = self.swc_path

        return os.path.abspath(self.swc_path)

class SwcCellModel(sciunit.Model, cap.ProducesSWC):
    """A model for cells defined using SWC files. Requires just a path to the SWC file."""

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
