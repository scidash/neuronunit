"""NeuronUnit abstract Capabilities for multicompartment cell models"""

import inspect
import sciunit

class ProducesSWC(sciunit.Capability):
    '''
    The capability to produce a morphology SWC file
    '''
    def produce_swc(self):
        '''
        Produces morphology description file in SWC file format

        :return: absolute path to the produced SWC file
        '''
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])