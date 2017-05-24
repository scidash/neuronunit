"""NeuronUnit abstract Capabilities for channel models"""

import inspect

import sciunit

class NML2_Channel_Runnable(sciunit.Capability):
    """Capability for models that can be run using functions available in pyNeuroML.analsysi.NML2ChannelAnalysis"""
    def NML2_channel_run(self,**run_params):
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
    

class ProducesIVCurve(sciunit.Capability):
    """The capability to produce a current-voltage plot for a set of voltage steps"""
    def produce_iv_curve(self, **run_params):
        """Produces steady-state and peak IV curve at voltages and conditions given according to 'run_params'"""
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
    
    def produce_iv_curve_ss(self, **run_params):
        """Produces steady-state IV curve at voltages and conditions given according to 'run_params'"""
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
    
    def produce_iv_curve_peak(self, **run_params):
        """Produces peak current IV curve at voltages and conditions given according to 'run_params'"""
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])
    
    def plot_iv_curve(self, v, i, *plt_args, **plt_kwargs):
        """Plots IV Curve using array-like voltage 'v' and array-like current 'i'"""
        return NotImplementedError("%s not implemented" % inspect.stack()[0][3])