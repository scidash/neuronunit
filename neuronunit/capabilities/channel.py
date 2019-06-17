"""NeuronUnit abstract Capabilities for channel models"""

import inspect

import sciunit


class NML2ChannelAnalysis(sciunit.Capability):
    """Capability for models that can be altered using functions available
    in pyNeuroML.analsysi.NML2ChannelAnalysis"""

    def ca_make_lems_file(self, **run_params):
        """Makes a LEMS file using the provided run parameters using
        the ChannelAnalysis module."""
        return NotImplementedError("%s not implemented" %
                                   inspect.stack()[0][3])

    def ca_run_lems_file(self):
        """Run the LEMS file using ChannelAnalysis module."""
        return NotImplementedError("%s not implemented" %
                                   inspect.stack()[0][3])

    def compute_iv_curve(self, results):
        """Compute an IV Curve from the iv data in `results`."""
        return NotImplementedError("%s not implemented" %
                                   inspect.stack()[0][3])

    def plot_iv_curve(self, v, i, *plt_args, **plt_kwargs):
        """Plots IV Curve using array-like voltage 'v'
        and array-like current 'i'"""
        return NotImplementedError("%s not implemented" %
                                   inspect.stack()[0][3])
