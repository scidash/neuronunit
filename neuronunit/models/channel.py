"""NeuronUnit model class for ion channels models"""

import os

import neuronunit.capabilities.channel as cap
from .lems import LEMSModel
from pyneuroml.analysis import NML2ChannelAnalysis as ca
import quantities as pq


class ChannelModel(LEMSModel, cap.NML2_Channel_Runnable,
                   cap.ProducesIVCurve):
    """A model for ion channels"""

    def __init__(self, channel_file_path_or_url, channel_index=0, name=None):
        """
        channel_file_path: Path to NML file.
        channel_index: Order of channel in NML file
                       (usually 0 since most files contain one channel).
        name: Optional model name.
        """
        if name is None:
            base, file_name = os.path.split(channel_file_path_or_url)
            name = file_name.split('.')[0]
        super(ChannelModel, self).__init__(channel_file_path_or_url, name=name,
                                           backend='jNeuroML')
        channels = ca.get_channels_from_channel_file(self.orig_lems_file_path)
        self.channel = channels[channel_index]
        self.a = None
        # Temperature, clamp parameters, etc.
        self.default_params = ca.DEFAULTS.copy()
        self.default_params.update({'nogui': True})
        #self.run_defaults = ca.build_namespace()

    """
    DEPRECATED
    def NML2_run(self, rerun=False, a=None, verbose=None, **params):
        self.params = self.default_params.copy()
        self.params.update(params)
        # Convert keyword args to a namespace.
        a = ca.build_namespace(a=a, **self.params)
        if verbose is None:
            verbose = a.v
        # Only rerun if params have changed.
        if self.a is None or a.__dict__ != self.a.__dict__ or rerun:
            self.a = a
            # Force the Channel Analysis module to write files to the
            # temporary directory
            ca.OUTPUT_DIR = self.temp_dir.name
            # Create a lems file.
            self.lems_file_path = ca.make_lems_file(self.channel, self.a)
            # Writes data to disk.
            self.results = ca.run_lems_file(self.lems_file_path, verbose)
    """

    def ca_run_lems_file(self, verbose=True):
        results = ca.run_lems_file(self.lems_file_path, verbose)
        return results

    def ca_produce_iv_curve(self, ca_namespace, results):
        iv_data = ca.compute_iv_curve(self.channel, ca_namespace, results)
        self.iv_data = {}
        for kind in ['i_peak', 'i_steady']:
            self.iv_data[kind] = {}
            for v, i in iv_data[kind].items():
                v = float((v * pq.V).rescale(pq.mV))
                self.iv_data[kind][v] = (i * pq.A).rescale(pq.pA)
        self.iv_data['hold_v'] = (iv_data['hold_v'] * pq.V).rescale(pq.mV)
        return self.iv_data

    def plot_iv_curve(self, v, i, *plt_args, **plt_kwargs):
        ca.plot_iv_curve(self.a, v, i, *plt_args, **plt_kwargs)
