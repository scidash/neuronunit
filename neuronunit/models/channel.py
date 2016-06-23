import os

import sciunit
from neuronunit.capabilities.channel import *
from pyneuroml.analysis import NML2ChannelAnalysis as ca
import quantities as pq

class ChannelModel(sciunit.Model, NML2_Channel_Runnable, ProducesIVCurve):
    """A model for ion channels"""
    
    def __init__(self, channel_file_path, channel_index=0, name=None):
        """
        channel_file_path: Path to NML file.
        channel_index: Order of channel in NML file (usually 0 since most files contain one channel).
        name: Optional model name.
        """
        self.nml_file_path = channel_file_path
        channels = ca.get_channels_from_channel_file(self.nml_file_path)
        self.channel = channels[channel_index]
        self.a = None
        self.run_defaults = ca.DEFAULTS.copy() # Temperature, clamp parameters, etc.  
        self.run_defaults.update({'nogui': True})
        self.run_defaults = ca.build_namespace()

        if name is None:
            name = os.path.split()[1].split('.')[0]
        super(ChannelModel,self).__init__(name=name)
    
    def NML2_run(self, rerun=False, a=None, **run_params):
        if not len(run_params):
            run_params = self.run_defaults
        a = ca.build_namespace(a=a,**run_params) # Convert keyword args to a namespace.  
        if self.a is None or a.__dict__ != self.a.__dict__ or rerun: # Only rerun if params have changed.  
            self.a = a
            self.lems_file_path = ca.make_lems_file(self.channel,self.a) # Create a lems file.
            self.results = ca.run_lems_file(self.lems_file_path,self.a) # Writes data to disk.  
    
    def produce_iv_curve(self, **run_params):
        run_params['ivCurve'] = True
        self.NML2_run(**run_params)
        iv_data = ca.compute_iv_curve(self.channel, self.a, self.results)
        self.iv_data = {}
        for kind in ['i_peak', 'i_steady']:
            self.iv_data[kind] = {}
            for v,i in iv_data[kind].items():
                v = float((v * pq.V).rescale(pq.mV))
                self.iv_data[kind][v] = (i * pq.A).rescale(pq.pA)
        self.iv_data['hold_v'] = (iv_data['hold_v'] * pq.V).rescale(pq.mV)
        return self.iv_data
    
    def produce_iv_curve_ss(self, **run_params):
        self.produce_iv_curve(**run_params)
        return {'v':self.iv_data['hold_v'], 
                'i':self.iv_data['i_steady']}
    
    def produce_iv_curve_peak(self, **run_params):
        self.produce_iv_curve(**run_params)
        return {'v':self.iv_data['hold_v'], 
                'i':self.iv_data['i_peak']}
        
    def plot_iv_curve(self, v, i, **plot_args):
        ca.plot_iv_curve(self.a, v, i, **plot_args)