"""jNeuroML Backend."""

import os

import eden_simulator

from .jNeuroML import jNeuroMLBackend


class EdenBackend(jNeuroMLBackend):
    """Use for simulation with the Eden simulator for NeuroML."""

    name = 'eden'
    
    def _get_results(self):
        f = eden_simulator.runEden
        
        # Needed to deal with a bug in Eden
        rel_path = os.path.relpath(self.model.lems_file_path, os.getcwd())
        
        results = f(rel_path)
        return results