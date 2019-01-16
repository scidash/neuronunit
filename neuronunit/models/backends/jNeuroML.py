"""jNeuroML Backend."""

import os
import tempfile
from sciunit.models.backends import Backend
from pyneuroml import pynml


class jNeuroMLBackend(Backend):
    """Use for simulation with jNeuroML, a reference simulator for NeuroML."""

    backend = 'jNeuroML'

    def init_backend(self, *args, **kwargs):
        self.model.create_lems_file_copy()
        super(jNeuroMLBackend, self).init_backend(*args, **kwargs)

    def set_attrs(self, **attrs):
        self.model.set_lems_attrs()

    def set_run_params(self, **run_params):
        self.model.set_lems_run_params()

    def inject_square_current(self, current):
        pass

    def _backend_run(self):
        f = pynml.run_lems_with_jneuroml
        self.exec_in_dir = tempfile.mkdtemp()
        lems_path = os.path.dirname(self.model.orig_lems_file_path)
        results = f(self.model.lems_file_path,
                    paths_to_include=[lems_path],
                    skip_run=self.model.skip_run,
                    nogui=self.model.run_params['nogui'],
                    load_saved_data=True, plot=False,
                    exec_in_dir=self.exec_in_dir,
                    verbose=self.model.run_params['v'])
        return results
