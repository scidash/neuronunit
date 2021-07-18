"""jNeuroML Backend."""

import os
import pathlib
import io
import tempfile

from pyneuroml import pynml

from sciunit.utils import redirect_stdout
from .base import Backend
from elephant.spike_train_generation import threshold_detection
        

class jNeuroMLBackend(Backend):
    """Use for simulation with jNeuroML, a reference simulator for NeuroML."""

    name = 'jNeuroML'
    
    def init_backend(self, *args, **kwargs):
        """Initialize the jNeuroML backend."""
        assert hasattr(self.model, 'set_lems_run_params'), \
            "A model using %s must implement `set_lems_run_params`" % \
            self.backend
        self.stdout = io.StringIO()
        self.model.create_lems_file_copy()  # Create a copy of the LEMS file
        super(jNeuroMLBackend, self).init_backend(*args, **kwargs)

    def set_attrs(self, **attrs):
        """Set the model attributes, i.e. model parameters."""
        self.model.set_lems_attrs()

    def set_run_params(self, **run_params):
        """Sey the backend runtime parameters, i.e. simulation parameters."""
        self.model.set_lems_run_params()

    def inject_square_current(self, **kwargs):
        """Inject a square current into the cell."""
        self.model.run_params['injected_square_current'] = kwargs
        self.set_run_params()  # Doesn't work yet.
        self._backend_run()

    def set_stop_time(self, t_stop):
        """Set the stop time of the simulation."""
        self.model.run_params['t_stop'] = t_stop
        self.set_run_params()

    def set_time_step(self, dt):
        """Set the time step of the simulation."""
        self.model.run_params['dt'] = dt
        self.set_run_params()

    def get_spike_count(self):
        thresh = threshold_detection(self.vm)
        return len(thresh)

    def _backend_run(self):
        """Run the simulation."""
        
        try:
            self.exec_in_dir = self.model.temp_dir.name
        except:
            self.exec_in_dir = tempfile.mkdtemp()
        
        path = pathlib.Path(self.exec_in_dir)
        (path / 'results').mkdir(exist_ok=True)
        
        with redirect_stdout(self.stdout):
            results = self._get_results()
        if results is None or not results:
            print(("No results returned: buffered error, warning, "
                   "and notice messages follow:\n"))
            print(self.stdout.getvalue())
        return results
    
    def _get_results(self):
        lems_path = os.path.dirname(self.model.orig_lems_file_path)
        f = pynml.run_lems_with_jneuroml
        results = f(self.model.lems_file_path,
                    paths_to_include=[lems_path],
                    skip_run=self.model.skip_run,
                    nogui=self.model.run_params['nogui'],
                    load_saved_data=True,
                    plot=False,
                    exec_in_dir=self.exec_in_dir,
                    exit_on_fail=False,
                    verbose=self.model.run_params['v'])
        return results