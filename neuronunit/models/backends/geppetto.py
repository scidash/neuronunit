"""jNeuroML Backend."""

from .jNeuroML import jNeuroMLBackend

class GeppettoBackend(jNeuroMLBackend):
    """Use for simulation with the Geppetto backend for SciDash."""

    backend = 'Geppetto'

    def init_backend(self, *args, **kwargs):
        """Initialize the Geppetto backend."""
        super(GeppettoBackend, self).init_backend(*args, **kwargs)

    def _backend_run(self):
        """Send the simulation to Geppetto to run.
        You have two options here.  Either: 
        (1) Run the simulation and return a dictionay of results, as other backends do.  
        (2) Implement nothing here and never call it, always writing to the backend's cache instead.
        """
        results = None
        return results