"""Static Backend."""

from .base import Backend


class StaticBackend(Backend):
    def _backend_run(self):
        pass
    
    def set_stop_time(self, t_stop):
        pass
