import subprocess

from scoop import futures, _control, utils, shared
from scoop._types import FutureQueue
from scoop.broker.structs import BrokerInfo

from .base import *

#def multiworker_set(self):
global subprocesses
worker = subprocess.Popen([sys.executable, "-m", "scoop.bootstrap.__main__",
                                                          "--brokerHostname", "127.0.0.1", "--taskPort", "5555",
                                                          "--metaPort", "5556", "test_optimization.py"])
subprocesses.append(worker)
