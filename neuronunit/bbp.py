"""NeuronUnit module for interaction with the Blue Brain Project data."""

import os
import zipfile
import requests

import matplotlib.pyplot as plt
from neo.io import IgorIO

try:  # Python 3
    from io import BytesIO
    MAJOR_VERSION = 3
except ImportError:  # Python 2
    from StringIO import StringIO
    MAJOR_VERSION = 2


def is_bbp_up():
    """Check whether the BBP microcircuit portal is up."""
    url = "http://microcircuits.epfl.ch/released_data/B95_folder.zip"
    request = requests.get(url)
    return request.status_code == 200


def list_curated_data():
    """List all curated datasets as of July 1st, 2017.

    Includes those found at
    http://microcircuits.epfl.ch/#/article/article_4_eph
    """
    return ['B95', 'C9', 'C12']


def get_curated_data(data_id, sweeps=None):
    """Download curated data (Igor files) from the microcircuit portal.

    data_id: An ID number like the ones in 'list_curated_data()' that appears
    in http://microcircuits.epfl.ch/#/article/article_4_eph.
    """
    url = "http://microcircuits.epfl.ch/data/released_data/%s.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_uncurated_data(data_id, sweeps=None):
    """Download uncurated data (Igor files) from the microcircuit portal."""
    url = "http://microcircuits.epfl.ch/data/uncurated/%s_folder.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_sweeps(url, sweeps=None):
    """Get sweeps of data from the given URL."""
    print("Getting data from %s" % url)
    path = find_or_download_data(url)  # Base path for this data
    assert type(sweeps) in [type(None), list], "Sweeps must be None or a list."
    sweep_paths = list_sweeps(path)  # Available sweeps
    if sweeps is None:
        sweeps = sweep_paths
    else:
        sweeps = []
        for sweep_path in sweep_paths:
            if any([sweep_path.endswith(sweep for sweep in sweeps)]):
                sweeps.append(sweep_path)
        sweeps = set(sweeps)
    data = {sweep: open_data(sweep) for sweep in sweeps}
    return data


def find_or_download_data(url):
    """Find or download data from the given URL.

    Return a path to a local directory containing the unzipped data found
    at the provided url.  The zipped file will be downloaded and unzipped if
    the directory cannot be found.  The path to the directory is returned.
    """
    zipped = url.split('/')[-1]  # Name of zip file
    unzipped = zipped.split('.')[0]  # Name when unzipped
    if not os.path.isdir(unzipped):  # If unzipped version not found
        if MAJOR_VERSION == 2:
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(StringIO(r.content))
        elif MAJOR_VERSION == 3:
            r = requests.get(url)
            z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(unzipped)


def list_sweeps(url, extension='.ibw'):
    """List all sweeps available in the file at the given URL."""
    path = find_or_download_data(url)  # Base path for this data
    sweeps = find_sweeps(path, extension=extension)
    return sweeps


def find_sweeps(path, extension='.ibw', depth=0):
    """Find sweeps available at the given path.

    Starting from 'path', recursively searches subdirectories and returns
    full paths to all files ending with 'extension'.
    """
    sweeps = []
    items = os.listdir(path)
    for item in items:
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            sweeps += find_sweeps(new_path, extension=extension, depth=depth+1)
        if os.path.isfile(new_path) and item.endswith(extension):
            sweeps += [new_path]
    return sweeps


def open_data(path):
    """Take a 'path' to an .ibw file and returns a neo.core.AnalogSignal."""
    igor_io = IgorIO(filename=path)
    analog_signal = igor_io.read_analogsignal()
    return analog_signal


def plot_data(signal):
    """Plot the data in a neo.core.AnalogSignal."""
    plt.plot(signal.times, signal)
    plt.xlabel(signal.sampling_period.dimensionality)
    plt.ylabel(signal.dimensionality)
