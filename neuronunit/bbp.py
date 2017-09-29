"""NeuronUnit module for interaction with the Allen Brain Insitute 
Cell Types database"""

import os
import sys
import shelve
import zipfile

import numpy as np

import matplotlib.pyplot as plt
import quantities as pq
from neo.io import IgorIO

try: # Python 2
    from urllib import urlencode, urlretrieve
    from urllib2 import urlopen, URLError, HTTPError
except ImportError: # Python 3
    from urllib.parse import urlencode
    from urllib.request import urlopen, urlretrieve, URLError, HTTPError

def is_bbp_up():
    url = "http://microcircuits.epfl.ch/released_data/B95_folder.zip"
    try:
        urlretrieve(url)
    except HTTPError:
        result = False
    else:
        result = True
    return result

def list_curated_data():
    """
    List of all curated datasets as of July 1st, 2017 found at
    http://microcircuits.epfl.ch/#/article/article_4_eph
    """
    
    return ['B95', 'C9', 'C12']


def get_curated_data(data_id, sweeps=None):
    """
    Downloads data (Igor files) from the microcircuit portal.
    data_id: An ID number like the ones in 'list_curated_data()' that appears in
    http://microcircuits.epfl.ch/#/article/article_4_eph.  
    """

    url = "http://microcircuits.epfl.ch/data/released_data/%s.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_uncurated_data(data_id, sweeps=None):
    url = "http://microcircuits.epfl.ch/data/uncurated/%s_folder.zip" % data_id
    data = get_sweeps(url, sweeps=sweeps)
    return data


def get_sweeps(url, sweeps=None):
    print("Getting data from %s" % url)
    path = find_or_download_data(url) # Base path for this data
    assert type(sweeps) in [type(None),list], "Sweeps must be None or a list."
    sweep_paths = list_sweeps(path) # Available sweeps
    if sweeps is None:
        sweeps = sweep_paths
    else:
        sweeps = []
        for sweep_path in sweep_paths:
            if any([sweep_path.endswith(sweep for sweep in sweeps)]):
                sweeps.append(sweep_path)
        sweeps = set(sweeps)
    data = {sweep:open_data(sweep) for sweep in sweeps}
    return data


def find_or_download_data(url):
    """Return a path to a local directory containing the unzipped data found
    at the provided url.  The zipped file will be downloaded and unzipped if
    the directory cannot be found.  The path to the directory is returned.
    """

    zipped = url.split('/')[-1] # Name of zip file
    unzipped = zipped.split('.')[0] # Name when unzipped
    if not os.path.isdir(unzipped): # If unzipped version not found
        downloaded_file, headers = urlretrieve(url)
        with zipfile.ZipFile(downloaded_file,"r") as zip_ref:
            zip_ref.extractall(unzipped)
    return unzipped


def list_sweeps(url, extension='.ibw'):
    path = find_or_download_data(url) # Base path for this data
    sweeps = find_sweeps(path, extension=extension)
    return sweeps


def find_sweeps(path, extension='.ibw', depth=0):
    """Starting from 'path', recursively searches subdirectories and returns
    full paths to all files ending with 'extension'.
    """  

    sweeps = []
    items = os.listdir(path)
    for item in items:
        new_path = os.path.join(path,item)
        if os.path.isdir(new_path):            
            sweeps += find_sweeps(new_path,extension=extension,depth=depth+1)
        if os.path.isfile(new_path) and item.endswith(extension):
            sweeps += [new_path]
    return sweeps


def open_data(path):
    """Take a 'path' to an .ibw file and returns a neo.core.AnalogSignal."""

    igor_io = IgorIO(filename=path)
    analog_signal = igor_io.read_analogsignal()
    return analog_signal


def plot_data(signal):
    """Plots the data in a neo.core.AnalogSignal."""

    plt.plot(signal.times,signal)
    plt.xlabel(signal.sampling_period.dimensionality)
    plt.ylabel(signal.dimensionality)
    #plt.show()
