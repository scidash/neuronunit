import os
import sys
import argparse
import csv
import subprocess

import numpy as np

USE_PYLEMS = False
if USE_PYLEMS:
    from lems.run import run
try:
    OPENWORM_HOME = os.environ['OPENWORM_HOME']
except KeyError:
    OPENWORM_HOME = '~/git'
LEMS_FILE_PATH = os.path.join(OPENWORM_HOME,'muscle_model/NeuroML2/LEMS_Test_k_fast.xml')
sys.path.append(os.path.join(OPENWORM_HOME,'BlueBrainProjectShowcase/Channelpedia'))
JNML_HOME = os.path.join(OPENWORM_HOME,'jNeuroML')
JNML_BIN = os.path.join(JNML_HOME,'jnml')
try:
    import NML2ChannelAnalyse
except ImportError:
    print("Could not import NML2 Channel Analyzer")

def run_lems(lems_file_path):
    """Runs a LEMS file with pylems (via jlems)"""
    if lems_file_path == '':
        lems_file_path = LEMS_FILE_PATH
    HERE = os.path.abspath(os.path.dirname(__file__))
    include_dirs = [os.path.join(HERE,'NeuroML2/NeuroML2CoreTypes')]
    if USE_PYLEMS:
        result = run(lems_file_path, include_dirs=include_dirs, nogui=True)
    else:
        p = subprocess.Popen(["%s %s -nogui" % (JNML_BIN,lems_file_path)], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, shell=True)
        result, err = p.communicate()
        result = str(result).split('\\n')
        if len(err):
            raise Exception(err)
    print("Ran LEMS file %s" % os.path.split(lems_file_path)[1])
    return result

def make_lems(nml_file_path,tool,**kwargs):
    """Creates a LEMS file from a NeuroML file with parameters."""
    assert tool in ['NML2ChannelAnalyse'], "Not a supported tool"
    lems_file_path = None
    if tool == 'NML2ChannelAnalyse':
        args = argparse.Namespace()
        args.channelFile = nml_file_path
        for key,value in kwargs.items():
            args.__dict__[key] = value
        for key,value in NML2ChannelAnalyse.DEFAULTS.items():
            if not hasattr(args,key):
                setattr(args,key,value)
        lems_file_path = NML2ChannelAnalyse.main(args=args)
    return lems_file_path

def run_nml(nml_file_path,tool,**args):
    lems_file_path = make_lems(nml_file_path,tool,**args)
    run_lems(lems_file_path)

def load_dat(dat_file_path):
    max_cols = []
    with open('%s.dat' % dat_file_path, 'r') as csvfile:
        for delimiter in [' ','\t']:
            csvfile.seek(0)
            reader = csv.reader(csvfile, delimiter=delimiter)
            cols = [col for col in list(zip(*reader)) \
                 if sum([len(_) for _ in col]) > 0]
            if len(cols) > len(max_cols):
                max_cols = cols
    result = np.array(max_cols).astype('float')
    return result

def main():
    run_nml(os.path.join(OPENWORM_HOME,'muscle_model/NeuroML2/k_fast.channel.nml'), # File.  
            'NML2ChannelAnalyse', # Tool.  
            channelId='k_fast', 
            temperature=34, 
            minV=55, 
            maxV=80, 
            duration=1000, 
            clampBaseVoltage=-55, 
            clampDuration=580, 
            stepTargetVoltage=10, 
            erev=55)



if __name__ == '__main__':
    main()
 