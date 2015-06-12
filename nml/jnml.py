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
    OPENWORM_HOME = '/Users/rgerkin/Dropbox/dev'#os.environ['OPENWORM_HOME']
except KeyError:
    OPENWORM_HOME = '~/git'
LEMS_FILE_PATH = os.path.join(OPENWORM_HOME,'muscle_model/NeuroML2/LEMS_Test_k_fast.xml')
sys.path.append(os.path.join(OPENWORM_HOME,'BlueBrainProjectShowcase/Channelpedia'))
sys.path.append(os.path.join(OPENWORM_HOME,'pyNeuroML'))
JNML_HOME = os.path.join(OPENWORM_HOME,'jNeuroML')
JNML_BIN = os.path.join(JNML_HOME,'jnml')

try:
    from pyneuroml.analysis import NML2ChannelAnalysis # Import NML2ChannelAnalysis, etc.  
except ImportError:
    print("Could not import pynml.analysis")
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
        p = subprocess.Popen([JNML_BIN,lems_file_path,"-nogui"], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         #shell=True,
                         env=dict(os.environ,JNML_HOME=JNML_HOME))
        result, err = p.communicate()
        result = str(result).split('\\n')
        if len(err):
            raise Exception(err)
    print("Ran LEMS file %s" % os.path.split(lems_file_path)[1])
    return result

def make_lems(nml_file_path,tool,**kwargs):
    """Creates a LEMS file from a NeuroML file with parameters."""
    print(tool.__name__)
    assert tool.__name__.split('.')[-1] in \
        ['NML2ChannelAnalyse', 'NML2ChannelAnalysis'], "Not a supported tool"
    lems_file_path = None
    args = argparse.Namespace()
    args.channelFiles = [nml_file_path]
    #args.target_file = nml_file_path
    args.nogui = True
    for key,value in kwargs.items():
        setattr(args,key,value)
    if hasattr(tool,'DEFAULTS'):
        for key,value in tool.DEFAULTS.items():
            if not hasattr(args,key):
                setattr(args,key,value)
    lems_file_path = tool.main(args=args)
    lems_file_path = os.path.join(os.getcwd(),lems_file_path)
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
            NML2ChannelAnalysis, # Tool.  
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
 