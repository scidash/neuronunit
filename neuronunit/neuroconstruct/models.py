"""
Implementation of a model built in neuroConstruct.
http://www.neuroconstruct.org/
"""

import os
from xml.etree.ElementTree import XML
from __init__ import *
from pythonnC.utils import putils # From the neuroConstruct pythonnC package.  
from sciunit import Model
from .capabilities import * # neurounit.neuroconstruct.capabilites
#from neurounit.capabilities import ReceivesCurrent
from neuronunit.capabilities import spike_functions
import numpy as np

class NeuroConstructModel(Model,
                          ProducesMembranePotential_NC,
                          ProducesSpikes_NC,
                          ReceivesCurrent_NC):
    """Implementation of a candidate model usable by neuroConstruct (written in neuroML).
    Execution takes places in the neuroConstruct program.
    Methods will be implemented using the neuroConstruct python
    API (in progress)."""

    def __init__(self, project_path, name=None, **kwargs):
        """file_path is the full path to an .ncx file."""
        #print("Instantiating a neuroConstruct model from %s." % project_path)
        self.project_path = project_path
        self.ran = False
        self.population_name = self.get_cell_group()+'_0'
        Model.__init__(self)
        Runnable_NC.__init__(self)
        ReceivesCurrent_NC.__init__(self)
        for key,value in kwargs.items():
            setattr(self,key,value)
        super().__init__(name=name, **kwargs)

    def get_ncx_file(self):
        # Get a list of .ncx (neuroConstruct) files.  Should be only one for most projects.
        ncx_files = [f for f in os.listdir(self.project_path) if f[-4:]=='.ncx']
        if not len(ncx_files):
            raise IOError("No .ncx files found in %s" % self.project_path)
        ncx_file = os.path.join(self.project_path,ncx_files[0]) # Get full path to .ncx file.
        return ncx_file

    def get_cell_group(self):
        ncx_file = self.get_ncx_file()
        with open(ncx_file,'r') as f:
            xml_str = f.read()
        neuroml = XML(xml_str) # The NeuroML file in parsable form.
        cell_group = neuroml.find("object/void[@property='allSimConfigs']/void/object/void[@property='cellGroups']/void/string").text
        return cell_group

class FakeNeuroConstructModel(NeuroConstructModel):
    """A fake neuroConstruct model that generates a gaussian noise
    membrane potential with some 'spikes'. Eventually I will make the membrane
    potential and the spikes change as a function of the current."""

    def __init__(*args,**kwargs):
        NeuroConstructModel.__init__(self,*args,**kwargs)
        self.current_ampl = 0

    def run(self,**kwargs):
        n_samples = getattr(self,'n_samples',10000)
        self.vm = np.random.randn(n_samples)-65.0 # -65 mV with gaussian noise.
        for i in range(200,n_samples,200): # Make 50 spikes.
            self.vm[i:i+10] += 10.0*np.array(range(10)) # Shaped like right triangles.
        super(FakeNeuroConstructModel,self).run(**kwargs)

    def set_current_ampl(self,current):
        self.current_ampl = current

class OSBModel(NeuroConstructModel):
    """A model hosted on Open Source Brain (http://www.opensourcebrain.org).
    Will be in NeuroML format, and run using neuroConstruct."""

    def __init__(self,brain_area,cell_type,model_name,**kwargs):
        project_path = os.path.join(self.models_path,
                                    brain_area,
                                    cell_type,
                                    model_name,
                                    "neuroConstruct")
        if 'name' not in kwargs.keys():
            self._name = u'%s/%s/%s' % (brain_area,cell_type,model_name)
        NeuroConstructModel.__init__(self,project_path,**kwargs)
        self.name = model_name

    @classmethod
    def make_model(self,brain_area,cell_type,model_name,**kwargs):
        project_path = os.path.join(self.models_path,
                                    brain_area,
                                    cell_type,
                                    model_name,
                                    "neuroConstruct")
        return NeuroConstructModel(project_path,**kwargs)

    models_path = putils.OSB_MODELS

# DEPRECATED. 
''' 
class NeuroML2Model(NeuroConstructModel):
    """A model hosted on Open Source Brain (http://www.opensourcebrain.org).
    Will be in NeuroML format, and run using neuroConstruct."""

    def __init__(self,model_name,**kwargs):
        project_path = os.path.join(self.models_path,
                                    model_name)
        if 'name' not in kwargs.keys():
            self._name = u'%s' % (model_name)
        #self.create_nc_project()
        NeuroConstructModel.__init__(self,project_path,**kwargs)

    models_path = putils.NEUROML2_MODELS
 
    def create_nc_project(self):
        """Creates a neuroConstruct project from the NeuroML2 file(s)."""
        from java.io import File
        from ucl.physiol.neuroconstruct.project import Project
        from ucl.physiol.neuroconstruct.cell.converters import MorphMLConverter
        project = Project.createNewProject('/Users/rgerkin/', self.model_name, None)
        morphDir = File("%s/osb/showcase/neuroConstructShowcase/Ex3_Morphology/importedMorphologies/" % NC_HOME)
        morphmlFile = File(morphDir, "SimplePurkinjeCell.morph.xml")
        converter = MorphMLConverter()
        cell = converter.loadFromMorphologyFile(morphmlFile, "NewCell")
        project.cellManager.addCellType(cell) # Actually add it to the project
        project.cellGroupsInfo.setCellType("DefaultCellGroup", cell.getInstanceName()) # Set the type of an existing cell group to this
        project.saveProject()
'''
