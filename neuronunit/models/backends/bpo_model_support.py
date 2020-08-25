"""Cell template class"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""


# pylint: disable=W0511

# TODO take into account that one might want to run protocols on different
# machines
# TODO rename this to 'CellModel' -> definitely

import sys
import os
import collections
import string

from . import create_hoc
from . import morphologies

import logging
logger = logging.getLogger(__name__)


from .very_reduced_sans_lems import VeryReducedModel
import neuronunit.capabilities as cap
from sciunit.models.runnable import RunnableModel
from neuronunit.optimisation.data_transport_container import DataTC
import copy

import bluepyopt.ephys as ephys

import numpy as np
class Model(object):

    """Model"""

    def __init__(self, name):
        """Constructor
        Args:
            name (str): name of the model
        """
        self.name = name

    def instantiate(self, sim=None):
        """Instantiate model in simulator"""
        pass

    def destroy(self, sim=None):
        """Destroy instantiated model in simulator"""
        pass

import gc

import asciiplotlib as apl

class ReducedCellModel(VeryReducedModel,
                       RunnableModel,
                       cap.ReceivesSquareCurrent,
                       cap.ProducesActionPotentials,
                       cap.ProducesMembranePotential):

    """Cell model class"""

    def __init__(
            self,
            name,
            params=None,
            gid=0,
            backend=None, 
            tests = None):
        """Constructor

        Args:
            name (str): name of this object
                        should be alphanumeric string, underscores are allowed,
                        first char should be a letter
            morph (Morphology):
                underlying Morphology of the cell
            mechs (list of Mechanisms):
                Mechanisms associated with the cell
            params (list of Parameters):
                Parameters of the cell model
            backend a NeuronUnit model backend/model simulator combination
            tests: NeuronUnit test suite used to extract features.
        """
        #super(CellModel, self).__init__(name)
        super(VeryReducedModel, self).__init__(name=name,backend=backend)
        
        #super(VeryReducedModel, self).__init__(name=name,backend=backend, attrs=attrs)
        self.backend = backend
        #self._backend = str('')
        self.attrs = {}
        self.run_number = 0
        self.tstop = None
        self.rheobse = None
        self.check_name()
        self.params = collections.OrderedDict(**params)
        self.tests = tests

        # Cell instantiation in simulator
        self.icell = None
        self.param_values = None
        self.gid = gid
        self.NU = True

    def model_to_dtc(self,attrs=None):
        """
        Args:
            self
        Returns: 
            dtc
            DTC is a simulator indipendent data transport container object.
        """
            
        dtc = DataTC(backend=self.backend)
        if hasattr(self,'tests'):
            if type(self.tests) is not type(None):
                dtc.tests = self.tests

        if type(attrs) is not type(None):
            if len(attrs):
                dtc.attrs = attrs
                self.attrs = attrs
            assert self._backend is not None
            return dtc
        else:
            if type(self.attrs) is not type(None):
                if len(self.attrs):
                    try:
                        dynamic_attrs = {str(k):float(v) for k,v in self.attrs.items()}
                    except:
                        dynamic_attrs = {str(k):float(v.value) for k,v in self.attrs.items()}

        if self._backend is None:
            super(VeryReducedModel, self).__init__(name=self.name,backend=self.backend)#,attrs=dtc.attrs)
            assert self._backend is not None
        frozen_attrs = self._backend.default_attrs
        if 'dynamic_attrs' in locals():
            frozen_attrs.update(dynamic_attrs)
        all_attrs = frozen_attrs
        dtc.attrs = all_attrs
        assert dtc.attrs is not None
        return dtc

    def inject_model(self,
                     DELAY=100,
                     DURATION=500): 
        dynamic_attrs = {str(k):float(v) for k,v in self.params.items()}
        #assert len(dynamic_attrs)
        frozen_attrs = self._backend.default_attrs

        frozen_attrs.update(dynamic_attrs)
        all_attrs = frozen_attrs
        #attrs = copy.copy(dfa)
        #attrs.update(copy.copy(param_values))
        dtc = self.model_to_dtc(attrs=all_attrs)
        assert len(dtc.attrs)
        from neuronunit.optimisation.optimization_management import dtc_to_rheo  

        dtc = dtc_to_rheo(dtc)
        self.rheobase = dtc.rheobase
            
            
        vm = [np.nan]
        if self.rheobase is not None:
            uc = {'amplitude':self.rheobase,'duration':DURATION,'delay':DELAY}
            self._backend.attrs = all_attrs
            try:
                self._backend.inject_square_current(uc)
                vm = self.get_membrane_potential()
                self.vm = vm
            except:
                vm = [np.nan]
                #print('frozne',frozen_attrs,'dyn',dynamic_attrs,'self',self.attrs)            
        else:
            self.vm = vm
        return vm    

    def check_name(self):
        """Check if name complies with requirements"""

        allowed_chars = string.ascii_letters + string.digits + '_'

        if sys.version_info[0] < 3:
            translate_args = [None, allowed_chars]
        else:
            translate_args = [str.maketrans('', '', allowed_chars)]

        if self.name == '' \
                or self.name[0] not in string.ascii_letters \
                or not str(self.name).translate(*translate_args) == '':
            raise TypeError(
                'CellModel: name "%s" provided to constructor does not comply '
                'with the rules for Neuron template name: name should be '
                'alphanumeric '
                'non-empty string, underscores are allowed, '
                'first char should be letter' % self.name)

    def params_by_names(self, param_names):
        """Get parameter objects by name"""

        return [self.params[param_name] for param_name in param_names]

    def freeze(self, param_dict):
        """Set params"""

        for param_name, param_value in param_dict.items():
            self.params[param_name].freeze(param_value)

    def unfreeze(self, param_names):
        """Unset params"""

        for param_name in param_names:
            self.params[param_name].unfreeze()

    def instantiate(self, sim=None):
        """
        Instantiate model in simulator
        As if called from a genetic algorithm.
        """
        #self.icell.gid = self.gid
        if self.params is not None:
            self.attrs = self.params

        dtc = self.model_to_dtc()
        for k,v in self.params.items():
            v = float(v.value)
            dtc.attrs[k] = v
            self.attrs[k] = v
        return dtc

            #for param in self.params.values():
            #    model.attrs[param] = 
                #param.instantiate(sim=sim, icell=self.icell)
    def destroy(self, sim=None):  # pylint: disable=W0613
        """Destroy instantiated model in simulator"""

        # Make sure the icell's destroy() method is called
        # without it a circular reference exists between CellRef and the object
        # this prevents the icells from being garbage collected, and
        # cell objects pile up in the simulator
        self.icell.destroy()

        # The line below is some M. Hines magic
        # DON'T remove it, because it will make sure garbage collection

        del self.icell# = None
        for param in self.params.values():
            param.destroy(sim=sim)
            print('destroyed param')

    def check_nonfrozen_params(self, param_names):  # pylint: disable=W0613
        """Check if all nonfrozen params are set"""
        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)




from sciunit.models import Model as SciUnitModel
from neuronunit.models.static import ExternalModel 
import quantities as pq

class CellModel(Model):
                #cap.ReceivesSquareCurrent,
                #cap.ProducesActionPotentials,
                #cap.ProducesMembranePotential):

    """Cell model class"""

    def __init__(
            self,
            name,
            morph=None,
            mechs=None,
            params=None,
            gid=0):
        """Constructor

        Args:
            name (str): name of this object
                        should be alphanumeric string, underscores are allowed,
                        first char should be a letter
            morph (Morphology):
                underlying Morphology of the cell
            mechs (list of Mechanisms):
                Mechanisms associated with the cell
            params (list of Parameters):
                Parameters of the cell model
        """
        super(CellModel, self).__init__(name)
        self.check_name()
        self.morphology = morph
        self.mechanisms = mechs
        self.params = collections.OrderedDict()
        if params is not None:
            for param in params:
                self.params[param.name] = param

        # Cell instantiation in simulator
        self.icell = None

        self.param_values = None
        self.gid = gid
        self.seclist_names = \
            ['all', 'somatic', 'basal', 'apical', 'axonal', 'myelinated']
        self.secarray_names = \
            ['soma', 'dend', 'apic', 'axon', 'myelin']

    def check_name(self):
        """Check if name complies with requirements"""

        allowed_chars = string.ascii_letters + string.digits + '_'

        if sys.version_info[0] < 3:
            translate_args = [None, allowed_chars]
        else:
            translate_args = [str.maketrans('', '', allowed_chars)]

        if self.name == '' \
                or self.name[0] not in string.ascii_letters \
                or not str(self.name).translate(*translate_args) == '':
            raise TypeError(
                'CellModel: name "%s" provided to constructor does not comply '
                'with the rules for Neuron template name: name should be '
                'alphanumeric '
                'non-empty string, underscores are allowed, '
                'first char should be letter' % self.name)

    def params_by_names(self, param_names):
        """Get parameter objects by name"""

        return [self.params[param_name] for param_name in param_names]

    def freeze(self, param_dict):
        """Set params"""

        for param_name, param_value in param_dict.items():
            self.params[param_name].freeze(param_value)

    def unfreeze(self, param_names):
        """Unset params"""

        for param_name in param_names:
            self.params[param_name].unfreeze()

    def inject_square_current(self,current):
        import l5pc_evaluator

        sweep_protocols = []
        fitness_protocols = l5pc_evaluator.define_protocols()
        protocol = fitness_protocols['Step3']
        if 'injected_square_current' in current.keys():
            current = current['injected_square_current']
        protocol.stimuli[0].step_amplitude = float(current['amplitude'])
        protocol.stimuli[0].step_delay = float(current['delay'])
        protocol.stimuli[0].step_duration = float(current['duration'])
        #protocol.stimuli[0].
        total_duration = float(current['delay']) + \
                        float(current['duration']) \
                        + 200       
        protocol.stimuli[0].total_duration = total_duration          
        fitness_calculator = l5pc_evaluator.define_fitness_calculator(fitness_protocols)
        param_names = [param for param in self.params.keys()]

        sim = ephys.simulators.NrnSimulator()
        evalu = ephys.evaluators.CellEvaluator(
            cell_model=self,
            param_names=param_names,
            fitness_protocols=fitness_protocols,
            fitness_calculator=fitness_calculator,
            sim=sim)
        response = evalu.run_protocol(evalu.fitness_protocols['Step3'],evalu.params)   

        return response
        

    @staticmethod
    def create_empty_template(
            template_name,
            seclist_names=None,
            secarray_names=None):
        '''create an hoc template named template_name for an empty cell'''

        objref_str = 'objref this, CellRef'
        newseclist_str = ''

        if seclist_names:
            for seclist_name in seclist_names:
                objref_str += ', %s' % seclist_name
                newseclist_str += \
                    '             %s = new SectionList()\n' % seclist_name

        create_str = ''
        if secarray_names:
            create_str = 'create '
            create_str += ', '.join(
                '%s[1]' % secarray_name
                for secarray_name in secarray_names)
            create_str += '\n'

        template = '''\
        begintemplate %(template_name)s
          %(objref_str)s
          proc init() {\n%(newseclist_str)s
            forall delete_section()
            CellRef = this
          }

          gid = 0

          proc destroy() {localobj nil
            CellRef = nil
          }

          %(create_str)s
        endtemplate %(template_name)s
               ''' % dict(template_name=template_name, objref_str=objref_str,
                          newseclist_str=newseclist_str,
                          create_str=create_str)

        return template

    @staticmethod
    def create_empty_cell(
            name,
            sim,
            seclist_names=None,
            secarray_names=None):
        """Create an empty cell in Neuron"""

        # TODO minize hardcoded definition
        # E.g. sectionlist can be procedurally generated
        hoc_template = CellModel.create_empty_template(
            name,
            seclist_names,
            secarray_names)
        sim.neuron.h(hoc_template)

        template_function = getattr(sim.neuron.h, name)

        return template_function()

    def instantiate(self, sim=None):
        """Instantiate model in simulator"""

        # TODO replace this with the real template name
        if not hasattr(sim.neuron.h, self.name):
            self.icell = self.create_empty_cell(
                self.name,
                sim=sim,
                seclist_names=self.seclist_names,
                secarray_names=self.secarray_names)
        else:
            self.icell = getattr(sim.neuron.h, self.name)()

        self.icell.gid = self.gid

        self.morphology.instantiate(sim=sim, icell=self.icell)

        if self.mechanisms is not None:
            for mechanism in self.mechanisms:
                mechanism.instantiate(sim=sim, icell=self.icell)

        if self.params is not None:
            for param in self.params.values():
                param.instantiate(sim=sim, icell=self.icell)

    def destroy(self, sim=None):  # pylint: disable=W0613
        """Destroy instantiated model in simulator"""

        # Make sure the icell's destroy() method is called
        # without it a circular reference exists between CellRef and the object
        # this prevents the icells from being garbage collected, and
        # cell objects pile up in the simulator
        self.icell.destroy()

        # The line below is some M. Hines magic
        # DON'T remove it, because it will make sure garbage collection
        # is called on the icell object
        sim.neuron.h.Vector().size()

        self.icell = None

        self.morphology.destroy(sim=sim)
        for mechanism in self.mechanisms:
            mechanism.destroy(sim=sim)
        for param in self.params.values():
            param.destroy(sim=sim)

    def check_nonfrozen_params(self, param_names):  # pylint: disable=W0613
        """Check if all nonfrozen params are set"""

        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)

    def create_hoc(self, param_values,
                   ignored_globals=(), template='cell_template.jinja2',
                   disable_banner=False,
                   template_dir=None):
        """Create hoc code for this model"""

        to_unfreeze = []
        for param in self.params.values():
            if not param.frozen:
                param.freeze(param_values[param.name])
                to_unfreeze.append(param.name)

        template_name = self.name
        morphology = os.path.basename(self.morphology.morphology_path)
        if self.morphology.do_replace_axon:
            replace_axon = self.morphology.replace_axon_hoc
        else:
            replace_axon = None

        if (
            self.morphology.morph_modifiers is not None
            and self.morphology.morph_modifiers_hoc is None
        ):
            logger.warning('You have provided custom morphology modifiers, \
                            but no corresponding hoc files.')
        elif (
            self.morphology.morph_modifiers is not None
            and self.morphology.morph_modifiers_hoc is not None
        ):
            if replace_axon is None:
                replace_axon = ''
            for morph_modifier_hoc in self.morphology.morph_modifiers_hoc:
                replace_axon += '\n'
                replace_axon += morph_modifier_hoc

        ret = create_hoc.create_hoc(mechs=self.mechanisms,
                                    parameters=self.params.values(),
                                    morphology=morphology,
                                    ignored_globals=ignored_globals,
                                    replace_axon=replace_axon,
                                    template_name=template_name,
                                    template_filename=template,
                                    template_dir=template_dir,
                                    disable_banner=disable_banner)

        self.unfreeze(to_unfreeze)

        return ret

    def __str__(self):
        """Return string representation"""

        content = '%s:\n' % self.name

        content += '  morphology:\n'

        if self.morphology is not None:
            content += '    %s\n' % str(self.morphology)

        content += '  mechanisms:\n'
        if self.mechanisms is not None:
            for mechanism in self.mechanisms:
                content += '    %s\n' % mechanism

        content += '  params:\n'
        if self.params is not None:
            for param in self.params.values():
                content += '    %s\n' % param

        return content


class HocMorphology(morphologies.Morphology):

    '''wrapper for Morphology so that it has a morphology_path'''

    def __init__(self, morphology_path):
        super(HocMorphology, self).__init__()
        if not os.path.exists(morphology_path):
            raise Exception('HocCellModel: Morphology not found at: %s'
                            % morphology_path)
        self.morphology_path = morphology_path


class HocCellModel(CellModel):

    '''Wrapper class for a hoc template so it can be used by BluePyOpt'''

    def __init__(self, name, morphology_path, hoc_path=None, hoc_string=None):
        """Constructor

        Args:
            name(str): name of this object
            sim(NrnSimulator): simulator in which to instatiate hoc_string
            hoc_path(str): Path to a hoc file
                (hoc_path and hoc_string can't be used simultaneously,
                but one of them has to specified)
            hoc_string(str): String that of hoc code that defines a template
                (hoc_path and hoc_string can't be used simultaneously,
                but one of them has to specified))
            morphology_path(str path): path to morphology that can be loaded by
                                       Neuron
        """
        super(HocCellModel, self).__init__(name,
                                           morph=None,
                                           mechs=[],
                                           params=[])

        if hoc_path is not None and hoc_string is not None:
            raise TypeError('HocCellModel: cant specify both hoc_string '
                            'and hoc_path argument')
        if hoc_path is not None:
            with open(hoc_path) as hoc_file:
                self.hoc_string = hoc_file.read()
        else:
            self.hoc_string = hoc_string

        self.morphology = HocMorphology(morphology_path)
        self.cell = None
        self.icell = None

    def params_by_names(self, param_names):
        pass

    def freeze(self, param_dict):
        pass

    def unfreeze(self, param_names):
        pass

    def instantiate(self, sim=None):
        sim.neuron.h.load_file('stdrun.hoc')
        template_name = self.load_hoc_template(sim, self.hoc_string)

        morph_path = self.morphology.morphology_path
        assert os.path.exists(morph_path), \
            'Morphology path does not exist: %s' % morph_path
        if os.path.isdir(morph_path):
            # will use the built in morphology name, if the init() only
            # gets one parameter
            self.cell = getattr(sim.neuron.h, template_name)(morph_path)
        else:
            morph_dir = os.path.dirname(morph_path)
            morph_name = os.path.basename(morph_path)
            self.cell = getattr(sim.neuron.h, template_name)(morph_dir,
                                                             morph_name)
        self.icell = self.cell.CellRef

    def destroy(self, sim=None):
        self.cell = None
        self.icell = None

    def check_nonfrozen_params(self, param_names):
        pass

    def __str__(self):
        """Return string representation"""
        return (
            '%s: %s of %s(%s)' %
            (self.__class__,
             self.name,
             self.get_template_name(self.hoc_string),
             self.morphology.morphology_path,))

    @staticmethod
    def get_template_name(hoc_string):
        """Find the template name from hoc_string

        Note: this will fail if there is a begintemplate in a /* */ style
        comment before the real begintemplate
        """
        for i, line in enumerate(hoc_string.split('\n')):
            if 'begintemplate' in line:
                line = line.strip().split()
                assert line[0] == 'begintemplate', \
                    'begintemplate must come first, line %d' % i
                template_name = line[1]
                logger.info('Found template %s on line %d', template_name, i)
                return template_name
        else:  # pylint: disable=W0120
            raise Exception('Could not find begintemplate in hoc file')

    @staticmethod
    def load_hoc_template(sim, hoc_string):
        """Have neuron hoc template, and detect what the name template name is

        The template must have an init that takes two parameters, the second of
        which is the path to a morphology.

        It must also have a CellRef member that is the result of
            `Import3d_GUI(...).instantiate()`
        """
        template_name = HocCellModel.get_template_name(hoc_string)
        if not hasattr(sim.neuron.h, template_name):
            sim.neuron.h(hoc_string)
            assert hasattr(sim.neuron.h, template_name), \
                'NEURON does not have template: ' + template_name

        return template_name

class NU_HocCellModel(HocCellModel,RunnableModel,
                       cap.ReceivesSquareCurrent,
                       cap.ProducesActionPotentials,
                       cap.ProducesMembranePotential):

    def __init__(self, name, morphology_path, hoc_path=None, hoc_string=None):
        """Constructor

        Args:
            name(str): name of this object
            sim(NrnSimulator): simulator in which to instatiate hoc_string
            hoc_path(str): Path to a hoc file
                (hoc_path and hoc_string can't be used simultaneously,
                but one of them has to specified)
            hoc_string(str): String that of hoc code that defines a template
                (hoc_path and hoc_string can't be used simultaneously,
                but one of them has to specified))
            morphology_path(str path): path to morphology that can be loaded by
                                    Neuron
        """
        super(NU_HocCellModel, self).__init__(name,
                                        morph=None,
                                        mechs=[],
                                        params=[])
        self.NU = True

