# Obtains the cell threshold, rheobase, resting v, and bias currents for
# steady state v of a cell defined in a hoc file in the given directory.
# Usage: python getCellProperties /path/To/dir/with/.hoc
<<<<<<< HEAD
#from neuronunit.tests.druckmann2013 import *
try:
    import cPickle
except:
=======

try:  # Python 2
    import cPickle
except ImportError:  # Python 3
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
    import _pickle as cPickle
import csv
import json
import os
import re
import shutil
import string
import urllib
from abc import abstractmethod, ABCMeta
<<<<<<< HEAD
#from decimal import Decimal
#import nmldbutils
=======
from decimal import Decimal
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
import inspect
import multiprocessing
cpus = multiprocessing.cpu_count

import numpy as np
from matplotlib import pyplot as plt
<<<<<<< HEAD
#from sklearn.decomposition import PCA

#from collector import Collector

#from neuronrunner import NeuronRunner, NumericalInstabilityException
#from runtimer import RunTimer
#from neuronunit.tables import Cells, Model_Waveforms, Morphometrics, Cell_Morphometrics, db_proxy, Models
#from neuronunit.nmldbmodel import NMLDB_Model
#cpus = multiprocessing.cpu_count
import dask.bag as dbag # a pip installable module, usually installs without complication
import dask
import urllib.request, json
import os
import requests
from neo.core import AnalogSignal
from quantities import mV, ms, nA
from neuronunit import models
import pickle
from neuronunit.optimisation import get_neab

from types import MethodType

try:
    #assert 1==2
    with open('static_models.p','rb') as f:
        sms = pickle.load(f)

except:
    def get_wave_forms(cell_id):
        url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
        waveids = requests.get(url)
        waveids = json.loads(waveids.text)
        wlist = waveids['waveform_list']
        waves_to_get = []
        for wl in wlist:
            waves_to_test = {}
            wid = wl['ID']
            url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
            waves = requests.get(url)
            temp = json.loads(waves.text)
            if temp['Spikes'] >= 1:
                if 'NOISE' in temp['Protocol_ID']:
                    print((temp['Waveform_Label']))

                    pass
                if 'RAMP' in temp['Protocol_ID']:
                    print((temp['Waveform_Label']))

                    pass
                if 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
                    print((temp['Waveform_Label']))
                    pass

                if 'SHORT_SQUARE' in temp['Protocol_ID'] and not 'SHORT_SQUARE_TRIPPLE' in temp['Protocol_ID']:
                    try:
                        parts = temp['Waveform_Label'].split(' ')
                        #import pdb; pdb.set_trace()
                        print(parts[0])
                        print(parts[1])
                        waves_to_test['prediction'] = float(parts[0])*nA# '1.1133 nA',
                        print('yes')

                        temp_vm = list(map(float, temp['Variable_Values'].split(',')))

                        waves_to_test['Times'] = list(map(float,temp['Times'].split(',')))
                        waves_to_test['DURATION'] = temp['Time_End'] -temp['Time_Start']
                        waves_to_test['DELAY'] = temp['Time_Start']
                        waves_to_test['Time_End'] = temp['Time_End']
                        waves_to_test['Time_Start'] = temp['Time_Start']
                        dt = waves_to_test['Times'][1]- waves_to_test['Times'][0]
                        waves_to_test['vm'] = AnalogSignal(temp_vm,sampling_period=dt*ms,units=mV)
                        waves_to_test['everything'] = temp
                        waves_to_get.append(waves_to_test)

                    except:
                        if temp['Waveform_Label'] is None:
                            pass
                        pass
        return waves_to_get
    waves = get_wave_forms(str('NMLCL001129'))
    sms = []
    for w in waves:

        sm = models.StaticModel(w['vm'])
        sm.rheobase = {}
        sm.rheobase['mean'] = w['prediction']
        sm.complete = w
        sms.append(sm)
    with open('static_models.p','wb') as f:
        pickle.dump(sms,f)

electro_path = str(os.getcwd())+'/examples/pipe_tests.p'
#import pdb
#pdb.set_trace()
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    test_frame = pickle.load(f)

def generate_prediction(self,model):
    prediction = {}
    prediction['n'] = 1
    prediction['std'] = 1.0
    prediction['mean'] = model.rheobase['mean']
    return prediction
    #model.get_membrane_potential()

#import pdb; pdb.set_trace()
test_scores = []
tt = [tests for tests in test_frame[0][0] ]
#vtest[k] = active_values(keyed,dtc.rheobase)
#for t in tt: t.generate_prediction = MethodType(generate_prediction,t)
params = {}


def get_m_p(cls,params = {}):
    return model.get_membrane_potential()
for model in sms: model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential

#def get_membrane_potential():

#for model in sms: model.inject_square_current = MethodType(model.get_membrane_potential,model)

#for model in sms: model.inject_square_current = MethodType(model.get_membrane_potential,model)
#for model in sms: print(model.inject_square_current())# = MethodType(model.get_membrane_potential,params)

tt[0].generate_prediction = MethodType(generate_prediction,tt[0])
from neuronunit.optimisation.optimisation_management import switch_logic#, active_values
#print(active_values.__file__)
def active_values(keyed,rheobase,square = None):
    keyed['injected_square_current'] = {}
    if square == None:
        DURATION = 1000.0*pq.ms
        DELAY = 100.0*pq.ms
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
    else:
        DURATION = square['Time_End'] -square['Time_Start']
        DELAY = square['Time_Start']
        keyed['injected_square_current']['amplitude'] = square['prediction']#value'])*pq.pA

    keyed['injected_square_current']['delay']= DELAY
    keyed['injected_square_current']['duration'] = DURATION

    return keyed



tt = switch_logic(tt) # for tests in tt ]
for t in tt:
    for sm in sms:
        #rheobase = sm.complete
        rheobase = sm.complete['prediction']
        t.params = {}
        t.params = active_values(t.params,rheobase,square=sm.complete)


flat_iter = iter((t,sm) for t in tt for sm in sms)
#print(flat_iter)
import efel
for t,sm in flat_iter:
    sm._backend = None
    if t.active:
        t.params = {}
        t.params['injected_square_current'] = None

        #score = rtest.judge(model)
        results = sm.get_membrane_potential()
        trace = {}

        trace['T'] = sm.complete['Times']
        trace['V'] = results
        trace['stim_start'] = [sm.complete['Time_Start']]#rtest.run_params[]
        DURATION = sm.complete['Time_End'] -sm.complete['Time_Start']

        trace['stim_end'] = [sm.complete['Time_End'] ]# list(sm.complete['duration'])
        traces = [trace]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
        traces_results = efel.getFeatureValues(traces,list(efel.getFeatureNames()))#
        for v in traces_results:
            for key,value in v.items():
                if type(value) is not type(None):
                    print(key,value)
            #import pdb; pdb.set_trace()
            #if type(v) is not type(None):
            #    print(v)
        #print(traces_results)
        '''
        try:
            test_scores.append(t.judge(sm))
            print(test_scores[-1],t.name)
        except:
            test_scores.append(t.judge(sm))

            print(test_scores[-1],t.name)

            print('active skipped: ',t.name)
        '''
    #else:
    #    print('passive skipped: ',t.name)


        #all_models = json.loads(url.read().decode())
def get_all(Model_ID = str('NMLNT001592')):
    if Model_ID == None:
        try:
            # Obtains the cell threshold, rheobase, resting v, and bias currents for
            #with urllib.request.urlopen("https://www.neuroml-db.org/api/models") as url:
            #    all_models = json.loads(url.read().decode())


            url = str("https://www.neuroml-db.org/api/models")
            all_models = requests.get(url)
            all_models = json.loads(all_models.text)
            print(all_models)
            for d in all_models:
                print(d.keys())
            for d in all_models[0]:
                print(d['Model_ID'],d['Directory_Path'])
                url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
                urllib.request.urlretrieve(url,Model_ID)
                #https://www.neuroml-db.org/api/models'
                #url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
                #os.system('wget '+str(url))
                os.system(str('unzip *')+str(d['Model_ID'])+('*'))
                os.system(str('pynml hhneuron.cell.nml -neuron'))
            return data

        except:
            pass
    else:
        d = {}
        d['Model_ID'] = Model_ID
        #print(d['Model_ID'],d['Directory_Path'])
        #https://www.neuroml-db.org/api/models'
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        #url = str('https://www.neuroml-db.org/GetModelZip?modelID=')+str(d['Model_ID'])+str('&version=NeuroML')
        urllib.request.urlretrieve(url,Model_ID)
        print(url)
        url = "https://www.neuroml-db.org/GetModelZip?modelID=NMLNT001592&version=NeuroML"
        os.system('wget '+str(url))
        os.system(str('unzip ')+str(d['Model_ID'])+('*'))
        os.system(str('pynml hhneuron.cell.nml -neuron'))


def run_cell():
    from neuron import h
    h.load_file('hhneuron.hoc')
    cell = h.hhneuron
    d = {}
    d['Model_ID'] = str('NT001592')
    with urllib.request.urlopen(str('https://www.neuroml-db.org/api/model?id=')+str(d['Model_ID'])) as url:
        data_on_model = json.loads(url.read().decode())


class CellModel():
=======
try:
    from playhouse.db_url import connect
    from sshtunnel import SSHTunnelForwarder
    from collector import Collector
    from neuronrunner import NeuronRunner, NumericalInstabilityException
    from sklearn.decomposition import PCA
    from runtimer import RunTimer
    from tables import Cells, Model_Waveforms, Morphometrics, Cell_Morphometrics, db_proxy, Models
    from nmldbmodel import NMLDB_Model
except:  # Hack to allow import to occur so unit tests can pass
    print("Many modules required by `cellmodelp` not found")
    NMLDB_Model = object
import dask.bag as db # a pip installable module, usually installs without complication
import dask

from neuronunit.tests.druckmann2013 import *

class CellModel(NMLDB_Model):
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
    def __init__(self, *args, **kwargs):
        super(CellModel, self).__init__(*args, **kwargs)

        self.init_cell_record()

        if self.cell_record.Steady_State_Delay is None:
            self.steady_state_delay = 1000
        else:
            self.steady_state_delay = self.cell_record.Steady_State_Delay

        self.pickle_file_cache = {}

        self.all_properties.extend([
            'NEURON_conversion',
            'equation_count',
            'runtime_per_step',
            'structural_metrics',
            'tolerances',
            'stability_range',
            'resting_voltage',
            'threshold',
            'rheobase',
            'DT_SENSITIVITY',
            'stability_range',
            'threshold',
            'rheobase',
            'bias_current',
            'tolerances_with_stim',
            'CVODE_STEP_FREQUENCIES',
            'STEADY_STATE',
            'RAMP',
            'SHORT_SQUARE',
            'SQUARE',
            'LONG_SQUARE',
            'SHORT_SQUARE_HOLD',
            'SHORT_SQUARE_TRIPPLE',
            'SQUARE_SUBTHRESHOLD',
            'DRUCKMANN_PROPERTIES'
            # 'NOISE',
            # 'NOISE_RAMP',
            # 'morphology_data'
        ])

        self.init_cell_record()

    def init_cell_record(self):
        #self.server.connect()

        self.cell_record = Cells.get_or_none(Cells.Model_ID == self.get_model_nml_id())

        if self.cell_record is None:
            self.cell_record = Cells(
                Model_ID=self.get_model_nml_id(),
                Stability_Range_Low=None,
                Stability_Range_High=None,
                Is_Intrinsically_Spiking=False,
                Resting_Voltage=None,
                Rheobase_Low=None,
                Rheobase_High=None,
                Threshold_Current_Low=None,
                Threshold_Current_High=None,
                Bias_Current=None,
                Bias_Voltage=None,
                Errors=None
            )

            # Create cell record if it doesn't exist (using the NMLDB ID as the pkey)
            self.cell_record.save(force_insert=True)

            # Retrieve the freshly created record
            self.cell_record = Cells.get_or_none(Cells.Model_ID == self.get_model_nml_id())

    def save_stability_range(self):

        if self.is_nosim():
            return

        self.use_optimal_dt_if_available()

        print("Getting stability range...")
        self.cell_record.Stability_Range_Low, self.cell_record.Stability_Range_High = self.get_stability_range()


        assert self.cell_record.Stability_Range_Low < self.cell_record.Stability_Range_High

        self.cell_record.save()

    def save_resting_voltage(self):

        if self.is_nosim():
            return

        self.use_optimal_dt_if_available()

        print("Getting resting voltage...")
        self.cell_record.Resting_Voltage = self.getRestingV(self.steady_state_delay, save_resting_state=True)["rest"]


        # No resting v means cell is intrinsically spiking
        self.cell_record.Is_Intrinsically_Spiking = self.cell_record.Resting_Voltage is None

        if not self.cell_record.Is_Intrinsically_Spiking:
            assert self.cell_record.Resting_Voltage < 1.0 # Allen Glif models rest at 0

        self.cell_record.save()


    def save_threshold(self):

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        print("Getting threshold...")

        th = self.getThreshold(0, self.cell_record.Stability_Range_High)
        self.cell_record.Threshold_Current_Low = np.min(th)
        self.cell_record.Threshold_Current_High = np.max(th)



        assert self.cell_record.Threshold_Current_Low < self.cell_record.Threshold_Current_High

        self.cell_record.save()


    def save_rheobase(self):
        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        print("Getting rheobase...")
        rb = self.getRheobase(0, self.cell_record.Threshold_Current_High)
        self.cell_record.Rheobase_Low = np.min(rb)
        self.cell_record.Rheobase_High = np.max(rb)



        assert self.cell_record.Rheobase_Low < self.cell_record.Rheobase_High
        assert self.cell_record.Rheobase_High < self.cell_record.Threshold_Current_High

        self.cell_record.save()

    def save_bias_current(self):
        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        print("Getting current for bias voltage...")
        roundedRest = round(self.cell_record.Resting_Voltage / 10) * 10

        if roundedRest == -80:
            bias_v = -70
        else:
            bias_v = -80

        bias_i = self.getBiasCurrent(targetV=bias_v)

        self.cell_record.Bias_Voltage = bias_v
        self.cell_record.Bias_Current = bias_i

        assert self.cell_record.Bias_Current < self.cell_record.Rheobase_High

        if self.cell_record.Bias_Voltage < self.cell_record.Resting_Voltage:
            assert self.cell_record.Bias_Current < 0
        else:
            assert self.cell_record.Bias_Current > 0

        self.cell_record.save()

    def get_number_of_compartments(self, h):
        if self.is_abstract_cell():
            return 1

        return sum(s.nseg for s in h.allsec())

    def save_to_SWC(self, h):
        import xml.etree.ElementTree

        root = xml.etree.ElementTree.parse(self.model_record.File_Name).getroot()

        seg_tags = root.findall(".//{http://www.neuroml.org/schema/neuroml2}segment")

        point_ids = {}
        self.current_id = 1

        def add_point(prox_dist):
            if len(prox_dist) > 0:
                point_str = str(prox_dist[0].attrib)

                if point_str not in point_ids:
                    point_ids[point_str] = str(self.current_id)
                    self.current_id += 1

        for seg_tag in seg_tags:
            proximal = seg_tag.findall('{http://www.neuroml.org/schema/neuroml2}proximal')
            distal   = seg_tag.findall('{http://www.neuroml.org/schema/neuroml2}distal')

            add_point(proximal)
            add_point(distal)

        segment_distal_point_ids = {}

        for seg_tag in seg_tags:
            distal = seg_tag.findall('{http://www.neuroml.org/schema/neuroml2}distal')
            point_id = point_ids[str(distal[0].attrib)]
            segment_distal_point_ids[seg_tag.attrib["id"]] = point_id

        swc_points = []

        def get_type(seg_tag):
            seg_name = seg_tag.attrib["name"].lower()

            if "dend" in seg_name:
                return "3"

            if "axon" in seg_name:
                return "2"

            if "soma" in seg_name:
                return "1"

            return "5"

        for tag in seg_tags:
            parent_tag = tag.findall('{http://www.neuroml.org/schema/neuroml2}parent')
            proximal = tag.findall('{http://www.neuroml.org/schema/neuroml2}proximal')
            distal   = tag.findall('{http://www.neuroml.org/schema/neuroml2}distal')

            if parent_tag:

                # parent - with prox - use proximal as parent id
                if proximal:
                    parent_id = point_ids[str(proximal[0].attrib)]

                    # If diameter of proximal is not the same as parent's distal - add as separate point
                    if parent_id not in [pt["id"] for pt in swc_points]:
                        if parent_tag[0].attrib["segment"] not in segment_distal_point_ids:
                            raise Exception("Segment refers to non-existent parent segment: " + str(parent_tag[0].attrib["segment"]))

                        swc_point = {
                            "id": parent_id,
                            "type": get_type(tag),
                            "parent": segment_distal_point_ids[parent_tag[0].attrib["segment"]],
                            "x": proximal[0].attrib["x"],
                            "y": proximal[0].attrib["y"],
                            "z": proximal[0].attrib["z"],
                            "radius": str(float(proximal[0].attrib["diameter"]) / 2.0)
                        }

                        swc_points.append(swc_point)

                # parent - no prox - use parent's distal as parent id
                else:
                    parent_id = segment_distal_point_ids[parent_tag[0].attrib["segment"]]

            # no parent - add proximal - will become distal's parent
            else:
                swc_point = {
                    "id": point_ids[str(proximal[0].attrib)],
                    "type": get_type(tag),
                    "parent": "-1",
                    "x": proximal[0].attrib["x"],
                    "y": proximal[0].attrib["y"],
                    "z": proximal[0].attrib["z"],
                    "radius": str(float(proximal[0].attrib["diameter"]) / 2.0)
                }

                parent_id = swc_point["id"]

                swc_points.append(swc_point)


            # Always add distal
            swc_point = {
                "id": point_ids[str(distal[0].attrib)],
                "type": get_type(tag),
                "parent": str(parent_id),
                "x": distal[0].attrib["x"],
                "y": distal[0].attrib["y"],
                "z": distal[0].attrib["z"],
                "radius": str(float(distal[0].attrib["diameter"]) / 2.0)
            }

            swc_points.append(swc_point)

        swc_file_path = os.path.join(self.get_conversion_dir("swc"),"cell.swc")

        with open(swc_file_path, "w") as file:
            for point in swc_points:
                file.write(
                    point["id"] + " " +
                    point["type"] + " " +
                    point["x"] + " " +
                    point["y"] + " " +
                    point["z"] + " " +
                    point["radius"] + " " +
                    point["parent"] + "\n")

        return os.path.abspath(swc_file_path)

    def save_LMeasure_metrics(self, swc_file):

        #db = self.server.connect()
        cell_id = self.get_model_nml_id()

        # Do all the work within a transaction
        with db.atomic():
            # Clear out existing cell metrics
            Cell_Morphometrics.delete().where(Cell_Morphometrics.Cell == cell_id).execute()

            # Get a list of metrics
            metrics = Morphometrics.select()

            for metric in metrics:
                # Compute the metric with lmeasure
                f = metric.Function_ID
                swc_file = swc_file.replace(os.path.abspath(os.getcwd()) + "/", "")
                os.system('../../lmeasure -f'+str(f)+',0,0,10.0 -slmeasure_out.csv '+swc_file+' -C')

                # Read the result
                with open('lmeasure_out.csv') as f:
                    line = list(csv.reader(f, delimiter="\t"))[0]

                    # Make sure the db function id corresponds to the Lmeasure function name
                    assert line[1].startswith(metric.ID)

                    # Save to DB
                    record = Cell_Morphometrics(
                        Cell=cell_id,
                        Metric=metric,
                        Total = float(line[2]),
                        Compartments_Considered=int(line[3]),
                        Compartments_Discarded=int(line[4].replace("(", "").replace(")", "")),
                        Minimum = float(line[5]),
                        Average = float(line[6]),
                        Maximum = float(line[7]),
                        StDev = float(line[8])
                    )

                    record.save()

            # Cleanup lmeasure files
            os.system("rm lmeasure_out.csv")

    def save_morphology_data(self):

        # Load the model
        h = self.build_model(restore_tolerances=False)

        if self.is_abstract_cell() or self.get_number_of_compartments(h) <= 1:
            print("Cell is ABSTRACT or SINGLE COMPARTMENT, skipping morphometrics and 3D visualization...")
            return

        # Compute morphometrics
        self.save_morphometrics(h)

        # Render 3D GIF
        # Rotate the cell to be upright along the x,y,z coord PCA axes
        self.rotate_cell_along_PCA_axes(h)

        import sys
        sys.path.append('/home/justas/Repositories/BlenderNEURON/ForNEURON')
        from blenderneuron import BlenderNEURON

        bl = BlenderNEURON(h)
        bl.prepare_for_collection()

        # Skip simulation if no basic properties are present
        if self.cell_record.Is_Intrinsically_Spiking is not None:
            # Inject continuous above rheobase current
            if self.cell_record.Rheobase_High is not None:
                self.current.delay = 0
                self.current.dur = 100
                self.current.amp = self.cell_record.Rheobase_High * 1.5

            self.use_optimal_dt_if_available()
            h.steps_per_ms = 10
            h.cvode_active(self.config.cvode_active)
            h.dt = self.config.dt

            # No additional stim for intrinsic spikers
            print("Simulating current injection...")
            h.tstop = 100.0
            h.newPlotV()
            h.run()

        self.save_rotating_gif(bl)


    def save_rotating_gif(self, bl):
        bl.enqueue_method("clear")
        bl.enqueue_method('set_render_params', file_format="JPEG2000")
        bl.send_model()
        bl.enqueue_method('link_objects')
        bl.enqueue_method('show_full_scene')
        bl.enqueue_method('color_by_unique_materials')
        bl.enqueue_method('orbit_camera_around_model')

        # Remove previous blend file
        os.system("rm " + os.path.join(self.get_conversion_dir("Blender"), "*.blend"))

        bl.run_method('save_scene',os.path.join(self.get_conversion_dir("Blender"),"cell.blend"))


        print("RENDERING... Check progress in Blender command line window...")

        # Wait till prev tasks and rendering is finished
        bl.run_method('render_animation', self.get_conversion_dir("gif"))

        print("Creating GIF from rendered frames...")
        self.make_gif_from_frames(self.get_conversion_dir("gif"))

    def save_morphometrics(self, h):
        swc = self.save_to_SWC(h)
        self.save_LMeasure_metrics(swc)

    def rotate_cell_along_PCA_axes(self, h):
        sections = [s for s in h.allsec()]

        # Using the first and last coords of sections
        coords = [[h.x3d(0, sec=s),
                   h.y3d(0, sec=s),
                   h.z3d(0, sec=s)] for s in sections] \
                 + \
                 [[h.x3d(h.n3d(sec=s) - 1, sec=s),
                   h.y3d(h.n3d(sec=s) - 1, sec=s),
                   h.z3d(h.n3d(sec=s) - 1, sec=s)] for s in sections]

        coords = np.array(coords)

        # Get the PCA components
        pca = PCA()
        pca.fit(coords)

        # Rotate each section point to be along the PCA axes
        for sec in sections:
            for i in range(int(h.n3d(sec=sec))):
                transformed = pca.transform([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)]])

                x = transformed[0][2]
                y = transformed[0][1]
                z = transformed[0][0]
                diam = h.diam3d(i, sec=sec)

                h.pt3dchange(i, x, y, z, diam, sec=sec)

    def save_tolerances_with_stim(self):
        if self.is_nosim():
            return

        self.save_tolerances(current_amp=0 if self.cell_record.Is_Intrinsically_Spiking else self.cell_record.Threshold_Current_High)

    def save_STEADY_STATE(self):
        """
        Reach steady state and save model state
        :return: None
        """

        self.remove_protocol_waveforms("STEADY_STATE")

        if self.is_nosim():
            return

        self.use_optimal_dt_if_available()

        result = self.getRestingV(save_resting_state=True, run_time=self.steady_state_delay)
        self.save_tvi_plot(label="STEADY STATE", tvi_dict=result)
        self.save_vi_waveforms(protocol="STEADY_STATE", tvi_dict=result)


    def save_RAMP(self):
        """
        From steady state, run ramp injection
        :return: None
        """

        self.remove_protocol_waveforms("RAMP")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        result = self.get_ramp_response(ramp_delay=self.steady_state_delay,
                                        ramp_max_duration=5 * 1000,
                                        ramp_increase_rate_per_second=self.cell_record.Rheobase_High,
                                        stop_after_n_spikes_found=10,
                                        restore_state=True)

        self.save_tvi_plot(label="RAMP", tvi_dict=result)
        self.save_vi_waveforms(protocol="RAMP", tvi_dict=result)

    def save_SHORT_SQUARE(self):
        """
        # Short square is a brief, threshold current pulse after steady state
        :return: None
        """

        self.remove_protocol_waveforms("SHORT_SQUARE")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_square_current_set(protocol="SHORT_SQUARE",
                                     square_low=self.cell_record.Threshold_Current_Low,
                                     square_high=self.cell_record.Threshold_Current_High,
                                     square_steps=2,
                                     delay=self.steady_state_delay,
                                     duration=3)


    def save_SQUARE(self):

        self.remove_protocol_waveforms("SQUARE")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_square_current_set(protocol="SQUARE",
                                     square_low=-self.cell_record.Rheobase_High * 0.5,  # Note the "-"
                                     square_high=self.cell_record.Rheobase_High * 1.5,
                                     square_steps=11,
                                     delay=self.steady_state_delay,
                                     duration=1000)


    def save_LONG_SQUARE(self):
        """
        Long square is a 2s current pulse after steady state
        :return: None
        """

        self.remove_protocol_waveforms("LONG_SQUARE")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        # Up to Druckmann 2013 150% of RB - "standard stimulus"
        self.save_square_current_set(protocol="LONG_SQUARE",
                                         square_low=self.cell_record.Rheobase_High,
                                         square_high=self.cell_record.Rheobase_High * 1.5,
                                         square_steps=3,
                                         delay=self.steady_state_delay,
                                         duration=2000)

        # Up to Druckmann 2013 300% of RB - "strong stimulus"
        self.save_square_current_set(protocol="LONG_SQUARE",
                                         square_low=self.cell_record.Rheobase_High * 3.0,
                                         square_high=self.cell_record.Rheobase_High * 3.0,
                                         square_steps=1,
                                         delay=self.steady_state_delay,
                                         duration=2000)


    def save_SHORT_SQUARE_HOLD(self):
        """
        SHORT_SQUARE_HOLD is a short threshold stimulus, while under bias current
        :return:
        """

        self.remove_protocol_waveforms("SHORT_SQUARE_HOLD")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        def get_current_ti():
            ramp_t = [
                0,
                self.steady_state_delay,
                self.steady_state_delay,
                self.steady_state_delay + 3.0,
                self.steady_state_delay + 3.0,
                self.steady_state_delay + 250
            ]
            ramp_i = [
                self.cell_record.Bias_Current,
                self.cell_record.Bias_Current,
                -self.cell_record.Bias_Current + self.cell_record.Threshold_Current_High,
                -self.cell_record.Bias_Current + self.cell_record.Threshold_Current_High,
                self.cell_record.Bias_Current,
                self.cell_record.Bias_Current
            ]

            return ramp_t, ramp_i

        self.save_arb_current(protocol="SHORT_SQUARE_HOLD",
                              delay=self.steady_state_delay,
                              duration=250,
                              get_current_ti=get_current_ti,
                              restore_state=False)  # Holding v, not resting


    def save_SHORT_SQUARE_TRIPPLE(self):

        self.remove_protocol_waveforms("SHORT_SQUARE_TRIPPLE")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_square_tuple_set(delay=self.steady_state_delay,
                                   threshold_current=self.cell_record.Threshold_Current_High)

    def save_SQUARE_SUBTHRESHOLD(self):
        """
        Subthreshold pulses to measure capacitance
        :return: None
        """

        self.remove_protocol_waveforms("SQUARE_SUBTHRESHOLD")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_square_current_set(protocol="SQUARE_SUBTHRESHOLD",
                                     square_low=-self.cell_record.Threshold_Current_Low,
                                     square_high=self.cell_record.Threshold_Current_Low,
                                     square_steps=2,
                                     delay=self.steady_state_delay,
                                     duration=0.5)

    def save_NOISE(self):

        self.remove_protocol_waveforms("NOISE")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_noise_response_set(protocol="NOISE",
                                     meta_protocol="SEED1",
                                     delay=self.steady_state_delay,
                                     duration=3000,
                                     post_delay=250,
                                     rheobase=self.cell_record.Rheobase_High,
                                     multiples=[0.75, 1.0, 1.25],
                                     noise_pickle_file="noise1.pickle",
                                     restore_state=True)

        self.save_noise_response_set(protocol="NOISE",
                                     meta_protocol="SEED2",
                                     delay=self.steady_state_delay,
                                     duration=3000,
                                     post_delay=250,
                                     rheobase=self.cell_record.Rheobase_High,
                                     multiples=[0.75, 1.0, 1.25],
                                     noise_pickle_file="noise2.pickle",
                                     restore_state=True)

    def save_NOISE_RAMP(self):

        self.remove_protocol_waveforms("NOISE_RAMP")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.use_optimal_dt_if_available()

        self.save_noise_response_set(protocol="NOISE_RAMP",
                                     delay=self.steady_state_delay,
                                     duration=32000,
                                     post_delay=250,
                                     rheobase=self.cell_record.Rheobase_High,
                                     multiples=[1.0],
                                     noise_pickle_file="noisyRamp.pickle",
                                     restore_state=True)



    def save_DT_SENSITIVITY(self):

        self.remove_protocol_waveforms("DT_SENSITIVITY")
        self.remove_protocol_waveforms("OPTIMAL_DT_BENCHMARK")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        smallest_dt_result = self.save_dt_sensitivity_set(rheobase=self.cell_record.Rheobase_High)

        # Compute the optimal dt based on runtime and error costs
        optimal_dt = self.save_optimal_time_step()

        # Get the waveform at the optimal dt and its error
        if optimal_dt is not None:
            print('Starting OPTIMAL_DT_BENCHMARK protocol...')

            optimal_dt_result = self.save_dt_sensitivity_set(
                rheobase=self.cell_record.Rheobase_High,
                protocol="OPTIMAL_DT_BENCHMARK",
                steps_per_ms_set=[1.0 / optimal_dt],
                save_max_stable_dt=False
            )

            # Interpolate the values of the optimal dt waveform to compare to the 0-error waveform
            from scipy.interpolate import interp1d
            optimal_interpolated = interp1d(optimal_dt_result["t"], optimal_dt_result["v"], kind="cubic", fill_value='extrapolate')
            optimal_v_sub = optimal_interpolated(smallest_dt_result["t_sub"])

            optimal_dt_error = self.compute_waveform_error(smallest_dt_result["v_sub"],
                                                           smallest_dt_result["range"],
                                                           optimal_v_sub)

            self.model_record.Optimal_DT_Error = optimal_dt_error
            self.model_record.save()


    def save_CVODE_STEP_FREQUENCIES(self):

        self.remove_protocol_waveforms("CVODE_STEP_FREQUENCIES")

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive:
            return

        self.save_cvode_step_frequencies(protocol="CVODE_STEP_FREQUENCIES",
                                         delay=self.steady_state_delay,
                                         sub_rheobase=self.cell_record.Rheobase_Low,
                                         rheobase=self.cell_record.Rheobase_High)

        self.save_cvode_runtime_complexity_metrics()

    def save_DRUCKMANN_PROPERTIES(self):
        """
        Tests of features described in Druckmann et. al. 2013
        (https://academic.oup.com/cercor/article/23/12/2994/470476)

        These tests use SQUARE and LONG_SQUARE waveforms obtained from the NMLDB Web API
        The waveforms should be uploaded to production server (dendrite) before running these
        tests.
        :return: None
        """

        if self.is_nosim():
            return

        if self.cell_record.Is_Intrinsically_Spiking or self.cell_record.Is_Passive or self.cell_record.Is_GLIF:
            return

        import sciunit, neuronunit, quantities
<<<<<<< HEAD

=======
>>>>>>> 51529ae8e9a02874e8b1d050bb812f2aec8d41d9
        from neuronunit.neuromldb import NeuroMLDBStaticModel

        model = NeuroMLDBStaticModel(self.get_model_nml_id())

        standard = model.nmldb_model.get_druckmann2013_standard_current()
        strong = model.nmldb_model.get_druckmann2013_strong_current()
        ir_currents = model.nmldb_model.get_druckmann2013_input_resistance_currents()

        tests = [
            AP12AmplitudeDropTest(standard),
            AP1SSAmplitudeChangeTest(standard),
            AP1AmplitudeTest(standard),
            AP1WidthHalfHeightTest(standard),
            AP1WidthPeakToTroughTest(standard),
            AP1RateOfChangePeakToTroughTest(standard),
            AP1AHPDepthTest(standard),
            AP2AmplitudeTest(standard),
            AP2WidthHalfHeightTest(standard),
            AP2WidthPeakToTroughTest(standard),
            AP2RateOfChangePeakToTroughTest(standard),
            AP2AHPDepthTest(standard),
            AP12AmplitudeChangePercentTest(standard),
            AP12HalfWidthChangePercentTest(standard),
            AP12RateOfChangePeakToTroughPercentChangeTest(standard),
            AP12AHPDepthPercentChangeTest(standard),
            InputResistanceTest(injection_currents=ir_currents),
            AP1DelayMeanTest(standard),
            AP1DelaySDTest(standard),
            AP2DelayMeanTest(standard),
            AP2DelaySDTest(standard),
            Burst1ISIMeanTest(standard),
            Burst1ISISDTest(standard),
            InitialAccommodationMeanTest(standard),
            SSAccommodationMeanTest(standard),
            AccommodationRateToSSTest(standard),
            AccommodationAtSSMeanTest(standard),
            AccommodationRateMeanAtSSTest(standard),
            ISICVTest(standard),
            ISIMedianTest(standard),
            ISIBurstMeanChangeTest(standard),
            SpikeRateStrongStimTest(strong),
            AP1DelayMeanStrongStimTest(strong),
            AP1DelaySDStrongStimTest(strong),
            AP2DelayMeanStrongStimTest(strong),
            AP2DelaySDStrongStimTest(strong),
            Burst1ISIMeanStrongStimTest(strong),
            Burst1ISISDStrongStimTest(strong),
        ]

        for i, test in enumerate(tests):
            mean = test.generate_prediction(model)['mean']
            field = test.__class__.__name__[:-4]
            setattr(self.cell_record, field, mean)
            print('Test ' + str(i+1).rjust(2,' ') + ' ' + field.rjust(50, ' ') + ": " + str(mean))

        self.cell_record.save()


    def save_square_tuple_set(self, delay, threshold_current,
                              intervals=[7, 11, 15, 19, 23, 27, 31, 35], stim_width=3, tuples=3):

        # Create a short square triple waveform
        def get_current_ti(interval):
            ramp_ti = [(0, 0)]

            for ti in range(tuples):
                ramp_ti.append((delay + interval * ti,              0))
                ramp_ti.append((delay + interval * ti,              threshold_current))
                ramp_ti.append((delay + interval * ti + stim_width, threshold_current))
                ramp_ti.append((delay + interval * ti + stim_width, 0))

            ramp_ti.append((delay + interval * tuples + 100, 0))

            # Split the tuple list into t and i lists
            return zip(*ramp_ti)

        for interval in intervals:
            freq = round(1000.0 / interval)
            self.save_arb_current(protocol="SHORT_SQUARE_TRIPPLE",
                                  label=str(freq) + " Hz",
                                  delay=delay,
                                  duration=interval * tuples + 100,
                                  get_current_ti=lambda: get_current_ti(interval),
                                  restore_state=True)

    def save_square_current_set(self, protocol, square_low, square_high, square_steps, delay, duration, post_delay=250):

        # Create current amplitude set
        amps = np.linspace(
                    max(square_low, self.cell_record.Stability_Range_Low),
                    min(square_high, self.cell_record.Stability_Range_High),
                    num=square_steps).tolist()

        # Run each injection as a separate simulation, resuming from steady state
        for amp in amps:
            result = self.get_square_response(delay=delay,
                                              duration=duration,
                                              post_delay=post_delay,
                                              amp=amp,
                                              restore_state=True)

            self.save_tvi_plot(label=protocol, case=self.short_string(amp) + " nA", tvi_dict=result)

            self.save_vi_waveforms(protocol=protocol,
                                   label=self.short_string(amp) + " nA",
                                   tvi_dict=result)

    def get_square_response(self,
                            delay,
                            duration,
                            post_delay,
                            amp,
                            restore_state=False):

        def square_protocol(time_flag):
            print('Starting SQUARE PROTOCOL...' + str(amp))
            self.time_flag = time_flag
            h = self.build_model()
            print('Cell model built, starting current injection...')

            # Set the sqauare current injector
            self.current.dur = duration
            self.current.delay = delay
            self.current.amp = amp

            with RunTimer() as timer:
                if restore_state:
                    self.restore_state()
                    t, v = self.runFor(duration + post_delay)
                else:
                    h.stdinit()
                    t, v = self.runFor(delay + duration + post_delay)

            result = {
                "t": t.tolist(),
                "v": v.tolist(),
                "i": self.ic_i_collector.get_values_list(),
                "run_time": timer.get_run_time(),
                "steps": int(self.tvec.size()),
                "cvode_active": int(self.config.cvode_active),
                "dt_or_atol": self.config.abs_tolerance if self.config.cvode_active else self.config.dt
            }

            return result

        runner = NeuronRunner(square_protocol)
        runner.DONTKILL = True
        result = runner.run()
        return result

    def get_ramp_response(self,
                          ramp_delay,
                          ramp_max_duration,
                          ramp_increase_rate_per_second,
                          stop_after_n_spikes_found,
                          restore_state=False):

        def test_condition(t, v):
            num_spikes = self.getSpikeCount(v)

            if num_spikes >= stop_after_n_spikes_found:
                print("Got " + str(num_spikes) + " spikes at " + str(t[-1]) + " ms. Stopping ramp current injection.")
                return True

            return False

        def get_current_ti():
            ramp_i = [0, 0, ramp_max_duration / 1000.0 * ramp_increase_rate_per_second, 0]
            ramp_t = [0, ramp_delay, ramp_delay + ramp_max_duration, ramp_delay + ramp_max_duration]
            return ramp_t, ramp_i

        return self.get_arb_current_response(delay=ramp_delay,
                                             duration=ramp_max_duration,
                                             get_current_ti=get_current_ti,
                                             test_condition=test_condition,
                                             restore_state=restore_state)

    def save_cvode_step_frequencies(self, protocol, delay, sub_rheobase, rheobase):
        # Ensure this is run using the variable step method
        orig_cvode = self.config.cvode_active
        self.config.cvode_active = 1

        # First two evaluate no stim and sub-threshold current CVODE step frequencies
        self.save_square_current_set(protocol=protocol,
                                     square_low=0,
                                     square_high=sub_rheobase,
                                     square_steps=2,
                                     delay=delay,
                                     post_delay=0,
                                     duration=1000)

        # The rest are used to quantify CVODE step frequency per spike
        self.save_square_current_set(protocol=protocol,
                                     square_low=rheobase,
                                     square_high=rheobase*1.5,
                                     square_steps=11,
                                     delay=delay,
                                     post_delay=0,
                                     duration=1000)

        # Restore the integration method to as it was before
        self.config.cvode_active = orig_cvode


    def save_dt_sensitivity_set(self,
                                rheobase,
                                protocol="DT_SENSITIVITY",
                                steps_per_ms_set=[1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
                                save_max_stable_dt=True):
        """
        :param rheobase: Cell rheobase current
        :param protocol: The label of the protocol to use when saving the waveform to DB
        :param steps_per_ms_set: A sequence (power of 2 works well to ensure values can be compared at same time points
        :param save_max_stable_dt: Record the largest dt that does not blow up the simulation
        :return: Nothing
        """

        noise_pickle_file = "dtSensitivity.pickle"

        # Cache the files - they're slow to load
        if noise_pickle_file not in self.pickle_file_cache:
            with open(os.path.join("..", "..", noise_pickle_file), "r") as f:
                self.pickle_file_cache[noise_pickle_file] = cPickle.load(f)

        def get_current_ti():
            noise = self.pickle_file_cache[noise_pickle_file]

            # 100 ms of rest with 50 ms of 0.75 RB square
            ramp_ti = [
                (0,     0),
                (100,   0),
                (100,   0.75 * rheobase),
                (150,   0.75 * rheobase)
            ]

            # 50 ms of pink noise at 0.75 RB
            ramp_ti += zip((np.array(noise["t"]) + 150.0).tolist(), (np.array(noise["i"]) * rheobase * 0.75).tolist())

            # 100 ms of square at 1.5 RB
            ramp_ti += [
                (200, 1.5 * rheobase),
                (300, 1.5 * rheobase),
            ]

            # Another 50 ms of pink noise at 1.5 RB
            ramp_ti += zip((np.array(noise["t"]) + 300.0).tolist(), (np.array(noise["i"]) * rheobase * 1.5).tolist())
            ramp_ti += zip((np.array(noise["t"]) + 350.0).tolist(), (np.array(noise["i"]) * rheobase * 1.5).tolist())

            # 50 ms of square at -0.25 RB
            ramp_ti += [
                (400, -0.25 * rheobase),
                (450, -0.25 * rheobase),
            ]

            # Another 50 ms of pink noise at -0.25 RB
            ramp_ti += zip((np.array(noise["t"]) + 350.0).tolist(), (np.array(noise["i"]) * rheobase * -0.25).tolist())

            # Finally a 100ms recovery
            ramp_ti += [
                (500, 0),
                (600, 0),
            ]

            return zip(*ramp_ti)

        smalest_dt_result = None
        max_stable_dt = 0

        for steps_per_ms in steps_per_ms_set:
            try:
                result = self.get_arb_current_response(delay=0,
                                                       duration=600,
                                                       post_delay=0,
                                                       get_current_ti=get_current_ti,
                                                       restore_state=False,
                                                       dt=1.0/steps_per_ms,
                                                       sampling_period=1.0/steps_per_ms)


                # Save the smallest dt waveform
                if steps_per_ms == max(steps_per_ms_set):
                    # Comparing everything to smallest dt waveform - its error is 0
                    result["error"] = 0
                    smalest_dt_result = result

                    # Compute the range of the waveform voltages
                    smalest_dt_result["range"] = max(result["v"]) - min(result["v"])

                    # Subsample the waveform using largest dt interval (1ms)
                    # (these values will be compared across waveforms)
                    try:
                        smalest_dt_result["v_sub"] = np.array(smalest_dt_result["v"])[::max(steps_per_ms_set)]
                        smalest_dt_result["t_sub"] = np.array(smalest_dt_result["t"])[::max(steps_per_ms_set)]
                    except:
                        pass

                else:
                    # Error is average of differences from baseline waveform expressed as percentages of the waveform range
                    result["error"] = self.compute_waveform_error(smalest_dt_result["v_sub"],
                                                                  smalest_dt_result["range"],
                                                                  np.array(result["v"])[::steps_per_ms])

                if save_max_stable_dt:
                    max_stable_dt = 1.0/steps_per_ms
                    max_stable_dt_error = result["error"]

                dt_str = str(1.0/steps_per_ms) + " ms"

                self.save_tvi_plot(label=protocol,
                                   case=dt_str,
                                   tvi_dict=result)

                self.save_vi_waveforms(protocol=protocol,
                                       label=dt_str,
                                       tvi_dict=result)

            except Exception:
                break

        if save_max_stable_dt:
            print("Saving max stable time step " + str(max_stable_dt))
            self.model_record.Max_Stable_DT = max_stable_dt
            self.model_record.Max_Stable_DT_Error = max_stable_dt_error
            self.model_record.save()

        # Clear the cache for this file
        self.pickle_file_cache.pop(noise_pickle_file)

        return smalest_dt_result

    def compute_waveform_error(self, lowest_error_v, lowest_error_range, v):
        return np.average(np.abs(v - lowest_error_v) / lowest_error_range * 100.0)


    def interpolate_time_signal(self, t, signal, new_interval):
        from scipy.interpolate import interp1d
        signal_function = interp1d(t, signal, kind="cubic", fill_value='extrapolate')

        new_t = np.arange(min(t),max(t),step=new_interval).tolist()
        new_signal = signal_function(new_t).tolist()

        return new_t, new_signal

    def save_noise_response_set(self,
                                protocol,
                                delay,
                                duration,
                                post_delay,
                                rheobase,
                                noise_pickle_file,
                                multiples,
                                meta_protocol=None,
                                restore_state=False):

        # Cache the files - they're slow to load
        if noise_pickle_file not in self.pickle_file_cache:
            with open(os.path.join("..", "..", noise_pickle_file), "r") as f:
                print('Reading noise .pickle file...')
                signal_orig =  cPickle.load(f)
                t_int, i_int = self.interpolate_time_signal(signal_orig['t'], signal_orig['i'], self.config.dt)
                self.pickle_file_cache[noise_pickle_file] = { 't':t_int, 'i': i_int }
                signal_orig = None
                print('DONE')


        def get_current_ti():
            noise = self.pickle_file_cache[noise_pickle_file]

            ramp_t = [0, delay] + (np.array(noise["t"]) + delay).tolist() + [delay + duration,
                                                                             delay + duration + post_delay]
            ramp_i = [0, 0] + (np.array(noise["i"]) * rheobase * multiple).tolist() + [0, 0]

            return ramp_t, ramp_i

        for multiple in multiples:
            result = self.get_arb_current_response(delay=delay,
                                                   duration=duration,
                                                   post_delay=post_delay,
                                                   get_current_ti=get_current_ti,
                                                   restore_state=restore_state)

            multiple_str = str(multiple) + "xRB"

            self.save_tvi_plot(label=protocol,
                               case=(meta_protocol if meta_protocol is not None else "") + " " + multiple_str,
                               tvi_dict=result)

            self.save_vi_waveforms(protocol=protocol,
                                   label=multiple_str,
                                   meta_protocol=meta_protocol,
                                   tvi_dict=result)

        # Clear the cache for this file
        self.pickle_file_cache.pop(noise_pickle_file)

    def save_arb_current(self,
                         protocol,
                         delay,
                         duration,
                         get_current_ti,
                         meta_protocol=None,
                         label=None,
                         restore_state=False):

        result = self.get_arb_current_response(delay=delay,
                                               duration=duration,
                                               get_current_ti=get_current_ti,
                                               restore_state=restore_state)

        self.save_tvi_plot(label=protocol,
                           case=(meta_protocol if meta_protocol is not None else "") + " " + (label if label is not None else ""),
                           tvi_dict=result)

        self.save_vi_waveforms(protocol=protocol,
                               label=label,
                               tvi_dict=result)

    def get_arb_current_response(self,
                                 delay,
                                 duration,
                                 get_current_ti,
                                 post_delay=0,
                                 test_condition=None,
                                 restore_state=False,
                                 dt=None,
                                 sampling_period=None):

        def arb_current_protocol(flag):
            self.time_flag = flag

            if dt is not None:
                self.config.cvode_active = 0
                self.config.dt = dt

            if sampling_period is not None:
                self.config.collection_period_ms = sampling_period

            h = self.build_model()

            # Set up IClamp for arbitrary current
            self.current.dur = 1e9
            self.current.delay = 0

            # Create ramp waveform
            ramp_t, ramp_i = get_current_ti()

            rv = h.Vector(ramp_i)
            tv = h.Vector(ramp_t)

            # Play ramp waveform into the IClamp (last param is continuous=True)
            rv.play(self.current._ref_amp, tv, 1)

            with RunTimer() as timer:
                if restore_state:
                    self.restore_state(keep_events=True)  # Keep events ensures .play() works
                    t, v = self.runFor(duration + post_delay, test_condition)
                else:
                    h.stdinit()
                    t, v = self.runFor(delay + duration + post_delay, test_condition)

            result = {
                "t": t.tolist(),
                "v": v.tolist(),
                "i": self.ic_i_collector.get_values_list(),
                "run_time": timer.get_run_time(),
                "steps": int(self.tvec.size()),
                "cvode_active": int(self.config.cvode_active),
                "dt_or_atol": self.config.abs_tolerance if self.config.cvode_active else self.config.dt
            }

            return result

        runner = NeuronRunner(arb_current_protocol)
        runner.DONTKILL = True
        result = runner.run()
        return result

    def load_model(self):
        # Load cell hoc and get soma
        os.chdir(self.temp_model_path)
        print("Loading NEURON... If this step 'freezes', ensure there are no hung NEURON processes with 'pkill -9 nrn*'")
        from neuron import h, gui
        print("DONE")

        if self.model_record.Publication.Temperature is None:
            print("Using default temperature " + str(self.config.default_temperature))
            h.celsius = self.config.default_temperature
        else:
            print("Using temperature from publication " + str(self.model_record.Publication.Temperature))
            h.celsius = self.model_record.Publication.Temperature

        # Create the cell
        if self.is_abstract_cell():
            self.test_cell = self.get_abstract_cell(h)
        elif len(self.get_hoc_files()) > 0:
            self.test_cell = self.get_cell_with_morphology(h)
        else:
            raise Exception("Could not find cell .hoc or abstract cell .mod file in: " + self.temp_model_path)

        # Get the root sections and try to find the soma
        self.roots = h.SectionList()
        self.roots.allroots()
        self.roots = [s for s in self.roots]
        self.somas = [sec for sec in self.roots if "soma" in sec.name().lower()]
        if len(self.somas) == 1:
            self.soma = self.somas[0]
        elif len(self.somas) == 0 and len(self.roots) == 1:
            self.soma = self.roots[0]
        else:
            raise Exception("Problem finding the soma section")

        return h

    def build_model(self, restore_tolerances=True):
        print("Loading cell: " + self.temp_model_path)
        h = self.load_model()

        # set up stim
        self.current = h.IClamp(self.soma(0.5))
        self.current.delay = 50.0
        self.current.amp = 0
        self.current.dur = 100.0

        self.vc = h.SEClamp(self.soma(0.5))
        self.vc.dur1 = 0


        # Set up variable collectors
        self.t_collector = Collector(self.config.collection_period_ms, h._ref_t)
        if self.cell_record.V_Variable is None:
            self.v_collector = Collector(self.config.collection_period_ms, self.soma(0.5)._ref_v)
        else:
            self.v_collector = Collector(self.config.collection_period_ms, getattr(self.abstract_mod,"_ref_" + self.cell_record.V_Variable))

        self.vc_i_collector = Collector(self.config.collection_period_ms, self.vc._ref_i)
        self.ic_i_collector = Collector(self.config.collection_period_ms, self.current._ref_i)

        self.tvec = h.Vector()
        self.tvec.record(h._ref_t)

        # h.nrncontrolmenu()
        self.nState = h.SaveState()
        self.sim_init()
        self.set_abs_tolerance(self.config.abs_tolerance)

        if not self.is_abstract_cell() and restore_tolerances:
            self.restore_tolerances()

        return h

    def is_abstract_cell(self):
        return len(self.get_hoc_files()) == 0 and len(self.get_mod_files()) == 1

    def get_abstract_cell(self, h):
        cell_mod_file = self.get_mod_files()[0]
        cell_mod_name = cell_mod_file.replace(".mod", "")

        soma = h.Section()
        soma.L = 10
        soma.diam = 10
        soma.cm = 318.31927  # Magic number, see: https://github.com/NeuroML/org.neuroml.export/issues/60

        mod = getattr(h, cell_mod_name)(0.5, sec=soma)

        self.abstract_soma = soma
        self.abstract_mod = mod

        return soma

    def get_cell_with_morphology(self, h):
        cell_hoc_file = self.get_hoc_files()[0]
        cell_template = cell_hoc_file.replace(".hoc", "")
        h.load_file(cell_hoc_file)
        cell = getattr(h, cell_template)()
        return cell

    def sim_init(self):
        from neuron import h
        h.stdinit()
        h.tstop = 1000
        self.current.amp = 0
        self.vc.dur1 = 0

    def setCurrent(self, amp, delay, dur):
        self.current.delay = delay
        self.current.amp = amp
        self.current.dur = dur

    def get_stability_range(self, testLow=-10, testHigh=15):

        print("Searching for UPPER boundary...")
        current_range, found_once = self.find_border(
            lowerLevel=0,
            upperLevel=testHigh,
            current_delay=self.steady_state_delay,
            current_duration=3,
            run_for_after_delay=10,
            test_condition=lambda t, v: False if np.max(np.abs(v)) < 150 else True,
            on_unstable=lambda: True,
            max_iterations=7,
            fig_file="stabilityHigh.png",
            skip_current_delay=False
        )

        high_edge = min(current_range)

        print("Searching for LOWER boundary...")
        current_range, found_once = self.find_border(
            lowerLevel=testLow,
            upperLevel=0,
            current_delay=self.steady_state_delay,
            current_duration=3,
            run_for_after_delay=10,
            test_condition=lambda t, v: True if np.max(np.abs(v)) < 150 else False,
            on_unstable=lambda: False,
            max_iterations=7,
            fig_file="stabilityLow.png",
            skip_current_delay=True
        )

        low_edge = max(current_range)

        return low_edge, high_edge

    def getThreshold(self, minCurrent, maxI):

        def test_condition(t, v):
            num_spikes = self.getSpikeCount(v)
            print("Got " + str(num_spikes) + " spikes")
            return num_spikes > 0

        current_range, found_once = self.find_border(
            lowerLevel=minCurrent,
            upperLevel=maxI,
            current_delay=self.steady_state_delay,
            current_duration=3,
            run_for_after_delay=50,
            test_condition=test_condition,
            max_iterations=10,
            fig_file="threshold.png",
            skip_current_delay=True
        )

        if not found_once:
            raise Exception("Did not find threshold with currents " + str(current_range))

        return current_range

    def recompute_range(iteration,upperLevel,lowerLevel):
        intervals = np.arange(upperLevel,lowerLevel,cpus)
        return intervals

    def list_process(simulate_iteration):
        runner = NeuronRunner(simulate_iteration)
        try:
            found = runner.run()
            return found
        except NumericalInstabilityException:
            if on_unstable is not None:
                found = on_unstable()
                return found
            else:
                return None

    def simulate_iteration(time_flag):
        self.time_flag = time_flag
        h = self.build_model()
        self.restore_state(state_file=state_file)

        self.setCurrent(amp=currentAmp, delay=current_delay, dur=current_duration)
        print("Trying " + str(currentAmp) + " nA...")

        if not test_early:
            t, v = self.runFor(run_for_after_delay)
            found = test_condition(t, v)
        else:
            t, v = self.runFor(run_for_after_delay, test_condition)
            found = test_condition(t, v)

        plt.plot(t, v, label=str(round(currentAmp, 4)) + ", Found: " + str(found))
        plt.legend(loc='upper left')
        plt.savefig(str(iteration) + " " + fig_file)

        print("FOUND" if found else "NOT FOUND")

        return found



    def find_border(self, lowerLevel, upperLevel,
                    current_delay, current_duration,
                    run_for_after_delay, test_condition, max_iterations, fig_file,
                    skip_current_delay=False, on_unstable=None, test_early=False):

        state_file = 'border_state.bin'

        if not skip_current_delay:
            def reach_resting_state(time_flag):
                self.time_flag = time_flag
                self.build_model()

                print("Simulating till current onset...")
                self.sim_init()
                self.setCurrent(amp=0, delay=current_delay, dur=current_duration)
                self.runFor(current_delay)
                self.save_state(state_file=state_file)
                print("Resting state reached. State saved.")

            runner = NeuronRunner(reach_resting_state)
            result = runner.run()

        iterate = True
        iteration = 0
        found_once = False

        upperLevel_start = upperLevel
        lowerLevel_start = lowerLevel

        first_interval = np.arange(lowerLevel_start,upperLevel_start,cpus)
        scattered_iterable = db.from_sequence(first_interval,npartitions=cpus)
        gathered_list = scattered_iterable.map(list_process)
        while iterate:
            scattered_iterable = db.from_sequence(intervals,npartitions=cpus)
            gathered_list = scattered_iterable.map(list_process)
            for g in gathered_list:
                if g == True:
                    break
            intervals = recompute_range(iteration,upperLevel,lowerLevel)
            if found:
                upperLevel = currentAmp
                found_once = True
            else:
                lowerLevel = currentAmp

            iteration = iteration + 1

            if iteration >= max_iterations or lowerLevel == upperLevel_start or upperLevel == lowerLevel_start:
                iterate = False
        current_range = (lowerLevel, upperLevel)
        return current_range, found_once

    def getRheobase(self, minCurrent, maxI):

        def test_condition(t, v):
            return self.getSpikeCount(v) > 0

        current_range, found_once = self.find_border(
            lowerLevel=minCurrent,
            upperLevel=maxI,
            current_delay=self.steady_state_delay,
            current_duration=1000,
            run_for_after_delay=500,
            test_condition=test_condition,
            test_early=True,
            max_iterations=10,
            fig_file="rheobase.png",
            skip_current_delay=True
        )

        if not found_once:
            raise Exception("Did not find rheobase with currents " + str(current_range))

        return current_range

    def getBiasCurrent(self, targetV):
        def bias_protocol(flag):
            self.time_flag = flag
            self.build_model()
            self.sim_init()

            self.vc.amp1 = targetV
            self.vc.dur1 = 10000

            t, v = self.runFor(1000)

            i = self.vc_i_collector.get_values_np()
            crossings = self.getSpikeCount(i, threshold=0)

            if crossings > 2:
                print(
                    "No bias current exists for steady state at " + str(
                        targetV) + " mV membrane potential (only spikes)")
                result = None
            else:
                result = self.vc.i

            self.vc.dur1 = 0

            plt.clf()
            plt.plot(t[np.where(t > 50)], i[np.where(t > 50)],
                     label="Bias Current for " + str(targetV) + "mV = " + str(result))
            plt.legend(loc='upper left')
            plt.savefig("biasCurrent" + str(targetV) + ".png")

            return result

        runner = NeuronRunner(bias_protocol)
        return runner.run()

    def getRestingV(self, run_time, save_resting_state=False):
        def rest_protocol(flag):
            self.time_flag = flag
            self.build_model()
            self.sim_init()

            with RunTimer() as timer:
                t, v = self.runFor(run_time)

            result = {
                "t": t.tolist(),
                "v": v.tolist(),
                "i": self.ic_i_collector.get_values_list(),
                "run_time": timer.get_run_time(),
                "steps": int(self.tvec.size()),
                "cvode_active": int(self.config.cvode_active),
                "dt_or_atol": self.config.abs_tolerance if self.config.cvode_active else self.config.dt
            }

            crossings = self.getSpikeCount(v)

            if crossings > 1:
                print("No rest - cell produces multiple spikes without stimulation.")
                result["rest"] = None
            else:
                result["rest"] = v[-1]

            if save_resting_state:
                self.save_state()

            return result

        runner = NeuronRunner(rest_protocol)
        runner.DONTKILL = True
        result = runner.run()
        return result

    def get_structural_metrics(self):

        def run_struct_analysis():
            h = self.build_model(restore_tolerances=False)

            result = {}

            if self.is_abstract_cell():
                result["section_count"] = 1
                result["compartment_count"] = 1

            else:
                result["section_count"] = len([sec for sec in h.allsec()])
                result["compartment_count"] = self.get_number_of_compartments(h)

            return result

        runner = NeuronRunner(run_struct_analysis, kill_slow_sims=False)
        metrics = runner.run()

        return metrics

    def save_structural_metrics(self):
        metrics = self.get_structural_metrics()

        self.cell_record.Sections = metrics["section_count"]
        self.cell_record.Compartments = metrics["compartment_count"]

        self.cell_record.save()


    def get_id_from_nml_file(self, nml):
        return re.search('<.*cell.*?id.*?=.*?"(.*?)"', nml, re.IGNORECASE).groups(1)[0]

    def get_tv(self):
        from neuron import h
        v_np = self.v_collector.get_values_np()
        t_np = self.t_collector.get_values_np()

        if np.isnan(v_np).any():
            raise NumericalInstabilityException(
                "Simulation is numericaly unstable with dt of " + str(h.dt) + " ms")

        return (t_np, v_np)

    def use_optimal_dt_if_available(self):
        if (self.cell_record.CVODE_Active is None or self.cell_record.CVODE_Active == 0) and self.model_record.Optimal_DT is not None:
            print("Using optimal DT " + str(self.model_record.Optimal_DT))
            self.config.cvode_active = 0
            self.config.dt = self.model_record.Optimal_DT
