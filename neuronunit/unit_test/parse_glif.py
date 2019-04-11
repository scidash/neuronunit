usage='''
Provenance: This file is originally from
https://github.com/vrhaynes/AllenInstituteNeuroML
https://github.com/OpenSourceBrain/AllenInstituteNeuroML
It is authored by pgleeson@github.com, vrhaynes@github.com and russelljjarvis@github.com

This file can be used to generate LEMS components for each of a number of GLIF models

Usage:

    python parse_glif.py -all

'''

import sys
import os
import json

from pyneuroml import pynml

def generate_lems(glif_package, curr_pA=None, show_plot=False):
    if curr_pA == None:
        curr_pA = 10
    glif_dir = os.getcwd()

    model_metadata,neuron_config,ephys_sweeps = glif_package

    template_cell = '''<Lems>

      <%s %s/>

    </Lems>
    '''

    type = '???'
    print(model_metadata['name'])
    if '(LIF)' in model_metadata['name']:
        type = 'glifCell'
    if '(LIF-ASC)' in model_metadata['name']:
        type = 'glifAscCell'
    if '(LIF-R)' in model_metadata['name']:
        type = 'glifRCell'
    if '(LIF-R-ASC)' in model_metadata['name']:
        type = 'glifRAscCell'
    if '(LIF-R-ASC-A)' in model_metadata['name']:
        type = 'glifRAscATCell'

    cell_id = 'GLIF_%s'%glif_dir

    #model_metadata['name']

    attributes = ""

    attributes +=' id="%s"'%cell_id
    attributes +='\n            C="%s F"'%neuron_config["C"]
    attributes +='\n            leakReversal="%s V"'%neuron_config["El"]
    attributes +='\n            reset="%s V"'%neuron_config["El"]
    attributes +='\n            thresh="%s V"'%( float(neuron_config["th_inf"]) * float(neuron_config["coeffs"]["th_inf"]))
    attributes +='\n            leakConductance="%s S"'%(1/float(neuron_config["R_input"]))

    if 'Asc' in type:
        attributes +='\n            tau1="%s s"'%neuron_config["asc_tau_array"][0]
        attributes +='\n            tau2="%s s"'%neuron_config["asc_tau_array"][1]
        attributes +='\n            amp1="%s A"'% ( float(neuron_config["asc_amp_array"][0]) * float(neuron_config["coeffs"]["asc_amp_array"][0]) )
        attributes +='\n            amp2="%s A"'% ( float(neuron_config["asc_amp_array"][1]) * float(neuron_config["coeffs"]["asc_amp_array"][1]) )

    if 'glifR' in type:
        attributes +='\n            bs="%s per_s"'%neuron_config["threshold_dynamics_method"]["params"]["b_spike"]
        attributes +='\n            deltaThresh="%s V"'%neuron_config["threshold_dynamics_method"]["params"]["a_spike"]
        attributes +='\n            fv="%s"'%neuron_config["voltage_reset_method"]["params"]["a"]
        attributes +='\n            deltaV="%s V"'%neuron_config["voltage_reset_method"]["params"]["b"]

    if 'glifRAscATCell' in type:
        attributes +='\n            bv="%s per_s"'%neuron_config["threshold_dynamics_method"]["params"]["b_voltage"]
        attributes +='\n            a="%s per_s"'%neuron_config["threshold_dynamics_method"]["params"]["a_voltage"]


    file_contents = template_cell%(type, attributes)

    print(file_contents)

    #cell_file_name = '%s.xml'%(cell_id)
    cell_file_name = '{0}{1}.xml'.format(os.getcwd(),str(model_metadata['name']))
    cell_file = open(cell_file_name,'w')
    cell_file.write(file_contents)
    cell_file.close()
    return cell_file_name

    '''
    import opencortex.build as oc


    nml_doc, network = oc.generate_network("Test_%s"%glif_dir)

    pop = oc.add_single_cell_population(network,
                                         'pop_%s'%glif_dir,
                                         cell_id)


    pg = oc.add_pulse_generator(nml_doc,
                           id="pg0",
                           delay="100ms",
                           duration="1000ms",
                           amplitude="%s pA"%curr_pA)


    oc.add_inputs_to_population(network,
                                "Stim0",
                                pop,
                                pg.id,
                                all_cells=True)



    nml_file_name = '%s.net.nml'%network.id
    oc.save_network(nml_doc, nml_file_name, validate=True)


    thresh = 'thresh'
    if 'glifR' in type:
        thresh = 'threshTotal'

    lems_file_name = oc.generate_lems_simulation(nml_doc,
                                network,
                                nml_file_name,
                                include_extra_lems_files = [cell_file_name,'../GLIFs.xml'],
                                duration =      1200,
                                dt =            0.01,
                                gen_saves_for_quantities = {'thresh.dat':['pop_%s/0/GLIF_%s/%s'%(glif_dir,glif_dir,thresh)]},
                                gen_plots_for_quantities = {'Threshold':['pop_%s/0/GLIF_%s/%s'%(glif_dir,glif_dir,thresh)]})

    results = pynml.run_lems_with_jneuroml(lems_file_name,
                                     nogui=True,
                                     load_saved_data=True)

    print("Ran simulation; results reloaded for: %s"%results.keys())

    info = "Model %s; %spA stimulation"%(glif_dir,curr_pA)

    times = [results['t']]
    vs = [results['pop_%s/0/GLIF_%s/v'%(glif_dir,glif_dir)]]
    labels = ['LEMS - jNeuroML']

    original_model_v = 'original.v.dat'
    if os.path.isfile(original_model_v):
        data, indices = pynml.reload_standard_dat_file(original_model_v)
        times.append(data['t'])
        vs.append(data[0])
        labels.append('Allen SDK')


    pynml.generate_plot(times,
                        vs,
                        "Membrane potential; %s"%info,
                        xaxis = "Time (s)",
                        yaxis = "Voltage (V)",
                        labels = labels,
                        grid = True,
                        show_plot_already=False,
                        save_figure_to='Comparison_%ipA.png'%(curr_pA))

    times = [results['t']]
    vs = [results['pop_%s/0/GLIF_%s/%s'%(glif_dir,glif_dir,thresh)]]
    labels = ['LEMS - jNeuroML']

    original_model_th = 'original.thresh.dat'
    if os.path.isfile(original_model_th):
        data, indeces = pynml.reload_standard_dat_file(original_model_th)
        times.append(data['t'])
        vs.append(data[0])
        labels.append('Allen SDK')


    pynml.generate_plot(times,
                        vs,
                        "Threshold; %s"%info,
                        xaxis = "Time (s)",
                        yaxis = "Voltage (V)",
                        labels = labels,
                        grid = True,
                        show_plot_already=show_plot,
                        save_figure_to='Comparison_Threshold_%ipA.png'%(curr_pA))

    readme =
    '''
