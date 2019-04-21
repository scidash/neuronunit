
import os
os.listdir(".")
import copy
import sys
import numpy as np
from numpy import arange
import pyNN
from pyNN.utility import get_simulator, init_logging, normalized_filename
import random
import socket
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import re
try:
    import pyNN.spiNNaker as sim
    spiNNaker = True
except:
    import pyNN.neuron as sim
    spiNNaker = False

if spiNNaker == True:
    import pacman
    print(dir(pacman))
    import pyNN.spiNNaker as sim
    import matplotlib.pyplot as plt
    sim.setup(timestep=1.0, min_delay=1.0)
    from pyNN.spiNNaker import STDPMechanism
    from pyNN.spiNNaker import STDPMechanism, SpikePairRule, AdditiveWeightDependence, FromListConnector
    from pyNN.spiNNaker import Projection, OneToOneConnector
    #from pyNN.spiNNaker import ParameterSpace
    import pyNN.spiNNaker as sim


if spiNNaker == False:
    #from pyNN.random import RandomDistribution, NumpyRNG
    import pyNN.neuron as neuron
    from pyNN.neuron import h	#from pyNN.spiNNaker import h
    from pyNN.neuron import StandardCellType, ParameterSpace
    from pyNN.random import RandomDistribution, NumpyRNG
    from pyNN.neuron import STDPMechanism, SpikePairRule, AdditiveWeightDependence, FromListConnector, TsodyksMarkramSynapse
    from pyNN.neuron import Projection, OneToOneConnector
    import socket
    import pyNN.neuron as sim

    nproc = sim.num_processes()
    nproc = 8
    host_name = socket.gethostname()
    node_id = sim.setup(timestep=0.01, min_delay=1.0)#, **extra)
    print("Host #%d is on %s" % (node_id + 1, host_name))

    threads = 1
    rngseed  = 98765
    parallel_safe = False


from pyNN.random import RandomDistribution, NumpyRNG



def get_sets(xx):
    xx = xx.conn_list
    xx_srcs = set([ int(e[0]) for e in xx ])
    xx_tgs = set([ int(e[1]) for e in xx ])
    return xx_srcs, xx_tgs

def con_check_one(xx_cl,xx_srcs,xx_tgs):
    for xo in xx_cl.conn_list:
        assert xo[0] in xx_srcs
        assert xo[1] in xx_tgs


def net_sim_runner(wg,sim,synpases,current):
    # inputs wg (weight gain factor)
    # outputs neo epys recording vectors.
    if spiNNaker == False:
        import pyNN.neuron as sim
    if spiNNaker == True:
        import pyNN.spiNNaker as sim
    all_cells, pop_exc, pop_inh, NEXC, NINH  = finalize_wiring(conn_ee, conn_ie, conn_ei, conn_ii)
    data,vms,binary_trains,t_spike_axis = run_network(current, tstop, all_cells, pop_exc, pop_inh, NEXC, NINH)
    return (data,vms,binary_trains,t_spike_axis)

def obtain_synapses(wiring_plan):

    rng = NumpyRNG(seed=64754)
    delay_distr = RandomDistribution('normal', [2, 1e-1], rng=rng)
    weight_distr = RandomDistribution('normal', [45, 1e-1], rng=rng)


    flat_iter = [ (i,j,k,xaxis) for i,j in enumerate(filtered) for k,xaxis in enumerate(j) ]
    index_exc = list(set( source for (source,j,target,xaxis) in flat_iter if xaxis==1 or xaxis == 2 ))
    index_inh = list(set( source for (source,j,target,xaxis) in flat_iter if xaxis==-1 or xaxis == -2 ))

    EElist = []
    IIlist = []
    EIlist = []
    IElist = []
    for (source,j,target,xaxis) in flat_iter:
        delay = delay_distr.next()
        weight = 1.0 # will be updated later.
        if xaxis==1 or xaxis == 2:
            if target in index_inh:
                EIlist.append((source,target,delay,weight))
            else:
                EElist.append((source,target,delay,weight))

        if xaxis==-1 or xaxis == -2:
            if target in index_exc:
                IElist.append((source,target,delay,weight))
            else:
                IIlist.append((source,target,delay,weight))

    conn_ee = sim.FromListConnector(EElist)
    conn_ie = sim.FromListConnector(IElist)
    conn_ei = sim.FromListConnector(EIlist)
    conn_ii = sim.FromListConnector(IIlist)

    return (conn_ee, conn_ie, conn_ei, conn_ii,index_exc,index_inh)

def prj_change(prj,wg):
    prj.setWeights(wg)

def prj_check(prj):
    for w in prj.weightHistogram():
        for i in w:
            print(i)



def finalize_wiring(conn_ee, conn_ie, conn_ei, conn_ii):
    ii_srcs, ii_tgs = get_sets(conn_ii)
    ei_srcs, ei_tgs = get_sets(conn_ei)
    ee_srcs, ee_tgs = get_sets(conn_ee)
    ie_srcs, ie_tgs = get_sets(conn_ie)

    _ = con_check_one(conn_ee,ee_srcs, ee_tgs)
    _ = con_check_one(conn_ii,ii_srcs,ii_tgs)
    _ = con_check_one(conn_ei,ei_srcs,ei_tgs)
    _ = con_check_one(conn_ie,ie_srcs,ie_tgs)

    len_es_srcs = len(list(ee_srcs))


    # the network is dominated by inhibitory neurons, which is unusual for modellers.
    # Plot all the Projection pairs as a connection matrix (Excitatory and Inhibitory Connections)
    rng = NumpyRNG(seed=64754)
    delay_distr = RandomDistribution('normal', [2, 1e-1], rng=rng)

    all_cells = sim.Population(len(index_exc)+len(index_inh), sim.Izhikevich(a=0.02, b=0.2, c=-65, d=8, i_offset=0))

    pop_exc = sim.PopulationView(all_cells,index_exc)
    pop_inh = sim.PopulationView(all_cells,index_inh)
    NEXC = len(index_exc)
    NINH = len(index_inh)
    # add random variation into Izhi parameters
    for pe in index_exc:
        pe = all_cells[pe]
        r = random.uniform(0.0, 1.0)
        pe.set_parameters(a=0.02, b=0.2, c=-65+15*r, d=8-r**2, i_offset=0)

    for pi in index_inh:
        pi = all_cells[pi]
        r = random.uniform(0.0, 1.0)
        pi.set_parameters(a=0.02+0.08*r, b=0.25-0.05*r, c=-65, d= 2, i_offset=0)

    [ all_cells[i].get_parameters() for i,_ in enumerate(all_cells) ]
    exc_syn = sim.StaticSynapse(weight = wg, delay=delay_distr)
    assert np.any(conn_ee.conn_list[:,0]) < len_es_srcs
    prj_exc_exc = sim.Projection(all_cells, all_cells, conn_ee, exc_syn, receptor_type='excitatory')
    prj_exc_inh = sim.Projection(all_cells, all_cells, conn_ei, exc_syn, receptor_type='excitatory')
    inh_syn = sim.StaticSynapse(weight = wg, delay=delay_distr)
    delay_distr = RandomDistribution('normal', [1, 100e-3], rng=rng)
    prj_inh_inh = sim.Projection(all_cells, all_cells, conn_ii, inh_syn, receptor_type='inhibitory')
    prj_inh_exc = sim.Projection(all_cells, all_cells, conn_ie, inh_syn, receptor_type='inhibitory')
    inh_distr = RandomDistribution('normal', [1, 2.1e-3], rng=rng)

    prj_change(prj_exc_exc,wg)
    prj_change(prj_exc_inh,wg)
    prj_change(prj_inh_exc,wg)
    prj_change(prj_inh_inh,wg)

    prj_check(prj_exc_exc)
    prj_check(prj_exc_inh)
    prj_check(prj_inh_exc)
    prj_check(prj_inh_inh)



    try:
        others = [prj_exc_exc, prj_exc_inh, inh_syn, prj_inh_inh, prj_inh_exc, inh_distr ]
    except:
        pass

    return ( all_cells, pop_exc, pop_inh, NEXC, NINH )

    all_cells, pop_exc, pop_inh, NEXC, NINH  = finalize_wiring(conn_ee, conn_ie, conn_ei, conn_ii)


    def run_network(current, tstop, all_cells, pop_exc, pop_inh, NEXC, NINH):
        noisee,noisei = current
        pop_exc.inject(noisee)
        pop_inh.inject(noisei)

        ##
        # Setup and run a simulation. Note there is no current injection into the neuron.
        # All cells in the network are in a quiescent state, so its not a surprise that xthere are no spikes
        ##

        arange = np.arange
        all_cells.record(['v','spikes'])  # , 'u'])
        all_cells.initialize(v=-65.0, u=-14.0)
        # === Run the simulation =====================================================
        #tstop = 2000.0
        all_cells.record("spikes")

        sim.run(tstop)
        vms = np.array(data.analogsignals[0].as_array().T)
        cleaned = []
        for i,vm in enumerate(vms):
            if np.max(vm) > 900.0 or np.min(vm) <- 900.0:
            else:
                cleaned.append(vm)
        vms = cleaned
                #vm = s#.as_array()[:,
        cnt = 0
        vm_spiking = []
        vm_not_spiking = []
        spike_trains = []
        binary_trains = []
        for spiketrain in data.spiketrains:
            y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
            # argument edges is the time interval you want to be considered.
            pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,len(ass)))
            spike_trains.append(pspikes)
            if len(spiketrain) > max_spikes:
                max_spikes = len(spiketrain)

            if np.max(ass[spiketrain.annotations['source_id']]) > 0.0:
                vm_spiking.append(vms[spiketrain.annotations['source_id']])
            else:
                vm_not_spiking.append(vms[spiketrain.annotations['source_id']])
            cnt+= 1

        for spiketrain in data.spiketrains:
            x = conv.BinnedSpikeTrain(spiketrain, binsize=1 * pq.ms, t_start=0 * pq.s)
            binary_trains.append(x)
        end_floor = np.floor(float(mdf1.t_stop))
        dt = float(mdf1.t_stop) % end_floor
        data.t_start
        #v = mdf1.take_slice_of_analogsignalarray_by_unit()
        t_spike_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)

        return data,vms,binary_trains,t_spike_axis



def get_dummy_synapses():
    try:
        os.system('wget https://github.com/russelljjarvis/HippNetTE/blob/master/internal_connectivities.p?raw=true')
        with open('internal_connectivities.p?raw=true','rb') as f:
            conn_ee,conn_ie,conn_ei,conn_ii,index_exc,index_inh = pickle.load(f)
            synapses = (conn_ee, conn_ie, conn_ei, conn_ii,index_exc,index_inh)
    except:

        # Get some hippocampus connectivity data, based on a conversation with
        # academic researchers on GH:
        # https://github.com/Hippocampome-Org/GraphTheory/issues?q=is%3Aissue+is%3Aclosed
        # scrape hippocamome connectivity data, that I intend to use to program neuromorphic hardware.
        # conditionally get files if they don't exist.
        # This is literally the starting point of the connection map
        path_xl = '_hybrid_connectivity_matrix_20171103_092033.xlsx'
        if not os.path.exists(path_xl):
            os.system('wget https://github.com/Hippocampome-Org/GraphTheory/files/1657258/_hybrid_connectivity_matrix_20171103_092033.xlsx')
        xl = pd.ExcelFile(path_xl)
        dfall = xl.parse()
        dfall.loc[0].keys()
        dfm = dfall.as_matrix()
        rcls = dfm[:,:1] # real cell labels.
        rcls = rcls[1:]
        rcls = { k:v for k,v in enumerate(rcls) } # real cell labels, cast to dictionary

        pd.DataFrame(rcls).to_csv('cell_names.csv', index=False)
        filtered = dfm[:,3:]
        wire_plan = filtered[1:]
        (conn_ee, conn_ie, conn_ei, conn_ii,index_exc,index_inh) = obtain_synapses(wire_plan)
        synapses = (conn_ee, conn_ie, conn_ei, conn_ii,index_exc,index_inh)
    return synapsess

(data,vms,binary_trains,t_spike_axis) = net_sim_runner(wg,sim,synpases,current)

    # with open('internal_connectivities.p','wb') as f:
    #    pickle.dump([conn_ee,conn_ie,conn_ei,conn_ii,index_exc,index_inh],f,protocol=2)


#data = sim_runner(0.5,sim)

if not os.path.exists("pickles"):
    os.mkdir("pickles")

with open('pickles/qi'+str(wg)+'.p', 'wb') as f:
    pickle.dump(data,f)


import pandas as pd
from scipy.sparse import coo_matrix
import pickle

def data_dump(plot_inhib,plot_excit,plot_EE,plot_IE,plot_II,plot_EI,filtered):
    num_exc = [ i for i,e in enumerate(plot_excit) if sum(e) > 0 ]
    num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]
    assert num_inh > num_exc

    assert len(num_exc) < ml
    assert len(num_inh) < ml

    assert np.sum(plot_inhib) > np.sum(plot_excit)

    with open('cell_indexs.p','wb') as f:
        returned_list = [index_exc, index_inh]
        pickle.dump(returned_list,f)

    with open('graph_inhib.p','wb') as f:
       pickle.dump(plot_inhib,f, protocol=2)

    with open('graph_excit.p','wb') as f:
       pickle.dump(plot_excit,f, protocol=2)

    pd.DataFrame(plot_EE).to_csv('ee.csv', index=False)
    pd.DataFrame(plot_IE).to_csv('ie.csv', index=False)
    pd.DataFrame(plot_II).to_csv('ii.csv', index=False)
    pd.DataFrame(plot_EI).to_csv('ei.csv', index=False)


    m = np.matrix(filtered[1:])

    bool_matrix = np.add(plot_excit,plot_inhib)
    with open('bool_matrix.p','wb') as f:
       pickle.dump(bool_matrix,f, protocol=2)

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    Gexc_ud = nx.Graph(plot_excit)
    avg_clustering = nx.average_clustering(Gexc_ud)#, nodes=None, weight=None, count_zeros=True)[source]

    rc = nx.rich_club_coefficient(Gexc_ud,normalized=False)
    print('This graph structure as rich as: ',rc[0])
    gexc = nx.DiGraph(plot_excit)

    gexcc = nx.betweenness_centrality(gexc)
    top_exc = sorted(([ (v,k) for k, v in dict(gexcc).items() ]), reverse=True)

    in_degree = gexc.in_degree()
    top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
    in_hub = top_in[-1][1]
    out_degree = gexc.out_degree()
    top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
    out_hub = top_out[-1][1]
    mean_out = np.mean(list(out_degree.values()))
    mean_in = np.mean(list(in_degree.values()))

    mean_conns = int(mean_in + mean_out/2)

    k = 2 # number of neighbouig nodes to wire.
    p = 0.25 # probability of instead wiring to a random long range destination.
    ne = len(plot_excit)# size of small world network
    small_world_ring_excit = nx.watts_strogatz_graph(ne,mean_conns,0.25)



    k = 2 # number of neighbouring nodes to wire.
    p = 0.25 # probability of instead wiring to a random long range destination.
    ni = len(plot_inhib)# size of small world network
    small_world_ring_inhib   = nx.watts_strogatz_graph(ni,mean_conns,0.25)

    with open('cell_names.p','wb') as f:
        pickle.dump(rcls,f)


    plot_EE = np.zeros(shape=(ml,ml), dtype=bool)
    plot_II = np.zeros(shape=(ml,ml), dtype=bool)
    plot_EI = np.zeros(shape=(ml,ml), dtype=bool)
    plot_IE = np.zeros(shape=(ml,ml), dtype=bool)

    for i in EElist:
        plot_EE[i[0],i[1]] = int(0)
        if i[0]!=i[1]: # exclude self connections
            plot_EE[i[0],i[1]] = int(1)
            pre_exc.append(i[0])
            post_exc.append(i[1])

    for i in IIlist:
        plot_II[i[0],i[1]] = int(0)
        if i[0]!=i[1]:
            plot_II[i[0],i[1]] = int(1)
            pre_inh.append(i[0])
            post_inh.append(i[1])

    for i in IElist:
        plot_IE[i[0],i[1]] = int(0)
        if i[0]!=i[1]: # exclude self connections
            plot_IE[i[0],i[1]] = int(1)
            pre_inh.append(i[0])
            post_inh.append(i[1])

    for i in EIlist:
        plot_EI[i[0],i[1]] = int(0)
        if i[0]!=i[1]:
            plot_EI[i[0],i[1]] = int(1)
            pre_exc.append(i[0])
            post_exc.append(i[1])

    plot_excit = plot_EI + plot_EE
    plot_inhib = plot_IE + plot_II


#iter_sim = [ (i,wg) for i,wg in enumerate(weight_gain_factors.keys()) ]
