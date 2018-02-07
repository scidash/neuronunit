"""Tools for plotting (contributed by Blue Brain Project)"""

import matplotlib
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import colorsys
import numpy
import collections

import os
import matplotlib
import pandas as pd

#import matplotlib.pyplot as plt
import numpy as np


import sys
KERNEL = ('ipykernel' in sys.modules)

import numpy as np
import pandas as pd

#import bs4
#from IPython.display import HTML,Javascript,display

def adjust_spines(ax, spines, color='k', d_out=10, d_down=None):

    if d_down in [None,[]]:
        d_down = d_out

    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    for loc, spine in ax.spines.items():
        if loc in spines:
            if loc == 'bottom':
                spine.set_position(('outward', d_down))  # outward by 10 points
            else:
                spine.set_position(('outward', d_out))  # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_visible(False) # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')

        if color != 'k':

            ax.spines['left'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)


    elif 'right' not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')

        if color != 'k':

            ax.spines['right'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)

    if 'bottom' in spines:
        #pass
        ax.xaxis.set_ticks_position('bottom')
        #ax.axes.get_xaxis().set_visible(True)

    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)


def light_palette(color, n_colors=6, reverse=False, lumlight=0.8, light=None):

    rgb = mplcol.colorConverter.to_rgb(color)

    if light is not None:
        light = mplcol.colorConverter.to_rgb(light)
    else:
        vals = list(colorsys.rgb_to_hls(*rgb))
        vals[1] = lumlight
        light = colorsys.hls_to_rgb(*vals)

    colors = [color, light] if reverse else [light, color]
    pal = mplcol.LinearSegmentedColormap.from_list("blend", colors)

    palette = pal(numpy.linspace(0, 1, n_colors))

    return palette


def tiled_figure(figname='', frames=1, columns=2,
                 figs=collections.OrderedDict(), axs=None, orientation='page',
                 width_ratios=None, height_ratios=None, top=0.97,
                 bottom=0.05, left=0.07, right=0.97, hspace=0.6,
                 wspace=0.2, dirname=''):

    if figname not in figs:

        if axs is None:
            axs = []

        if orientation == 'landscape':
            figsize=(297/25.4, 210/25.4)
        elif orientation == 'page':
            figsize=(210/25.4, 297/25.4)

        # Some of these are not valid params in the current version
        # of matplotlib so I've commented them out.
        params = {'backend': 'ps',
                  'axes.labelsize': 6,
                  'axes.linewidth' : 0.5,
                  #'title.fontsize': 8,
                  #'text.fontsize': 8,
                  'font.size': 8,
                  'axes.titlesize': 8,
                  'legend.fontsize': 8,
                  'xtick.labelsize': 6,
                  'ytick.labelsize': 6,
                  'legend.borderpad': 0.2,
                  #'legend.linewidth': 0.1,
                  'legend.loc': 'best',
                  #'legend.ncol': 4,
                  'text.usetex': False,
                  'pdf.fonttype': 42,
                  'figure.figsize': figsize}
        matplotlib.rcParams.update(params)

        fig = plt.figure(figname, facecolor='white')
        figs[figname] = {}
        figs[figname]['fig'] = fig
        figs[figname]['dirname'] = dirname

        if width_ratios is None:
            width_ratios=[1]*columns

        rows = int(numpy.ceil(frames/float(columns)))
        if height_ratios is None:
            height_ratios=[1]*rows

        gs = matplotlib.gridspec.GridSpec(rows, columns, height_ratios=height_ratios, width_ratios=width_ratios)
        gs.update(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

        for fi in range(frames):
            axs.append(fig.add_subplot(gs[int(fi/columns), int(fi%columns)]))
            adjust_spines(axs[-1], ['left', 'bottom'], d_out=0)

        figs[figname]['axs'] = axs

    else:
        axs = figs[figname]['axs']

    return axs

def plot_surface(model_param0,model_param1,td,history):
    '''
    Inputs should be are two model parameter keys two parameters are needed to constitute a surface,
    The third positional argument is a list or dictionary of all the model parameters used, such that the
    provided two parameters can be located in the dictionary.
    that are parameters see new function definition below
    '''
    import numpy as np
    import matplotlib
    matplotlib.rcParams.update({'font.size':16})
    import matplotlib.pyplot as plt
    x = [ i for i,j in enumerate(td) if str(model_param0) == j ][0]
    y = [ i for i,j in enumerate(td) if str(model_param1) == j ][0]
    all_inds = history.genealogy_history.values()
    sums = np.array([np.sum(ind.fitness.values) for ind in all_inds])
    xs = np.array([ind[x] for ind in all_inds])
    ys = np.array([ind[y] for ind in all_inds])
    min_ys = ys[np.where(sums == np.min(sums))]
    min_xs = xs[np.where(sums == np.min(sums))]
    plt.clf()
    fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')
    trip_axis = ax_trip.tripcolor(xs,ys,sums,20,norm=matplotlib.colors.LogNorm())
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue',label='global minima')
    fig_trip.colorbar(trip_axis, label='Sum of Objective Errors ')
    ax_trip.set_xlabel('Parameter '+str((td[x])))
    ax_trip.set_ylabel('Parameter '+str((td[y])))
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue')
    fig_trip.tight_layout()
    if not KERNEL:
        plt.savefig('surface'+str((td[z])+str('.png')))
    else:
        plt.show()


# shadow(dtcpop,best_vm):#This method must be pickle-able for ipyparallel to work.
# A method to plot the best and worst candidate solution waveforms side by side
# Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
# is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
# value, that was found in a previous rheobase search.
# Outputs: This method only has side effects, not datatype outputs from the method.
# The most important side effect being a plot in png format.

def use_dtc_to_plotting(dtcpop):
    from neuronunit.capabilities import spike_functions
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    stored_min = []
    stored_max = []
    for dtc in dtcpop[1:-1]:
        plt.plot(dtc.tvec, dtc.vm0,linewidth=3.5, color='grey')
        stored_min.append(np.min(dtc.vm0))
        stored_max.append(np.max(dtc.vm0))
    plt.plot(dtcpop[0].tvec,dtcpop[0].vm0,linewidth=1, color='red',label='best candidate')
    plt.legend()
    plt.ylabel('$V_{m}$ mV')
    plt.xlabel('ms')

    if not KERNEL:
        plt.savefig(str('rheobase')+'vm_versus_t.png', format='png')#, dpi=1200)
    else:
        plt.show()


    plt.clf()
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    stored_min = []
    stored_max = []
    for dtc in dtcpop[1:-1]:
        plt.plot(dtc.tvec, dtc.vm1,linewidth=3.5, color='grey')
        stored_min.append(np.min(dtc.vm1))
        stored_max.append(np.max(dtc.vm1))

    plt.plot(dtcpop[0].tvec,dtcpop[0].vm1,linewidth=1, color='red',label='best candidate')
    plt.legend()
    plt.ylabel('$V_{m}$ mV')
    plt.xlabel('ms')
    if not KERNEL:
        plt.savefig(str('AP_width_test')+'vm_versus_t.png', format='png')#, dpi=1200)
    else:
        plt.show()


def plot_history_statistically(log,hof,gen_vs_hof):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook
    Outputs: A graph that is either visualized in the context of an ipython notebook, if
    a notebook kernel is present.
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    gen_numbers =[ i for i in range(0,len(log.select('gen'))) ]
    mean = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('avg')])
    std = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('std')])
    minimum = np.array([ np.sqrt(np.mean(np.square(i))) for i in log.select('min')])
    best_line = np.array([ np.sqrt(np.mean(np.square(list(p.fitness.values))))  for p in hof])
    dynamic_hof = np.array([ sum(gen_vs_hof[i].fitness) for i in gen_vs_hof ])
    #best_changed = np.array([ np.sqrt(np.mean(((p))))  for p in gen_vs_hof ])

    blg = [ best_line[h] for i, h in enumerate(gen_numbers) ]
    print(blg)

    print(len(best_changed),len(gen_numbers))

    stdminus = mean - std
    stdplus = mean + std
    try:
        assert len(gen_numbers) == len(stdminus) == len(stdplus)
    except:
        pass

    axes.plot(
        gen_numbers,
        mean,
        color='black',
        linewidth=2,
        label='population average')
    axes.fill_between(gen_numbers, stdminus, stdplus)
    axes.plot(gen_numbers, blg,'y--', linewidth=2,  label='grid search error')
    axes.plot(gen_numbers, best_changed, 'go', linewidth=2, label='hall of fame error')

    #try:
    axes.plot(gen_numbers, dynamic_hof, 'y^', linewidth=2, label='hall of fame error')
    #except:
    #    pass
    axes.plot(gen_numbers, stdminus, label='std variation lower limit')
    axes.plot(gen_numbers, stdplus, label='std variation upper limit')
    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([-0.1, maximum])
    axes.legend()

    fig.tight_layout()
    if not KERNEL:
        fig.savefig('Izhikevich_evolution_components.png', format='png')#, dpi=1200)
    else:
        fig.show()


def plotly_graph(history,vmhistory):
	# TODO experiment with making the plot output style a dendro
	# dendrograms
    import os
    os.system('conda install -c anaconda python-graphviz plotly')
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    import graphviz

    import matplotlib as mpl
    # setting of an appropriate backend.

    mpl.use('Agg')

    #from plotly.graph_objs import *
    import matplotlib.pyplot as plt
    import numpy as np

    from networkx.drawing.nx_agraph import graphviz_layout
    import plotly
    import plotly.plotly as py
    import networkx as nx
    import networkx
    G = networkx.DiGraph(history.genealogy_tree)
    G = G.reverse()
    labels = {}
    for i in G.nodes():
        labels[i] = i
    node_colors = np.log([ np.sum(history.genealogy_history[i].fitness.values) for i in G ])
    import networkx as nx


    positions = graphviz_layout(G, prog="dot")

    # adjust circle size was
    # 1 now 1.5
    dmin=1.5
    ncenter=0
    for n in positions:
        x,y=positions[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
    edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

    for edge in G.edges():
        source = G.nodes()[edge[0]-1]
        target = G.nodes()[edge[1]-1]

        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            #colorscale='YIGnBu',
            colorscale='Hot',

            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Summed Error of the gene',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():

        x,y = positions[G.nodes()[node-1]]
        node_trace['x'].append(x)
        node_trace['y'].append(y)

    for k, node in enumerate(G):
        node_trace['marker']['color'].append(node_colors[k])
        try:
            node_info = 'gene id: {0} threshold current {1} pA model attributes {2}'.format( str(int(k)), str(vmhistory[k].rheobase), str(vmhistory[k].attrs))
        except:
            node_info = 'gene id: {0} threshold currrent {1} pA model attributes {2} pA '.format( str(int(k)), str(vmhistory[k].rheobase), str(vmhistory[k].attrs))
        node_trace['text'].append(node_info)

    fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Genetic History Inheritence Tree with NeuronUnit/DEAP',
                titlefont=dict(size=22),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="</a>",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=True)))
    py.sign_in('RussellJarvis','FoyVbw7Ry3u4N2kCY4LE')
    py.iplot(fig, filename='networkx_new_DEAP.svg',image='svg')


def pca(best_worst,vmpop,fitnesses,td):

#def pca(final_population,vmpop,td):

    '''
    Principle Component Analysis.
    Use PCA to find out which of the model parameters are
    the most resonsible for causing variations in error/fitness
    '''

    import plotly
    import plotly.plotly as py
    from plotly.graph_objs import Scatter, Marker, Line, Data, Layout, Figure, XAxis, YAxis
    from sklearn.preprocessing import StandardScaler
    # need to standardise the data since each parameter consists of different variables.
    #p_plus_f = [ ind.append(np.sum(fitnesses[k])) for k, ind in enumerate(final_population) ]
    errors = np.array([ np.sum(f.fitness.values) for f in enumerate(best_worst) ])

    #p_plus_f = final_population
    #X_std = StandardScaler().fit_transform(final_population)
    X_std = StandardScaler().fit_transform(best_worst)

    from sklearn.decomposition import PCA as sklearnPCA
    sklearn_pca = sklearnPCA(n_components=10)
    Y_sklearn = sklearn_pca.fit_transform(X_std)

    # Make a list of (eigenvalue, eigenvector) tuples
    cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))
    Y = X_std.dot(errors)
    '''
    points = []
    proj0 = []
    proj1 = []

    for ind in final_population:
        proj0.append(ind * Y_sklearn[:,0])
        proj1.append(ind * Y_sklearn[:,1])
    '''
    #x_points = [ ind * Y_sklearn[:,0] for ind in final_population ]
    #y_points = [ ind * Y_sklearn[:,1] for ind in final_population ]

    #iter_list = list(zip(x_points,y_points))

    #for counter,ind in enumerate(final_population):
    for k,v in enumerate(Y_sklearn):
        #print(Y_sklearn[k,0])
        x=Y_sklearn[k,0] #Component 1
        y=Y_sklearn[k,1] #Component 2
        print(x,y,td[k])

        point = Scatter(
        x=Y_sklearn[k,0], #Component 1
        y=Y_sklearn[k,1], #Component 2
        mode='markers',
        name = str(final_pop[counter])+str(td[k]),
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
        points.append(point)

    data = Data(points)
    layout = Layout(xaxis=XAxis(title='PC1', showline=True),
                    yaxis=YAxis(title='PC2', showline=True))
    fig = Figure(data=data, layout=layout)
    py.iplot(fig)
    py.sign_in('RussellJarvis','FoyVbw7Ry3u4N2kCY4LE')
    py.iplot(fig, filename='improved_names.svg',image='svg')

'''

def plot_evaluate(vms_best,vms_worst,names=['best','worst']):#This method must be pickle-able for ipyparallel to work.
    #A method to plot the best and worst candidate solution waveforms side by side


    #Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    #is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    #value, that was found in a previous rheobase search.

    #Outputs: This method only has side effects, not datatype outputs from the method.
    #The most important side effect being a plot in png format.

    import os
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
    vmslist = [vms_best, vms_worst]
    import copy
    ##
    # The attribute 'get_neab.tests[0].prediction'
    # must be declared before use can occur.
    ##
    get_neab.tests[0].prediction = None
    get_neab.tests[0].prediction = {}
    get_neab.tests[0].prediction['value'] = None
    #vms = best_worst[0]
    tests = copy.copy(get_neab.tests)
    for k,v in enumerate(tests):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.style.use('ggplot')
        stored_min = []
        stored_max = []
        sc_for_frame_best = []
        sc_for_frame_worst = []
        for iterator, vms in enumerate(vmslist):
            get_neab.tests[0].prediction['value'] = vms.rheobase * pq.pA

            plt.plot(vms.results[str(v)]['ts'],vms.results[str(v)]['v_m'])#,label=str(v)+str(names[iterator])+str(score))
            plt.xlim(0,float(v.params['injected_square_current']['duration']) )
            stored_min.append(np.min(model.results['vm']))
            stored_max.append(np.max(model.results['vm']))
            plt.ylim(np.min(stored_min),np.max(stored_max))
            plt.tight_layout()
            model.results = None
            plt.ylabel('$V_{m}$ mV')
            plt.xlabel('ms')
        plt.savefig(str('test_')+str(v)+'vm_versus_t.png', format='png', dpi=1200)
        import pandas as pd
        sf_best = pd.DataFrame(sc_for_frame_best)
        sf_worst = pd.DataFrame(sc_for_frame_worst)

'''

'''
def sp_spike_width(best_worst):#This method must be pickle-able for ipyparallel to work.

    #A method to plot the best and worst candidate solution waveforms side by side
    #Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    #is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    #value, that was found in a previous rheobase search.
    #Outputs: This method only has side effects, not datatype outputs from the method.
    #The most important side effect being a plot in png format.

    import os

    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal


    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    ##
    # The attribute 'get_neab.tests[0].prediction'
    # must be declared before use can occur.
    ##
    get_neab.tests[0].prediction = None
    get_neab.tests[0].prediction = {}
    get_neab.tests[0].prediction['value'] = None
    vms = best_worst[0]
    get_neab.tests[0].prediction['value'] = vms.rheobase * pq.pA

    stored_min = []
    stored_max = []
    sc_for_frame_best = []
    sc_for_frame_worst = []

    sindexs = []
    # visualize
    # amplitude tests
    fig, ax = plt.subplots(1, figsize=(10, 5), facecolor='white')

    v = get_neab.tests[5]

    waves = []

    for iterator, vms in enumerate(best_worst):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        assert type(vms.rheobase) is not type(None)

        v.params['injected_square_current']['duration'] = 1000 * pq.ms
        v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
        v.params['injected_square_current']['delay'] = 100 * pq.ms
        import neuron
        model.load_model()

        model.reset_h(neuron)
        model.update_run_params(vms.attrs)
        score = v.judge(model,stop_on_error = False, deep_error = True)

        v_m = model.get_membrane_potential()
        from neuronunit import capabilities as cap
        threshold = cap.spikes2thresholds(v_m)
        ts = model.results['t'] # time signal
        if iterator == 0:
            waves.append(ts)
        waves.append(v_m)
		##
		# since the threshold value derived from
		# capabilities, spike functions, spikes2thresholds
		# has a different precision to
		# the neo analogue signal v_m,
		# there is no: v in v_m that exactly equals
		# threshold, so an approximately equals will have to do
		# 1e-4 is a nominally low tolerance for the approximation.
		##
        threshold_time = [ ts[index] for index,v in enumerate(v_m) if np.abs(float(threshold)-float(v)) < 1e-4 ]
        threshold_time = threshold_time[0]

        dt = float(v_m.sampling_period)

        ts = model.results['t'] # time signal
        st = spike_functions.get_spike_train(v_m) #spike times

        start = int((float(threshold_time)/ts[-1])*len(ts))  # The index corresponding to the time offset, for

	    # when the models v_m crosses its threshold.

        stop = start + int(2500)
        time_sequence = np.arange(start , stop, 1)
        ptvec = np.array(model.results['t'])[time_sequence]

        other_stop = ptvec[-1]-ptvec[0]
        lined_up_time = np.arange(0,other_stop,float(dt))
        pvm = np.array(model.results['vm'])[time_sequence]
        ans = model.get_membrane_potential()

        sw = cap.spike_functions.get_spike_waveforms(ans)
        sa = cap.spike_functions.spikes2amplitudes(sw)

        plt.plot(lined_up_time , pvm, label=str(sa), linewidth=1.5)

    plt.savefig(str('from_threshold_test_')+str(v)+'vm_versus_t.png', format='png', dpi=1200)#
    import pickle
    with open('waveforms.p','wb') as handle:
        pickle.dump(waves,handle)

    # visualize
    # threshold test
    plt.style.use('ggplot')
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(10, 5), facecolor='white')
    v = get_neab.tests[7]
    #if k == 5: # Only interested in InjectedCurrentAPWidthTest this time.
    for iterator, vms in enumerate(best_worst):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        assert type(vms.rheobase) is not type(None)

        v.params['injected_square_current']['duration'] = 1000 * pq.ms
        v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
        v.params['injected_square_current']['delay'] = 100 * pq.ms
        import neuron
        model.load_model()

        model.reset_h(neuron)
        model.update_run_params(vms.attrs)
        score = v.judge(model,stop_on_error = False, deep_error = True)
        v_m = model.get_membrane_potential()
        dt = float(v_m.sampling_period)

        v_m = model.get_membrane_potential()
        ts = model.results['t'] # time signal
        st = spike_functions.get_spike_train(v_m) #spike times
        v_m = model.get_membrane_potential()
        start = int((float(st)/ts[-1])*len(ts))-750  #index offset from spike
        stop = int((float(st)/ts[-1])*len(ts))+1500

        time_sequence = np.arange(start , stop, 1)
        ptvec = np.array(model.results['t'])[time_sequence]
        other_stop = ptvec[-1]-ptvec[0]
        lined_up_time = np.arange(0,other_stop,float(dt))
        pvm = np.array(model.results['vm'])[time_sequence]
        print(len(pvm),len(lined_up_time),float(dt))
        assert len(pvm)==len(lined_up_time)


        if 'value' in v.observation.keys():
            unit_observations = v.observation['value']

        if 'value' in v.prediction.keys():
            unit_predictions = v.prediction['value']


        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']

        if 'mean' in v.prediction.keys():
            unit_predictions = v.prediction['mean']

        to_r_s = unit_observations.units
        unit_predictions = unit_predictions.rescale(to_r_s)
        if len(lined_up_time)==len(pvm):
            plt.plot(lined_up_time , pvm, label=str(unit_predictions), linewidth=1.5)
        #    pass
        #ax[iterator].legend(labels=str(unit_predictions),loc="upper right")
        threshold_line = []# [ float(unit_predictions)
        for i in lined_up_time:
            if i < 1000:
                threshold_line.append(float(unit_predictions))
            else:
                append(0.0)
        plt.plot(lined_up_time ,threshold_line)
        #plt.legend(loc="lower left")
        #score = None
    #plt.legend()
    #fig.text(0.5, 0.04, 'ms', ha='center', va='center')
    #fig.text(0.06, 0.5, '$V_{m}$ mV', ha='center', va='center', rotation='vertical')
    fig.savefig(str('threshold')+str(v)+'vm_versus_t.png', format='png', dpi=1200)#,
    ##
    # Amplitude
    ##
    plt.style.use('ggplot')
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(10, 5), facecolor='white')
    v = get_neab.tests[6]

    for iterator, vms in enumerate(best_worst):
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel

        print(get_neab.LEMS_MODEL_PATH)
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

        assert type(vms.rheobase) is not type(None)

        v.params['injected_square_current']['duration'] = 1000 * pq.ms
        v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
        v.params['injected_square_current']['delay'] = 100 * pq.ms
        import neuron
        model.load_model()

        model.reset_h(neuron)
        model.update_run_params(vms.attrs)
        print(v.params)
        score = v.judge(model,stop_on_error = False, deep_error = True)
        v_m = model.get_membrane_potential()
        dt = float(v_m.sampling_period)

        ts = model.results['t'] # time signal
        st = spike_functions.get_spike_train(v_m) #spike times
        print(st)
        assert len(st) == 1

        start = int((float(st)/ts[-1])*len(ts))-750  #index offset from spike
        stop = int((float(st)/ts[-1])*len(ts))+1500

        time_sequence = np.arange(start , stop, 1)
        ptvec = np.array(model.results['t'])[time_sequence]
        other_stop = ptvec[-1]-ptvec[0]
        lined_up_time = np.arange(0,other_stop,float(dt))

        pvm = np.array(model.results['vm'])[time_sequence]
        print(len(pvm),len(lined_up_time),float(dt))
        assert len(pvm)==len(lined_up_time)

        print(len(pvm),len(lined_up_time))


        if 'value' in v.observation.keys():
            unit_observations = v.observation['value']

        if 'value' in v.prediction.keys():
            unit_predictions = v.prediction['value']


        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']

        if 'mean' in v.prediction.keys():
            unit_predictions = v.prediction['mean']

        to_r_s = unit_observations.units
        unit_predictions = unit_predictions.rescale(to_r_s)
        if len(lined_up_time)==len(pvm):
            plt.plot(lined_up_time , pvm, label=str(unit_predictions), linewidth=1.5)
        #except:
        #    pass
        #plt.legend(loc="lower left")
        #score = None
    #plt.legend()
    #fig.text(0.5, 0.04, 'ms', ha='center', va='center')
    #fig.text(0.06, 0.5, '$V_{m}$ mV', ha='center', va='center', rotation='vertical')
    fig.savefig(str('amplitude')+str(v)+'vm_versus_t.png', format='png', dpi=1200)#,
'''
def not_just_mean(log,hypervolumes):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in png format.

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    gen_numbers = log.select('gen')
    mean_many = np.array(log.select('avg'))
    axes.plot(
        gen_numbers,
        mean_many,
        color='black',
        linewidth=2,
        label='population average')
    axes.plot(
        gen_numbers,
        hypervolumes,
        color='red',
        linewidth=2,
        label='Solution Hypervolume')

    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([0, max(max(mean_many),max(hypervolumes))])
    axes.legend()
    fig.tight_layout()
    fig.savefig('Izhikevich_evolution_just_mean.png', format='png')#, dpi=1200)

def bar_chart(vms,name=None):
    '''
    A method to plot raw predictions
    versus observations
    Outputs: This method only has side effects, not datatype outputs from the method.
    The most important side effect being a plot in png format.
    '''
    import plotly.plotly as py
    from plotly.graph_objs import Bar


    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
    import copy
    from itertools import repeat
    #TODO move install into docker
    #os.system('sudo /opt/conda/bin/pip install cufflinks')
    #import cufflinks as cf

    import plotly.tools as tls
    tls.embed('https://plot.ly/~cufflinks/8')
    from neuronunit.optimization.nsga_parallel import pre_format
    import pandas as pd
    traces = []

    delta = []
    scores = []
    tests = copy.copy(get_neab.tests)
    labels = [ '{0}_{1}'.format(str(t),str(t.observation['value'].units)) for t in tests if 'mean' not in t.observation.keys() ]
    labels.extend([ '{0}_{1}'.format(str(t),str(t.observation['mean'].units))  for t in tests if 'mean' in t.observation.keys() ])
    test_dic = {}
    columns1 = [] # a list of test labels to use as column labels.

    for k,v in enumerate(tests):

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
        assert type(vms.rheobase) is not type(None)

        model.update_run_params(vms.attrs)
        dtc = pre_format(dtc)
        score = v.judge(model,stop_on_error = False, deep_error = True)
        scores.append(score)


        if 'mean' in v.observation.keys():
            unit_observations = v.observation['mean']

        if 'value' in v.observation.keys():
            unit_observations = v.observation['value']

        if 'mean' in v.prediction.keys():
            unit_predictions = v.prediction['mean']

        if 'value' in v.prediction.keys():
            unit_predictions = v.prediction['value']

        print('observations: {0} predictions: {1}'.format(unit_observations, unit_predictions))

        try:
            to_r_s = unit_observations.units
            unit_predictions = unit_predictions.rescale(to_r_s)
            unit_delta = np.abs(np.abs(unit_observations)-np.abs(unit_predictions))
        except:
            unit_delta = 0.0
        delta.append(unit_delta)
        print('observation {0} versus prediction {1}'.format(unit_observations,unit_predictions))
        print('unit delta', unit_delta)
        sv = score.sort_key
        test_dic[str(v)] = (float(unit_observations), float(unit_predictions), unit_delta)


        if 'value' in v.observation.keys():
            columns1.append(str(v)+str(v.observation['value'].units))
            #labels_dic[str(v)] = str(v.observation['value'].units)
        if 'mean' in v.observation.keys():
            #labels_dic[str(v)] = str(v.observation['mean'].units)
            columns1.append(str(v)+str(v.observation['mean'].units))
    threed = []
    columns2 = []
    iterator = 0
    average = np.mean([ np.sum(v) for v in test_dic.values()])
    for i,t in enumerate(tests):
        v = test_dic[str(t)]
        if not(np.sum(v) > 3.5 * average) and not(v[2] > 25.0) :
            threed.append((float(v[0]),float(v[1]),float(v[2])))
            columns2.append(columns1[i])
    stacked = np.column_stack(np.array(threed))
    df = pd.DataFrame(np.array(stacked), columns=columns2)
    df.index = ['observation','prediction','difference']
    # uncoment this line to make a bar chart on plotly.
    # df.iplot(kind='bar', barmode='stack', yTitle='NeuronUnit Test Agreement', title='test agreement every test', filename='grouped-bar-chart')
    html = df.to_html()
    html_file = open("tests_agreement_table.html","w")
    html_file.write(html)
    html_file.close()
    return df, threed, columns1 ,stacked, html, test_dic
