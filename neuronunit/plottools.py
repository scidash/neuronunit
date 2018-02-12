"""Tools for plotting (contributed by Blue Brain Project)"""

import matplotlib
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import colorsys
import numpy
import collections

import os
import matplotlib
#import pandas as pd
#import numpy as np


import sys
KERNEL = ('ipykernel' in sys.modules)
import numpy as np
import pandas as pd

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


def use_dtc_to_plotting(dtcpop):
    '''
    To be used to visualize outputs generated by first running
    nsga_parallel.dtc_to_plotting on data_transport_container objects.
    outputs: a plot of voltage traces of a gene population against one time axis.
    '''
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
    dynamic_hof = np.array([ sum(i.fitness) for i in gen_vs_hof ])

    dynamic_hof = np.array([ sum(i.fitness) for i in gen_vs_hof ])


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
    axes.plot(gen_numbers, dynamic_hof, 'y^', linewidth=2, label='hall of fame error')
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
