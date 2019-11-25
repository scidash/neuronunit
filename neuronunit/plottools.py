"""Tools for plotting (contributed by Blue Brain Project)"""
import seaborn as sns
import matplotlib
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt
import matplotlib as mpl

import colorsys
import numpy
import collections
import cython
import copy

import sys
KERNEL = ('ipykernel' in sys.modules)
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO
from neuronunit.capabilities.spike_functions import get_spike_waveforms

import numpy as np
import pandas as pd
import bs4
from IPython.display import HTML,Javascript,display
#KERNEL = ('ipykernel' in sys.modules)
from collections.abc import Iterable
#from neuronunit.tests.base import AMPL, DELAY, DURATION

#from collections.abc import Iterable
import seaborn as sns
from neuronunit.tests.base import AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
import matplotlib as mpl

@cython.boundscheck(False)
@cython.wraparound(False)
def inject_and_plot(dtc,second_pop=None,third_pop=None,figname='problem',snippets=False,experimental_cell_type="neo_cortical",ground_truth = None,BPO=True):
    sns.set_style("darkgrid")
    from neuronunit.optimisation.optimization_management import mint_generic_model

    if not isinstance(dtc, Iterable):
        model = mint_generic_model(dtc.backend)
        if hasattr(dtc,'rheobase'):
            try:
                rheobase = dtc.rheobase['value']
            except:
                rheobase = dtc.rheobase
                uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
        if hasattr(dtc,'ampl'):
            uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

        model.set_attrs(**dtc.attrs)
        model.inject_square_current(uc)
        if str(dtc.backend) is str('ADEXP'):
            model.finalize()
        else:
            pass
        #vm = model.get_membrane_potential().magnitude
        sns.set_style("darkgrid")
        plt.plot(model.get_membrane_potential().times,model.get_membrane_potential().magnitude)#,label='ground truth')
        dtc.vm = model.get_membrane_potential()
        plot_backend = mpl.get_backend()
        if plot_backend == str('agg'):
            plt.savefig(figname+str('debug.png'))
        else:
            plt.show()

    else:

        if type(second_pop) is not type(None):
            dtcpop = copy.copy(dtc)
            dtc = None
            fig = plt.figure(figsize=(11,11),dpi=100)
            ax = fig.add_subplot(111)
            for index,dtc in enumerate(dtcpop):
                color = 'lightblue'
                if index == 0:
                    color = 'blue'
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    # this ugly hack can be fixed in the file tests/fi.py
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}
                model.set_attrs(**dtc.attrs)
                if rheobase is None:
                    break
                model.inject_square_current(uc)
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                vm = model.get_membrane_potential()#.magnitude
                dtc.vm = vm

                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')
                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')

                if str("BHH") in dtc.backend:
                    label=str('Hodgkin Huxley Model')
                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')
                sns.set_style("darkgrid")
                if snippets:
                    snippets_ = get_spike_waveforms(vm)
                    dtc.snippets = snippets_
                    plt.plot(snippets_.times,snippets_,color=color,label=str('model type: ')+label)#,label='ground truth')
                else:
                    plt.plot(vm.times,vm,color=color,label=str('model type: ')+label)#,label='ground truth')
                ax.legend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

            for index,dtc in enumerate(second_pop):
                color = 'lightcoral'
                if index == 0:
                    color = 'red'
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                #uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                #dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                #if model.get_spike_count()>1:
                #    break
                if str(dtc.backend) in str('ADEXP'):
                    model.finalize()
                else:
                    pass
                vm = model.get_membrane_potential()#.magnitude
                vm = model.get_membrane_potential()#.magnitude
                dtc.vm = vm
                if str("RAW") in dtc.backend:
                    label=str('Izhikevich Model')

                if str("ADEXP") in dtc.backend:
                    label=str('Adaptive Exponential Model')

                if str("GLIF") in dtc.backend:
                    label=str('Generalized Leaky Integrate and Fire')
                #label = label+str(latency)

                sns.set_style("darkgrid")
                if snippets:
                    snippets_ = get_spike_waveforms(vm)
                    plt.plot(snippets_.times,snippets_,color=color,label=str('model type: ')+label)#,label='ground truth')
                else:
                    plt.plot(vm.times,vm,color=color,label=str('model type: ')+label)#,label='ground truth')
                #plt.plot(model.get_membrane_potential().times,vm,color='blue',label=label)#,label='ground truth')
                #ax.legend(['A simple line'])
                ax.legend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

            if third_pop is None:
                pass
            else:

                for index,dtc in enumerate(second_pop):
                    color = 'lightgreen'
                    if index == 0:
                        color = 'green'
                    model = mint_generic_model(dtc.backend)
                    if hasattr(dtc,'rheobase'):
                        try:
                            rheobase = dtc.rheobase['value']
                        except:
                            rheobase = dtc.rheobase
                            uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                    if hasattr(dtc,'ampl'):
                        uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                    model.set_attrs(**dtc.attrs)
                    model.inject_square_current(uc)
                    #if model.get_spike_count()>1:
                    #    break
                    if str(dtc.backend) in str('ADEXP'):
                        model.finalize()
                    else:
                        pass
                    vm = model.get_membrane_potential()#.magnitude
                    #vm = model.get_membrane_potential()#.magnitude
                    dtc.vm = vm
                    sns.set_style("darkgrid")

                    if str("RAW") in dtc.backend:
                        label=str('Izhikevich Model')

                    if str("ADEXP") in dtc.backend:
                        label=str('Adaptive Exponential Model')
                    if str("GLIF") in dtc.backend:
                        label=str('Generalized Leaky Integrate and Fire')
                    #label = label+str(latency)
                    if snippets:

                        snippets_ = get_spike_waveforms(vm)
                        plt.plot(snippets_.times,snippets_,color=color,label=str('model type: ')+label)#,label='ground truth')
                        #ax.legend(['A simple line'])
                        ax.legend()
                    else:
                        plt.plot(vm.times,vm,color=color,label=str('model type: ')+label)#,label='ground truth')

                plot_backend = mpl.get_backend()
                sns.set_style("darkgrid")

                plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (mV)')

                if str('agg') in plot_backend:

                    plt.savefig(figname+str('all_traces.png'))
                else:
                    plt.show()
        else:
            dtcpop = copy.copy(dtc)
            dtc = None
            for dtc in dtcpop[0:2]:
                model = mint_generic_model(dtc.backend)
                if hasattr(dtc,'rheobase'):
                    try:
                        rheobase = dtc.rheobase['value']
                    except:
                        rheobase = dtc.rheobase
                        uc = {'amplitude':rheobase,'duration':DURATION,'delay':DELAY}
                if hasattr(dtc,'ampl'):
                    uc = {'amplitude':dtc.ampl,'duration':DURATION,'delay':DELAY}

                #dtc.run_number += 1
                model.set_attrs(**dtc.attrs)
                model.inject_square_current(uc)
                vm = model.get_membrane_potential().magnitude
                #vm = model.get_membrane_potential()#.magnitude
                dtc.vm =  model.get_membrane_potential()
                sns.set_style("darkgrid")

                plt.plot(model.get_membrane_potential().times,vm)#,label='ground truth')
            plot_backend = mpl.get_backend()
            sns.set_style("darkgrid")

            plt.title(experimental_cell_type)#+str(' Model Type: '+str(second_pop[0].backend)+str(dtc.backend)))
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude (mV)')

            if plot_backend == str('agg'):
                plt.savefig(figname+str('all_traces.png'))
            else:
                plt.show()
    return [dtc,second_pop,third_pop]

def elaborate_plots(self,ga_out):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
    '''
    Move to slides
    The plot shows the mean error value of the population as the GA evolves it's population. The red interval at any instant is the standard deviation of the error. The fact that the mean GA error is able to have a net upwards trajectory, after experiencing a temporary downwards trajectory, demonstrates that the GA retains a drive to explore, and is resiliant against being stuck in a local minima. Also in the above plot population variance in error stays remarkably constant, in this way BluePyOpts selection criteria SELIBEA contrasts with DEAPs native selection strategy NSGA2
    #for index, val in enumerate(ga_out.values()):
    '''
    import pdb
    pdb.set_trace()
    try:
       temp = copy.copy(ga_out['pf'][0].dtc.scores)
    except:
       temp = copy.copy(ga_out['dtc_pop'][0].scores)

    if not self.use_rheobase_score:
        temp.pop("RheobaseTest",None)
    objectives = list(temp.keys())

    logbook = ga_out['log']
    ret = logbook.chapters
    logbook = ret['fitness']


    gen_numbers =[ i for i in range(0,len(logbook.select('gen'))) ]
    pf = ga_out['pf']
    mean = np.array(logbook.select('stats_fit'))

    mean = np.array(logbook.select('avg'))
    std = np.array(logbook.select('std'))
    minimum = logbook.select('min')
    try:
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

        axes.plot(gen_numbers, stdminus, label='std variation lower limit')
        axes.plot(gen_numbers, stdplus, label='std variation upper limit')
        axes.set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers) + 1)
        axes.set_xlabel('Generations')
        axes.set_ylabel('Sum of objectives')
        axes.legend()
        fig.tight_layout()
        plt.savefig(str('classic_evolution_')+str(self.cell_name)+str(self.backend)+str('.png'))
    except:
        print('mistake, plotting routine access logbook stats')

    #sns.set_style("darkgrid")
    plt.figure()
    sns.set_style("darkgrid")
    logbook = ga_out['log']
    ret = logbook.chapters
    gen = ret['every'].select("gen")
    everything = ret["every"]
    fig2, ax2 = plt.subplots(len(objectives)+2,1,figsize=(10,10))

    for i,k in enumerate(objectives):
        if i < len(everything.select("avg")[0]):
            #[j[i] for j in everything.select("avg")]
            line2 = ax2[i].plot([j[i] for j in everything.select("avg")], "r-", label="Evolution objectives")

            [j[i] for j in everything.select("avg")]
            ax2[i].set_ylim([0.0, 1.0])
            if i!=len(objectives):
                ax2[i].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            ax2[len(objectives)].plot(
                gen_numbers,
                mean,
                color='black',
                linewidth=2,
                label='population average')
            ax2[len(objectives)].fill_between(gen_numbers, stdminus, stdplus)
            ax2[len(objectives)].plot(gen_numbers, stdminus, label='std variation lower limit')
            ax2[len(objectives)].plot(gen_numbers, stdplus, label='std variation upper limit')
            ax2[len(objectives)].set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers) + 1)
            h = ax2[len(objectives)].set_xlabel("NeuronUnit Test: {0}".format(k))
    plt.savefig(str('reliable_history_plot_')+str(self.cell_name)+str(self.backend)+str('.png'))



    #logbook = ga_out['log']
    logbook = ga_out['log']
    ret = logbook.chapters
    logbook = ret['fitness']

    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_max = logbook.select("max")
    fit_avgs = logbook.select("avg")


    fig, ax1 = plt.subplots(figsize=(10,10))
    line1 = ax1.plot(gen, fit_mins, "r-", label="Minimum Fitness")
    line2 = ax1.plot(gen, fit_avgs, "g-", label="Average Fitness")
    line3 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness out of: {0}".format(len(objectives)))
    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    plt.savefig(str('evolution_')+str(self.cell_name)+str(self.backend)+str('.png'))
    ga_out['stats_plot_1'] = ax1, fig# ax2, ax3, fig

    avg, max_, min_, std_ = logbook.select("avg", "max", "min","std")
    all_over_gen = {}
    for i,k in enumerate(objectives):
        all_over_gen[k] = []
        for v in ga_out['history'].genealogy_history.values():
            all_over_gen[k].append(v.fitness.values[i])# if type(ind.dtc) is not type(None) ]
    plt.figure()
    sns.set_style("darkgrid")


    fig2, ax2 = plt.subplots(len(objectives)+1,1,figsize=(10,10))
    for i,k in enumerate(objectives):
        #if i < len(everything.select("avg")[0]):

        ax2[i].plot(list(range(0,len(all_over_gen[k]))),all_over_gen[k])
        temp = [0 for i in range(0,len(all_over_gen[k])) ]
        ax2[i].plot(list(range(0,len(all_over_gen[k]))),temp)
        ax2[i].set_ylim([0.0, 1.0])
        if i!=len(objectives)-1:
            ax2[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        h = ax2[i].set_xlabel("NeuronUnit Test: {0}".format(k))
        #h.set_rotation(90)
        plt.savefig(str('history_plot_')+str(self.cell_name)+str(self.backend)+str('.png'))
        ga_out['history_plot_'] = (fig2,plt)


    all_over_gen = {}
    for i,k in enumerate(objectives):
        all_over_gen[k] = []
        for gen in ga_out['gen_vs_pop']:
            sorted_pop = sorted([(np.sum(ind.fitness.values),ind) for ind in gen if len(ind.fitness.values)==len(objectives) ], key=lambda tup: tup[0])
            ind = sorted_pop[0][1]
            all_over_gen[k].append(ind.fitness.values[i])# if type(ind.dtc) is not type(None) ]
    plt.figure()
    sns.set_style("darkgrid")

    fig2, ax2 = plt.subplots(len(objectives)+1,1,figsize=(10,10))
    for i,k in enumerate(objectives):
        ax2[i].plot(list(range(0,len(all_over_gen[k]))),all_over_gen[k])
        temp = [0 for i in range(0,len(all_over_gen[k])) ]
        ax2[i].plot(list(range(0,len(all_over_gen[k]))),temp)
        ax2[i].set_ylim([0.0, 1.0])
        if i!=len(objectives)-1:
            ax2[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        h = ax2[i].set_xlabel("NeuronUnit Test: {0}".format(k))
        #h.set_rotation(90)
    plt.savefig(str('crazy_plot_')+str(self.cell_name)+str(self.backend)+str('.png'))
    ga_out['crazy_plot_'] = (fig2,plt)

    return ga_out


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



import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size':16})
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import pylab

def plot_surface(fig_trip,ax_trip,model_param0,model_param1,history):
    '''

    Move this method back to plottools
    Inputs should be keys, that are parameters see new function definition below
    '''
    td = list(history.genealogy_history[1].dtc.attrs.keys())
    x = [ i for i,j in enumerate(td) if str(model_param0) == j ][0]
    y = [ i for i,j in enumerate(td) if str(model_param1) == j ][0]
    z = [ i for i,j in enumerate(td) if str(model_param1) == j ][0]

    all_inds = history.genealogy_history.values()
    sums = np.array([np.sum(ind.fitness.values) for ind in all_inds])

    xs = np.array([ind[x] for ind in all_inds])
    ys = np.array([ind[y] for ind in all_inds])
    zs = np.array([ind[z] for ind in all_inds])

    min_ys = ys[np.where(sums == np.min(sums))]
    min_xs = xs[np.where(sums == np.min(sums))]
    min_zs = xs[np.where(sums == np.min(sums))]

    data = np.zeros((len(xs),3))
    data[:,0] = xs
    data[:,1] = ys
    data[:,2] = zs


    #data = np.random.random((12,3))            # arbitrary 3D data set
    #tri = scipy.spatial.Delaunay( data[:,:2] ) # take the first two dimensions

    #pylab.triplot( data[:,0], data[:,1], tri.simplices.copy() )
    #pylab.plot( data[:,0], data[:,1], 'ro' ) ;

    #fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')
    trip_axis = ax_trip.tripcolor(xs,ys,sums,20,norm=matplotlib.colors.LogNorm())
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue',label='global minima')
    #plot_axis.colorbar(trip_axis, label='Sum of Objective Errors ')
    if type(td) is not type(None):
        ax_trip.set_xlabel('Parameter '+str((td[x])))
        ax_trip.set_ylabel('Parameter '+str((td[y])))
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue')
    #plot_axis.tight_layout()
    return ax_trip,plot_axis

def plot_vm(hof,ax,key):
    ax.cla()
    ax.set_title(' {0} vs  $V_{M}$'.format(key[0]))
    best_dtc = hof[0].dtc
    best_rh = hof[0].dtc.rheobase
    neuron = None
    model = ReducedModel(path_params['model_path'],name = str('regular_spiking'),backend =('NEURON',{'DTC':best_dtc}))
    params = {'injected_square_current':
            {'amplitude': best_rh, 'delay':DELAY, 'duration':DURATION}}
    results = modelrs.inject_square_current(params)
    vm = model.get_membrane_potential()
    times = vm.times
    ax.plot(times,vm)
    return ax

def plotss(matrix,hof):
    dim = np.shape(matrix)[0]
    print(dim)
    cnt = 0
    fig,ax = plt.subplots(dim,dim,figsize=(10,10))
    flat_iter = []
    for i,k in enumerate(matrix):
        for j,r in enumerate(k):
            keys = list(r[0])
            gr = r[1]
            if i==j:
                ax[i,j] = plot_vm(hof,ax[i,j],keys)
            if i>j:
                ax[i,j] = plot_surface(gr,ax[i,j],keys,imshow=False)
            if i < j:
                ax[i,j] = plot_scatter(hof,ax[i,j],keys)
        print(i,j)
    plt.savefig(str('surface_and_vm.png'))
    return None

def scatter_surface(fig_trip,ax_trip,model_param0,model_param1,history):
    '''

    Move this method back to plottools
    Inputs should be keys, that are parameters see new function definition below
    '''
    td = list(history.genealogy_history[1].dtc.attrs.keys())
    x = [ i for i,j in enumerate(td) if str(model_param0) == j ][0]
    y = [ i for i,j in enumerate(td) if str(model_param1) == j ][0]

    all_inds = history.genealogy_history.values()
    z = np.array([np.sum(ind.fitness.values) for ind in all_inds])

    xs = np.array([ind[x] for ind in all_inds])
    ys = np.array([ind[y] for ind in all_inds])

    return xs,ys,z

def shadow(dtcpop,best_vm):#This method must be pickle-able for ipyparallel to work.
    '''
    A method to plot the best and worst candidate solution waveforms side by side


    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in png format.

    '''
    from neuronunit.optimization.nsga_parallel import dtc_to_plotting
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    import os

    import quantities as pq
    import numpy as np
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal
    import matplotlib as mpl
    # setting of an appropriate backend.


    #color='lightblue'
    dtcpop.append(best_vm)

    import copy
    from neuronunit.optimization import get_neab
    tests = copy.copy(get_neab.tests)
    for k,v in enumerate(tests):
        import matplotlib.pyplot as plt

        plt.clf()
        plt.style.use('ggplot')
     	# following variables possibly are
        # going to become depreciated
        stored_min = []
        stored_max = []
        sc_for_frame_best = []
        sc_for_frame_worst = []

        sindexs = []
        for iterator, vms in enumerate(dtcpop):


            from neuronunit.models import backends
            from neuronunit.models.reduced import ReducedModel

            print(get_neab.LEMS_MODEL_PATH)
            #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')

            #import pdb; pdb.set_trace()
            assert type(vms.rheobase) is not type(None)
            if k == 0:
                v.prediction = {}
                v.prediction['value'] = vms.rheobase * pq.pA

                print(v.prediction)
            if k != 0:
                v.prediction = None

            if k == 1 or k == 2 or k == 3:
                # Negative square pulse current.
                v.params['injected_square_current']['duration'] = 100 * pq.ms
                v.params['injected_square_current']['amplitude'] = -10 *pq.pA
                v.params['injected_square_current']['delay'] = 30 * pq.ms
            if k == 0 or k ==4 or k == 5 or k == 6 or k == 7:
                # Threshold current.
                v.params['injected_square_current']['duration'] = 1000 * pq.ms
                v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
                v.params['injected_square_current']['delay'] = 100 * pq.ms
            import neuron
            model.reset_h(neuron)
            #model.load_model()
            model.update_run_params(vms.attrs)
            print(v.params)
            score = v.judge(model,stop_on_error = False, deep_error = True)

            if k == 0 or k ==4 or k == 5 or k == 6 or k == 7:
                v_m = model.get_membrane_potential()
                dt = float(v_m.sampling_period)
                ts = model.results['t'] # time signal
                st = spike_functions.get_spike_train(v_m) #spike times
                if model.get_spike_count() == 1:
                    print(st)
                    assert len(st) == 1
                    # st = float(st)
                    # get the approximate integer index into the array of membrane potential corresponding to when the spike time
                    # occurs, and store it in a list
                    # minimums and maximums of this list will be calculated on a piecemeal basis.

                    stored_min.append(np.min(model.results['vm']))
                    stored_max.append(np.max(model.results['vm']))
                    sindexs.append(int((float(st)/ts[-1])*len(ts)))
                    time_sequence = np.arange(np.min(sindexs)-5 , np.max(sindexs)+5, 1)
                    ptvec = np.array(model.results['t'])[time_sequence]
                    pvm = np.array(model.results['vm'])[time_sequence]
                    assert len(pvm) == len(ptvec)
                    plt.plot(ptvec, pvm, label=str(v)+str(score), linewidth=1.5)
                    #plt.xlim(np.min(sindexs)-11,np.min(sindexs)+11 )
                    #plt.ylim(np.min(stored_min)-4,np.max(stored_max)+4)

            else:
                stored_min.append(np.min(model.results['vm']))
                stored_max.append(np.max(model.results['vm']))
                plt.plot(model.results['t'],model.results['vm'],label=str(v)+str(score), linewidth=1.5)
                plt.xlim(0,float(v.params['injected_square_current']['duration']) )
                #plt.ylim(np.min(stored_min)-4,np.max(stored_max)+4)
                #model.results = None
        #inside the tests loop but outside the model loop.
        #plt.tight_layout()
        plt.legend()
        plt.ylabel('$V_{m}$ mV')
        plt.xlabel('ms')
        if not KERNEL:
            plt.savefig(str('test_')+str(v)+'vm_versus_t.png', format='png')#, dpi=1200)
        else:
            plt.show()



def surfaces(history,td):
    import numpy as np
    import matplotlib
    matplotlib.rcParams.update({'font.size':16})

    import matplotlib.pyplot as plt

    all_inds = history.genealogy_history.values()
    sums = np.array([np.sum(ind.fitness.values) for ind in all_inds])
    keep = set()
    quads = []
    for k in range(1,9):
        for i,j in enumerate(td):
            print(i,k)
            if i+k < 10:
                quads.append((td[i],td[i+k],i,i+k))

    #for q in quads:
        #print(k)
        #(x,y,w,z) = q
        #print(x,y,w,z,i)
    all_inds1 = history.genealogy_history.values()

    ab = [ (all_inds1[y][4],all_inds1[y][-3]) for y in all_inds1 ]

    xs = np.array([ind[4] for ind in all_inds])
    ys = np.array([ind[-3] for ind in all_inds])
    min_ys = ys[np.where(sums == np.min(sums))]
    min_xs = xs[np.where(sums == np.min(sums))]
    plt.clf()
    fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')
    trip_axis = ax_trip.tripcolor(xs,ys,sums,20,norm=matplotlib.colors.LogNorm())
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue',label='global minima')
    fig_trip.colorbar(trip_axis, label='Sum of Objective Errors ')
    ax_trip.set_xlabel('Parameter $ b$')
    ax_trip.set_ylabel('Parameter $ a$')
    plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue')
    fig_trip.tight_layout()
    fig_trip.show()
    #fig_trip.legend()
    #fig_trip.savefig('surface'+str('a')+str('b')+'.pdf',format='pdf')#', dpi=1200)


    matrix_fill = [ (i,j) for i in range(0,len(modelp.model_params['a'])) for j in range(0,len(modelp.model_params['b'])) ]
    mf = list(zip(matrix_fill,summed))
    empty = np.zeros(shape=(int(len(modelp.model_params['a'])),int(len(modelp.model_params['a']))))
    max_x = np.max(modelp.model_params['a'])
    max_y = np.min(modelp.model_params['b'])
    x_mapped_ind = [int((ind[4]/max_x)*len(modelp.model_params['a'])) for ind in all_inds1]
    y_mapped_ind = [int((np.abs(ind[-3])/max_y)*len(modelp.model_params['a'])) for ind in all_inds1]

    #y_mapped_ind = np.array([int(ind[-3]/max_y) for ind in all_inds1])
    #int((all_inds1[1][4]/max_x)*len(modelp.model_params['a']))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmin = np.min(mf2)
    vmax = np.max(mf2)
    from matplotlib.colors import LogNorm
    cax = ax.matshow(dfimshow, interpolation='nearest',norm=LogNorm(vmin=vmin,vmax=vmax))
    fig.colorbar(cax)

    ax.set_xticklabels(modelp.model_params['a'])
    ax.set_yticklabels(modelp.model_params['b'])
    plt.title(str('$a$')+' versus '+str('$b$'))
    if not KERNEL:
        plt.savefig('2nd_approach_d_error_'+str('a')+str('b')+'surface.png')
    else:
        plt.show()

    return ab




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




def plot_log(log): #logbook
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
    gen_numbers =[ i for i in range(0,len(log.select('gen'))) ]
    mean = np.array([ np.sum(i) for i in log.select('avg')])
    std = np.array([ np.sum(i) for i in log.select('std')])
    minimum = np.array([ np.sum(i) for i in log.select('min')])

    stdminus = mean - std
    stdplus = mean + std
    try:
        assert len(gen_numbers) == len(stdminus) == len(stdplus)
    except:
        pass
        #raise Exception

    axes.plot(
        gen_numbers,
        mean,
        color='black',
        linewidth=2,
        label='population average')
    axes.fill_between(gen_numbers, stdminus, stdplus)
    axes.plot(gen_numbers, stdminus)
    axes.plot(gen_numbers, stdplus)
    axes.set_xlim(np.min(gen_numbers) - 1, np.max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([np.min(stdminus), np.max(stdplus)])
    axes.legend()

    fig.tight_layout()
    if not KERNEL:
        fig.savefig('Izhikevich_history_evolution.png', format='png')#, dpi=1200)
    else:
        fig.show()

def dtc_to_plotting(dtc):
    dtc.vm0 = None
    dtc.vm1 = None
    from neuronunit.models.reduced import ReducedModel
    from neuronunit.optimization.get_neab import tests as T
    from neuronunit.optimization import get_neab
    from neuronunit.optimization import evaluate_as_module
    from neuronunit.optimization.evaluate_as_module import pre_format
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    import neuron
    model._backend.reset_neuron(neuron)
    model.set_attrs(dtc.attrs)
    #model.rheobase = dtc.rheobase#['value']
    dtc = pre_format(dtc)
    parameter_list = list(dtc.vtest.values())
    print(parameter_list[0])
    #print(dtc.rheobase)
    #print(dtc.rheobase['value'])

    #import pdb; pdb.set_trace()

    model.inject_square_current(parameter_list[0])
    model._backend.local_run()
    assert model.get_spike_count() == 1 or model.get_spike_count() == 0

    dtc.vm0 = list(model.results['vm'])
    dtc.tvec = list(model.results['t'])
    return dtc

    #model = None
    '''
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    import neuron
    model._backend.reset_neuron(neuron)
    model.set_attrs(dtc.attrs)
    print(parameter_list[-1])
    model.inject_square_current(parameter_list[-1])
    print(model.get_spike_count())
    dtc.vm1 = list(model.results['vm'])
    '''

def plot_objectives_history(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook
    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in png format.

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from neuronunit.optimization import get_neab
    plt.clf()
    plt.style.use('ggplot')


    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')

    gen_numbers = log.select('gen')
    minimum = log.select('min')
    mean = log.select('mean')

    objective_labels = [ str(t) for t in get_neab.tests ]
    mins_components_plot = log.select('min')
    components = {}
    for i in range(0,7):
        components[i] = []
        for l in mins_components_plot:
            components[i].append(l[i])
    maximum = 0.0
    #import pdb; pdb.set_trace()
    for keys in components:

        axes.plot(
            gen_numbers,
            components[keys],
            linewidth=2,
            label=str(objective_labels[keys])
            )
        if np.max(components[keys]) > maximum:
            maximum = np.max(components[keys])

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

'''
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

    from plotly.graph_objs import *
    import matplotlib.pyplot as plt
    import numpy as np

    from IPython.lib.deepreload import reload
    import ipyparallel as ipp
    rc = ipp.Client(profile='default')
    rc[:].use_cloudpickle()
    dview = rc[:]

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
'''


def best_worst(history):
    '''
    Query the GA's DEAP history object to get the  best and worst candidates ever evaluated.

    inputs DEAP history object
    best output should be the same as the first index of the ParetoFront list (pf[0]).
    outputs best and worst candidates evaluated.
    '''
    import numpy as np
    badness = [ np.sum(ind.fitness.values) for ind in history.genealogy_history.values() if ind.fitness.valid ]
    maximumb = np.max(badness)
    minimumb = np.min(badness)
    history_dic = history.genealogy_history
    worst = []
    best = []
    # there may be multiple best and worst candidates, just take the first ones found.
    for k,v in history_dic.items():
        if np.sum(v.fitness.values) == maximumb:
            worst.append(v)
        if np.sum(v.fitness.values) == minimumb:
            best.append(v)
    return best[0], worst[0]


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

def sp_spike_width(best_worst):
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

def load_data():
    a = pickle.load(open('for_pandas.p','rb'))
    df = pd.DataFrame(np.transpose(stacked),columns=columns1)
    stacked = opened[0]
    columns1 = opened[1]


def pandas_rh_search(vmoffspring):
    searchedd = {}
    from get_neab import tests
    v = tests[0]
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import pandas as pd
    import quantities as qt

    new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
    model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
    for x, i in enumerate(vmoffspring[0].searched):
       searchedd[x]={}
       for j in i:
          v.params['injected_square_current']['amplitude'] = j
          model.inject_square_current(v.params)
          searchedd[x][float(j)] = model.get_spike_count()
    with open('rheoframe.p','wb') as handle:
        pickle.dump([df0,df1,df2,df3,df4,vmoffspring],handle)
	##
  	# Obviously the four variables should be made as
    # dictionary keys, rather than repitious code below
    # however, no time.
    ##
    to_stack0 = [ (k,v) for k,v in searchedd[0].items() ]
    to_stack1 = [ (k,v) for k,v in searchedd[1].items() ]
    to_stack2 = [ (k,v) for k,v in searchedd[2].items() ]
    to_stack3 = [ (k,v) for k,v in searchedd[3].items() ]
    to_stack4 = [ (k,v) for k,v in searchedd[4].items() ]
    stacked0 = np.column_stack(np.array(to_stack0))
    df0 = pd.DataFrame(np.transpose(stacked0),columns=['pA Injection','Spike Count'])
    stacked1 = np.column_stack(np.array(to_stack1))
    df1 = pd.DataFrame(np.transpose(stacked1),columns=['pA Injection','Spike Count'])
    stacked2 = np.column_stack(np.array(to_stack2))
    df2 = pd.DataFrame(np.transpose(stacked2),columns=['pA Injection','Spike Count'])
    stacked3 = np.column_stack(np.array(to_stack3))
    df3 = pd.DataFrame(np.transpose(stacked3),columns=['pA Injection','Spike Count'])
    stacked4 = np.column_stack(np.array(to_stack4))
    df4 = pd.DataFrame(np.transpose(stacked4),columns=['pA Injection','Spike Count'])
    dfs = [df0,df1,df2,df3,df4]
    #for i in dfs:
    df0 = df0.sort(['pA Injection','Spike Count'],ascending=[1,0])
    df1 = df1.sort(['pA Injection','Spike Count'],ascending=[1,0])
    df2 = df2.sort(['pA Injection','Spike Count'],ascending=[1,0])
    df3 = df3.sort(['pA Injection','Spike Count'],ascending=[1,0])
    df4 = df4.sort(['pA Injection','Spike Count'],ascending=[1,0])
    df0.index = ['CPU 0','CPU 1','CPU 2','CPU 3', 'CPU 4', 'CPU 5', 'CPU 6']
    df1.index = ['CPU 0','CPU 1','CPU 2','CPU 3', 'CPU 4', 'CPU 5']
    df2.index = ['CPU 0','CPU 1','CPU 2','CPU 3', 'CPU 4', 'CPU 5']
    df3.index = ['CPU 0','CPU 1','CPU 2','CPU 3', 'CPU 4', 'CPU 5']
    df4.index = ['CPU 0','CPU 1','CPU 2','CPU 3', 'CPU 4', 'CPU 5']

    orig_cmap = sns.light_palette("red",as_cmap=True)
    shrunk_cmap = shiftedColorMap(orig_cmap, start=np.min(vmoffspring[0].searched), midpoint=vmoffspring[0].rheobase, stop=np.max(vmoffspring[0].searched), name='shrunk')
    s0 = df0.style.background_gradient(cmap = shrunk_cmap)
    s1 = df1.style.background_gradient(cmap = shrunk_cmap)
    s2 = df2.style.background_gradient(cmap = shrunk_cmap)
    s3 = df3.style.background_gradient(cmap = shrunk_cmap)
    s4 = df4.style.background_gradient(cmap = shrunk_cmap)

    return df0,df1,df2,df3,df4

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

    import plotly.plotly as py
    from plotly.graph_objs import Bar
    '''
    A method to plot raw predictions
    versus observations

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in png format.

    '''
    import os
    import matplotlib
    import pandas as pd

    #import matplotlib.pyplot as plt
    import numpy as np

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
        model.load_model()
        assert type(vms.rheobase) is not type(None)



        if k == 0:
            v.prediction = {}
            v.prediction['value'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms
            print(v.prediction)
        if k != 0:
            v.prediction = None

        if k == 1 or k == 2 or k == 3:
            # Negative square pulse current.
            v.params['injected_square_current']['duration'] = 100 * pq.ms
            v.params['injected_square_current']['amplitude'] = -10 *pq.pA
            v.params['injected_square_current']['delay'] = 30 * pq.ms
        if k == 5 or k == 6 or k == 7:
            # Threshold current.
            v.params['injected_square_current']['duration'] = 1000 * pq.ms
            v.params['injected_square_current']['amplitude'] = vms.rheobase * pq.pA
            v.params['injected_square_current']['delay'] = 100 * pq.ms
        import neuron
        model.reset_h(neuron)
        model.load_model()
        model.update_run_params(vms.attrs)
        print(v.params)
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
        sv = score.norm_score
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
    #import pdb;
    #pdb.set_trace()
    # uncoment this line to make a bar chart on plotly.
    # df.iplot(kind='bar', barmode='stack', yTitle='NeuronUnit Test Agreement', title='test agreement every test', filename='grouped-bar-chart')
    html = df.to_html()
    html_file = open("tests_agreement_table.html","w")
    html_file.write(html)
    html_file.close()
    return df, threed, columns1 ,stacked, html, test_dic
