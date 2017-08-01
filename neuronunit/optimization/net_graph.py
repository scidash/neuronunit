
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import graphviz

#import graphviz_layout
import matplotlib as mpl
mpl.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work

import matplotlib.pyplot as plt
import numpy as np

def plot_performance_profile():
    '''
    makes a plot about this programs performance profile
    It needs ProfExit to be called at the beggining of the program.
    https://github.com/jrfonseca/gprof2dot

    No inputs and outputs only side effects.
    Actually this probably won't work, as the performance profiler
    Requires all the process IDs to cleanly exit due to finishing their work.
    Instead the essence of this method should be rehashed as BASH script (which should be easy, since the origin is bash script)
    '''
    import os
    import subprocess
    #os.system('sudo /opt/conda/bin/pip install gprof2dot')
    os.system('cp /opt/conda/lib/python3.5/site-packages/gprof2dot.py .')
    prof_f_name = '{0}'.format(os.getpid())
    #Open and close the file, as a quick and dirty way to garuntee that an appropriate file path exists.
    f = open('NeuroML2/{0}'.format(prof_f_name),'wb')
    file_name = 'NeuroML2/{0}'.format(prof_f_name)
    f.close()
    #subprocess.Popen('python','gprof2dot.py' -f -n0 -e0 pstats {0}  | dot -Tsvg -o {0}.svg'.format(prof_f_name))
    os.system('python gprof2dot.py -f -n0 -e0 pstats {0}  | dot -Tsvg -o {0}.svg'.format(prof_f_name))

def graph_s(history,graph):
    '''
    Make a directed graph (family tree) plot of the genealogy history
    Bottom is the final generation, top is the initial population.
    Extreme left and right nodes are terminating, ie they have no children, and
    have been discarded from the breeding population.
    NB: authors this should be obvious, but GA authors do not
    necessarily that this is a model that
    reflects actual genes and or actual breading.
    '''
    plt.clf()
    #try:
    import networkx
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()
    labels = {}
    for i in graph.nodes():
        labels[i] = i
    node_colors = [ np.sum(history.genealogy_history[i].fitness.values) for i in graph ]
    positions = graphviz_layout(graph, prog="dot")
    networkx.draw(graph, positions, node_color=node_colors, labels = labels)
    nodes=networkx.draw_networkx_nodes(graph,positions,node_color=node_colors, node_size=4.5, labels = labels)
    edges=networkx.draw_networkx_edges(graph,positions,width=1.5,edge_cmap=plt.cm.Blues)
    plt.sci(nodes)
    cbar = plt.colorbar(nodes,fraction=0.046, pad=0.04, ticks=range(4))
    plt.sci(edges)
    plt.axis('on')
    plt.savefig('genealogy_history_{0}_.eps'.format(len(graph)), format='eps', dpi=1200)



def best_worst(history):
    '''
    Query the GA's DEAP history object to get the worst candidate ever evaluated.

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
    for k,v in history_dic.items():
        if np.sum(v.fitness.values) == maximumb:
            worst.append(v)
        if np.sum(v.fitness.values) == minimumb:
            best.append(v)
    return best[0], worst[0]


def plot_evaluate(vms_best,vms_worst,names=['best','worst']):#This method must be pickle-able for ipyparallel to work.
    '''
    A method to plot the best and worst candidate solution waveforms side by side


    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

    '''

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
    import os
    #import net_graph
    vmslist = [vms_best, vms_worst]
    import copy
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
            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(new_file_path,name=str(vms.attrs),backend='NEURON')
            #model.load_model()
            assert type(vms.rheobase) is not type(None)
            #tests = get_neab.suite.tests
            #model.update_run_params(vms.attrs)

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
            if iterator == 0:
                sc_for_frame_best.append(score)
            else:
                sc_for_frame_worst.append(score)

            #vms.score = score



            #for t in get_neab.tests:
            plt.plot(model.results['t'],model.results['vm'],label=str(v)+str(names[iterator])+str(score))
            plt.xlim(0,float(v.params['injected_square_current']['duration']) )
            stored_min.append(np.min(model.results['vm']))
            stored_max.append(np.max(model.results['vm']))
            plt.ylim(np.min(stored_min),np.max(stored_max))
            plt.tight_layout()
            model.results = None
            plt.ylabel('$V_{m}$ mV')
            plt.xlabel('mS')
        plt.savefig(str('test_')+str(v)+'vm_versus_t.eps', format='eps', dpi=1200)
        import pandas as pd
        sf_best = pd.DataFrame(sc_for_frame_best)
        sf_worst = pd.DataFrame(sc_for_frame_worst)


def just_mean(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

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
    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([0, max(mean_many)])
    axes.legend()
    fig.tight_layout()
    fig.savefig('Izhikevich_evolution_just_mean.eps', format='eps', dpi=1200)

def plot_db(vms,name=None):
    '''
    A method to plot raw predictions
    versus observations

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

    '''
    import matplotlib
    import pandas as pd

    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
    matplotlib.style.use('ggplot')
    #f, axarr = plt.subplots(2, sharex=True)
    fig, axarr = plt.subplots(5, sharex=True, figsize=(10, 10), facecolor='white')

    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    import copy
    from itertools import repeat
    #import net_graph
    #vmslist = [vms_best, vms_worst]
    delta = []
    scores = []
    tests = copy.copy(get_neab.tests)
    for k,v in enumerate(tests):
        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(new_file_path,name=str(vms.attrs),backend='NEURON')
        #model.load_model()
        assert type(vms.rheobase) is not type(None)
        #tests = get_neab.suite.tests
        #model.update_run_params(vms.attrs)

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

        #if unit_observations.units == unit_predictions.units:
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


        if k == 0:
            axarr[3].scatter(k, sv,  c='black', label = 'the score')
            axarr[4].scatter(k,0.0,  c='w',label ='the score')

            axarr[k].scatter(k,float(unit_delta),label = 'difference')
            axarr[k].scatter(k,float(unit_observations),label = 'observation')
            axarr[k].scatter(k,float(unit_predictions),label = 'prediction')

            #axarr[k].legend()
        elif k ==1:
            axarr[3].scatter(k, sv,  c='black', label = 'the score')
            axarr[4].scatter(k,0.0,  c='w',label ='the score')

            axarr[k].scatter(k,float(unit_delta),label = 'difference')
            axarr[k].scatter(k,float(unit_observations),label = 'observation')
            axarr[k].scatter(k,float(unit_predictions),label = 'prediction')
            #axarr[k].legend()

        else:
            axarr[4].scatter(k, float(sv),  c='black', label = 'the score')

            axarr[2].scatter(k,float(unit_delta),label = 'difference')
            axarr[2].scatter(k,float(unit_observations),label = 'observation')
            axarr[2].scatter(k,float(unit_predictions),label = 'prediction')
            #axarr[2].legend()



    vms.delta.append(np.mean(delta))

    labels = [ '{0}_{1}'.format(str(t),str(t.observation['value'].units)) for t in tests if 'mean' not in t.observation.keys() ]
    labels.extend([ '{0}_{1}'.format(str(t),str(t.observation['mean'].units))  for t in tests if 'mean' in t.observation.keys() ])
    #labels = labels[1:-1]
    labels = tuple(labels)
    #labels = tuple([str(t.observations.values.units) for t in tests ])
    plt.xlabel('test type')
    plt.ylabel('observation versus prediction')
    tick_locations = tuple(range(0,len(tests)))
    plt.xticks(tick_locations , labels)
    #plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.savefig('obsevation_versus_prediction_{0}.eps'.format(name), format='eps', dpi=1200)
    #import pandas as pd
    #pd.DataFrame(scores).plot(kind='bar', stacked=True)
    #df2.plot(kind='bar', stacked=True);

    return vms

def plot_log(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook
    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.style.use('ggplot')


    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')

    gen_numbers = log.select('gen')
    mean = np.array(log.select('avg'))
    std = np.array(log.select('std'))
    minimum = log.select('min')
    # maximum = log.select('max')
    import get_neab
    objective_labels = [ str(t) for t in get_neab.tests ]

    stdminus = mean - std
    stdplus = mean + std
    axes.plot(
        gen_numbers,
        mean,
        color='black',
        linewidth=2,
        label='population average')

    axes.fill_between(
        gen_numbers,
        stdminus,
        stdplus,
        color='lightgray',
        linewidth=2,
        label='population standard deviation')

    axes.plot(
        gen_numbers,
        minimum,
        linewidth=2,
        label='population objectives')
        # want objective labels to be label.
        # problem is vector scalar mismatch.



    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([0, max(stdplus)])
    axes.legend()

    fig.tight_layout()
    fig.savefig('Izhikevich_history_evolution.eps', format='eps', dpi=1200)


def plot_objectives_history(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Input: DEAP Plot logbook
    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

    '''
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.style.use('ggplot')


    fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')

    gen_numbers = log.select('gen')
    minimum = log.select('min')
    import get_neab
    objective_labels = [ str(t) for t in get_neab.tests ]
    mins_components_plot = log.select('min')
    components = {}
    for i in range(0,7):
        components[i] = []
        for l in mins_components_plot:
            components[i].append(l[i])
    maximum = 0.0
    for keys in components:

        axes.semilogy(
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
    #axes.set_ylim([0, max(maximum[0])])
    axes.legend()

    fig.tight_layout()
    fig.savefig('Izhikevich_evolution_components.eps', format='eps', dpi=1200)
    '''
    def plot_test_waveforms(tests):
        judges = [ i.judge for i in tests ]
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        from itertools import repeat
        import get_neab

    def test_to_model(judges,model):
        import matplotlib.pyplot as plt
        plt.clf()
        matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
        matplotlib.style.use('ggplot')
        for j in judges:
            j(model)

            print(t.observation, t.prediction)
            for t in j.tests:
                v = t.related_data['vm'].rescale('mV')
                time = j.tests.related_data['t']
                plt.plot(v,time)
        plt.savefig('voltage_tests_{0}_{1}.eps'.format(os.pid,j))

    for v in vmpop:
        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(new_file_path,name=str(v.attrs),backend='NEURON')
        model.load_model()
        model.update_run_params(v.attrs)

        plt = list(dview.map(test_to_model,judges,repeat(model)))
    #plot_test_waveforms(get_neab.tests)


    def plot_test_obpre(tests):
        judges = [ i.judge for i in tests ]
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        from itertools import repeat
        def test_to_model(judges,model):
            import matplotlib.pyplot as plt
            plt.clf()
            matplotlib.use('Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
            matplotlib.style.use('ggplot')

            for j in judges:
                j(model)
                obs = []
                pre = []
                print(t.observation, t.prediction)
                for t in j.tests:
                    obs.append(t.observation)
                    pre.append(t.prediction)
                plt.plot(obs,pre)
            plt.savefig('observation_vs_prediction.eps'.format(os.pid,j))

        for v in vmpop:
            new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(new_file_path,name=str(v.attrs),backend='NEURON')
            model.load_model()
            model.update_run_params(v.attrs)

            plt = list(dview.map(test_to_model,judges,repeat(model)))
    plot_test_obpre(get_neab.tests)
    '''



    '''

    import plotly.plotly as py
    from plotly.graph_objs import *
    import plotly.plotly as py
    import plotly.graph_objs as go

    import igraph
    from igraph import *
    igraph.__version__
    g2 = igraph.Graph.Adjacency((nx.to_numpy_matrix(G) > 0).tolist())
    layout = g2.layout('rt')
    pos = [ lay.coords for lay in layout ]
    #pos = graphviz_layout(G, prog="dot")
    print(pos)

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
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
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)


    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
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
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)


    for node, adjacencies in enumerate(G.adjacency_list()):
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = '# of connections: '+str(len(adjacencies))
        node_trace['text'].append(node_info)


    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                    title='<br>Network graph made with Python',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename='networkx')
'''
