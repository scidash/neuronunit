
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
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
    '''


    import os
    import subprocess
    os.system('sudo /opt/conda/bin/pip install gprof2dot')
    os.system('cp /opt/conda/lib/python3.5/site-packages/gprof2dot.py .')
    prof_f_name = '{0}'.format(os.getpid())
    #Open and close the file, as a quick and dirty way to confirm that exists.
    f = open('NeuroML2/{0}'.format(prof_f_name),'wb')
    file_name = 'NeuroML2/{0}'.format(prof_f_name)
    f.close()

    #subprocess.Popen('python','gprof2dot.py' -f -n0 -e0 pstats {0}  | dot -Tsvg -o {0}.svg'.format(prof_f_name))

    os.system('python gprof2dot.py -f -n0 -e0 pstats {0}  | dot -Tsvg -o {0}.svg'.format(prof_f_name))

def graph_s(graph):

    plt.clf()
    import networkx
    import numpy as np
    graph = networkx.DiGraph(history.genealogy_tree)
    graph = graph.reverse()     # Make the grah top-down
    labels ={}
    for i in graph.nodes():
        labels[i] = i
    colors = [ np.sum(history.genealogy_history[i].fitness.values) for i in graph ]
    positions = graphviz_layout(graph, prog="dot")
    networkx.draw(graph, positions, node_color=colors, node_size=1.5, labels = labels)#, interpolation='none')
    #plt.colorbar()
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig('genealogy_history_{0}_.eps'.format(gen))


def bpyopt(pf):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/graupnerbrunelstdp.ipynb
    '''
    import numpy as np
    #import run_fit
    best_ind_dict_vm = update_vm_pop(pf)
    best_ind_dict_vm[0].attrs
    #best_ind_dict
    gs = [ ind for ind in history.genealogy_history.itervalues() if np.all(np.array(ind.fitness.values) < 2.0) ]
    good_solutions = update_vm_pop(gs)



def just_mean(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Plot logbook
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
    fig.savefig('Izhikevich_evolution_just_mean.eps')

def plot_log(log):
    '''
    https://github.com/BlueBrain/BluePyOpt/blob/master/examples/graupnerbrunelstdp/run_fit.py
    Plot logbook
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
        label=r'population standard deviation')

    axes.plot(
        gen_numbers,
        minimum,
        color='red',
        linewidth=2,
        label='population minimum')

    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([0, max(stdplus)])
    axes.legend()

    fig.tight_layout()
    fig.savefig('Izhikevich_evolution.eps')

    def plot_test_waveforms(tests):
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
    plot_test_waveforms(get_neab.tests)

    '''
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
