
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import graphviz

import matplotlib as mpl
# setting of an appropriate backend.
try:
    mpl.use('Qt5Agg') # Need to do this before importing neuronunit on a Mac, because OSX backend won't work
except:
    mpl.use('Agg')

from plotly.graph_objs import *
import matplotlib.pyplot as plt
import numpy as np



def plotly_graph(history,vmhistory):
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
    positions = graphviz_layout(G, prog="dot")

    dmin=1
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
                title='Genes summed error (absence of fitness)',
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
                title='<br>Geneeology History made with NeuronUnit/DEAP optimizer outputing into networkx and plotly',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='the code this derives from.' \
                    https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/optimization/net_graph.py#L470-L569</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    py.sign_in('RussellJarvis','FoyVbw7Ry3u4N2kCY4LE')
    py.iplot(fig, filename='networkx.svg',image='svg')



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
    '''
    A method to plot the best and worst candidate solution waveforms side by side


    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    Outputs: This method only has side effects, not datatype outputs from the method.
    The most important side effect being a plot in eps format.

    '''
    import os
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat
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
            model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
            #model.load_model()
            assert type(vms.rheobase) is not type(None)
            #tests = get_neab.suite.tests
            #model.update_run_params(vms.attrs)

            if k == 0:
                v.prediction = {}
                v.prediction['value'] = vms.rheobase * pq.pA
            if k != 0:
                v.prediction = None

            if k == 1 or k == 2 or k == 3:
                # Negative square pulse current.
                v.params['injected_square_current']['duration'] = 100 * pq.ms
                v.params['injected_square_current']['amplitude'] = -10 *pq.pA
                v.params['injected_square_current']['delay'] = 30 * pq.ms
            if k==0 or k == 4 or k == 5 or k == 6 or k == 7:
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

def speed_up(not_optional_list):
    import os
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal
    import matplotlib.pyplot as plt
    tests = copy.copy(get_neab.tests)
    the_ks = list(np.arange(0,len(tests),1))

    def nested_function(not_optional_list,k):
        for iterator, vms in enumerate(not_optional_list):
            new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
            print(new_file_path)

            os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
            model = ReducedModel(new_file_path,name='vanilla',backend='NEURON')
            #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
            assert type(vms.rheobase) is not type(None)
            if k == 0:
                v.prediction = {}
                v.prediction['value'] = vms.rheobase * pq.pA
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
            model.load_model()
            model.update_run_params(vms.attrs)
            print(v.params)
            score = v.judge(model,stop_on_error = False, deep_error = True)
            dt = model.results['t'][1] - model.results['t'][0]
            dt = dt*pq.s
            v_m = AnalogSignal(copy.copy(model.results['vm'].to_python()),units=pq.V,sampling_rate=1.0/dt)
            ts = copy.copy(model.results['t'].to_python()) # time signal
            vms.results['ts'] = ts
            vms.results['v_m'] = v_m
            not_optional_list[iterator] = vms
            ts = None
            v_m = None
            model.results = None
        return not_optional_list
        not_optional_list = dview.map_sync(nested_function,not_optional_list,the_ks)
        assert type(not_optional_list[0].results['v_m']) is not type(None)
    return not_optional_list

def shadow(not_optional_list,best_vm):#This method must be pickle-able for ipyparallel to work.
    '''
    A method to plot the best and worst candidate solution waveforms side by side


    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

    '''
    import os
    from neuronunit.models import backends
    from neuronunit.models.reduced import ReducedModel
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal
    import matplotlib.pyplot as plt

    color='lightblue'
    not_optional_list.append(best_vm)

    import copy
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

        #stimes = []
        sindexs = []
        for iterator, vms in enumerate(not_optional_list):

            if vms is not_optional_list[-1]:
                color = 'blue'
            elif iterator == len(not_optional_list):
                color = 'blue'
            else:
                color = 'lightblue'

            #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
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
            model.load_model()
            model.update_run_params(vms.attrs)
            print(v.params)
            score = v.judge(model,stop_on_error = False, deep_error = True)

            if k == 0 or k ==4 or k == 5 or k == 6 or k == 7:

                dt = model.results['t'][1] - model.results['t'][0]
                dt = dt*pq.s
                v_m = AnalogSignal(model.results['vm'],units=pq.V,sampling_rate=1.0/dt)
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
                    plt.plot(ptvec, pvm, label=str(v)+str(score), color=color, linewidth=1)
                    #plt.xlim(np.min(sindexs)-11,np.min(sindexs)+11 )
                    plt.ylim(np.min(stored_min)-4,np.max(stored_max)+4)

            else:
                stored_min.append(np.min(model.results['vm']))
                stored_max.append(np.max(model.results['vm']))
                plt.plot(model.results['t'],model.results['vm'],label=str(v)+str(score), color=color, linewidth=1)
                plt.xlim(0,float(v.params['injected_square_current']['duration']) )
                plt.ylim(np.min(stored_min)-4,np.max(stored_max)+4)
                model.results = None
        #inside the tests loop but outside the model loop.
        plt.tight_layout()
        plt.legend()
        plt.ylabel('$V_{m}$ mV')
        plt.xlabel('mS')
        plt.savefig(str('test_')+str(v)+'vm_versus_t.png')#, format='eps', dpi=1200)



def surfaces(history,td):

    all_inds = history.genealogy_history.values()
    sums = numpy.array([np.sum(ind.fitness.values) for ind in all_inds])
    keep = set()
    quads = []
    for k in range(1,9):
        for i,j in enumerate(td):
            print(i,k)
            if i+k < 10:
                quads.append((td[i],td[i+k],i,i+k))

    for q in quads:
        print(k)
        (x,y,w,z) = q
        print(x,y,w,z,i)
        xs = numpy.array([ind[w] for ind in all_inds])
        ys = numpy.array([ind[z] for ind in all_inds])
        min_ys = ys[numpy.where(sums == numpy.min(sums))]
        min_xs = xs[numpy.where(sums == numpy.min(sums))]
        plt.clf()
        fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')
        trip_axis = ax_trip.tripcolor(xs,ys,sums+1,20,norm=matplotlib.colors.LogNorm())
        plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue',label='global minima')
        fig_trip.colorbar(trip_axis, label='sum of objectives + 1')
        ax_trip.set_xlabel('Parameter '+ str(td[w]))
        ax_trip.set_ylabel('Parameter '+ str(td[z]))
        plot_axis = ax_trip.plot(list(min_xs), list(min_ys), 'o', color='lightblue')
        fig_trip.tight_layout()
        fig_trip.legend()
        fig_trip.savefig('surface'+str(td[w])+str(td[z])+'.eps')


def load_data():
    a = pickle.load(open('for_pandas.p','rb'))
    df = pd.DataFrame(np.transpose(stacked),columns=columns1)
    stacked = opened[0]
    columns1 = opened[1]




def not_just_mean(log,hypervolumes):
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

def bar_chart(vms,name=None):

    import plotly.plotly as py
    from plotly.graph_objs import Bar
    '''
    A method to plot raw predictions
    versus observations

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in eps format.

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
    os.system('sudo /opt/conda/bin/pip install cufflinks')
    import cufflinks as cf

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
    for k,v in enumerate(tests):

        new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
        model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
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

    labels = [ '{0}_{1}'.format(str(t),str(t.observation['value'].units)) for t in tests if 'mean' not in t.observation.keys() ]
    labels.extend([ '{0}_{1}'.format(str(t),str(t.observation['mean'].units))  for t in tests if 'mean' in t.observation.keys() ])


    for t in tests:
        if 'mean' not in t.observation.keys():
            labels[str(t)] = str(t.observation['value'].units)
        if 'mean' in t.observation.keys():
            labels[str(t)] = str(t.observation['mean'].units)
    for k,v in enumerate(columns1):
       columns1[k]=str(v)+labels[v]

    threed = []
    for k,v in test_dic.items():
		# these v[0] ... v[2], types
		# may need to be cast to float
		# for panda ie float(v[0]) etc.
		# hopefuly not though.
        threed.append((v[0],v[1],v[2]))

    # What if the data was normalized first?
    X_std = StandardScaler().fit_transform(threed)

    trans = np.array(threed)
    stacked = np.column_stack(trans)
    with open('for_pandas.p','wb') as handle:
        pickle.dump([stacked,columns1],handle)

    df = pd.DataFrame(np.transpose(np.array(stacked)), columns=columns1)
    df.index = ['observation','prediction','difference']
    df = df.transpose()
    df.loc['CapacitanceTest1.0 F'] *= 10 ** 9
    df.loc['InputResistanceTest1.0 ohm'] *= 10 **-9
    #df = df.drop(['InputResistanceTest1.0 ohm'])
    #df.index(['CapacitanceTest1.0 F'])
    py.sign_in('RussellJarvis','FoyVbw7Ry3u4N2kCY4LE')


    df.iplot(kind='bar', barmode='stack', yTitle='NeuronUnit Test Agreement', title='test agreement', filename='grouped-bar-chart')

    return test_dic


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
