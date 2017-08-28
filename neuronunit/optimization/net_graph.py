
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

def plotly_graph(history,vmhistory):
	# TODO experiment with making the plot output style a dendro 
	# dendrograms
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
    The most important side effect being a plot in png format.

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


def speed_up(not_optional_list):
    '''
    This will be used in conjunction with rgerkin@github 's latest unpickable method
    To achieve a big speed up by storing sciunit scores inside models, such that they don't need to
    be reavaluated, each time.

    Also awaiting another workaround.
    '''
    import ipyparallel as ipp

    rc = ipp.Client(profile='default')
    rc[:].use_cloudpickle()
    dview = rc[:]

    import os
    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal
    import matplotlib.pyplot as plt
    import copy
    tests = get_neab.tests
    the_ks = list(np.arange(0,len(tests),1))
    print(the_ks)
    for i in not_optional_list:
         for j in tests:
            i.results[str(j)] = {}
    tests = None
    print([i.results for i in not_optional_list])
    #vms = not_optional_list
    def second_nesting(k,vms):
        import get_neab

        new_file_path = '{0}{1}'.format(str(get_neab.LEMS_MODEL_PATH),int(os.getpid()))
        #print(new_file_path)
        os.system('cp ' + str(get_neab.LEMS_MODEL_PATH)+str(' ') + new_file_path)
        # These imports have to happen very locally otherwise all hell breaks loose.
        from neuronunit.models import backends
        from neuronunit.models.reduced import ReducedModel
        #model = ReducedModel(get_neab.LEMS_MODEL_PATH,name='vanilla',backend='NEURON')
        model = ReducedModel(new_file_path,name=str('vanilla'),backend='NEURON')
        tests = get_neab.tests

        v = tests[k]
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

        v_m = model.get_membrane_potential()
        ts = model.results['t']# time signal
        if type(vms.results[str(v)]) is type(None):
            vms.results[str(v)] = {}
        vms.results[str(v)]['ts'] = copy.copy(ts)
        vms.results[str(v)]['v_m'] = copy.copy(v_m)
        #not_optional_list[iterator] = vms
        #ts = None
        #v_m = None
        #model.results = None
        return vms

    def nested_function(vms,the_ks):
        print(vms,k)
        from itertools import repeat
        print(type(k))
        print(k)
        #for k in the_ks:
        vms = list(map(second_nesting,the_ks,repeat(vms)))
        #print(vms.results)
        return vms

    from itertools import repeat
    not_optional_list = list(dview.map_sync(nested_function,not_optional_list,repeat(the_ks)))

    return not_optional_list




def sp_spike_width(best_worst):#This method must be pickle-able for ipyparallel to work.
    '''
    A method to plot the best and worst candidate solution waveforms side by side
    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.
    Outputs: This method only has side effects, not datatype outputs from the method.
    The most important side effect being a plot in png format.
    '''
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




def shadow(not_optional_list,best_vm):#This method must be pickle-able for ipyparallel to work.
    '''
    A method to plot the best and worst candidate solution waveforms side by side


    Inputs: An individual gene from the population that has compound parameters, and a tuple iterator that
    is a virtual model object containing an appropriate parameter set, zipped togethor with an appropriate rheobase
    value, that was found in a previous rheobase search.

    Outputs: This method only has side effects, not datatype outputs from the method.

    The most important side effect being a plot in png format.

    '''
    import os

    import quantities as pq
    import numpy as np
    import get_neab
    from itertools import repeat

    from neuronunit.capabilities import spike_functions
    import quantities as pq
    from neo import AnalogSignal
    import matplotlib.pyplot as plt

    #color='lightblue'
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

        sindexs = []
        for iterator, vms in enumerate(not_optional_list):


            from neuronunit.models import backends
            from neuronunit.models.reduced import ReducedModel

            print(get_neab.LEMS_MODEL_PATH)
            #new_file_path = str(get_neab.LEMS_MODEL_PATH)+str(os.getpid())
            model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend='NEURON')
            print(dir(model))
            print(dir(ReducedModel))

            print(os.getcwd())

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
        plt.savefig(str('test_')+str(v)+'vm_versus_t.png', format='png', dpi=1200)



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
    #fig_trip.legend()
    fig_trip.savefig('surface'+str('a')+str('b')+'.pdf',format='pdf', dpi=1200)


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
    plt.savefig('2nd_approach_d_error_'+str('a')+str('b')+'surface.png')


    return ab

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
    fig.savefig('Izhikevich_evolution_just_mean.png', format='png', dpi=1200)

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
    #import pdb;
    #pdb.set_trace()
    # uncoment this line to make a bar chart on plotly.
    # df.iplot(kind='bar', barmode='stack', yTitle='NeuronUnit Test Agreement', title='test agreement every test', filename='grouped-bar-chart')
    html = df.to_html()
    html_file = open("tests_agreement_table.html","w")
    html_file.write(html)
    html_file.close()
    return df, threed, columns1 ,stacked, html, test_dic

def plot_log(log,hypervolumes):
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

    axes.plot(
        gen_numbers,
        hypervolumes,
        color='red',
        linewidth=2,
        label='Solution Hypervolume')
        # want objective labels to be label.
        # problem is vector scalar mismatch.



    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    axes.set_ylim([0, max(stdplus)])
    axes.legend()

    fig.tight_layout()
    fig.savefig('Izhikevich_history_evolution.png', format='png', dpi=1200)


def plot_objectives_history(log):
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
    minimum = log.select('min')
    mean = log.select('mean')

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
    '''
    axes.semilogy(
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
    '''
    axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    axes.set_xlabel('Generation #')
    axes.set_ylabel('Sum of objectives')
    #axes.set_ylim([0, max(maximum[0])])
    axes.legend()

    fig.tight_layout()
    fig.savefig('Izhikevich_evolution_components.png', format='png', dpi=1200)
