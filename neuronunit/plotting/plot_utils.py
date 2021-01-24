import os

def plot_as_normal(dtc):
    import streamlit as st
    collect_z_offset = []
    for i,t in enumerate(dtc.tests):
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        x1 = -1.01
        x2 = 1.0
        sigma = 1.0
        mu = 0
        x = np.arange(-sigma, sigma, 0.001) # range of x in spec
        x_all = np.arange(-sigma, sigma, 0.001)
        y_point = norm.pdf(mu+float(score.raw),0,1)
        y2 = norm.pdf(x_all,0,1)

        y = norm.pdf(x,0,1)
        y2 = norm.pdf(x_all,0,1)



        x_point = mu+float(score.raw)
        collect_z_offset.append(score.raw)
        name = t.name.split('Test')[0]
        title = str(name)+str(' ')+str(t.observation['mean'].units)

        zplot(x_point,y_point,title)
        st.pyplot()
def plot_as_normal_all(dtc,random):
    import streamlit as st
    collect_z_offset = []
    collect_z_offset_random = []
    for i,t in enumerate(dtc.tests):
        #ax = axes.flat[i]
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        collect_z_offset.append(np.abs(float(score.raw)))

    for i,t in enumerate(random.tests):
        #ax = axes.flat[i]
        t.score_type = ZScore
        model = dtc.dtc_to_model()
        score = t.judge(model)
        collect_z_offset_random.append(np.abs(float(score.raw)))


    x1 = -1.01
    x2 = 1.0
    sigma = 1.0
    mu = 0
    x = np.arange(-sigma, sigma, 0.001) # range of x in spec
    x_all = np.arange(-sigma, sigma, 0.001)
    y_point = norm.pdf(mu+float(np.mean(collect_z_offset)),0,1)
    y2 = norm.pdf(x_all,0,1)

    y = norm.pdf(x,0,1)
    y2 = norm.pdf(x_all,0,1)
    x_point = mu+float(np.mean(collect_z_offset))

    x_point_random = mu+float(np.mean(collect_z_offset_random))
    y_point_random = norm.pdf(mu+float(np.mean(collect_z_offset_random)),0,1)
    best_random = [x_point_random,y_point_random]

    zplot(x_point,y_point,'all_tests',more=best_random)


def zplot(x_point,y_point,title,area=0.68, two_tailed=True, align_right=False, more=None):
    """Plots a z distribution with common annotations
    Example:
        zplot(area=0.95)
        zplot(area=0.80, two_tailed=False, align_right=True)
    Parameters:
        area (float): The area under the standard normal distribution curve.
        align (str): The area under the curve can be aligned to the center
            (default) or to the left.
    Returns:
        None: A plot of the normal distribution with annotations showing the
        area under the curve and the boundaries of the area.
    """
    # create plot object
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    # create normal distribution
    norm = scs.norm()
    # create data points to plot
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)

    ax.plot(x, y)
    ax.scatter(x_point,y_point,c='g',s=150,marker='o')

    if more is not None:
        ax.scatter(more[0],more[1],c='b',s=150,marker='o')


    # code to fill areas for two-tailed tests
    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

        ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                 fontsize=12, rotation=90, va="bottom", ha="left")
    # for one-tailed tests
    else:
        # align the area to the right
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        # align the area to the left
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25',
                            where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")

    # annotate the shaded area
    plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12,
             ha='center')
    # axis labels
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.title(title)
    plt.show()

try:
    import plotly.offline as py
except:
    warnings.warn("plotly")
try:
    import plotly

    plotly.io.orca.config.executable = "/usr/bin/orca"
except:
    print("silently fail on plotly")
try:
    import seaborn as sns
except:
    warnings.warn("Seaborne plotting sub library not available, consider installing")

def check_bin_vm_soma(target,opt):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import quantities as qt
    sns.set(context="paper", font="monospace")

    plt.plot(target.vm_soma.times,target.vm_soma.magnitude,label='Allen Experiment')
    plt.plot(opt.vm_soma.times,opt.vm_soma.magnitude,label='Optimized Model')
    signal = target.vm_soma
    plt.xlabel(qt.s)
    plt.ylabel(signal.dimensionality)
    plt.legend()
    plt.show()

def display_fitting_data():
    cells = pickle.load(open("processed_multicellular_constraints.p","rb"))

    purk = TSD(cells['Cerebellum Purkinje cell'])#.tests
    purk_vr = purk["RestingPotentialTest"].observation['mean']

    ncl5 = TSD(cells["Neocortex pyramidal cell layer 5-6"])
    ncl5.name = str("Neocortex pyramidal cell layer 5-6")
    ncl5_vr = ncl5["RestingPotentialTest"].observation['mean']

    ca1 = TSD(cells['Hippocampus CA1 pyramidal cell'])
    ca1_vr = ca1["RestingPotentialTest"].observation['mean']


    olf = TSD(pickle.load(open("olf_tests.p","rb")))
    olf.use_rheobase_score = False
    cells.pop('Olfactory bulb (main) mitral cell',None)
    cells['Olfactory bulb (main) mitral cell'] = olf


    list_of_dicts = []
    for k,v in cells.items():
        observations = {}
        for k1 in ca1.keys():
            vsd = TSD(v)
            if k1 in vsd.keys():
                vsd[k1].observation['mean']
                observations[k1] = float(vsd[k1].observation['mean'])##,2)

                observations[k1] = np.round(vsd[k1].observation['mean'],2)
                observations['name'] = k
        list_of_dicts.append(observations)
    df = pd.DataFrame(list_of_dicts)
    df = df.set_index('name').T
    return df

def inject_and_plot_passive_model(pre_model,second=None,figname=None,plotly=True):
    model = pre_model.dtc_to_model()
    uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()


    if second is not None:
        model2 = second.dtc_to_model()
        uc = {'amplitude':-10*pq.pA,'duration':500*pq.ms,'delay':100*pq.ms}
        model2.inject_square_current(uc)
        vm2 = model2.get_membrane_potential()
    if plotly and second is None:
        fig = px.line(x=vm.times, y=vm.magnitude, labels={'x':'t (ms)', 'y':'V (mV)'})
        return fig
    if plotly and second is not None:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm.times[0:-1]], y=[float(i) for i in vm.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm2.times[0:-1]], y=[float(i) for i in vm2.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )
        # Add figure title
        fig.update_layout(
            title_text="Compare traces"
        )
        # Set x-axis title
        fig.update_xaxes(title_text="time (ms)")
        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 1", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 2", secondary_y=True)
        return fig
    if not plotly:
        matplotlib.rcParams.update({'font.size': 15})
        plt.figure()
        if pre_model.backend in str("HH"):
            plt.title('Hodgkin-Huxley Neuron')
        else:
            plt.title('Membrane Potential')
        plt.plot(vm.times, vm.magnitude, c='b')#+str(model.attrs['a']))

        plt.plot(vm2.times, vm2.magnitude, c='g')
        plt.ylabel('Time (sec)')

        plt.ylabel('V (mV)')
        plt.legend(loc="upper left")

        if figname is not None:
            plt.savefig('thesis_simulated_data_match.png')
    return vm,plt

def inject_and_not_plot_model(pre_model,known_rh=None):

    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    model = pre_model.dtc_to_model()

    if known_rh is None:
        pre_model = dtc_to_rheo(pre_model)
        if type(model.rheobase) is type(dict()):
            uc = {'amplitude':model.rheobase['value'],'duration':DURATION,'delay':DELAY}
        else:
            uc = {'amplitude':model.rheobase,'duration':DURATION,'delay':DELAY}

    else:
        if type(known_rh) is type(dict()):
            uc = {'amplitude':known_rh['value'],'duration':DURATION,'delay':DELAY}
        else:
            uc = {'amplitude':known_rh,'duration':DURATION,'delay':DELAY}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    return vm

def plotly_version(vm0,vm1,figname=None,snippets=False):

    import plotly.graph_objects as go
    if snippets:
        snippets1 = get_spike_waveforms(vm1)
        snippets0 = get_spike_waveforms(vm0)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in snippets0.times[0:-1]], y=[float(i) for i in snippets0.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=[float(i) for i in snippets1.times[0:-1]], y=[float(i) for i in snippets1.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Double Y Axis Example"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="xaxis title")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
        fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

        fig.show()
        if figname is not None:
            fig.write_image(str(figname)+str('.png'))
        else:
            fig.show()

    else:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm0.times[0:-1]], y=[float(i) for i in vm0.magnitude[0:-1]], name="yaxis data"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=[float(i) for i in vm1.times[0:-1]], y=[float(i) for i in vm1.magnitude[0:-1]], name="yaxis2 data"),
            secondary_y=True,
        )

        # Add figure title

        # Set x-axis title
        fig.update_xaxes(title_text="xaxis title")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 1", secondary_y=False)
        fig.update_yaxes(title_text="<b>Vm (mv)</b> model 2", secondary_y=True)

        if figname is not None:
            fig.write_image(str(figname)+str('.png'))
        else:
            fig.show()

def model_trace(pre_model):
    from neuronunit.tests.base import AMPL, DELAY, DURATION

    # get rheobase injection value
    # get an object of class ReducedModel with known attributes and known rheobase current injection value.
    pre_model = dtc_to_rheo(pre_model)
    model = pre_model.dtc_to_model()
    uc = {'amplitude':model.rheobase,'duration':DURATION,'delay':DELAY}
    model.inject_square_current(uc)
    vm = model.get_membrane_potential()
    return vm
def check_binary_match(dtc0,dtc1,figname=None,snippets=True,plotly=True):

    vm0 = model_trace(dtc0)
    vm1 = model_trace(dtc1)

    if plotly:
        plotly_version(vm0,vm1,figname,snippets)
    else:
        matplotlib.rcParams.update({'font.size': 8})

        plt.figure()

        if snippets:
            plt.figure()

            snippets1 = get_spike_waveforms(vm1)
            snippets0 = get_spike_waveforms(vm0)
            plt.plot(snippets0.times,snippets0.magnitude,label=str('model type: '))#+label)#,label='ground truth')
            plt.plot(snippets1.times,snippets1.magnitude,label=str('model type: '))#+label)#,label='ground truth')
            if dtc0.backend in str("HH"):
                plt.title('Check for waveform Alignment variance exp: {0}'.format(basic_expVar(snippets1, snippets0)))
            else:
                plt.title('membrane potential: variance exp: {0}'.format(basic_expVar(snippets1, snippets0)))
            plt.ylabel('V (mV)')
            plt.legend(loc="upper left")

            if figname is not None:
                plt.savefig(figname)

        else:
            if dtc0.backend in str("HH"):
                plt.title('Check for waveform Alignment')
            else:
                plt.title('membrane potential plot')
            plt.plot(vm0.times, vm0.magnitude,label="target")
            plt.plot(vm1.times, vm1.magnitude,label="solutions")
            plt.ylabel('V (mV)')
            plt.legend(loc="upper left")

            if figname is not None:
                plt.savefig(figname)
