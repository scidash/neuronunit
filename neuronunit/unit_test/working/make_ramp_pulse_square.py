
from __future__ import division
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyNN.utility import normalized_filename
import izhikevich as izhi
from numba import jit
from tqdm.auto import tqdm
import time
import numba

global_time_step = 0.25

plt.rcParams.update({
    'lines.linewidth': 0.5,
    'legend.fontsize': 'small',
    'axes.titlesize': 'small',
    'font.size': 6,
    'savefig.dpi': 200,
})


def run_simulation(time_step=global_time_step, a=0.02, b=0.2, c=-65.0, d=6.0,
                   C=100, k=0.7, vr=-60, vt=-40, vpeak=35,
                   u_init=None, v_init=-70.0, waveform=None, t_stop=100.0,
                   title="", scalebar_level=0, label_scalebar=False,
                   save_data=False):
    """
    Run a simulation of a single neuron.

    Arguments:
        time_step - time step used in solving the differential equations
        a - time scale of the recovery variable u
        b - sensitivity of u to the subthreshold fluctuations of the membrane potential v
        c - after-spike reset value of v
        d - after-spike reset of u
        u_init - initial value of u
        v_init - initial value of v
        waveform - a tuple of two NumPy arrays, containing time and amplitude data for the injected current
        t_stop - duration of the simulation
        title - a title to be added to the figure panel for this simulation
        scalebar_level - a value between 0 and 1, controlling the vertical placement of the scalebar
        label_scalebar - True or False, whether to add a label to the scalebar
    """
    global j, fig, gs

    # create a neuron and current source

    #sim.setup(timestep=time_step)

    if u_init is None:
        u_init = b * v_init
    #initialValues = {'u': u_init, 'v': v_init}

    #cell_type = sim.Izhikevich(a=a, b=b, c=c, d=d, i_offset=0.0)
    attrs = {}
    model = izhi.IZHIModel()
    #attrs = model.default_attrs
    attrs['vr'] = v_init
    attrs['u_init'] = u_init

    attrs['a'] = a
    attrs['b'] = b
    attrs['c'] = c
    attrs['d'] = d
    model.attrs = {}
    model.attrs = attrs
    #neuron = sim.create(cell_type)
    #neuron.initialize(**initialValues)

    #neuron.record('v')

    times, amps = waveform
    t1 = time.time()

    vm = model.wrap_known_i(amps,times,two_thousand_and_three=True)
    print(len(vm),np.std(vm),np.mean(vm))
    t2 = time.time()
    print('time taken on block {0} '.format(t2-t1))
    print(len(vm),len(vm.times))

    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1,
                                           subplot_spec=gs[j//4, j%4],
                                           height_ratios=[8, 1],
                                           hspace=0.0)
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])

    j += 1
    for ax in (ax1, ax2):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_color('None')
        ax.spines['right'].set_color('None')
        ax.spines['bottom'].set_color('None')
        ax.spines['top'].set_color('None')
        #ax.set_xlim(0.0, float(vm.times[-1]))

    ax1.set_title(title)

    #vm = data.filter(name='v')[0]
    #i_times, i_vars = stepify(times, amps)
    #print(len(i_times),len(i_vars))
    ##
    #
    ###
    ax1.plot(vm.times, vm)
    ax1.set_ylim(-90, 30)
    #plt.plot(vm.times, vm)
    #plt.show()
    #amps,times
    ax2.plot(times, amps, 'g')

    ymin, ymax = amps.min(), amps.max()
    padding = (ymax - ymin)/10
    ax2.set_ylim(ymin - padding, ymax + padding)

    # scale bar
    scalebar_y = ymin + (ymax - ymin) * scalebar_level
    ax2.plot([t_stop - 20, t_stop], [scalebar_y, scalebar_y],
             color='k', linestyle='-', linewidth=1)
    if label_scalebar:
        ax.text(t_stop, ymin + padding, "20 ms", fontsize=4, horizontalalignment='right')

    plt.show(block=True)
    fig.canvas.draw()
    if save_data:
        datfilename = 'results/results_file.dat'#"results/%s_%s.dat" % (title.replace("(","").replace(")","").replace(" ","_"),options.simulator)

        datfile = open('results/results_file.dat','w')
        for i in range(len(vm)):
            datfile.write('%s\t%s\n'%(vm.times[i].magnitude,vm[i][0].magnitude))
        datfile.close()
        print('Saved data to %s'%datfilename)

@jit
def make_step(amplitude, t_stop,time_step=global_time_step):
    """
    Generate the waveform for a current
    that starts at zero and is stepped up
    to the given amplitude at time t_stop/10.
    """

    times = np.array([0, t_stop/10, t_stop])
    amps = np.array([0, amplitude, amplitude])
    delay = t_stop/10
    duration = t_stop
    tMax = t_stop#delay + duration #+ 200.0#*pq.ms
    times = np.arange(0,tMax,global_time_step)
    N = int(tMax/time_step)
    Iext = np.zeros(N)
    delay_ind = int((delay/tMax)*N)
    duration_ind = int((duration/tMax)*N)

    Iext[0:delay_ind-1] = 0.0
    Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
    Iext[delay_ind+duration_ind::] = 0.0

    return times, Iext


#@jit
def make_pulse(amplitude, onsets, width, t_stop, baseline=0.0):
    """
    Generate the waveform for a series of current pulses.

    Arguments:
        amplitude - absolute current value during each pulse
        onsets - a list or array of times at which pulses begin
        width - duration of each pulse
        t_stop - total duration of the waveform
        baseline - the current value before, between and after pulses.
    """
    times = [0]
    amps = [baseline]
    for onset in onsets:
        times += [onset, onset + width]
        amps += [amplitude, baseline]
    times += [t_stop]
    amps += [baseline]

    #times = np.array([0, t_stop/10, t_stop])
    #amps = np.array([0, amplitude, amplitude])
    delay = t_stop/10
    duration = t_stop
    tMax = t_stop#delay + duration #+ 200.0#*pq.ms
    times = np.arange(0,tMax,global_time_step)
    N = int(tMax/global_time_step)
    Iext = np.zeros(N)

    on_indexs = []
    off_indexs = []
    contribution = baseline
    for onset in onsets:
        #times += [onset, onset + width]
        #amps += [amplitude, baseline]

        on_indexs.append(int((onset/tMax)*N))
        off_indexs.append(int(((onset+width)/tMax)*N))
        #duration_ind = int((duration/tMax)*N)
        contribution += amplitude
        Iext[on_indexs[-1]:off_indexs[-1]] = contribution
        #Iext[delay_ind+duration_ind::] = 0.0

    Iext[0:on_indexs[0]] = 0.0

    return np.array(times), np.array(Iext)

@jit
def make_ramp(gradient, onset, t_stop, baseline=0.0, time_step=global_time_step, t_start=0.0):
    """
    Generate the waveform for a current which is initially constant
    and then increases linearly with time.

    Arguments:
        gradient - gradient of the ramp
        onset - time at which the ramp begins
        t_stop - total duration of the waveform
        baseline - current value before the ramp
        time_step - interval between increments in the ramp current
        t_start - time at which the waveform begins (used to construct waveforms
                  containing multiple ramps).
    """
    if onset > t_start:
        times = np.hstack((np.array((t_start, onset)),  # flat part
                           np.arange(onset + time_step, t_stop + time_step, time_step)))  # ramp part
    else:
        times = np.arange(t_start, t_stop + time_step, time_step)
    amps = baseline + gradient*(times - onset) * (times > onset)
    return times, amps


# == Get command-line options, import simulator backend =====================



# == Initialize figure ======================================================

j = 0
plt.ion()
fig = plt.figure(1, facecolor='white', figsize=(6, 6))
gs = gridspec.GridSpec(6, 4)
gs.update(hspace=0.5, wspace=0.4)

pbar = tqdm(total=21)



#%%%%%%%%%%%%%%% thalamo-cortical burst (TC) %%%%%%%%%%%%%%%%%%%%%%
#subplot(2,4,6)
#V=-87;  u=b*V;
#VV=[];  uu=[];
#tau = 0.25; tspan = 0:tau:150;
#T1=3*tspan(end)/10;
#for t=tspan#
#    if (t>T1)
#        I=0.0;
#    else
#        I=-25;
#    end;


#%%%%%%%%%%%%%%% chattering (CH) %%%%%%%%%%%%%%%%%%%%%%
#subplot(2,4,3)
#a=0.02; b=0.2;  c=-50;  d=2;
#V=-70;  u=b*V;
#VV=[];  uu=[];
#tau = 0.25; tspan = 0:tau:150;
#T1=tspan(end)/10;

t_stop = 150.0
run_simulation(a=0.02, b=0.2,  c=-50,  d=2,v_init=-70,
               waveform=make_step(10.0, t_stop),
               t_stop=t_stop, title='chattering (CH)',
               label_scalebar=True, save_data=True)

tau = 0.25;
tspan = np.arange(0,150/tau,tau);

T1=3*tspan[-1]/10;
I = []
for t in tspan:
    if (t>T1):
        I.append(0.0);
    else:
        I.append(-25.0);
wv=[I,tspan]
t_stop = 150.0
run_simulation(a=0.02, b=0.25,  c=-65,  d=0.05,v_init=-87,
               waveform=wv,#step(-25, t_stop),
               t_stop=t_stop, title='thalamo-cortical burst (TC)',
               label_scalebar=True, save_data=True)

# == Sub-plot A: Tonic spiking ==============================================
pbar.update(1)

t_stop = 100.0
run_simulation(a=0.02, b=0.2, c=-65.0, d=6.0, v_init=-70.0,
               waveform=make_step(0.014, t_stop),
               t_stop=t_stop, title='(A) Tonic spiking',
               label_scalebar=True, save_data=True)
#filename = "results_izhikevich2004_1.png"#normalized_filename()
#try:
#    os.makedirs(os.path.dirname(filename))
#except OSError:
#    pass
#fig.savefig("results_izhikevich2004_1.png")

# == Sub-plot B: Phasic spiking =============================================
pbar.update(1)

t_stop = 200.0
run_simulation(a=0.02, b=0.25, c=-65.0, d=6.0, v_init=-64.0,
               waveform=make_step(0.0005, t_stop),
               t_stop=t_stop, title='(B) Phasic spiking')
fig.savefig("results_izhikevich2004_12png")

# == Sub-plot C: Tonic bursting =============================================
pbar.update(1)

_stop = 220.0
run_simulation(a=0.02, b=0.2, c=-50.0, d=2.0, v_init=-70.0,
               waveform=make_step(0.015, t_stop),
               t_stop=t_stop, title='(C) Tonic bursting', save_data=True)

# == Sub-plot D: Phasic bursting ============================================
pbar.update(1)

t_stop = 200.0
run_simulation(a=0.02, b=0.25, c=-55.0, d=0.05, v_init=-64.0,
               waveform=make_step(0.0006, t_stop),
               t_stop=t_stop, title='(D) Phasic bursting')

# == Sub-plot E: Mixed mode =================================================
pbar.update(1)

t_stop = 160.0
run_simulation(a=0.02, b=0.2, c=-55.0, d=4.0, v_init=-70.0,
               waveform=make_step(0.01, t_stop),
               t_stop=t_stop, title='(E) Mixed mode')

# == Sub-plot F: Spike Frequency Adaptation (SFA) ===========================
pbar.update(1)

t_stop = 85.0
run_simulation(a=0.01, b=0.2, c=-65.0, d=8.0, v_init=-70.0,
               waveform=make_step(0.03, t_stop),
               t_stop=t_stop, title='(F) SFA')

# == Sub-plot G: Class 1 excitable ==========================================

'''
Note: This simulation is supposed to use a different parameterization of the
      model, i.e.
            V' = tau*(0.04*V^2 + 4.1*V + 108 -u + I)
      as opposed to
            V' = tau*(0.04*V^2 + 5*V + 140 - u + I)
The alternative parameterization is not currently available in PyNN, therefore
the results of this simulation are not expected to match the original figure.
'''
pbar.update(1)

t_stop = 300.0
run_simulation(a=0.02, b=0.2, c=-65.0, d=6.0, v_init=-70.0,
               waveform=make_ramp(0.000075, 30.0, t_stop),
               t_stop=t_stop, title='(G) Class 1 excitable')

# == Sub-plot H: Class 2 excitable ==========================================
pbar.update(1)

t_stop = 300.0
run_simulation(a=0.2, b=0.26, c=-65.0, d=0.0, v_init=-64.0,
               waveform=make_ramp(0.000015, 30.0, t_stop, baseline=-0.0005),
               t_stop=t_stop, title='(H) Class 2 excitable')

# == Sub-plot I: Spike latency ==============================================
pbar.update(1)

t_stop = 100.0
run_simulation(a=0.02, b=0.2, c=-65.0, d=6.0, v_init=-70.0,
               waveform=make_pulse(0.00671,  # 0.00704 in original
                              [10], 3, t_stop),
               t_stop=t_stop, title='(I) Spike latency',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot J: Subthreshold oscillation ===================================

t_stop = 200.0
run_simulation(a=0.05, b=0.26, c=-60.0, d=0.0, v_init=-62.0,
               waveform=make_pulse(0.002, [20], 5, t_stop),
               t_stop=t_stop, title='(J) Subthreshold oscillation',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot K: Resonator ==================================================

t_stop = 400.0
T1 = t_stop / 10
T2 = T1 + 20
T3 = 0.7 * t_stop
T4 = T3 + 40
run_simulation(a=0.1, b=0.26, c=-60.0, d=-1.0, v_init=-62.0,
               waveform=make_pulse(0.00065, [T1, T2, T3, T4], 4, t_stop),
               t_stop=t_stop, title='(K) Resonator',
               scalebar_level=0.5)

# == Sub-plot L: Integrator =================================================
pbar.update(1)

'''
Note: This simulation is supposed to use a different parameterization of the
      model, i.e.
            V' = tau*(0.04*V^2 + 4.1*V + 108 -u + I)
      as opposed to
            V' = tau*(0.04*V^2 + 5*V + 140 - u + I)
The alternative parameterization is not currently available in PyNN, therefore
the results of this simulation are not expected to match the original figure.
'''

t_stop = 100.0
T1 = t_stop / 11
T2 = T1 + 5
T3 = 0.7 * t_stop
T4 = T3 + 10
run_simulation(a=0.02, b=-0.1, c=-55.0, d=6.0, v_init=-60.0,
               waveform=make_pulse(0.009, [T1, T2, T3, T4], 2, t_stop),
               t_stop=t_stop, title='(L) Integrator',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot M: Rebound spike ==============================================

t_stop = 200.0
run_simulation(a=0.03, b=0.25, c=-60.0, d=4.0, v_init=-64.0,
               waveform=make_pulse(-0.015, [20], 5, t_stop),
               t_stop=t_stop, title='(M) Rebound spike')

# == Sub-plot N: Rebound burst ==============================================
pbar.update(1)

t_stop = 200.0
run_simulation(a=0.03, b=0.25, c=-52.0, d=0.0, v_init=-64.0,
               waveform=make_pulse(-0.015, [20], 5, t_stop),
               t_stop=t_stop, title='(N) Rebound burst')

# == Sub-plot O: Threshold variability ======================================
pbar.update(1)

t_stop = 100.0
times = np.array([0, 10, 15, 70, 75, 80, 85, t_stop])
amps = np.array([0, 0.001, 0, -0.006, 0, 0.001, 0, 0])
run_simulation(a=0.03, b=0.25, c=-60.0, d=4.0, v_init=-64.0,
               waveform=(times, amps),
               t_stop=t_stop, title='(O) Threshold variability')
pbar.update(1)

# == Sub-plot P: Bistability ================================================

t_stop = 300.0
T1 = t_stop/8
T2 = 208  # 216.0 in original
run_simulation(a=0.1, b=0.26, c=-60.0, d=0.0, v_init=-61.0,
               waveform=make_pulse(0.00124, [T1, T2], 5, t_stop, baseline=0.00024),
               t_stop=t_stop, title='(P) Bistability',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot Q: Depolarizing after-potential ===============================

t_stop = 50.0
run_simulation(a=1.0, b=0.18,  # 0.2 in original
               c=-60.0, d=-21.0, v_init=-70.0,
               waveform=make_pulse(0.02, [9], 2, t_stop),
               t_stop=t_stop, title='(Q) DAP',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot R: Accomodation ===============================================

'''
Note: This simulation is supposed to use a different parameterization of the
      model, i.e.
            u' = tau*a*(b*(V + 65))
      as opposed to
            u' = tau*a*(b*V - u)
The alternative parameterization is not currently available in PyNN, therefore
the results of this simulation are not expected to match the original figure.
'''

t_stop = 400.0

parts = (make_ramp(0.00004, 0.0, 200.0),
         (np.array([200.0 + global_time_step, 300.0 - global_time_step]), np.array([0.0, 0.0])),
         make_ramp(0.00032, 300.0, 312.5, t_start=300.0),
         (np.array([312.5 + global_time_step, t_stop]), np.array([0.0, 0.0])))
totalTimes, totalAmps = np.hstack(parts)
pbar.update(1)

run_simulation(a=0.02, b=1.0, c=-55.0, d=4.0, v_init=-65.0, u_init=-16.0,
               waveform=(totalTimes, totalAmps),
               t_stop=t_stop, title='(R) Accomodation',
               scalebar_level=0.5)
pbar.update(1)

# == Sub-plot S: Inhibition-induced spiking =================================

t_stop = 350.0
run_simulation(a=-0.02, b=-1.0, c=-60.0, d=8.0, v_init=-63.8,
               waveform=make_pulse(0.075, [50], 170,  # 200 in original
                              t_stop, baseline=0.08),
               t_stop=t_stop, title='(S) Inhibition-induced spiking')
pbar.update(1)

# == Sub-plot T: Inhibition-induced bursting ================================

'''
Modifying parameter d from -2.0 to -0.7 in order to reproduce Fig. 1
'''

t_stop = 350.0
run_simulation(a=-0.026, b=-1.0, c=-45.0, d=-0.7, v_init=-63.8,
               waveform=make_pulse(0.075, [50], 200, t_stop, baseline=0.08),
               t_stop=t_stop, title='(T) Inhibition-induced bursting')
pbar.update(1)


# == Export figure in PNG format ============================================

filename = "results_izhikevich2004.png"#normalized_filename()
#try:
#    os.makedirs(os.path.dirname(filename))
#except OSError:
#    pass
fig.savefig("results_izhikevich2004.png")

print("\n  Simulation complete. Results can be seen in figure at %s\n"%(filename))
