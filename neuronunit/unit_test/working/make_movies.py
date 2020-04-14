#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys    

import os
get_ipython().run_line_magic('matplotlib', 'inline')
os.system('jupyter trust progress_report_10thJuly.ipynb')
print('Running on Python version: {}'.format(sys.version))
import pickle
hc = pickle.load(open('held_constant_grid.p','rb'))




# In[2]:


from neuronunit.optimization import get_neab
import copy

electro_path = str(os.getcwd())+'/pipe_tests.p'
print(os.getcwd())
assert os.path.isfile(electro_path) == True
with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)

electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]


# In[3]:




print(hc.keys())

print(hc['vrb'][0])
print(hc['vrb'][0][-1])
print(hc['vra'][0][-1])
print(hc['ba'][0][-1])

print('\n\n\n\n')
      
      
print(hc['vrb'][-1][-1]) 
print(hc['vra'][-1][-1])
print(hc['ba'][-1][-1])


print([h[-1] for h in hc])


# #All I can tell is that changing VR does not do anythingg

# In[4]:


try:
    from prettyplotlib import plt
except: 
    import matplotlib.pyplot as plt

from neuronunit.optimization import get_neab
import copy
import os
import pickle
electro_path = str(os.getcwd())+'/pipe_tests.p'
from neuronunit import plottools
import numpy as np
ax = None
from neuronunit.optimization import exhaustive_search as es

plot_surface = plottools.plot_surface
scatter_surface = plottools.plot_surface

with open(electro_path,'rb') as f:
    electro_tests = pickle.load(f)
with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)
    
with open('pre_grid_reports.p','rb') as f:#
    grid_results = pickle.load(f)

with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)
pop = package[0]
print(pop[0].dtc.attrs.items())
history = package[4]
gen_vs_pop =  package[6]
    
pop = package[0]
electro_tests = get_neab.replace_zero_std(electro_tests)
electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
test, observation = electro_tests[0]

npoints = 6
tests = copy.copy(electro_tests[0][0][0:2])

fig,ax = plt.subplots(3,3,figsize=(10,10))
attrs_list = list(pop[0].dtc.attrs.keys())

    
hc['vra']    
hckeys = list(hc.keys())
with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)

grid_results = {}
hof = package[1]
from matplotlib.colors import LogNorm


from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params

for i,ki in enumerate(pop[0].dtc.attrs.keys()):
    for j,kj in enumerate(pop[0].dtc.attrs.keys()):

        ax[i,j].set_title(' {0} vs  {1}'.format(str(ki),str(kj)))
        ss = set([ki,kj])
        bs = set(attrs_list)
        diff = bs.difference(ss)
        bd =  {}
        bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]
        print(bd)
        for k,v in hc.items():
            if str(ki) in k and str(kj) in k:
                key = k
                
        tuples = np.array([ (np.sum(list(p.dtc.scores.values())), p.dtc.attrs[str(ki)],p.dtc.attrs[str(ji)]) for p in hc[key] ])        
        
        z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
        # x = np.array([ p.dtc.attrs[str(ki)] for p in hc[key] ])
        # y = np.array([ p.dtc.attrs[str(kj)] for p in hc[key] ])

        N = int(np.sqrt(len(x)))
        x = x.reshape((N, N))
        y = y.reshape((N, N))
        z = z.reshape((N, N))
        
        #plt.pcolor(X, Y, Z1)

        ax[i,j].pcolormesh(x, y, z)#, norm=LogNorm(vmin=z.min(), vmax=z.max()), cmap='PuBu_r')
        if i==j:
            
            ax[i,j].set_title(' {0} vs  score'.format(attrs_list[i],attrs_list[j]))

            print(key)
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
            x = [i for i in range(0,len(z))]
            ax[i,j].plot(x,z)
            
            ax[i,j].set_xlim(np.min(x),np.max(x))
            ax[i,j].set_ylim(np.min(z),np.max(z))
            
        
            #if 'vr' in key:
                #print(z,np.mean(z),np.std(z))
            
print(hc.keys())
     


# In[ ]:


#from matplotlib.colors import LogNorm
plt.clf()
fig,ax = plt.subplots(3,3,figsize=(10,10))

visited = []
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid, mock_grid
one_d = {}
two_d = {}
cnt = 0

#from neuronunit.optimization.model_parameters import model_params
      
from neuronunit.optimization import model_parameters as modelp
mp = modelp.model_params
for i,ki in enumerate(pop[0].dtc.attrs.keys()):
    for j,kj in enumerate(pop[0].dtc.attrs.keys()):

            

        free_param = set([ki,kj]) # construct a small-set out of the indexed keys 2.
        fp = { k:None for k in list(free_param) }

        bs = set(attrs_list) # construct a full set out of all of the keys available 3. 

        diff = bs.difference(free_param) # diff is simply the key that is not indexed.
        visited.append(diff)

        # BD is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.

        bd =  {}
        for d in diff:
            bd[d] = hof[0].dtc.attrs[d]

        for k,v in hc.items():
            if str(ki) in k and str(kj) in k:
                key = k
                
        if ki == kj:
            #pass
            gr = run_grid(10,tests,provided_keys = fp ,hold_constant = bd)#, use_cache = True, cache_name='complex')
            one_d[str(fp.keys())] = gr
            ax[i,j].plt()
        if ki != kj:
            
      
            reduced = { k:mp[k] for k in fp.keys() }
            gr = None
            gr = run_grid(10,tests,provided_keys = reduced ,hold_constant = bd)
            if str('kj') not in two_d[ki].keys():
                two_d[ki] = {}
                two_d[ki][kj] = gr
            else:
                two_d[ki][kj] = gr
            
            print(key)
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])

            
            x = [ p.dtc.attrs[ki] for p in hc[key] ]
            y = [ p.dtc.attrs[kj] for p in hc[key] ]
            
            N = int(np.sqrt(len(x)))
            x = np.array(x).reshape((N, N))
            y = np.array(y).reshape((N, N))
            z = np.array(z).reshape((N, N))
            print(len(x),len(y),len(z), np.shape(z))
            ax[i,j].pcolormesh(x, y, z)
            
with open('surfaces.p','wb') as f:           
    pickle.dump(f,[two_d,one_d,ax])


# In[ ]:


one_d


# In[ ]:


plt.clf()
fig,ax = plt.subplots(3,3,figsize=(10,10))

for i in range(len(attrs_list)):
    for j in range(len(attrs_list)):
        if i>j:
            ax[i,j].set_title(' {0} vs  {1}'.format(attrs_list[i],attrs_list[j]))
            ss = set([attrs_list[j],attrs_list[i]])
            bs = set(attrs_list)
            diff = bs.difference(ss)
            bd =  {}
            bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]
    
        
            #ax_trip,plot_axis = plot_surface(fig,ax[i,j],attrs_list[j],attrs_list[i],history)
            #ax[i,j].plot_axis = plot_axis
            
            for k,v in hc.items():
                if str(attrs_list[j]) in k and str(attrs_list[i]) in k:
                    key = k
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
            x = np.array([ p.dtc.attrs[str(attrs_list[i])] for p in hc[key] ])
            y = np.array([ p.dtc.attrs[str(attrs_list[j])] for p in hc[key] ])
            
            print(x[0],'x')
            print(y[0],'y')
            print(z[0],'z')
            
            for k,l in enumerate(x):
                if k!=0:
                    diff = l - y[i-1]
                    #print(diff)
            
            N = int(np.sqrt(len(x)))
            x = x.reshape((N, N))
            y = y.reshape((N, N))
            z = z.reshape((N, N))
        if i == j:
            for k,v in hc.items():
                if str(attrs_list[j]) in k and str(attrs_list[i]) in k:
                    key = k
            ax[i,j].set_title(' {0} vs score'.format(attrs_list[i]))
            #y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]
            x = np.array([ p.dtc.attrs[str(attrs_list[j])] for p in hc[key] ])
            y = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
            ax[i,j].plot(x,y)


     


# In[ ]:


plt.clf()
fig,ax = plt.subplots(3,3,figsize=(10,10))

for i in range(len(attrs_list)):
    for j in range(len(attrs_list)):
        if i<j:
            
            ax[i,j].set_title(' {0} vs  {1}'.format(attrs_list[i],attrs_list[j]))
            ss = set([attrs_list[j],attrs_list[i]])
            bs = set(attrs_list)
            diff = bs.difference(ss)
            bd =  {}
            bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]
            ax_trip,plot_axis = plot_surface(fig,ax[i,j],attrs_list[j],attrs_list[i],history)
            ax[i,j].plot_axis = plot_axis

            
        if i == j:
            for k,v in hc.items():
                if str(attrs_list[j]) in k and str(attrs_list[i]) in k:
                    key = k
            ax[i,j].set_title(' {0} vs score'.format(attrs_list[i]))
            #y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]
            x = np.array([ p.dtc.attrs[str(attrs_list[j])] for p in hc[key] ])
            y = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
            ax[i,j].plot(y,x)
            print(x)
            #x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            #z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            #ax[i,i].scatter(x,z)
            #ax[i,j].set_xlim(np.min(x),np.max(x))
            
            
        elif i>j:
            ax[i,j].set_title(' {0} vs  {1}'.format(attrs_list[i],attrs_list[j]))
            ss = set([attrs_list[j],attrs_list[i]])
            bs = set(attrs_list)
            diff = bs.difference(ss)
            bd =  {}
            bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]
    
        
            #ax_trip,plot_axis = plot_surface(fig,ax[i,j],attrs_list[j],attrs_list[i],history)
            #ax[i,j].plot_axis = plot_axis
            
            for k,v in hc.items():
                if str(attrs_list[j]) in k and str(attrs_list[i]) in k:
                    key = k
            z = np.array([ np.sum(list(p.dtc.scores.values())) for p in hc[key] ])
            x = np.array([ p.dtc.attrs[str(attrs_list[j])] for p in hc[key] ])
            y = np.array([ p.dtc.attrs[str(attrs_list[i])] for p in hc[key] ])
            
     
            
            N = int(np.sqrt(len(x)))
            x = x.reshape((N, N))
            y = y.reshape((N, N))
            z = z.reshape((N, N))
            

            ax[i,j].pcolormesh(x, y, z)


# In[ ]:


fig,ax = plt.subplots(3,3,figsize=(10,10))

for i in range(len(attrs_list)):
    for j in range(len(attrs_list)):
        if i<j:
            ax[i,j].set_title(' {0} vs  {1}'.format(attrs_list[i],attrs_list[j]))

            x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]
            z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            ax[i,j].scatter(y,x,c=z)
            ax[i,j].set_xlim(np.min(y),np.max(y))
            ax[i,j].set_ylim(np.min(x),np.max(x))

            
        if i == j:
            ax[i,j].set_title(' {0} vs score'.format(attrs_list[i]))
            y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]

            x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            ax[i,i].scatter(x,z)
            ax[i,j].set_xlim(np.min(x),np.max(x))
            
            
        elif i>j:
            ax[i,j].set_title(' {0} vs  {1}'.format(attrs_list[i],attrs_list[j]))
            ss = set([attrs_list[j],attrs_list[i]])
            bs = set(attrs_list)
            diff = bs.difference(ss)
            bd =  {}
            bd[list(diff)[0]] = hof[0].dtc.attrs[list(diff)[0]]
    
        
            #ax_trip,plot_axis = plot_surface(fig,ax[i,j],attrs_list[j],attrs_list[i],history)
            #ax[i,j].plot_axis = plot_axis


            ax[i,j].scatter(x, y, c=z)
            ax[i,j].set_xlim(np.min(x),np.max(x))
            ax[i,j].set_ylim(np.min(y),np.max(y))


# 

# In[ ]:


import pickle
get_ipython().system('pip install prettyplotlib')
import prettyplotlib as ppl
from prettyplotlib import plt
from prettyplotlib import brewer2mpl
import numpy as np
import string

green_purple = brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math as math
from pylab import rcParams
from neuronunit.optimization.results_analysis import make_report, min_max
with open('pre_gridport_reports.p','rb') as f:#
    grid_results = pickle.load(f)

with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)
pop = package[0]
print(pop[0].dtc.attrs.items())
history = package[4]
gen_vs_pop =  package[6]


# In[ ]:


attrs_ = [ list(p.dtc.attrs.keys()) for i,p in history.genealogy_history.items() ]
attrs = attrs_[0]
print(attrs)

scores_ = [ list(p.dtc.scores.keys()) for i,p in history.genealogy_history.items() ]
scores = scores_[0]
from collections import OrderedDict

urlDats = []
hi = [ (i,p) for i,p in history.genealogy_history.items() ]
pops_ = [ (i,p) for i,p in enumerate(pop) ]

sc = [ (i,p) for i,p in enumerate(grid_results) ]

import quantities as pq

def history_iter(mapped):
    i,p = mapped
    gene_contents = OrderedDict()
    gene_contents['gene_number'] = i
    
    attrs = list(p.dtc.attrs.keys()) 
    scores = list(p.dtc.scores.keys()) 
    for a in attrs:
        gene_contents[a] = p.dtc.attrs[a]    
    scores0 = scores[0]
    for s in scores:
        gene_contents[s] = p.dtc.scores[s]
    gene_contents[str('total')] = sum(p.dtc.scores.values())
    for test in p.dtc.score.keys():
        if 'prediction' in p.dtc.score[test]:
            
            
           
            gene_contents['observation'] = p.dtc.score[test]['observation']['mean']
            try:
                pass
                #gene_contents['prediction'] = p.dtc.score[test]['prediction']['value']
                #x = pq.Quantity(gene_contents['prediction'])
                
                #print(x.simplified)
                
            except:
                gene_contents['prediction'] = p.dtc.score[test]['prediction']['mean']
                x = pq.Quantity(gene_contents['prediction'])
                print(x.simplified)
                
                gene_contents['disagreement'] = np.abs(float(gene_contents['observation'])) - np.abs(float(gene_contents['prediction']))#p.dtc.score[test]['agreement']
    return gene_contents


    
def process_dics(contents):
    dfs = []
    for gene_contents in contents:
        # pandas Data frames are best data container for maths/stats, but steep learning curve.
        # Other exclusion criteria. Exclude reading levels above grade 100,
        # as this is most likely a problem with the metric algorithm, and or rubbish data in.

        if len(dfs) == 0:
            dfs = pd.DataFrame(pd.Series(gene_contents)).T
        dfs = pd.concat([ dfs, pd.DataFrame(pd.Series(gene_contents)).T ])
    return dfs

genes = list(map(history_iter,hi))    
dfg = process_dics(genes)

grids = list(map(history_iter,sc))    
dfs = process_dics(grids)


dfg.set_index('gene_number', inplace=True)
dfg = dfg.drop(dfg.index[0])
#dfg = dfg.reset_index()

def highlight_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]


def highlight_col(x):
    #copy df to new - original data are not changed
    df = x.copy()

    mask = df['total'] == df['total'].argmin()
    #if mask:
    print(df['total'].argmin())
    df.loc[mask, :] = 'background-color: yellow'
    df.loc[~mask,:] = 'background-color: ""'
    return df,x    




#colors,dfg = highlight_col(dfg)

#dfg


# In[ ]:




def highlight_col(x):
    df = x.copy()
    mask = df['total'] ==  df['total'].min()
    df.loc[mask, :] = 'background-color: yellow'
    df.loc[~mask,:] = 'background-color: ""'
    return df    

dfg = dfg.style.apply(highlight_col, axis=None)
#dfg = dfg.style.background_gradient(cmap='viridis')

'''
import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)

s = dfg.style.background_gradient(cmap=cm)
s


'''

#dfg = dfg.style.bar(subset=['total'], color=['#d65f5f', '#5fba7d'])
dfg 

#s = dfg.style.background_gradient(cmap=cm)
#s


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import scipy.ndimage as ndi
import neuronunit.optimization.exhaustive_search as es

data = np.zeros((6, 6, 6))
#data = ndi.filters.gaussian_filter(data, sigma=1)

axis = [ [ str('vr'), str('a'), str('b') ], [ str('vr'), str('b'), str('a')], [ str('b'), str('a'), str('vr') ] ]

gen_vs_pop = package[6]    

k = axis[0]
ee = [ np.sum(list(i.dtc.scores.values())) for i in grid_results ]
zz = [ i.dtc.attrs[k[2]] for i in grid_results ]
yy = [ i.dtc.attrs[k[1]] for i in grid_results ]
xx = [ i.dtc.attrs[k[0]] for i in grid_results ]
hvc = es.tfc2i(xx,yy,zz,ee)


# In[ ]:


#from neuronunit imp
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization.model_parameters import model_params, path_params
import quantities as pq

super_pop = []

for gp in gen_vs_pop:
    super_pop.extend(gp)
minga,maxga = min_max(super_pop)
mingr,maxgr = min_max(grid_results)
abs_min = np.min((mingr[1],minga[1]))
abs_max = np.max((maxgr[1],maxga[1]))
v = list(np.linspace(abs_min, abs_max, 15, endpoint=True))

def proc_xargs(dtc,dtcpop = None):
    model = ReducedModel(path_params['model_path'],name=str('vanilla'),backend='NEURON')
    xargs = {}
    xargs['injected_square_current'] = {}
    xargs['injected_square_current']['duration'] = 1000 * pq.ms
    xargs['injected_square_current']['amplitude'] = dtc.rheobase['value']
    xargs['injected_square_current']['delay'] = 250 * pq.ms # + 150
    model.set_attrs(dtc.attrs)
    model.inject_square_current(xargs)
    v_m = model.get_membrane_potential()
    ts = model.results['t'] # time signal
    return v_m,ts,model

def realign(contents):
    plt.clf()

    for c in contents:            
        v_m,ts,model = c
        try:
            spt = float(model.get_spike_train())
            #thr = float(model.get_threshold())

            new_time = [ t-spt for t in ts ]        
            plt.plot(new_time,v_m)
            plt.xlim(-0.03,0.03)
        except:
            print(model.get_spike_train())
            
    plt.show()        
    return _

'''
contents = []   
dtcpop = [ p.dtc for p in pop ]
dtcpop.extend([ p.dtc for p in gen_vs_pop[0] ])
for d in dtcpop:
    contents.append(proc_xargs(d))

contents.append(proc_xargs(mingr[2]))
contents.append(proc_xargs(minga[2]))

realign(contents)    
'''


# In[ ]:


from collections import OrderedDict
zz = [ list(i.dtc.scores.items()) for i in grid_results ]
xx = [ list(i.dtc.scores.items()) for i in pop ]

def sample_points(iter_dict, npoints=2):
    replacement = {}
    for p in range(0,len(iter_dict)):
        k,v = iter_dict.popitem(last=False)
        sample_points = list(np.linspace(v.max(),v.min(),npoints))
        replacement[k] = sample_points
    return replacement

from neuronunit.optimization import model_parameters as modelp
mp = OrderedDict(modelp.model_params)
smaller = {}
smaller = OrderedDict(sample_points(mp, npoints=2))
ranges = OrderedDict( {k:smaller[k] for k in smaller})
        
lims = []
for k,v in pop[0].dtc.attrs.items():
    lims.append([np.min(ranges[k]), np.max(ranges[k])])


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from matplotlib import animation, rc
from IPython.display import HTML

#rc('animation', html='html5')
#options = {'c1':0.5, 'c2':0.3, 'w':0.9}


#moviewriter = animation.MovieWriter(...)
#moviewriter.setup(fig=fig, 'my_movie.ext', dpi=100)


super_set = []
length = len(gen_vs_pop)

#frame_number = [i for i, pop in enumerate(gen_vs_pop)]        
for i, pop in enumerate(gen_vs_pop):
#def to_movie(frame_number):
    other_points = []
    pf_points = []
    hof_points = []    
    labels = []
    for p in pop:
        #print(p)
        xyz = []
        for k,v in p.dtc.attrs.items():
            xyz.append(v)
            labels.append(k)
        other_points.append(xyz)

    #lims = []
    #for k,v in pop[0].dtc.attrs.items():
    #    lims.append((np.min(ranges[k]), np.max(ranges[k])))

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots(1, 1)#, figsize=figsize)
    ax = Axes3D(fig)
    

    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])

    # Set plot title
    #labels=('x-axis', 'y-axis', 'z-axis')#,
    interval=80#,
    title='Particle Movement in 3D space'#,
    title_fontsize="large"#,
    text_fontsize="medium"
    ax.set_title(title, fontsize=title_fontsize)



    # Set plot axes labels

    # Set plot limits

    errors = [ np.sum(list(p.dtc.scores.values())) for p in pop ]
    xx = [ i[0] for i in other_points ]
    yy = [ i[1] for i in other_points ]
    zz = [ i[2] for i in other_points ]
    if len(super_set) !=0 :
        for ss in super_set:
            ops, ers = ss
            p = ax.scatter3D([i[0] for i in ops], [ i[1] for i in ops], [i[2] for i in ops], alpha=0.0925, c=ers, cmap='jet', marker='o', vmin=abs_min, vmax=abs_max)
            #ax.set_autoscalez_on(False)

    p = ax.scatter3D(xx, yy, zz, c=errors, cmap='jet', marker='o', vmin=abs_min, vmax=abs_max)
    #print(xx,yy,zz)
    #print([l for l in lims[0]], [l for l in lims[1]], [l for l in lims[2]])
    #p = ax.scatter3D([l for l in lims[0]], [l for l in lims[1]], [l for l in lims[2]], c=[2.0,2.0], marker='*', vmin=abs_min, vmax=abs_max)
    #ax.set_autoscalez_on(False)

    cb = fig.colorbar(p)
    cb.set_label('summed scores')

    zz = [ np.sum(list(i.dtc.scores.values())) for i in grid_results ]
    zz_sorted = sorted([( np.sum(list(i.dtc.scores.values())), index) for index,i in enumerate(grid_results) ])
    gbest = zz_sorted[0]
    gworst = zz_sorted[-1]

    gworst_grid_attrs = grid_results[gworst[1]].dtc.attrs
    gbest_grid_attrs = grid_results[gbest[1]].dtc.attrs

    ax.set_xlabel(str(labels[0]))
    ax.set_ylabel(str(labels[1]))
    ax.set_zlabel(str(labels[2]))

    plt.savefig('particle_cube_'+str(float(i/length))+str('.png'))
    super_set.append((other_points,errors))    
    plt.show()

    #return ax


frame_number = [i for i, pop in enumerate(gen_vs_pop)]        
#fig = plt.figure()

#_ = list(map(to_movie,frame_number))

#anim = animation.FuncAnimation(fig, func=to_movie,
#                              frames=frame_number,
#                              interval=30)


# In[ ]:





# In[ ]:





# In[ ]:


axis = [ [ str('vr'), str('a'), str('b') ], [ str('vr'), str('b'), str('a')], [ str('b'), str('a'), str('vr') ] ]

gen_vs_pop = package[6]    
pop = gen_vs_pop[-1]

for k in axis:    
    
    zz = [ np.sum(list(i.dtc.scores.values())) for i in grid_results ]
    yy = [ i.dtc.attrs[k[1]] for i in grid_results ]
    xx = [ i.dtc.attrs[k[0]] for i in grid_results ]

    last_frame = len(gen_vs_pop)
    other_points = []
    pf_points = []
    hof_points = []
    labels = []
    
    pf = package[2]
    hof = package[1]
    
    for p in pop:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            other_points.append(xy)
            
            
    for p in pf:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            pf_points.append(xy)            
       
    for p in hof:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            hof_points.append(xy)        

            
    zi, yi, xi = np.histogram2d(yy, xx, bins=(6,6), weights=zz, normed=False)
    counts, _, _ = np.histogram2d(yy, xx, bins=(6,6))
    #binned , _, _ = np.histogram(zce, bins=10)

    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    fig, ax = plt.subplots()
    #scat = ppl.pcolormesh(fig, ax, xi, yi, zi, edgecolors='black', cmap=green_purple)
    scat = ax.pcolormesh(xi, yi, zi, edgecolors='black', cmap=green_purple)
    


    fig.colorbar(scat)
    ax.margins(0.05)

    #if i == last_frame-1:
    for xy in hof_points:
        ax.plot(xy[0], xy[1],'y*',label ='hall of fame') 
    for xy in pf_points:
        ax.plot(xy[0], xy[1],'b*',label ='pareto front') 
        #legend = ax.legend([rect("r"), rect("g"), rect("b")], ["gene population","pareto front","hall of fame"])


    for xy in other_points:
        ax.plot(xy[0], xy[1],'ro',label ='gene population') 
    ax.margins(0.05)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


# In[ ]:



gen_vs_pop = package[6]    
pop = gen_vs_pop[-1]
for k in axis:    
    zz = [ np.sum(list(i.dtc.scores.values())) for i in grid_results ]
    zz_sorted = sorted([( np.sum(list(i.dtc.scores.values())), index) for index,i in enumerate(grid_results) ])
    gbest = zz_sorted[0]
    gworst = zz_sorted[-1]
    assert gbest!=gworst
    gworst_grid_attrs = grid_results[gworst[1]].dtc.attrs
    gbest_grid_attrs = grid_results[gbest[1]].dtc.attrs
    gba = gbest_grid_attrs[k[2]]   
    zz = [ np.sum(list(i.dtc.attrs.values())) for i in grid_results ]
    zce = [ np.sum(list(i.dtc.scores.values())) for i in grid_results if i.dtc.attrs[k[2]] == gba]
    yy = [ i.dtc.attrs[k[1]] for i in grid_results if i.dtc.attrs[k[2]] == gba ]
    xx = [ i.dtc.attrs[k[0]] for i in grid_results if i.dtc.attrs[k[2]] == gba ]
    last_frame = len(gen_vs_pop)
    other_points = []
    pf_points = []
    hof_points = []
    labels = []
    
    pf = package[2]
    hof = package[1]
    
    for p in pop:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            other_points.append(xy)
            
            
    for p in pf:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            pf_points.append(xy)            
       
    for p in hof:
        xy = []
        for key in k:
            v = p.dtc.attrs[key]
            xy.append(v)
            labels.append(key)
            hof_points.append(xy)        

            
    zi, yi, xi = np.histogram2d(yy, xx, bins=(6,6), weights=zce, normed=False)
    counts, _, _ = np.histogram2d(yy, xx, bins=(6,6))

    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    fig, ax = plt.subplots()
    scat = ax.pcolormesh(xi, yi, zi, edgecolors='black', cmap=green_purple)

    fig.colorbar(scat)
    ax.margins(0.05)

    #if i == last_frame-1:
    for xy in hof_points:
        ax.plot(xy[0], xy[1],'y*',label ='hall of fame') 
    for xy in pf_points:
        ax.plot(xy[0], xy[1],'b*',label ='pareto front') 
        #legend = ax.legend([rect("r"), rect("g"), rect("b")], ["gene population","pareto front","hall of fame"])


    for xy in other_points:
        ax.plot(xy[0], xy[1],'ro',label ='gene population') 
    ax.margins(0.05)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()

    
    


# In[ ]:


nparams=3
new_report = make_report(grid_results,pop, nparams)
from neuronunit.optimization import exhaustive_search as es
print(new_report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from prettyplotlib import plt
import numpy as np
fig,ax = plt.subplots(4,4,figsize=(10,10))
attrs_list = list(pop[0].dtc.attrs)
for i in range(4):
    for j in range(4):
        if i<j:
            # genes is a list of all the gene coordinates, with N coordinates per gene
            #ax[i,j].scatter(*genes[but only the two dimensions that are i and j])
            #x = [ g.dtc.attrs[attrs_list[i]] for g in pop ]
            #y = [ g.dtc.attrs[attrs_list[j]] for g in pop ]
            #z = [ sum(list(g.dtc.scores.values())) for g in pop ]
            #[g.dtc.attrs['a'] for g in pop ],[ sum(g.dtc.scores.values()) for g in pop ]
            ax[i,j].scatter(*np.random.rand(2,100),c=np.random.rand(100))
        elif i>j:
            ax[i,j].set_title('Param %d vs Param %d' % (i,j))
            # Z = 2D grid search with all params except i and j held constant at optimum
            Z = np.random.rand(25,25)
            ax[i,j].pcolormesh(Z)
        else: # i==j
            #z = 1D grid search with all params except i held contant at optimum
            z = np.random.rand(25)
            ax[i,i].plot(z)
plt.tight_layout()
plt.text(0,0,'Projections using the best value for other parameters')


# So imagine 6 2D plots.  The first three are:  XY at Z=z, XZ at Y=y, and YZ at X=x, where x, y, z 
# is the location of the global minimum.  The second three are XY with the minimum taken across Z, 
# XZ with the minumum taken across Y, YZ, with the minimum taken across X.
# With these 6 I think you would get a pretty good idea of the contours around the global minimum.

# In[ ]:




with open('pre_ga_reports.p','rb') as f:
    package = pickle.load(f)
log = package[3]    
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
fig, axes = plt.subplots(figsize=(10, 10), facecolor='white')
gen_numbers = [ i for i in range(0,len(log.select('gen'))) ]

mean = np.array([ np.sum(i) for i in log.select('avg')])
std = np.array([ np.sum(i) for i in log.select('std')])
minimum = np.array([ np.sum(i) for i in log.select('min')])
#minimum = np.array([ np.sum(i) for i in log.select('min')])
worst = np.max([ sum(g.dtc.scores.values()) for g in grid_results ])
#grid_min = np.min([ sum(g.dtc.scores.values()) for g in grid_results ])
gwl = [ worst for i in range(0,len(log.select('gen'))) ]
grid_min = np.min([ sum(g.dtc.scores.values()) for g in grid_results ])
gml = [ grid_min for i in range(0,len(log.select('gen'))) ]

if new_report[nparams]['success'] == True:
    axes.plot(gen_numbers,
              gml,
              color='blue',
            linewidth=2,
            label='exhaustive search best')

    axes.plot(gen_numbers,
              gwl,
              color='yellow',
            linewidth=2,
            label='exhaustive search worst')
    


    stdminus = mean - std
    stdplus = mean + std

    axes.plot(
        gen_numbers,
        minimum,
        color='black',
        linewidth=2,
        label='population minimum')
    axes.fill_between(gen_numbers, stdminus, stdplus)


axes.plot(
    gen_numbers,
    mean,
    color='black',
    linewidth=2,
    label='population average')
    



plt.xlabel('generation')
plt.ylabel('error')


plt.legend()
plt.show()
print(worst,grid_min)


# In[ ]:





# In[ ]:






# 

# In[ ]:



#fig.savefig('pcolormesh_prettyplotlib_labels_other_cmap_diverging.png')


# 
# 

# In[ ]:






# 

# 

# In[ ]:





# In[ ]:


import bokeh
import numpy as np
import matplotlib


# 

# In[ ]:




from neuronunit.optimization.optimization_management import run_ga
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
from neuronunit.optimization.model_parameters import model_params
import os
import pickle
from neuronunit.optimization import get_neab
reports = {}
npoints = 10



with open('pre_grid_reports.p','rb') as f:
    grid_results = pickle.load(f)
opt_keys = list(grid_results[0].dtc.attrs.keys())

with open('pre_ga_reports.p','rb') as f:
    ga_out = pickle.load(f)

pop = ga_out[0]
genes_vs_gen = ga_out[-1]


# In[ ]:



from neuronunit.optimization.exhaustive_search import create_grid
gp = create_grid(npoints = 6,nparams = 3)

    


# 
# Found tough parameters for which the GA is not able to perform particularily # well. Suspect b's error surface is not concave.
# Explore 1D cross section.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



plt.clf()

plt.scatter([g.dtc.attrs['a'] for g in pop ],[ sum(g.dtc.scores.values()) for g in pop ] ,label='ga pop')
plt.scatter([g.dtc.attrs['a'] for g in grid_results ],[ sum(g.dtc.scores.values()) for g in grid_results ] ,label='grid evaluations')
plt.ylabel('score')
plt.xlabel('gene attribute a')
plt.legend()
plt.show()


plt.scatter([g.dtc.attrs['vr'] for g in pop ],[ sum(g.dtc.scores.values()) for g in pop ] ,label='ga pop')
plt.scatter([g.dtc.attrs['vr'] for g in grid_results ],[ sum(g.dtc.scores.values()) for g in grid_results ] ,label='grid evaluations')
plt.ylabel('score')
plt.xlabel('gene attribute vr')
plt.legend()
plt.show()



plt.scatter([g.dtc.attrs['b'] for g in pop ],[ sum(g.dtc.scores.values()) for g in pop ] ,label='ga pop')
plt.scatter([g.dtc.attrs['b'] for g in grid_results ],[ sum(g.dtc.scores.values()) for g in grid_results ] ,label='grid evaluations')
plt.ylabel('score')
plt.xlabel('gene attribute b')
plt.legend()
plt.show()



# In[ ]:



nparams = 3
opt_keys = [str('a'),str('vr'),str('b')]


#grid_results = run_grid(nparams,npoints,tests,provided_keys = opt_keys)
from neuronunit.optimization import exhaustive_search #import run_grid, reduce_params, create_grid

from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
grid_points,maps = exhaustive_search.create_grid(npoints=6,nparams=3,provided_keys=opt_keys)
print(maps)


# In[ ]:


def blah():
    '''
    This works but I changed the grid
    '''
    grid = np.zeros((6,6,6))
    import matplotlib.pyplot as plt
    print(maps)
    for i in grid_results:
        xyz = []
        for k,v in i.dtc.attrs.items():
            print(k,v)
            #print(maps[v][k])
            xyz.append(maps[k][v])
        grid[xyz[0],xyz[1],xyz[2]] = sum(i.dtc.scores.values())

    for i in range(0,6):
        flat = grid[i,:,:]
        plt.imshow(flat)
        plt.show()

    for i in range(0,6):
        flat = grid[:,i,:]
        plt.imshow(flat)
        plt.show()

    for i in range(0,6):
        flat = grid[:,:,i]
        plt.imshow(flat)
        plt.show()
        
    


# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:


#new_list = genes_vs_gen[1:-1]
#print(new_list)


# In[ ]:


#![alt text](rick_style_guide.png "Ricks table guide")


# 

# In[ ]:





# In[ ]:





# In[ ]:


dfg


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# 

# 
# 

# 
# 
# 

# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
# 
# 

# In[ ]:





# 

# 

# In[ ]:





# 

# In[ ]:





# In[ ]:



    


# In[ ]:





# I am currently writing code that should enable the plotting of HOF values versus generation.

# In[ ]:





# In[ ]:



   


# For some reason the global minimum solution is not converged on, as shown by the evolution of errors below:

# In[ ]:





# In[ ]:





# For some reason, the GA population does not converge to the absolute minimum, although it does sample it.
# Perhaps the absolute minimum is a highly dominated solution, which is a testable hypthosis.
# 
# None the less because the GA samples the absolute minimum, this value can be corroborated with the GA.

# 

# In[ ]:





# 
# 
# 
# 

# In[ ]:




    



# In[ ]:



            
            


# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# # Quantize distance between minimimum error and maximum error.
# This step will allow the GA's performance to be located within or below the range of error found by grid search.
# 

# In[ ]:





# The code below reports on the differences between between attributes of best models found via grid versus attributes of best models found via GA search:
# 

# In[ ]:




    


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




