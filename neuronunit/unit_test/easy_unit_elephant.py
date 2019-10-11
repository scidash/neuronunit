from hide_imports import *
from neuronunit.optimisation.optimisations import run_ga
df = pd.DataFrame(rts)
ga_outad = {}
ga_outiz = {}
ga_outgl = {}
from neuronunit.optimisation.optimization_management import inject_and_plot
#from neuronunit.tests.allen_tests import pre_obs#, test_collection
#import pdb;
#pdb.set_trace()
NGEN = 10
from hide_imports import *
df = pd.DataFrame(rts)
for key,v in rts.items():
    helper_tests = [value for value in v.values() ]
    break
for key,v in rts.items():
    backend = str('RAW')
    filename = str(key)+backend+str('.p')
    ga_outiz[key], DO = run_ga(model_params.MODEL_PARAMS['RAW'],NGEN, local_tests, free_params = model_params.MODEL_PARAMS['RAW'],
                                    MU = 12, backend = str('RAW'),protocol={'allen':False,'dm':False,'elephant':True})
    pickle.dump(ga_outiz[key],open(filename,'wb'))
    dtcpop = [ ind.dtc for ind in ga_outiz[key]['pf'] ]
    filename = str(key)+backend+str('.p')
    d1 = [p.dtc for p in ga_outiz[key]['pop']]

    backend = str('ADEXP')
    filename = str(key)+backend+str('.p')
    ga_outad[key], DO = run_ga(model_params.MODEL_PARAMS['ADEXP'],NGEN, local_tests, free_params = model_params.MODEL_PARAMS['ADEXP'],
                                   MU = 10, backend = str('ADEXP'),protocol={'allen':False,'dm':False,'elephant':True})
    pickle.dump(ga_outad[key],open(filename,'wb'))

    d3 = [p.dtc for p in ga_outad[key]['pop']]




    #import pdb; pdb.set_trace()
    '''
    backend = str('GLIF')
    filename = str(key)+backend+str('.p')

    mp = model_params.MODEL_PARAMS['GLIF']
    mp = { k:v for k,v in mp.items() if type(v) is not dict }
    mp = { k:v for k,v in mp.items() if v is not None }
    #ga_outgl[key], DO = om.run_ga(mp ,NGEN, local_tests, free_params = mp, NSGA = True, MU = 10, model_type = str('GLIF'))#,seed_pop=seeds[key])
#d2 = [p.dtc for p in ga_outgl[key]['pf'][0:-1] if not p.dtc.rheobase is None]

    #while not len(d2):
    ga_outgl[key], DO = om.run_ga(mp ,10, local_tests, free_params = mp, NSGA = True, MU = 10, model_type = str('GLIF'))#,seed_pop=seeds[key])
    d2 = [p.dtc for p in ga_outgl[key]['pop'] if not p.dtc.rheobase is None]
    rh = [ p.dtc.rheobase  for p in ga_outgl[key]['pop']  ]

    #print('still looping')
    pickle.dump(ga_outgl[key],open(filename,'wb'))
    #import pdb
    #pdb.set_trace()
    '''






    #d2 = [p.dtc for p in ga_outgl[key]['pf'][0:-1]]
    #try:
    #    inject_and_plot(d1,second_pop=d2,third_pop=d2,figname=key+'quick_two')

    #except:
    #inject_and_plot(d1,second_pop=d2,third_pop=d3,figname=key+'quick_two')


    #try:
    #    inject_and_plot(d1,second_pop=d2,third_pop=d3,figname=key)
    #except:
    #inject_and_plot(d1,second_pop=d1,third_pop=d3,figname=key)

#for key,v in rts.items():
#    local_tests = [value for value in v.values() ]
