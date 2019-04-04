import numpy as np
import pickle

def param_distance(dtc_ga_attrs,dtc_grid_attrs,td):
    distances = {}
    # These imports are defined here to avoid cyclic importing, as the function provides a limited scope of the imports.
    #
    from neuronunit.optimization.optimization_management import model_params
    from neuronunit.optimization.exhaustive_search import reduce_params
    ranges = { k:model_params[k] for k,v in dtc_ga_attrs.items() }
    for k,v in dtc_ga_attrs.items():
        dimension_length = np.max(ranges[k]) - np.min(ranges[k])
        solution_distance_in_1D = np.abs(float(dtc_grid_attrs[k]))-np.abs(float(v))
        try:
            relative_distance = np.abs(solution_distance_in_1D/dimension_length)
        except:
            relative_distance = None
        distances.get(k, relative_distance)
        distances[k] = (relative_distance)
        print('the difference between brute force candidates model parameters and the GA\'s model parameters:')
        print(float(dtc_grid_attrs[k])-float(v),dtc_grid_attrs[k],v,k)
        print('the relative distance scaled by the length of the parameter dimension of interest:')
        print(relative_distance)
    return distances

def min_max(pop):
    garanked = [ (r.dtc.attrs , sum(r.dtc.scores.values()), r.dtc) for r in pop ]
    garanked = sorted(garanked, key=lambda w: w[1])
    miniga = garanked[0]
    maxiga = garanked[-1]
    return miniga, maxiga


def error_domination(dtc_ga,dtc_grid):
    distances = {}
    errors_ga = list(dtc_ga.scores.values())
    print(errors_ga)
    me_ga = np.mean(errors_ga)
    std_ga = np.std(errors_ga)

    errors_grid = list(dtc_grid.scores.values())
    print(errors_grid)
    me_grid = np.mean(errors_grid)
    std_grid = np.std(errors_grid)

    dom_grid = False
    dom_ga = False

    for e in errors_ga:
        if e <= me_ga + std_ga:
            dom_ga = True

    for e in errors_grid:
        if e <= me_grid + std_grid:
            dom_grid= True

    return dom_grid, dom_ga


def make_report(grid_results, ga_out, nparams, pop = None):
    from neuronunit.optimization.exhaustive_search import create_grid
    grid_points = create_grid(npoints = 2,nparams = nparams)
    td = list(grid_points[0][0].keys())

    reports = {}
    reports[nparams] = {}

    mini = min_max(grid_results)[0][1]
    maxi = min_max(grid_results)[1][1]
    if type(pop) is not type(None):
        miniga = min_max(pop)[0][1]
    else:
        miniga = min_max(ga_out)[0][1]

    reports[nparams]['miniga'] = miniga
    reports[nparams]['minigrid'] = mini
    quantize_distance = list(np.linspace(mini,maxi,21))
    success = bool(miniga < quantize_distance[2])
    better = bool(miniga < quantize_distance[0])

    print('Report: ')
    print('did it work? {0} was it better {1}'.format(success,better))


    reports[nparams]['success'] = success
    reports[nparams]['better'] = better
    dtc_ga = min_max(ga_out)[0][0]
    attrs_grid = min_max(grid_results)[0][0]
    attrs_ga = min_max(ga_out)[0][0]
    reports[nparams]['attrs_ga'] = attrs_ga
    reports[nparams]['attrs_grid'] = attrs_grid



    reports[nparams]['p_dist'] = param_distance(attrs_ga,attrs_grid,td)
    ##
    # mistake here
    ##
    dtc_grid = dtc_ga = min_max(ga_out)[0][2]
    dom_grid, dom_ga = error_domination(dtc_ga,dtc_grid)
    reports[nparams]['vind_domination'] = False
    # Was there vindicating domination in grid search but not GA?
    if dom_grid == True and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    elif dom_grid == False and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    # Was there incriminating domination in GA but not the grid, or in GA and Grid
    elif dom_grid == True and dom_ga == True:
        reports[nparams]['inc_domination'] = False
    elif dom_grid == False and dom_ga == True:
        reports[nparams]['inc_domination'] = False


    #reports[nparams]['success'] = bool(miniga < quantize_distance[2])
    dtc_ga = min_max(ga_out)[0][0]
    attrs_grid = min_max(grid_results)[0][0]
    attrs_ga = min_max(ga_out)[0][0]

    grid_points = create_grid(npoints = 1,nparams = nparams)#td = list(grid_points[0].keys())
    td = list(grid_points[0][0].keys())

    reports[nparams]['p_dist'] = param_distance(attrs_ga,attrs_grid,td)
    dtc_grid = dtc_ga = min_max(ga_out)[0][2]
    dom_grid, dom_ga = error_domination(dtc_ga,dtc_grid)
    reports[nparams]['vind_domination'] = False
    # Was there vindicating domination in grid search but not GA?
    if dom_grid == True and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    elif dom_grid == False and dom_ga == False:
        reports[nparams]['vind_domination'] = True
    # Was there incriminating domination in GA but not the grid, or in GA and Grid
    elif dom_grid == True and dom_ga == True:
        reports[nparams]['inc_domination'] = False
    elif dom_grid == False and dom_ga == True:
        reports[nparams]['inc_domination'] = False


    with open('reports.p','wb') as f:
        pickle.dump(reports,f)
    return reports
