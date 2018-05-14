#NLXWIKI:sao830368389
#NLXWIKI:sao471801888
#NLXWIKI:nifext_120
#SAO:830368389

from neuronunit.optimization import get_neab #import get_neuron_criteria, impute_criteria
fi_basket = {'nlex_id':'NLXCELL:100201'}
pvis_cortex = {'nlex_id': 'nifext_50'} # Layer V pyramidal cell

try:
    #assert 1==2
    contents = pickle.load(open('ne_neuron_criteria.p','rb'))
    pvis_criterion, inh_criterion = contents
    #print(inh_criterion, inh_observations)


except:
    pvis_criterion, pvis_observations = get_neab.get_neuron_criteria(pvis_cortex)
    inh_criterion, inh_observations = get_neab.get_neuron_criteria(fi_basket)
    #print(type(inh_observations),inh_observations)

    inh_observations = get_neab.impute_criteria(pvis_observations,inh_observations)

    inh_criterion, inh_observations = get_neab.get_neuron_criteria(fi_basket,observation = inh_observations)
    with open('ne_neuron_criteria.p','wb') as f:
       pickle.dump((pvis_criterion, inh_criterion),f)
