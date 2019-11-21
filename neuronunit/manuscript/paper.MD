# Introduction
The Izhikevich model was published with parameter sets believed to pertain to real biological cell types. However since publication much more specific cellular data and specific knowledge of electrical behavior has accumulated. When the “Regular Spiking”, neuron category is compared to specific electro-physiological data for pyramidal cells, it is better to optimize against specific data and find a set of model parameters that more accurately reproduce the data. Score ratios where calculated with the simple formula:

![image](images/table_scores.png)<!-- .element height="50%" width="50%" -->

![image](images/table_scores.png){ width=50% }

# Methods

types where used to guide evolution of a genetic algorithm. Here we show the data from just the cerebellar purkinje cell. Vertical lines represent waveform measurement found by the optimizer, blue vertical lines represent the mode of a probability kernel used to guide optimization, by constraining parameters.

Two reduced model types: (adaptive exponential model and Izhikevich model) where optimized against Cerebellar-Purkinje data. On the left we see isolated spike waveforms that are aligned by threshold crossing. On the right we see whole membrane potential recordings. The triangular shape of the adaptive Izhikevich waveform is an artifact.

In the left most plot. each plot we can see the average, maximum and minimum fitness of a gene population as it learns the optimal model to fit a given data set. On the left you can see an example of the gene pool, exploring unfit models, while retaining a memory of the best model fit. On the right you can see a more straight forward and model Evolution process.

On the middle and rightmost plot the progress of individual objective-errors, in a muti-objective context  (8 errors) are tracked. You can see that early on easy compatible errors are able to go down together, however, many errors go down at the expense of raising errors, and more exploration of parameter space is required, to bring down both errors together.
![image 2](images/ground_truth.png){:.thumbnail.bordered}
Caption: simulated ground truth, versus optimal.  Artificial data was simulated by sampling from randomly created models. A randomly chosen model is used to derive a spike shape. The optimizer was then tasked with recovering the random models. This approach demonstrates the theoretical best optimization results in a 10 dimensional, 8 objective function optimization problem. In Z-score driven optimization, variance in data can affect error gradients, and complicate model-data agreement. A spike waveform which constitutes less good fit, is also shown for contrast with the good fit green.


![image](images/Reduced_Druckman_space.png){:height="50%" width="50%"}

![image](images/firing_rate_simulated_data.png){:height="50%" width="50%"}

In order to test the generality and robustness of our optimization frame work I used an Allen Feature extraction protocol. The Allen protocol captures measurements of waveform shape, and spike train statistics on multi-spiking waveforms. Rather than optimizing models against experimental data. We simulated one multi-spiking waveform, to test if the optimizer could recover the same waveform using errors derived from Allen Features. Here you can see that this approach works generally well as the overall CV and ISI of both ground-truth and optimal are similar. In this case only the waveform shape of the first spike was included with regards to wave-shape optimization.

In order illustrate that the optimized models are better imitations of real data, four adaptive Exponential models, and four Izhikevich models each were fitted to four different classes of experimental cells (see slide 2). These eight fitted models where then fed into a Druckman feature extraction algorithm, and added as data points in
a dimension reduced plot of the feature space. Many pre-existing neural models, and some Allen Brain Data where also plotted as contextual data in the same feature space. Ordinarily models and data are easily seperable by using PCA on the druckman feature space.
