# -*- coding: utf-8 -*-
"""NeuronUnit Test classes for cell models with morphology"""

import quantities as pq
from pylmeasure import *

from neuronunit.capabilities.morphology import *
from sciunit.scores import ZScore


class MorphologyTest(sciunit.Test):
    '''
    An abstract class to hold common elements among the morphology tests below
    '''

    required_capabilities = (ProducesSWC,)
    score_type = ZScore

    def get_lmeasure(self, model_swc, measure, stat, specificity="Type > 1"):
        '''
        Computes the specified L-Measure measure and selects one of the statistics

        :param model_swc: A model that has ProducesSWC capability
        :param measure: One of the functions from the list: http://cng.gmu.edu:8080/Lm/help/index.htm
        :param stat: One of: Average, Maximum, Minimum, StdDev, TotalSum
        :return: The computed measure statistic
        '''

        pca = False
        if measure in ("Height", "Width", "Depth"):
            pca = True

        swc_path = model_swc.produce_swc()
        value = getOneMeasure(measure, swc_path, pca, specificity)[stat]
        return value


class SomaSurfaceAreaTest(MorphologyTest):
    """Test the agreement between soma surface area observed in the model and experiments"""

    name = 'Soma Surface Area Test'
    units = pq.um**2

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Surface', "TotalSum", specificity="Type == 1")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class NumberofStemsTest(MorphologyTest):
    """Test the agreement between number of stems observed in the model and experiments"""

    name = 'Number of Stems Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'N_stems', "TotalSum", specificity="Type > 1")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class NumberofBifurcationsTest(MorphologyTest):
    """Test the agreement between number of bifurcations observed in the model and experiments"""

    name = 'Number of Bifurcations Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'N_bifs', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class NumberofBranchesTest(MorphologyTest):
    """Test the agreement between number of branches observed in the model and experiments"""

    name = 'Number of Branches Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'N_branch', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class OverallWidthTest(MorphologyTest):
    """Test the agreement between overall width observed in the model and experiments"""

    name = 'Overall Width Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Width', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class OverallHeightTest(MorphologyTest):
    """Test the agreement between overall height observed in the model and experiments"""

    name = 'Overall Height Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Height', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class OverallDepthTest(MorphologyTest):
    """Test the agreement between overall depth observed in the model and experiments"""

    name = 'Overall Depth Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Depth', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class AverageDiameterTest(MorphologyTest):
    """Test the agreement between average diameter observed in the model and experiments"""

    name = 'Average Diameter Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Diameter', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class TotalLengthTest(MorphologyTest):
    """Test the agreement between total length observed in the model and experiments"""

    name = 'Total Length Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Length', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class TotalSurfaceTest(MorphologyTest):
    """Test the agreement between total surface observed in the model and experiments"""

    name = 'Total Surface Test'
    units = pq.um**2

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Surface', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class TotalVolumeTest(MorphologyTest):
    """Test the agreement between total volume (excl. soma) observed in the model and experiments"""

    name = 'Total Volume Test'
    units = pq.um**3

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Volume', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class MaxEuclideanDistanceTest(MorphologyTest):
    """Test the agreement between max euclidean distance observed in the model and experiments"""

    name = 'Max Euclidean Distance Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'EucDistance', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class MaxPathDistanceTest(MorphologyTest):
    """Test the agreement between max path distance observed in the model and experiments"""

    name = 'Max Path Distance Test'
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'PathDistance', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class MaxBranchOrderTest(MorphologyTest):
    """Test the agreement between max branch order observed in the model and experiments"""

    name = 'Max Branch Order Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Branch_Order', "Maximum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class AverageContractionTest(MorphologyTest):
    """Test the agreement between average contraction observed in the model and experiments"""

    name = 'Average Contraction Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Contraction', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class TotalFragmentationTest(MorphologyTest):
    """Test the agreement between total fragmentation observed in the model and experiments"""

    name = 'Total Fragmentation Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Fragmentation', "TotalSum")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class PartitionAsymmetryTest(MorphologyTest):
    """Test the agreement between partition asymmetry observed in the model and experiments"""

    name = 'Partition Asymmetry Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Partition_asymmetry', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class AverageRallsRatioTest(MorphologyTest):
    """Test the agreement between average Rall's ratio observed in the model and experiments"""

    name = 'Average Ralls Ratio Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Pk_classic', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class AverageBifurcationAngleLocalTest(MorphologyTest):
    """Test the agreement between average bifurcation angle local observed in the model and experiments"""

    name = 'Average Bifurcation Angle Local Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Bif_ampl_local', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class AverageBifurcationAngleRemoteTest(MorphologyTest):
    """Test the agreement between average bifurcation angle remote observed in the model and experiments"""

    name = 'Average Bifurcation Angle Remote Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Bif_ampl_remote', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }


class FractalDimensionTest(MorphologyTest):
    """Test the agreement between fractal dimension observed in the model and experiments"""

    name = 'Fractal Dimension Test'
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, 'Fractal_Dim', "Average")

        return {
            'mean': value * self.units,
            'std': 0 * self.units,
            'n': 1
        }






