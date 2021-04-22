# -*- coding: utf-8 -*-
"""NeuronUnit Test classes for cell models with morphology"""

import quantities as pq
from pylmeasure import *

from neuronunit.capabilities.morphology import *
from sciunit.scores import ZScore


class MorphologyTest(sciunit.Test):
    """
    An abstract class to hold common elements among the morphology tests below
    """

    required_capabilities = (ProducesSWC,)
    score_type = ZScore
    specificity = "Type > 1"
    pca = False

    def get_lmeasure(self, model_swc, measure, stat):
        """
        Computes the specified L-Measure measure and selects one of the statistics

        :param model_swc: A model that has ProducesSWC capability
        :param measure: One of the functions from the list: http://cng.gmu.edu:8080/Lm/help/index.htm
        :param stat: One of: Average, Maximum, Minimum, StdDev, TotalSum
        :return: The computed measure statistic
        """

        swc_path = model_swc.produce_swc()
        value = getOneMeasure(measure, swc_path, self.pca, self.specificity)[stat]
        return value


class SomaSurfaceAreaTest(MorphologyTest):
    """Test the agreement between soma surface area observed in the model and experiments"""

    name = "Soma Surface Area Test"
    units = pq.um ** 2

    def generate_prediction(self, model):
        self.specificity = "Type == 1"

        value = self.get_lmeasure(model, "Surface", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class NumberofStemsTest(MorphologyTest):
    """Test the agreement between number of stems observed in the model and experiments"""

    name = "Number of Stems Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        self.specificity = "Type > 0"

        value = self.get_lmeasure(model, "N_stems", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class NumberofBifurcationsTest(MorphologyTest):
    """Test the agreement between number of bifurcations observed in the model and experiments"""

    name = "Number of Bifurcations Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "N_bifs", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteNumberofBifurcationsTest(NumberofBifurcationsTest):
    """Test the agreement between number of basal dendrite bifurcations observed in the model and experiments"""

    name = "Number of Basal Dendrite Bifurcations Test"
    specificity = "Type==3"


class ApicalDendriteNumberofBifurcationsTest(NumberofBifurcationsTest):
    """Test the agreement between number of apical dendrite bifurcations observed in the model and experiments"""

    name = "Number of Apical Dendrite Bifurcations Test"
    specificity = "Type==4"


class NumberofBranchesTest(MorphologyTest):
    """Test the agreement between number of branches observed in the model and experiments"""

    name = "Number of Branches Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "N_branch", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteNumberofBranchesTest(NumberofBranchesTest):
    name = "Basal Dendrite Number of Branches Test"
    specificity = "Type==3"


class ApicalDendriteNumberofBranchesTest(NumberofBranchesTest):
    name = "Apical Dendrite Number of Branches Test"
    specificity = "Type==4"


class OverallWidthTest(MorphologyTest):
    """Test the agreement between overall width observed in the model and experiments"""

    name = "Overall Width Test"
    units = pq.um
    pca = True

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Width", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteOverallWidthTest(OverallWidthTest):
    name = "Basal Dendrite Overall Width Test"
    specificity = "Type==3"


class ApicalDendriteOverallWidthTest(OverallWidthTest):
    name = "Apical Dendrite Overall Width Test"
    specificity = "Type==4"


class OverallHeightTest(MorphologyTest):
    """Test the agreement between overall height observed in the model and experiments"""

    name = "Overall Height Test"
    units = pq.um
    pca = True

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Height", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteOverallHeightTest(OverallHeightTest):
    name = "Basal Dendrite Overall Height Test"
    specificity = "Type==3"


class ApicalDendriteOverallHeightTest(OverallHeightTest):
    name = "Apical Dendrite Overall Height Test"
    specificity = "Type==4"


class OverallDepthTest(MorphologyTest):
    """Test the agreement between overall depth observed in the model and experiments"""

    name = "Overall Depth Test"
    units = pq.um
    pca = True

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Depth", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteOverallDepthTest(OverallDepthTest):
    name = "Basal Dendrite Overall Depth Test"
    specificity = "Type==3"


class ApicalDendriteOverallDepthTest(OverallDepthTest):
    name = "Apical Dendrite Overall Depth Test"
    specificity = "Type==4"


class AverageDiameterTest(MorphologyTest):
    """Test the agreement between average diameter observed in the model and experiments"""

    name = "Average Diameter Test"
    units = pq.um
    pca = True

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Diameter", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteAverageDiameterTest(AverageDiameterTest):
    name = "Basal Dendrite Average Diameter Test"
    specificity = "Type==3"


class ApicalDendriteAverageDiameterTest(AverageDiameterTest):
    name = "Apical Dendrite Average Diameter Test"
    specificity = "Type==4"


class TotalLengthTest(MorphologyTest):
    """Test the agreement between total length observed in the model and experiments"""

    name = "Total Length Test"
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Length", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteTotalLengthTest(TotalLengthTest):
    name = "Basal Dendrite Total Length Test"
    specificity = "Type==3"


class ApicalDendriteTotalLengthTest(TotalLengthTest):
    name = "Apical Dendrite Total Length Test"
    specificity = "Type==4"


class TotalSurfaceTest(MorphologyTest):
    """Test the agreement between total surface observed in the model and experiments"""

    name = "Total Surface Test"
    units = pq.um ** 2

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Surface", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteTotalSurfaceTest(TotalSurfaceTest):
    name = "Basal Dendrite Total Surface Test"
    specificity = "Type==3"


class ApicalDendriteTotalSurfaceTest(TotalSurfaceTest):
    name = "Apical Dendrite Total Surface Test"
    specificity = "Type==4"


class TotalVolumeTest(MorphologyTest):
    """Test the agreement between total volume (excl. soma) observed in the model and experiments"""

    name = "Total Volume Test"
    units = pq.um ** 3

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Volume", "TotalSum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteTotalVolumeTest(TotalVolumeTest):
    name = "Basal Dendrite Total Volume Test"
    specificity = "Type==3"


class ApicalDendriteTotalVolumeTest(TotalVolumeTest):
    name = "Apical Dendrite Total Volume Test"
    specificity = "Type==4"


class MaxEuclideanDistanceTest(MorphologyTest):
    """Test the agreement between max euclidean distance observed in the model and experiments"""

    name = "Max Euclidean Distance Test"
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "EucDistance", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteMaxEuclideanDistanceTest(MaxEuclideanDistanceTest):
    name = "Basal Dendrite Max Euclidean Distance Test"
    specificity = "Type==3"


class ApicalDendriteMaxEuclideanDistanceTest(MaxEuclideanDistanceTest):
    name = "Apical Dendrite Max Euclidean Distance Test"
    specificity = "Type==4"


class MaxPathDistanceTest(MorphologyTest):
    """Test the agreement between max path distance observed in the model and experiments"""

    name = "Max Path Distance Test"
    units = pq.um

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "PathDistance", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteMaxPathDistanceTest(MaxPathDistanceTest):
    name = "Basal Dendrite Max Path Distance Test"
    specificity = "Type==3"


class ApicalDendriteMaxPathDistanceTest(MaxPathDistanceTest):
    name = "Apical Dendrite Max Path Distance Test"
    specificity = "Type==4"


class MaxBranchOrderTest(MorphologyTest):
    """Test the agreement between max branch order observed in the model and experiments"""

    name = "Max Branch Order Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Branch_Order", "Maximum")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteMaxBranchOrderTest(MaxBranchOrderTest):
    name = "Basal Dendrite Max Branch Order Test"
    specificity = "Type==3"


class ApicalDendriteMaxBranchOrderTest(MaxBranchOrderTest):
    name = "Apical Dendrite Max Branch Order Test"
    specificity = "Type==4"


class AverageContractionTest(MorphologyTest):
    """Test the agreement between average contraction observed in the model and experiments"""

    name = "Average Contraction Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Contraction", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteAverageContractionTest(AverageContractionTest):
    name = "Basal Dendrite Average Contraction Test"
    specificity = "Type==3"


class ApicalDendriteAverageContractionTest(AverageContractionTest):
    name = "Apical Dendrite Average Contraction Test"
    specificity = "Type==4"


class PartitionAsymmetryTest(MorphologyTest):
    """Test the agreement between partition asymmetry observed in the model and experiments"""

    name = "Partition Asymmetry Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Partition_asymmetry", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendritePartitionAsymmetryTest(PartitionAsymmetryTest):
    name = "Basal Dendrite Partition Asymmetry Test"
    specificity = "Type==3"


class ApicalDendritePartitionAsymmetryTest(PartitionAsymmetryTest):
    name = "Apical Dendrite Partition Asymmetry Test"
    specificity = "Type==4"


class AverageRallsRatioTest(MorphologyTest):
    """Test the agreement between average Rall's ratio observed in the model and experiments"""

    name = "Average Ralls Ratio Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Pk_classic", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteAverageRallsRatioTest(AverageRallsRatioTest):
    name = "Basal Dendrite Average Ralls Ratio Test"
    specificity = "Type==3"


class ApicalDendriteAverageRallsRatioTest(AverageRallsRatioTest):
    name = "Apical Dendrite Average Ralls Ratio Test"
    specificity = "Type==4"


class AverageBifurcationAngleLocalTest(MorphologyTest):
    """Test the agreement between average bifurcation angle local observed in the model and experiments"""

    name = "Average Bifurcation Angle Local Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Bif_ampl_local", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteAverageBifurcationAngleLocalTest(AverageBifurcationAngleLocalTest):
    name = "Basal Dendrite Average Bifurcation Angle Local Test"
    specificity = "Type==3"


class ApicalDendriteAverageBifurcationAngleLocalTest(AverageBifurcationAngleLocalTest):
    name = "Apical Dendrite Average Bifurcation Angle Local Test"
    specificity = "Type==4"


class AverageBifurcationAngleRemoteTest(MorphologyTest):
    """Test the agreement between average bifurcation angle remote observed in the model and experiments"""

    name = "Average Bifurcation Angle Remote Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Bif_ampl_remote", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteAverageBifurcationAngleRemoteTest(AverageBifurcationAngleRemoteTest):
    name = "Basal Dendrite Average Bifurcation Angle Remote Test"
    specificity = "Type==3"


class ApicalDendriteAverageBifurcationAngleRemoteTest(
    AverageBifurcationAngleRemoteTest
):
    name = "Apical Dendrite Average Bifurcation Angle Remote Test"
    specificity = "Type==4"


class FractalDimensionTest(MorphologyTest):
    """Test the agreement between fractal dimension observed in the model and experiments"""

    name = "Fractal Dimension Test"
    units = pq.dimensionless

    def generate_prediction(self, model):
        value = self.get_lmeasure(model, "Fractal_Dim", "Average")

        return {"mean": value * self.units, "std": 0 * self.units, "n": 1}


class BasalDendriteFractalDimensionTest(FractalDimensionTest):
    name = "Basal Dendrite Fractal Dimension Test"
    specificity = "Type==3"


class ApicalDendriteFractalDimensionTest(FractalDimensionTest):
    name = "Apical Dendrite Fractal Dimension Test"
    specificity = "Type==4"
