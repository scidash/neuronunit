"""Tests of the 38 Druckmann 2013 test classes"""

import unittest
from sciunit.suites import TestSuite
from neuronunit.models.morphology import SwcCellModel
from neuronunit.tests.morphology import *
import quantities as pq

# SWC file and expected values are from: http://neuromorpho.org/neuron_info.jsp?neuron_name=VD110330-IDC
model = SwcCellModel('neuronunit/unit_test/morphology/pyramidalCell.swc')

class MorphologyTestCase(unittest.TestCase):
    # -------------------- Somatic ------------------------------ #
    def test_SomaSurfaceAreaTest(self):
        test = SomaSurfaceAreaTest({"mean":1739.11*pq.um**2, "std":1*pq.um**2,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_NumberofStemsTest(self):
        test = NumberofStemsTest({"mean":10, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)
        

    # -------------------- Whole Cell ------------------------------ #
    def test_NumberofBifurcationsTest(self):
        test = NumberofBifurcationsTest({"mean":138, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_NumberofBranchesTest(self):
        test = NumberofBranchesTest({"mean":286, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_OverallWidthTest(self):
        test = OverallWidthTest({"mean":771.2*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_OverallHeightTest(self):
        test = OverallHeightTest({"mean":1672.09*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_OverallDepthTest(self):
        test = OverallDepthTest({"mean":198.75*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_AverageDiameterTest(self):
        test = AverageDiameterTest({"mean":0.38*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_TotalLengthTest(self):
        test = TotalLengthTest({"mean":16949.9*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_TotalSurfaceTest(self):
        test = TotalSurfaceTest({"mean":18620.5*pq.um**2, "std":1*pq.um**2,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_TotalVolumeTest(self):
        test = TotalVolumeTest({"mean":3558.79*pq.um**3, "std":1*pq.um**3,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_MaxEuclideanDistanceTest(self):
        test = MaxEuclideanDistanceTest({"mean":1099.26*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_MaxPathDistanceTest(self):
        test = MaxPathDistanceTest({"mean":1368.54*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_MaxBranchOrderTest(self):
        test = MaxBranchOrderTest({"mean":23, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_AverageContractionTest(self):
        test = AverageContractionTest({"mean":0.86, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_PartitionAsymmetryTest(self):
        test = PartitionAsymmetryTest({"mean":0.51, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_AverageRallsRatioTest(self):
        test = AverageRallsRatioTest({"mean":1.6, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_AverageBifurcationAngleLocalTest(self):
        test = AverageBifurcationAngleLocalTest({"mean":80.94, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_AverageBifurcationAngleRemoteTest(self):
        test = AverageBifurcationAngleRemoteTest({"mean":71.75, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_FractalDimensionTest(self):
        test = FractalDimensionTest({"mean":1.05, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    # -------------------- APICAL ------------------------------ #
    def test_ApicalDendriteNumberofBifurcationsTest(self):
        test = ApicalDendriteNumberofBifurcationsTest({"mean": 138, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-73, 0.1)

    def test_ApicalDendriteNumberofBranchesTest(self):
        test = ApicalDendriteNumberofBranchesTest({"mean": 286, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-155, 0.1)

    def test_ApicalDendriteOverallWidthTest(self):
        test = ApicalDendriteOverallWidthTest({"mean": 771.2 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-461.617, 0.1)

    def test_ApicalDendriteOverallHeightTest(self):
        test = ApicalDendriteOverallHeightTest({"mean": 1672.09 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-560.8, 0.1)

    def test_ApicalDendriteOverallDepthTest(self):
        test = ApicalDendriteOverallDepthTest({"mean": 198.75 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-8.16, 0.1)

    def test_ApicalDendriteAverageDiameterTest(self):
        test = ApicalDendriteAverageDiameterTest({"mean": 0.38 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteTotalLengthTest(self):
        test = ApicalDendriteTotalLengthTest({"mean": 7791.44 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteTotalSurfaceTest(self):
        test = ApicalDendriteTotalSurfaceTest({"mean": 11094.4 * pq.um ** 2, "std": 1 * pq.um ** 2, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteTotalVolumeTest(self):
        test = ApicalDendriteTotalVolumeTest({"mean": 2613.45 * pq.um ** 3, "std": 1 * pq.um ** 3, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteMaxEuclideanDistanceTest(self):
        test = ApicalDendriteMaxEuclideanDistanceTest({"mean": 1099.26 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteMaxPathDistanceTest(self):
        test = ApicalDendriteMaxPathDistanceTest({"mean": 1368.54 * pq.um, "std": 1 * pq.um, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteMaxBranchOrderTest(self):
        test = ApicalDendriteMaxBranchOrderTest({"mean": 23, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteAverageContractionTest(self):
        test = ApicalDendriteAverageContractionTest({"mean": 0.86, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendritePartitionAsymmetryTest(self):
        test = ApicalDendritePartitionAsymmetryTest({"mean": 0.51, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteAverageRallsRatioTest(self):
        test = ApicalDendriteAverageRallsRatioTest({"mean": 1.6, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_ApicalDendriteAverageBifurcationAngleLocalTest(self):
        test = ApicalDendriteAverageBifurcationAngleLocalTest({"mean": 80.94, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-0.52, 0.1)

    def test_ApicalDendriteAverageBifurcationAngleRemoteTest(self):
        test = ApicalDendriteAverageBifurcationAngleRemoteTest({"mean": 71.75, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-6.75, 0.1)

    def test_ApicalDendriteFractalDimensionTest(self):
        test = ApicalDendriteFractalDimensionTest({"mean": 1.05, "std": 1, "n": 1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)


    # -------------------- BASAL ------------------------------ #
    def test_BasalDendriteNumberofBifurcationsTest(self):
        test = BasalDendriteNumberofBifurcationsTest({"mean":138, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-110, 0.1)

    def test_BasalDendriteNumberofBranchesTest(self):
        test = BasalDendriteNumberofBranchesTest({"mean":286, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-222, 0.1)

    def test_BasalDendriteOverallWidthTest(self):
        test = BasalDendriteOverallWidthTest({"mean":771.2*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-527.4, 0.1)

    def test_BasalDendriteOverallHeightTest(self):
        test = BasalDendriteOverallHeightTest({"mean":1672.09*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-1454.7, 0.1)

    def test_BasalDendriteOverallDepthTest(self):
        test = BasalDendriteOverallDepthTest({"mean":198.75*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-119.26, 0.1)

    def test_BasalDendriteAverageDiameterTest(self):
        test = BasalDendriteAverageDiameterTest({"mean":0.38*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_BasalDendriteTotalLengthTest(self):
        test = BasalDendriteTotalLengthTest({"mean":16949.9*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-13958, 0.1)

    def test_BasalDendriteTotalSurfaceTest(self):
        test = BasalDendriteTotalSurfaceTest({"mean":18620.5*pq.um**2, "std":1*pq.um**2,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-15099.17, 0.1)

    def test_BasalDendriteTotalVolumeTest(self):
        test = BasalDendriteTotalVolumeTest({"mean":3558.79*pq.um**3, "std":1*pq.um**3,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-2982.2, 0.1)

    def test_BasalDendriteMaxEuclideanDistanceTest(self):
        test = BasalDendriteMaxEuclideanDistanceTest({"mean":1099.26*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-900.56, 0.1)

    def test_BasalDendriteMaxPathDistanceTest(self):
        test = BasalDendriteMaxPathDistanceTest({"mean":1368.54*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-1130.2, 0.1)

    def test_BasalDendriteMaxBranchOrderTest(self):
        test = BasalDendriteMaxBranchOrderTest({"mean":23, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-19, 0.1)

    def test_BasalDendriteAverageContractionTest(self):
        test = BasalDendriteAverageContractionTest({"mean":0.86, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)

    def test_BasalDendritePartitionAsymmetryTest(self):
        test = BasalDendritePartitionAsymmetryTest({"mean":0.51, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-0.186, 0.1)

    def test_BasalDendriteAverageRallsRatioTest(self):
        test = BasalDendriteAverageRallsRatioTest({"mean":1.6, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-0.37, 0.1)

    def test_BasalDendriteAverageBifurcationAngleLocalTest(self):
        test = BasalDendriteAverageBifurcationAngleLocalTest({"mean":80.94, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-16.33, 0.1)

    def test_BasalDendriteAverageBifurcationAngleRemoteTest(self):
        test = BasalDendriteAverageBifurcationAngleRemoteTest({"mean":71.75, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score)-11.2, 0.1)

    def test_BasalDendriteFractalDimensionTest(self):
        test = BasalDendriteFractalDimensionTest({"mean":1.05, "std":1,"n":1})
        z = test.judge(model)
        self.assertLess(abs(z.score), 0.1)


if __name__ == '__main__':
    unittest.main()