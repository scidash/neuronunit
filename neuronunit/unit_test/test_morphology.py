"""Tests of the 38 Druckmann 2013 test classes"""

import unittest
from sciunit.suites import TestSuite
from neuronunit.models.morphology import SwcCellModel
from neuronunit.tests.morphology import *
import quantities as pq

# SWC file and expected values are from: http://neuromorpho.org/neuron_info.jsp?neuron_name=VD110330-IDC
model = SwcCellModel('neuronunit/unit_test/morphology/pyramidalCell.swc')

class MorphologyTestCase(unittest.TestCase):
    def test_SomaSurfaceAreaTest(self):
        test = SomaSurfaceAreaTest({"mean":1739.11*pq.um**2, "std":1*pq.um**2,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_NumberofStemsTest(self):
        test = NumberofStemsTest({"mean":10*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_NumberofBifurcationsTest(self):
        test = NumberofBifurcationsTest({"mean":138*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_NumberofBranchesTest(self):
        test = NumberofBranchesTest({"mean":286*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_OverallWidthTest(self):
        test = OverallWidthTest({"mean":771.2*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_OverallHeightTest(self):
        test = OverallHeightTest({"mean":1672.09*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_OverallDepthTest(self):
        test = OverallDepthTest({"mean":198.75*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_AverageDiameterTest(self):
        test = AverageDiameterTest({"mean":0.38*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_TotalLengthTest(self):
        test = TotalLengthTest({"mean":16949.9*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_TotalSurfaceTest(self):
        test = TotalSurfaceTest({"mean":18620.5*pq.um**2, "std":1*pq.um**2,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_TotalVolumeTest(self):
        test = TotalVolumeTest({"mean":3558.79*pq.um**3, "std":1*pq.um**3,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_MaxEuclideanDistanceTest(self):
        test = MaxEuclideanDistanceTest({"mean":1099.26*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_MaxPathDistanceTest(self):
        test = MaxPathDistanceTest({"mean":1368.54*pq.um, "std":1*pq.um,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_MaxBranchOrderTest(self):
        test = MaxBranchOrderTest({"mean":23*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_AverageContractionTest(self):
        test = AverageContractionTest({"mean":0.86*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_TotalFragmentationTest(self):
        test = TotalFragmentationTest({"mean":4439*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_PartitionAsymmetryTest(self):
        test = PartitionAsymmetryTest({"mean":0.51*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_AverageRallsRatioTest(self):
        test = AverageRallsRatioTest({"mean":1.6*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_AverageBifurcationAngleLocalTest(self):
        test = AverageBifurcationAngleLocalTest({"mean":80.94*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_AverageBifurcationAngleRemoteTest(self):
        test = AverageBifurcationAngleRemoteTest({"mean":71.75*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)

    def test_FractalDimensionTest(self):
        test = FractalDimensionTest({"mean":1.05*pq.dimensionless, "std":1*pq.dimensionless,"n":1})
        z = test.judge(model)
        self.assertLess(z.score, 0.1)


if __name__ == '__main__':
    unittest.main()