import unittest
import torch

from nubo.test_functions import *


class TestAckley(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """
        # inputs
        dims = 4

        # run code
        func = Ackley(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))

        # run code
        func = Ackley(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))

class TestDixonPrice(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4

        # run code
        func = DixonPrice(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))

        # run code
        func = DixonPrice(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))

class TestGriewank(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4

        # run code
        func = Griewank(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Griewank(dims=dims)
        y = func(X)
        
        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestHartmann3D(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 3
        
        # run code
        func = Hartmann3D()
        
        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 3
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Hartmann3D()
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestHartmann6D(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 6
        
        # run code
        func = Hartmann6D()
        
        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 6
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Hartmann6D()
        y = func(X)
        
        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestLevy(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4
        
        # run code
        func = Levy(dims=dims)
        
        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Levy(dims=dims)
        y = func(X)
        
        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestRastrigin(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4
        
        # run code
        func = Rastrigin(dims=dims)
        
        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        x = torch.rand((points, dims))
        
        # run code
        func = Rastrigin(dims=dims)
        y = func(x)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestSchwefel(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4

        # run code
        func = Schwefel(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))

        # run code
        func = Schwefel(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestSphere(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4
        
        # run code
        func = Sphere(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Griewank(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestSumSquares(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4

        # run code
        func = SumSquares(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))

        # run code
        func = SumSquares(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


class TestZakharov(unittest.TestCase):

    def test_bounds_type(self):
        """
        Test that bounds are a 2 x dims torch.Tenosr.
        """

        # inputs
        dims = 4

        # run code
        func = Zakharov(dims=dims)

        # test
        self.assertIsInstance(func.bounds, torch.Tensor)
        self.assertEqual(func.bounds.size(), torch.Size([2, dims]))

    def test_output_shape_with_torch_input(self):
        """
        Test that function returns a 1D torch.Tensor when input is a
        torch.Tensor.
        """

        # inputs
        dims = 4
        points = 10
        X = torch.rand((points, dims))
        
        # run code
        func = Zakharov(dims=dims)
        y = func(X)

        # test
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.size(), torch.Size([points]))


if __name__ == "__main__":
    unittest.main()
