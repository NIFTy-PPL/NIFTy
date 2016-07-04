import nifty as nt
import numpy as np
import unittest
import d2o


class TestFFTWTransform(unittest.TestCase):
    def test_comm(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = d2o.distributed_data_object(a)
        b.comm = [1, 2, 3]  # change comm to something not supported
        with self.assertRaises(RuntimeError):
            x.fft_machine.transform(b, x, x.get_codomain())

    def test_shapemismatch(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = x.cast(a)
        with self.assertRaises(ValueError):
            x.fft_machine.transform(b, x, x.get_codomain(), axes=(0, 1, 2))

    def test_local_ndarray(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(a, x, x.get_codomain()),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_local_notzero(self):
        x = nt.RgSpace(8, fft_module='pyfftw')
        a = np.ones((8, 8))
        b = x.cast(a)
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(1,)),
                np.fft.fftn(a, axes=(1,))
            ), 'results do not match numpy.fft.fftn'
        )

    def test_not(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = d2o.distributed_data_object(a, distribution_strategy='not')
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain()),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_axesnone(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = x.cast(a)
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain()),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_axesnone_equal(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = d2o.distributed_data_object(a, distribution_strategy='equal')
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain()),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_axesall(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = x.cast(a)
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(0, 1)),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_axesall_equal(self):
        x = nt.RgSpace((8, 8), fft_module='pyfftw')
        a = np.ones((8, 8))
        b = d2o.distributed_data_object(a, distribution_strategy='equal')
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(0, 1)),
                np.fft.fftn(a)
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_zero(self):
        x = nt.RgSpace(8, fft_module='pyfftw')
        a = np.ones((8, 8)) + 1j*np.zeros((8, 8))
        b = x.cast(a)
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(0,)),
                np.fft.fftn(a, axes=(0,))
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_zero_equal(self):
        x = nt.RgSpace(8, fft_module='pyfftw')
        a = np.ones((8, 8)) + 1j*np.zeros((8, 8))
        b = d2o.distributed_data_object(a, distribution_strategy='equal')
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(0,)),
                np.fft.fftn(a, axes=(0,))
            ), 'results do not match numpy.fft.fftn'
        )

    def test_mpi_zero_not(self):
        x = nt.RgSpace(8, fft_module='pyfftw')
        a = np.ones((8, 8)) + 1j*np.zeros((8, 8))
        b = d2o.distributed_data_object(a, distribution_strategy='not')
        self.assertTrue(
            np.allclose(
                x.fft_machine.transform(b, x, x.get_codomain(), axes=(0,)),
                np.fft.fftn(a, axes=(0,))
            ), 'results do not match numpy.fft.fftn'
        )

if __name__ == '__main__':
    unittest.main()
