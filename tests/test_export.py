"""Test the PyVista/VTK export methods
"""
import unittest
import numpy as np
import os
import tempfile
import shutil

import gstools as gs

HAS_PYVISTA = False
try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    pass


class TestExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.x_grid = self.y_grid = self.z_grid = np.arange(25)
        model = gs.Gaussian(dim=3, var=0.6, len_scale=20)
        self.srf_structured = gs.SRF(model)
        self.srf_structured(
            (self.x_grid, self.y_grid, self.z_grid), mesh_type="structured"
        )
        # unstructured field
        seed = gs.random.MasterRNG(19970221)
        rng = np.random.RandomState(seed())
        self.x_tuple = rng.randint(0, 100, size=100)
        self.y_tuple = rng.randint(0, 100, size=100)
        model = gs.Exponential(
            dim=2, var=1, len_scale=[12.0, 3.0], angles=np.pi / 8.0
        )
        self.srf_unstructured = gs.SRF(model, seed=20170519)
        self.srf_unstructured([self.x_tuple, self.y_tuple])

    def tearDown(self):
        # Remove the test data directory after the test
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_pyvista_struct(self):
        mesh = self.srf_structured.convert()
        self.assertIsInstance(mesh, pv.RectilinearGrid)

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_pyvista_unstruct(self):
        mesh = self.srf_unstructured.convert()
        self.assertIsInstance(mesh, pv.UnstructuredGrid)

    def test_pyevtk_export_struct(self):
        filename = os.path.join(self.test_dir, "structured")
        self.srf_structured.export(filename, "vtk")
        self.assertTrue(os.path.isfile(filename + ".vtr"))

    def test_pyevtk_export_unstruct(self):
        filename = os.path.join(self.test_dir, "unstructured")
        self.srf_unstructured.export(filename, "vtk")
        self.assertTrue(os.path.isfile(filename + ".vtu"))

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_pyvista_vector_struct(self):
        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model, generator="VectorField")
        srf((self.x_grid, self.y_grid), mesh_type="structured", seed=19841203)
        mesh = srf.convert()
        self.assertIsInstance(mesh, pv.RectilinearGrid)

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_pyvista_vector_unstruct(self):
        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model, generator="VectorField")
        srf((self.x_tuple, self.y_tuple), mesh_type="unstructured", seed=19841203)
        mesh = srf.convert()
        self.assertIsInstance(mesh, pv.UnstructuredGrid)

    def test_pyevtk_vector_export_struct(self):
        filename = os.path.join(self.test_dir, "vector")
        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model, generator="VectorField")
        srf((self.x_grid, self.y_grid), mesh_type="structured", seed=19841203)
        srf.export(filename, "vtk")
        self.assertTrue(os.path.isfile(filename + ".vtr"))

    def test_pyevtk_vector_export_unstruct(self):
        filename = os.path.join(self.test_dir, "vector")
        model = gs.Gaussian(dim=2, var=1, len_scale=10)
        srf = gs.SRF(model, generator="VectorField")
        srf((self.x_tuple, self.y_tuple), mesh_type="unstructured", seed=19841203)
        srf.export(filename, "vtk")
        self.assertTrue(os.path.isfile(filename + ".vtu"))


if __name__ == "__main__":
    unittest.main()
