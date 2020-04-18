"""Test the PyVista/VTK export methods
"""
import unittest
import numpy as np
import os
import tempfile
import shutil

from gstools import SRF, Gaussian, Exponential
from gstools.random import MasterRNG

HAS_PYVISTA = False
try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    pass


class TestExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        x = y = z = range(25)
        model = Gaussian(dim=3, var=0.6, len_scale=20)
        self.srf_structured = SRF(model)
        self.srf_structured((x, y, z), mesh_type="structured")
        # unstrucutred field
        seed = MasterRNG(19970221)
        rng = np.random.RandomState(seed())
        x = rng.randint(0, 100, size=100)
        y = rng.randint(0, 100, size=100)
        model = Exponential(
            dim=2, var=1, len_scale=[12.0, 3.0], angles=np.pi / 8.0
        )
        self.srf_unstructured = SRF(model, seed=20170519)
        self.srf_unstructured([x, y])

    def tearDown(self):
        # Remove the test data directory after the test
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_pyvista(self):
        mesh = self.srf_structured.to_pyvista()
        self.assertIsInstance(mesh, pv.RectilinearGrid)
        mesh = self.srf_unstructured.to_pyvista()
        self.assertIsInstance(mesh, pv.UnstructuredGrid)

    def test_pyevtk_export(self):
        # Structured
        sfilename = os.path.join(self.test_dir, "structured")
        self.srf_structured.vtk_export(sfilename)
        self.assertTrue(os.path.isfile(sfilename + ".vtr"))
        # Unstructured
        ufilename = os.path.join(self.test_dir, "unstructured")
        self.srf_unstructured.vtk_export(ufilename)
        self.assertTrue(os.path.isfile(ufilename + ".vtu"))


if __name__ == "__main__":
    unittest.main()
