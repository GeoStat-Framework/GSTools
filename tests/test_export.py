"""Test the PyVista/VTK export methods"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from gstools import SRF, Exponential, Gaussian
from gstools.random import MasterRNG

HAS_PYVISTA = False
try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    pass


def get_first_line(file):
    with open(file) as f:
        first_line = f.readline().strip("\n")
    return first_line


class TestExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # structured field with a size 100x100x100 and a grid-size of 1x1x1
        x = y = z = range(10)
        model = Gaussian(dim=3, var=0.6, len_scale=20)
        self.srf_structured = SRF(model, seed=20170519)
        self.srf_structured((x, y, z), mesh_type="structured")
        # vector
        model = Gaussian(dim=2, var=0.6, len_scale=10)
        self.srf_vector = SRF(model, generator="VectorField", seed=19841203)
        self.srf_vector((x, y), mesh_type="structured")
        # latlon temporal
        model = Gaussian(latlon=True, temporal=True, var=0.6, len_scale=20)
        self.srf_latlon_temp = SRF(model, seed=20170519)
        self.srf_latlon_temp((x, y, z), mesh_type="structured")
        self.srf_latlon_temp((x, y, z), mesh_type="structured", store="other")
        # 4d
        x = y = z = v = range(3)
        model = Gaussian(dim=4, var=0.6, len_scale=1)
        self.srf_4d = SRF(model, seed=20170519)
        self.srf_4d((x, y, z, v), mesh_type="structured")
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

    def test_csv_export(self):
        # Structured
        sfilename = os.path.join(self.test_dir, "structured.csv")
        self.srf_structured.csv_export(sfilename)
        self.assertTrue(os.path.isfile(sfilename))
        self.assertTrue(get_first_line(sfilename) == "x,y,z,field")
        # Unstructured
        ufilename = os.path.join(self.test_dir, "unstructured.csv")
        self.srf_unstructured.csv_export(ufilename)
        self.assertTrue(os.path.isfile(ufilename))
        self.assertTrue(get_first_line(ufilename) == "x,y,field")
        # latlon temp
        lfilename = os.path.join(self.test_dir, "latlon.csv")
        self.srf_latlon_temp.csv_export(lfilename)
        self.assertTrue(os.path.isfile(lfilename))
        self.assertTrue(get_first_line(lfilename) == "lat,lon,t,field,other")
        # vector
        vfilename = os.path.join(self.test_dir, "vector.csv")
        self.srf_vector.csv_export(vfilename)
        self.assertTrue(os.path.isfile(vfilename))
        self.assertTrue(get_first_line(vfilename) == "x,y,field_x,field_y")
        # 4D
        dfilename = os.path.join(self.test_dir, "4D.csv")
        self.srf_4d.csv_export(dfilename)
        self.assertTrue(os.path.isfile(dfilename))
        self.assertTrue(get_first_line(dfilename) == "x0,x1,x2,x3,field")


if __name__ == "__main__":
    unittest.main()
