"""
GStools subpackage providing tools for Fields.

.. currentmodule:: gstools.field.tools

The following classes and functions are provided

.. autosummary::
   fmt_mean_norm_trend
   to_vtk_helper
   generate_on_mesh
"""

# pylint: disable=W0212, C0415
import meshio
import numpy as np

from gstools.normalizer import Normalizer
from gstools.tools.export import to_vtk, vtk_export
from gstools.tools.misc import list_format

__all__ = ["fmt_mean_norm_trend", "to_vtk_helper", "generate_on_mesh"]


def _fmt_func_val(f_cls, func_val):  # pragma: no cover
    if func_val is None:
        return str(None)
    if callable(func_val):
        return "<function>"  # or format(func_val.__name__)
    if np.size(func_val) > 1:
        return list_format(func_val, prec=f_cls.model._prec)
    return f"{float(func_val):.{f_cls.model._prec}}"


def _fmt_normalizer(f_cls):  # pragma: no cover
    norm = f_cls.normalizer
    return str(None) if norm.__class__ is Normalizer else norm.name


def fmt_mean_norm_trend(f_cls):  # pragma: no cover
    """Format string repr. for mean, normalizer and trend of a field."""
    args = [
        "mean=" + _fmt_func_val(f_cls, f_cls.mean),
        "normalizer=" + _fmt_normalizer(f_cls),
        "trend=" + _fmt_func_val(f_cls, f_cls.trend),
    ]
    return "".join([", " + arg for arg in args if not arg.endswith("None")])


def to_vtk_helper(
    f_cls, filename=None, field_select="field", fieldname="field"
):  # pragma: no cover
    """Create a VTK/PyVista grid of the field or save it as a VTK file.

    This is an internal helper that will handle saving or creating objects

    Parameters
    ----------
    f_cls : :any:`Field`
        Field class in use.
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtr or .vtu) will be added to the name. If ``None`` is
        passed, a PyVista dataset of the appropriate type will be returned.
    field_select : :class:`str`, optional
        Field that should be stored. Can be:
        "field", "raw_field", "krige_field", "err_field" or "krige_var".
        Default: "field"
    fieldname : :class:`str`, optional
        Name of the field in the VTK file. Default: "field"
    """
    field = f_cls[field_select] if field_select in f_cls.field_names else None
    if f_cls.value_type == "vector":
        if not (f_cls.pos is None or field is None or f_cls.mesh_type is None):
            suf = ["_X", "_Y", "_Z"]
            fields = {}
            for i in range(f_cls.model.dim):
                fields[fieldname + suf[i]] = field[i]
            if filename is None:
                return to_vtk(f_cls.pos, fields, f_cls.mesh_type)
            return vtk_export(filename, f_cls.pos, fields, f_cls.mesh_type)
        raise ValueError(f"Field.to_vtk: '{field_select}' not available.")
    if f_cls.value_type == "scalar":
        if not (f_cls.pos is None or field is None or f_cls.mesh_type is None):
            if filename is None:
                return to_vtk(f_cls.pos, {fieldname: field}, f_cls.mesh_type)
            return vtk_export(
                filename, f_cls.pos, {fieldname: field}, f_cls.mesh_type
            )
        raise ValueError(f"Field.to_vtk: '{field_select}' not available.")
    raise ValueError(f"Unknown field value type: {f_cls.value_type}")


def generate_on_mesh(
    f_cls, mesh, points="centroids", direction="all", name="field", **kwargs
):
    """Generate a field on a given meshio, ogs5py or pyvista mesh.

    Parameters
    ----------
    f_cls : :any:`Field`
        The field class in use.
    mesh : meshio.Mesh or ogs5py.MSH or PyVista mesh
        The given meshio, ogs5py, or PyVista mesh
    points : :class:`str`, optional
        The points to evaluate the field at.
        Either the "centroids" of the mesh cells
        (calculated as mean of the cell vertices) or the "points"
        of the given mesh.
        Default: "centroids"
    direction : :class:`str` or :class:`list`, optional
        Here you can state which direction should be chosen for
        lower dimension. For example, if you got a 2D mesh in xz direction,
        you have to pass "xz". By default, all directions are used.
        One can also pass a list of indices.
        Default: "all"
    name : :class:`str` or :class:`list` of :class:`str`, optional
        Name(s) to store the field(s) in the given mesh as point_data or
        cell_data. If to few names are given, digits will be appended.
        Default: "field"
    **kwargs
        Keyword arguments forwarded to `Field.__call__`.

    Notes
    -----
    This will store the field in the given mesh under the given name,
    if a meshio or PyVista mesh was given.

    See: https://github.com/nschloe/meshio

    See: https://github.com/GeoStat-Framework/ogs5py

    See: https://github.com/pyvista/pyvista
    """
    has_pyvista = False
    has_ogs5py = False

    try:
        import pyvista as pv

        has_pyvista = True
    except ImportError:
        pass
    try:
        import ogs5py as ogs

        has_ogs5py = True
    except ImportError:
        pass

    if isinstance(direction, str) and direction == "all":
        select = list(range(f_cls.dim))
    elif isinstance(direction, str):
        select = _get_select(direction)[: f_cls.dim]
    else:
        select = direction[: f_cls.dim]
    if len(select) < f_cls.dim:
        raise ValueError(
            f"Field.mesh: need at least {f_cls.dim} direction(s), "
            f"got '{direction}'"
        )
    # convert pyvista mesh
    if has_pyvista and pv.is_pyvista_dataset(mesh):
        if points == "centroids":
            pnts = mesh.cell_centers().points.T[select]
        else:
            pnts = mesh.points.T[select]
        out = f_cls.unstructured(pos=pnts, **kwargs)
        # Deal with the output
        fields = [out] if isinstance(out, np.ndarray) else out
        if f_cls.value_type == "vector":
            fields = [f.T for f in fields]
        for f_name, field in zip(_names(name, len(fields)), fields):
            mesh[f_name] = field
    # convert ogs5py mesh
    elif has_ogs5py and isinstance(mesh, ogs.MSH):
        if points == "centroids":
            pnts = mesh.centroids_flat.T[select]
        else:
            pnts = mesh.NODES.T[select]
        out = f_cls.unstructured(pos=pnts, **kwargs)
    # convert meshio mesh
    elif isinstance(mesh, meshio.Mesh):
        if points == "centroids":
            # define unique order of cells
            offset = []
            length = []
            mesh_dim = mesh.points.shape[1]
            if mesh_dim < f_cls.dim:
                raise ValueError("Field.mesh: mesh dimension too low!")
            pnts = np.empty((0, mesh_dim), dtype=np.double)
            for cell in mesh.cells:
                pnt = np.mean(mesh.points[cell.data], axis=1)
                offset.append(pnts.shape[0])
                length.append(pnt.shape[0])
                pnts = np.vstack((pnts, pnt))
            # generate pos for __call__
            pnts = pnts.T[select]
            out = f_cls.unstructured(pos=pnts, **kwargs)
            fields = [out] if isinstance(out, np.ndarray) else out
            if f_cls.value_type == "vector":
                fields = [f.T for f in fields]
            f_lists = []
            for field in fields:
                f_list = []
                for off, leng in zip(offset, length):
                    f_list.append(field[off : off + leng])
                f_lists.append(f_list)
            for f_name, f_list in zip(_names(name, len(f_lists)), f_lists):
                mesh.cell_data[f_name] = f_list
        else:
            out = f_cls.unstructured(pos=mesh.points.T[select], **kwargs)
            fields = [out] if isinstance(out, np.ndarray) else out
            if f_cls.value_type == "vector":
                fields = [f.T for f in fields]
            for f_name, field in zip(_names(name, len(fields)), fields):
                mesh.point_data[f_name] = field
    else:
        raise ValueError("Field.mesh: Unknown mesh format!")
    return out


def _names(name, cnt):
    name = [name] if isinstance(name, str) else list(name)[:cnt]
    if len(name) < cnt:
        name += [f"{name[-1]}{i + 1}" for i in range(cnt - len(name))]
    return name


def _get_select(direction):
    select = []
    if not 0 < len(direction) < 4:
        raise ValueError(
            f"Field.mesh: need 1 to 3 direction(s), got '{direction}'"
        )
    for axis in direction:
        if axis == "x":
            if 0 in select:
                raise ValueError(
                    f"Field.mesh: got duplicate directions {direction}"
                )
            select.append(0)
        elif axis == "y":
            if 1 in select:
                raise ValueError(
                    f"Field.mesh: got duplicate directions {direction}"
                )
            select.append(1)
        elif axis == "z":
            if 2 in select:
                raise ValueError(
                    f"Field.mesh: got duplicate directions {direction}"
                )
            select.append(2)
        else:
            raise ValueError(f"Field.mesh: got unknown direction {axis}")
    return select
