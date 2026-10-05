"""
GStools subpackage providing observation data for stochastic geometry.

.. currentmodule:: gstools.geometry.observations

The following classes and functions are provided

.. autosummary::
   BoreholeData
   from_well_logs
   from_surface_points
"""

import warnings

import numpy as np

__all__ = ["KIND", "BoreholeData", "from_well_logs", "from_surface_points"]

KIND = ["thickness", "elevation"]
""":class:`list` of :class:`str`: Supported data kinds."""

MERGE_TOL = 1e-9
""":class:`float`: Tolerance for agreeing duplicate equalities."""


class BoreholeData:
    """Aligned per-datum records for conditioning a stack.

    ``kind="thickness"`` data condition a
    :class:`gstools.geometry.LayerStack` (layer index ``-1`` addresses the
    base elevation), ``kind="elevation"`` data a
    :class:`gstools.geometry.SurfaceStack` (surface indices).

    Each datum is either an equality (finite ``value``) or an inequality
    (``value`` NaN, informative ``lower``/``upper``). A thickness equality
    of ``0`` means "layer absent" and is stored as the bound ``t <= 0``.

    Parameters
    ----------
    pos : :class:`numpy.ndarray`
        Lateral positions, shape ``(dim - 1, n)``, all finite.
    layer : :class:`numpy.ndarray`
        Layer (thickness) or surface (elevation) index per datum,
        shape ``(n,)``.
    value : :class:`numpy.ndarray`
        Thickness or elevation per datum, shape ``(n,)``, NaN where this
        is not an equality constraint.
    kind : :class:`str`, optional
        One of ``KIND``. Default: "thickness"
    lower : :class:`numpy.ndarray`, optional
        Lower bound per datum. Default: 0.0 for thickness, -inf for
        elevation (and for base data)
    upper : :class:`numpy.ndarray`, optional
        Upper bound per datum. Default: +inf
    err : :class:`numpy.ndarray`, optional
        Measurement variance per datum.
    well : :class:`numpy.ndarray`, optional
        Well id per datum, every error message names a well.
        Default: the datum index
    truncated : :class:`numpy.ndarray`, optional
        Boolean erosional-truncation flag per datum.

    Raises
    ------
    ValueError
        If invariants are violated:

            * non-finite ``pos``
            * a datum is neither an equality nor a bounded inequality
            * ``lower > value`` or ``value > upper``
            * ``lower >= upper``
            * ``lower < 0`` (thickness only)
            * a negative layer index
            * an unknown ``kind``
            * a negative or non-finite ``err``
            * mismatched array lengths
    """

    def __init__(
        self,
        pos,
        layer,
        value,
        kind="thickness",
        lower=None,
        upper=None,
        err=None,
        well=None,
        truncated=None,
    ):
        if kind not in KIND:
            raise ValueError(f"BoreholeData: kind must be one of {KIND}.")
        self._kind = kind
        thick = kind == "thickness"

        # lateral positions with shape (dim - 1, n)
        pos = np.asarray(pos, dtype=np.double)
        self._pos = pos.reshape(1, -1) if pos.ndim < 2 else pos
        n = self._pos.shape[1]

        # target and value per datum, layer -1 is the base elevation
        self._layer = _per_datum(layer, n, "layer").astype(np.int64)
        self._value = _per_datum(value, n, "value").astype(np.double)
        base = self._layer < 0

        # bounds: thicknesses are >= 0 by default, elevations unbounded
        if lower is None:
            lower = np.where(thick & ~base, 0.0, -np.inf)
        self._lower = _per_datum(lower, n, "lower").astype(np.double)
        if upper is None:
            upper = np.inf
        self._upper = _per_datum(upper, n, "upper").astype(np.double)

        # optional records per datum
        self._err = None if err is None else _per_datum(err, n, "err")
        if well is None:
            well = [str(i) for i in range(n)]
        self._well = _per_datum(np.asarray(well, dtype=str), n, "well")
        if truncated is None:
            truncated = False
        self._truncated = _per_datum(truncated, n, "truncated").astype(bool)

        # a thickness of 0 ("layer absent") is stored as the bound t <= 0
        if thick:
            absent = (self._value == 0) & ~base
            self._value[absent] = np.nan
            self._upper[absent] = np.minimum(self._upper[absent], 0.0)

        self._validate()

    def _validate(self):
        """Check for errors listed under ``Raises``."""
        wells = self._well

        def fail(mask, msg):
            if np.any(mask):
                names = sorted(set(wells[mask].tolist()))[:10]
                names = ", ".join(f"#{w}" for w in names)
                raise ValueError(f"BoreholeData: {msg} (wells: {names})")

        eq = np.isfinite(self._value)
        base = self._layer < 0
        # thickness data of layers (not of the base elevation)
        thick = (self._kind == "thickness") & ~base

        # positions and layer indices (-1 only for the base of a LayerStack)
        fail(~np.all(np.isfinite(self._pos), axis=0), "non-finite positions")
        min_layer = -1 if self._kind == "thickness" else 0
        fail(self._layer < min_layer, "invalid layer index")

        # every datum is an equality or an inequality with a real bound
        low_def = np.where(thick, 0.0, -np.inf)
        informative = (self._lower > low_def) | (self._upper < np.inf)
        fail(~eq & ~informative, "datum is neither equality nor bounded")

        # equalities lie within their bounds
        fail(eq & (self._lower > self._value), "lower bound above value")
        fail(eq & (self._value > self._upper), "value above upper bound")

        # inequalities need lower < upper, a point interval must be given
        # as an equality, except "layer absent" (stored as 0 <= t <= 0)
        fail(~eq & (self._lower > self._upper), "empty bounds (lower > upper)")
        absent = thick & (self._lower == 0) & (self._upper == 0)
        point = ~eq & (self._lower == self._upper) & ~absent
        fail(point, "lower == upper, give it as an equality instead")

        # thicknesses are non-negative
        fail(thick & (self._lower < 0), "negative thickness bound")
        fail(thick & eq & (self._value < 0), "negative thickness")

        # measurement errors are finite, non-negative variances
        if self._err is not None:
            bad = ~np.isfinite(self._err) | (self._err < 0)
            fail(bad, "negative or non-finite measurement error")

    def __getitem__(self, mask):
        """Subset every field by one boolean/index mask.

        Parameters
        ----------
        mask : :class:`numpy.ndarray`
            Boolean mask or integer indices selecting data.

        Returns
        -------
        :any:`BoreholeData`
            The selected data.
        """
        return BoreholeData(
            self._pos[:, mask],
            self._layer[mask],
            self._value[mask],
            kind=self._kind,
            lower=self._lower[mask],
            upper=self._upper[mask],
            err=None if self._err is None else self._err[mask],
            well=self._well[mask],
            truncated=self._truncated[mask],
        )

    def for_layer(self, i):
        """Extract the data belonging to a single layer.

        Parameters
        ----------
        i : :class:`int`
            Layer index.

        Returns
        -------
        :any:`BoreholeData`
            The data of layer ``i``.
        """
        return self[self._layer == i]

    def merge_duplicates(self):
        """Merge exactly coincident lateral positions per layer.

        "Same location" is a modelling decision, so positions are never
        merged by tolerance. Intersects bounds and averages equalities.

        Returns
        -------
        :any:`BoreholeData`
            The data with at most one datum per layer and position.

        Raises
        ------
        ValueError
            Naming the wells, on disagreeing equalities or an empty bounds
            intersection.
        """
        n = len(self)
        if n < 2:
            return self
        keys = np.concatenate((self._layer[None, :], self._pos), axis=0).T
        _, first, inv = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        inv = inv.reshape(-1)
        if len(first) == n:
            return self
        order = np.argsort(first)  # keep order of first occurrence
        rows = {"value": [], "lower": [], "upper": [], "err": []}
        rows.update({"well": [], "truncated": [], "idx": []})
        for grp in order:
            sel = np.flatnonzero(inv == grp)
            ids = list(dict.fromkeys(self._well[sel].tolist()))
            wells = "+".join(f"#{w}" for w in ids)  # for messages
            val = self._value[sel]
            eq = np.isfinite(val)
            low = np.max(self._lower[sel])
            upp = np.min(self._upper[sel])
            if np.any(eq):
                ref = val[eq]
                if not np.allclose(
                    ref, ref[0], rtol=MERGE_TOL, atol=MERGE_TOL
                ):
                    raise ValueError(
                        "BoreholeData.merge_duplicates: disagreeing values "
                        f"{ref.tolist()} at one position (wells: {wells})"
                    )
                value = float(np.mean(ref))
                if not low - MERGE_TOL <= value <= upp + MERGE_TOL:
                    raise ValueError(
                        "BoreholeData.merge_duplicates: value outside the "
                        f"merged bounds at one position (wells: {wells})"
                    )
            else:
                value = np.nan
                if low > upp:
                    raise ValueError(
                        "BoreholeData.merge_duplicates: empty bounds "
                        f"intersection at one position (wells: {wells})"
                    )
            rows["value"].append(value)
            rows["lower"].append(low)
            rows["upper"].append(upp)
            if self._err is not None:
                rows["err"].append(float(np.mean(self._err[sel])))
            rows["well"].append("+".join(ids))
            rows["truncated"].append(bool(np.any(self._truncated[sel])))
            rows["idx"].append(sel[0])
        idx = np.array(rows["idx"])
        return BoreholeData(
            self._pos[:, idx],
            self._layer[idx],
            rows["value"],
            kind=self._kind,
            lower=rows["lower"],
            upper=rows["upper"],
            err=None if self._err is None else rows["err"],
            well=rows["well"],
            truncated=rows["truncated"],
        )

    def __len__(self):
        return self._pos.shape[1]

    def __repr__(self):
        n_eq = int(np.sum(np.isfinite(self._value)))
        return (
            f"BoreholeData(kind='{self._kind}', n={len(self)}, "
            f"equalities={n_eq}, inequalities={len(self) - n_eq})"
        )

    @property
    def kind(self):
        """:class:`str`: Either "thickness" or "elevation"."""
        return self._kind

    @property
    def pos(self):
        """:class:`numpy.ndarray`: Lateral positions ``(dim - 1, n)``."""
        return self._pos

    @property
    def layer(self):
        """:class:`numpy.ndarray`: Layer or surface index per datum."""
        return self._layer

    @property
    def value(self):
        """:class:`numpy.ndarray`: Equality values (NaN for inequalities)."""
        return self._value

    @property
    def lower(self):
        """:class:`numpy.ndarray`: Lower bounds."""
        return self._lower

    @property
    def upper(self):
        """:class:`numpy.ndarray`: Upper bounds."""
        return self._upper

    @property
    def err(self):
        """:class:`numpy.ndarray` or :any:`None`: Measurement variances."""
        return self._err

    @property
    def well(self):
        """:class:`numpy.ndarray`: Well id per datum."""
        return self._well

    @property
    def truncated(self):
        """:class:`numpy.ndarray`: Erosional-truncation flags."""
        return self._truncated

    @property
    def is_equality(self):
        """:class:`numpy.ndarray`: Whether each datum is an equality."""
        return np.isfinite(self._value)


def _per_datum(value, n, name):
    """One value per datum: broadcast a scalar or check an array length.

    Always returns a copy, so the given array is never shared.
    """
    value = np.asarray(value)
    if value.ndim == 0:
        return np.full(n, value.item(), dtype=value.dtype)
    value = value.reshape(-1)
    if value.size != n:
        raise ValueError(
            f"BoreholeData: '{name}' has {value.size} entries, expected {n}."
        )
    return value.copy()


class _Rows:
    """Collector for data rows and rejected data."""

    def __init__(self, lat_dim):
        self.lat_dim = lat_dim
        self.rows = []
        self.rejected = []

    def add(self, well, layer, lat, value=np.nan, lower=None, upper=np.inf):
        self.rows.append((well, layer, lat, value, lower, upper, False))

    def add_trunc(self, well, layer, lat, lower):
        self.rows.append((well, layer, lat, np.nan, lower, np.inf, True))

    def reject(self, well, layer, reason):
        self.rejected.append({"well": well, "layer": layer, "reason": reason})

    def build(self, kind):
        n = len(self.rows)
        pos = np.empty((self.lat_dim, n))
        well, layer, value, lower, upper, trunc = [], [], [], [], [], []
        for i, (w, lay, lat, val, low, upp, tr) in enumerate(self.rows):
            pos[:, i] = lat
            well.append(str(w))
            layer.append(lay)
            value.append(val)
            if low is None:
                low = 0.0 if kind == "thickness" and lay >= 0 else -np.inf
            lower.append(low)
            upper.append(upp)
            trunc.append(tr)
        return BoreholeData(
            pos,
            np.array(layer, dtype=np.int64),
            np.array(value, dtype=np.double),
            kind=kind,
            lower=np.array(lower, dtype=np.double),
            upper=np.array(upper, dtype=np.double),
            well=np.array(well, dtype=str) if well else None,
            truncated=np.array(trunc, dtype=bool),
        )


def _finish(rows, kind, strict, func):
    """Build the data and report (never silently drop) rejected data."""
    if rows.rejected:
        lines = [
            f"well #{r['well']}, layer {r['layer']}: {r['reason']}"
            for r in rows.rejected
        ]
        msg = (
            f"{func}: {len(lines)} datum/data could not be reduced to "
            "single-layer constraints:\n  " + "\n  ".join(lines[:20])
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return rows.build(kind), rows.rejected


def _check_order(rows, well, s_val, a_val, b_val):
    """Check observed contacts are ordered and inside the hole."""
    obs = s_val[np.isfinite(s_val)]
    if np.any(np.diff(obs) < 0):
        rows.reject(well, None, "contacts out of stratigraphic order")
        return False
    if obs.size:
        tol = 1e-9 * max(1.0, np.max(np.abs(obs)))
        if (np.isfinite(a_val) and obs.min() < a_val - tol) or (
            np.isfinite(b_val) and obs.max() > b_val + tol
        ):
            rows.reject(well, None, "contact picked outside the drilled hole")
            return False
    return True


def _reduce_thickness(rows, well, s_val, lat, a_val, b_val, sign, eroded):
    """Reduce one well to thickness data (signed vertical space)."""
    n_lay = len(s_val) - 1
    obs = np.isfinite(s_val)
    if not np.any(obs):
        return
    idx_obs = np.flatnonzero(obs)
    lo_o, hi_o = idx_obs.min(), idx_obs.max()
    if obs[0]:  # base datum, world elevation
        rows.add(well, -1, lat[0], value=sign * s_val[0], lower=-np.inf)
    elif lo_o > 0:
        rows.reject(
            well,
            int(lo_o),
            f"contact {lo_o} elevation constrains a sum of thicknesses "
            "(lower contacts not observed). Use SurfaceStack for it",
        )
    for i in range(n_lay):
        o_bot, o_top = obs[i], obs[i + 1]
        if o_bot and o_top:
            thick = s_val[i + 1] - s_val[i]
            mid = 0.5 * (lat[i] + lat[i + 1])
            if eroded is not None and thick == 0 and well not in eroded:
                raise ValueError(
                    f"from_well_logs: well #{well}, layer {i} has zero "
                    "thickness, but erosion is enabled and 'eroded' does "
                    "not state whether it was eroded or never deposited."
                )
            if eroded is not None and i in eroded.get(well, ()):
                if thick > 0:
                    rows.add_trunc(well, i, mid, lower=thick)
                # absent and eroded: no constraint at all
                continue
            rows.add(well, i, mid, value=thick)
        elif o_top and not o_bot:
            if i + 1 != lo_o:
                rows.reject(well, i, f"contact {i} not picked")
            elif np.isfinite(a_val) and s_val[i + 1] - a_val > 0:
                rows.add(well, i, lat[i + 1], lower=s_val[i + 1] - a_val)
        elif o_bot and not o_top:
            if i != hi_o:
                rows.reject(well, i, f"contact {i + 1} not picked")
            elif np.isfinite(b_val) and b_val - s_val[i] > 0:
                rows.add(well, i, lat[i], lower=b_val - s_val[i])
        elif lo_o < i < hi_o:
            rows.reject(well, i, f"contacts {i} and {i + 1} not picked")


def _reduce_elevation(rows, well, z_val, lat, a_val, b_val, relation):
    """Reduce one well to single-surface elevation data.

    Every constraint is an equality or a box on one raw surface, see
    :class:`gstools.geometry.SurfaceStack`.
    """
    n_srf = len(z_val)
    obs = np.isfinite(z_val)
    if not np.any(obs):
        return
    idx_obs = np.flatnonzero(obs)
    lo_o, hi_o = idx_obs.min(), idx_obs.max()
    onlap = relation == "onlap"
    # observed contacts, coincident ones: onlap -> lowest carries the
    # equality, erode -> highest, the others get one-sided bounds
    for j in idx_obs:
        c_val = z_val[j]
        same = idx_obs[z_val[idx_obs] == c_val]
        carrier = same.min() if onlap else same.max()
        if j == carrier:
            rows.add(well, j, lat[j], value=c_val)
        elif onlap:
            rows.add(well, j, lat[j], upper=c_val)
        else:
            rows.add(well, j, lat[j], lower=c_val)
    for j in range(n_srf):
        if obs[j]:
            continue
        if onlap:
            if j < lo_o:
                bound = a_val if np.isfinite(a_val) else z_val[lo_o]
                rows.add(well, j, lat[lo_o], upper=bound)
            elif j < hi_o:
                nxt = idx_obs[idx_obs > j].min()
                rows.add(well, j, lat[nxt], upper=z_val[nxt])
            elif j == hi_o + 1 and np.isfinite(b_val) and b_val > z_val[hi_o]:
                rows.add(well, j, lat[hi_o], lower=b_val)
        else:
            if j > hi_o:
                bound = (
                    max(b_val, z_val[hi_o])
                    if np.isfinite(b_val)
                    else z_val[hi_o]
                )
                rows.add(well, j, lat[hi_o], lower=bound)
            elif j > lo_o:
                prv = idx_obs[idx_obs < j].max()
                rows.add(well, j, lat[prv], lower=z_val[prv])
            elif j == lo_o - 1 and np.isfinite(a_val) and a_val < z_val[lo_o]:
                rows.add(well, j, lat[lo_o], upper=a_val)


def from_well_logs(
    tops,
    *,
    layers,
    collar=None,
    tvd=None,
    td=None,
    eroded=None,
    stacking="up",
    kind="thickness",
    relation="onlap",
    strict=False,
):
    """Build :any:`BoreholeData` from raw well-log contacts.

    Contacts are the ``n_layers + 1`` interfaces of a stack, ordered along
    the stacking direction (interface ``i`` is the base of layer ``i``).

    ``kind="thickness"`` (for :class:`gstools.geometry.LayerStack`):
    consecutive contacts within a well are translated into thickness
    data:

        * two picked contacts give a thickness equality
        * a hole ending inside a layer gives ``t >= partial``
        * a zero thickness gives "absent" (``t <= 0``), unless ``eroded``
          is given, see below
        * with ``eroded``: a truncated layer gives ``t >= t_obs`` and a
          fully eroded one gives no constraint
        * the lowest contact (interface ``0``), if picked, gives a base
          datum (an elevation, layer index ``-1``).

    A contact that does not reduce, within one well, to a single-layer
    thickness or a base datum (e.g. a lowest picked contact above
    interface ``0``) is not used. It is reported in ``rejected`` together
    with its well, layer and reason.

    ``kind="elevation"`` (for :class:`gstools.geometry.SurfaceStack`):
    contacts are exact surface elevations, unobserved surfaces get the
    box constraints implied by ``relation``. Nothing is rejected on
    structural grounds.

    For both kinds, a well whose picked contacts are out of stratigraphic
    order or lie outside the drilled hole is rejected as a whole.

    Parameters
    ----------
    tops : :class:`dict`
        Mapping ``well -> array (n_layers + 1,)`` of measured depths (below
        collar) of each contact, NaN where not picked or not reached. May
        be ``None`` for wells given in ``tvd``.
    layers : :class:`int` or :class:`list`
        Number of layers, or layer names in stacking order.
    collar : :class:`dict`
        Required. Mapping ``well -> (x, [y,] z)`` collar position, with
        ``z`` the collar elevation. Depths are positive downwards.
    tvd : :class:`dict`, optional
        For deviated wells: mapping ``well -> array (n_layers + 1, dim)``
        of ``(dx, [dy,] tvd)`` per contact relative to the collar (NaN rows
        where not picked). Deviated wells are never guessed from ``tops``.
    td : :class:`dict`, optional
        Mapping ``well -> total depth`` (measured depth, or TVD for wells
        in ``tvd``), yielding inequality data where a hole ends inside a
        layer.
    eroded : :class:`dict`, optional
        Enables erosion: mapping ``well -> iterable of layer indices`` that
        are erosionally truncated in that well. With erosion enabled,
        every well with a zero-thickness layer must be listed (possibly
        with an empty iterable), since encoding "fully eroded" as "never
        deposited" produces a confidently wrong pinch-out.
        Default: None (no erosion)
    stacking : :class:`str`, optional
        Either "up" or "down" (must match the stack). ``kind="elevation"``
        requires "up". Default: "up"
    kind : :class:`str`, optional
        "thickness" or "elevation". Default: "thickness"
    relation : :class:`str`, optional
        "onlap" or "erode", for ``kind="elevation"``. Default: "onlap"
    strict : :class:`bool`, optional
        If True, raise on any rejected datum, otherwise warn and report it
        in ``rejected``. Default: False

    Returns
    -------
    data : :any:`BoreholeData`
        The reduced single-layer constraints of all wells.
    rejected : :class:`list` of :class:`dict`
        Every unreducible datum, with keys ``well``/``layer``/``reason``.

    Raises
    ------
    ValueError
        If ``kind``, ``stacking`` or ``relation`` is unknown, ``collar`` is
        missing or malformed, no wells are given, a well has the wrong
        number of contacts, ``eroded`` is given but a well with a
        zero-thickness layer is not listed in it, or (``strict=True``) any
        datum is rejected.

    Warns
    -----
    UserWarning
        If data are rejected and ``strict=False``, or if deviated wells are
        projected to the lateral midpoint of each layer, which breaks the
        one-position-per-(well, layer) assumption for wells crossing a layer
        over a long lateral distance.
    """
    if kind not in KIND:
        raise ValueError(f"from_well_logs: kind must be one of {KIND}.")
    if stacking not in ("up", "down"):
        raise ValueError("from_well_logs: stacking must be 'up' or 'down'.")
    if kind == "elevation" and stacking != "up":
        raise ValueError("from_well_logs: elevation data need stacking='up'.")
    if relation not in ("onlap", "erode"):
        raise ValueError("from_well_logs: relation must be onlap or erode.")
    if collar is None:
        raise ValueError("from_well_logs: 'collar' positions are required.")
    n_lay = layers if isinstance(layers, int) else len(layers)
    tops = {} if tops is None else dict(tops)
    tvd = {} if tvd is None else dict(tvd)
    td = {} if td is None else dict(td)
    wells = list(dict.fromkeys(list(tops) + list(tvd)))
    if not wells:
        raise ValueError("from_well_logs: no wells given.")
    lat_dim = len(np.atleast_1d(collar[wells[0]])) - 1
    rows = _Rows(lat_dim)
    sign = 1.0 if stacking == "up" else -1.0
    max_offset = 0.0
    for well in wells:
        if well not in collar:
            raise ValueError(f"from_well_logs: no collar for well #{well}.")
        col = np.asarray(collar[well], dtype=np.double).reshape(-1)
        if col.size != lat_dim + 1:
            raise ValueError(f"from_well_logs: bad collar for well #{well}.")
        if well in tvd:
            arr = np.asarray(tvd[well], dtype=np.double)
            arr = arr.reshape(n_lay + 1, lat_dim + 1)
            lat = col[:-1] + arr[:, :-1]
            z_val = col[-1] - arr[:, -1]
            if kind == "thickness":
                off = np.nanmax(np.abs(np.diff(lat, axis=0)), initial=0.0)
                max_offset = max(max_offset, off)
        else:
            md_val = np.asarray(tops[well], dtype=np.double).reshape(-1)
            if md_val.size != n_lay + 1:
                raise ValueError(
                    f"from_well_logs: well #{well} needs {n_lay + 1} contacts."
                )
            lat = np.repeat(col[None, :-1], n_lay + 1, axis=0)
            z_val = col[-1] - md_val
        lat = np.where(np.isfinite(lat), lat, col[:-1])
        z_td = col[-1] - float(td[well]) if well in td else np.nan
        z_top = col[-1]
        if sign > 0:
            a_val, b_val = z_td, z_top
        else:
            a_val, b_val = -z_top, -z_td
        s_val = sign * z_val
        if not _check_order(rows, well, s_val, a_val, b_val):
            continue
        if kind == "thickness":
            _reduce_thickness(
                rows, well, s_val, lat, a_val, b_val, sign, eroded
            )
        else:
            _reduce_elevation(rows, well, z_val, lat, a_val, b_val, relation)
    if max_offset > 0:
        warnings.warn(
            "from_well_logs: deviated wells are projected to the lateral "
            "midpoint of each layer (max lateral offset between contacts: "
            f"{max_offset:.3g}), a well crossing a layer over a long lateral "
            "distance breaks the one-position-per-(well, layer) assumption."
        )
    return _finish(rows, kind, strict, "from_well_logs")


def from_surface_points(
    x,
    y,
    z,
    surface,
    *,
    surfaces,
    well=None,
    td=None,
    relation="onlap",
    strict=False,
):
    """Build elevation :any:`BoreholeData` from a table of surface points.

    The table has one row per contact (``x``, ``y``, ``z``, ``surface``).
    Contacts are exact surface elevations and can be used directly for
    conditioning a :class:`gstools.geometry.SurfaceStack`, no differencing of
    thicknesses is needed.

    Contacts are grouped into wells. Within a well, surfaces without a
    contact get the box constraints implied by ``relation``. Where several
    surfaces share the same elevation in a well, only one of them carries
    the equality, the others get one-sided bounds:

        * ``relation="onlap"``: the lowest surface carries the equality,
          the ones above get upper bounds
        * ``relation="erode"``: the highest surface carries the equality,
          the ones below get lower bounds

    Parameters
    ----------
    x, y : array_like
        Lateral contact coordinates. For 2D stacks (no lateral ``y``
        direction) pass ``y=None``.
    z : array_like
        Contact elevations.
    surface : array_like
        Surface name or index per contact.
    surfaces : :class:`int` or :class:`list`
        Number of surfaces, or surface names ordered from bottom to top.
    well : array_like, optional
        Well id per contact.
        Default: contacts at identical lateral positions form one well.
    td : :class:`dict`, optional
        Mapping ``well -> hole-bottom elevation``, yielding inequality data
        for the surfaces below the last contact.
    relation : :class:`str`, optional
        "onlap" or "erode" (must match the stack). Default: "onlap"
    strict : :class:`bool`, optional
        If True, raise on inconsistent data, otherwise warn and report it
        in ``rejected``. Default: False

    Returns
    -------
    data : :any:`BoreholeData`
        The reduced constraints of all wells, with ``kind == "elevation"``.
    rejected : :class:`list` of :class:`dict`
        Every inconsistent well, with keys ``well``/``layer``/``reason``.
        A well is rejected as a whole if a surface is picked more than once,
        if its contacts are out of stratigraphic order, or if a contact lies
        below the hole bottom given in ``td``.

    Raises
    ------
    ValueError
        If ``relation`` is unknown, a ``surface`` is neither a name in
        ``surfaces`` nor a valid index, or (``strict=True``) any well is
        rejected.

    Warns
    -----
    UserWarning
        If wells are rejected and ``strict=False``.
    """
    if relation not in ("onlap", "erode"):
        raise ValueError("from_surface_points: relation must be onlap/erode.")
    z = np.asarray(z, dtype=np.double).reshape(-1)
    lat = [np.asarray(x, dtype=np.double).reshape(-1)]
    if y is not None:
        lat.append(np.asarray(y, dtype=np.double).reshape(-1))
    lat = np.array(lat)
    n_srf = surfaces if isinstance(surfaces, int) else len(surfaces)
    names = None if isinstance(surfaces, int) else list(surfaces)
    idx = []
    for srf in np.asarray(surface).reshape(-1).tolist():
        if names is not None and srf in names:
            idx.append(names.index(srf))
        elif isinstance(srf, (int, np.integer)) and 0 <= srf < n_srf:
            idx.append(int(srf))
        else:
            raise ValueError(f"from_surface_points: unknown surface {srf}")
    idx = np.array(idx, dtype=np.int64)
    if well is None:
        _, inv = np.unique(lat.T, axis=0, return_inverse=True)
        well = inv.reshape(-1).astype(str)
    well = np.asarray(well).reshape(-1)
    td = {} if td is None else dict(td)
    rows = _Rows(lat.shape[0])
    for w_id in dict.fromkeys(well.tolist()):
        sel = np.flatnonzero(well == w_id)
        z_val = np.full(n_srf, np.nan)
        lat_w = np.repeat(lat[:, sel[:1]].T, n_srf, axis=0)
        if len(set(idx[sel].tolist())) != sel.size:
            rows.reject(w_id, None, "surface picked more than once")
            continue
        z_val[idx[sel]] = z[sel]
        lat_w[idx[sel]] = lat[:, sel].T
        a_val = float(td[w_id]) if w_id in td else np.nan
        if not _check_order(rows, w_id, z_val, a_val, np.nan):
            continue
        _reduce_elevation(rows, w_id, z_val, lat_w, a_val, np.nan, relation)
    return _finish(rows, "elevation", strict, "from_surface_points")
