"""
GStools subpackage providing plurigaussian simulation (PGS)

.. currentmodule:: gstools.field.pgs

The following classes are provided

.. autosummary::
   :toctree:

   PGS
"""

# pylint: disable=C0103
import numpy as np

# very clunky way of supporting both np 1.x and 2.x exceptions
try:
    np.AxisError = np.exceptions.AxisError
except AttributeError:
    ...


class PGS:
    """A class to generate plurigaussian field simulations (PGS).

    See e.g. [Ricketts2023]_ and [Armstrong2011]_ for more details.

    Parameters
    ----------
    dim : :class:`int`
        dimension of the field
    fields : :class:`list` or :class:`numpy.ndarray`
        For `dim > 1` a list of spatial random fields (SRFs), with
        `len(fields) == dim`. For `dim == 1`, the SRF can be directly given,
        instead of a list. This class supports structured and unstructured meshes.
        All fields must have the same shapes.

    Notes
    -----
    Using plurigaussian fields for conditioning fields is still a beta feature.

    References
    ----------
    .. [Ricketts2023] Ricketts, E.J., Freeman, B.L., Cleall, P.J. et al.
        A Statistical Finite Element Method Integrating a Plurigaussian Random
        Field Generator for Multi-scale Modelling of Solute Transport in
        Concrete. Transp Porous Med 148, 95–121 (2023)
        https://doi.org/10.1007/s11242-023-01930-8
    .. [Armstrong2011] Armstrong, Margaret, et al.
        Plurigaussian simulations in geosciences.
        Springer Science & Business Media, 2011.
        https://doi.org/10.1007/978-3-642-19607-2
    """

    def __init__(self, dim, fields):
        # hard to test for 1d case
        for d in range(1, dim):
            if not fields[0].shape == fields[d].shape:
                raise ValueError("PGS: Not all fields have the same shape.")
        self._dim = dim
        self._fields = np.array(fields)
        self._lithotypes = None
        self._pos_lith = None
        self._tree = None
        self._field_names = [f"Z{i+1}" for i in range(len(self._fields))]
        try:
            self._mapping = np.stack(self._fields, axis=1)
        except np.AxisError:
            # if dim==1, `fields` is prob. a raw field & not a 1-tuple or
            # equivalent
            if self._dim == 1:
                self._fields = np.array([self._fields])
                self._mapping = np.stack(self._fields, axis=1)
            else:
                raise

    def __call__(self, lithotypes=None, tree=None):
        """
        Generate the plurigaussian field via spatial lithotype or decision tree.

        Either a lithotype field or a decision tree config must be provided.
        If `lithotypes` is given, map lithotype codes to the PGS via index
        scaling. If `tree` is given, build and apply a DecisionTree to assign
        phase labels.

        Parameters
        ----------
        lithotypes : :class:`numpy.ndarray`, optional
            `dim`-dimensional structured lithotype field. Shape may differ from
            `fields`, as indices are automatically scaled. Mutually exclusive
            with `tree`.
        tree : dict, optional
            Configuration dict for constructing a DecisionTree. Must contain
            node specifications. Mutually exclusive with `lithotypes`.

        Returns
        -------
        pgs : :class:`numpy.ndarray`
            Plurigaussian field array: either the mapped lithotype field or
            the labels assigned by the decision tree, matching the simulation
            domain.

        Raises
        ------
        ValueError
            If neither or both of `lithotypes` and `tree` are provided.
        ValueError
            If `lithotypes` shape does not match `dim` or number of `fields`.
        ValueError
            If `dim` != len(fields) when using `lithotypes`.
        """

        if lithotypes is not None or tree is not None:
            if lithotypes is not None:
                if self._dim > 1:
                    if self._dim != len(self._fields):
                        raise ValueError(
                            "PGS: Mismatch between dim. and no. of fields."
                        )

                self._lithotypes = np.array(lithotypes)
                if len(self._lithotypes.shape) != self._dim:
                    raise ValueError(
                        "PGS: Mismatch between dim. and facies shape."
                    )
                self._pos_lith = self.calc_lithotype_axes(
                    self._lithotypes.shape
                )
                P_dig = []
                for d in range(self._dim):
                    P_dig.append(
                        np.digitize(self._mapping[:, d], self._pos_lith[d])
                    )
                # once Py3.11 has reached its EOL, we can drop the 1-tuple :-)
                return self._lithotypes[(*P_dig,)]

            if tree is not None:
                self._tree = self.DecisionTree(tree)
                self._tree = self._tree.build_tree()

                coords_P = np.stack(
                    [
                        self._fields[d].ravel()
                        for d in range(len(self._fields))
                    ],
                    axis=1,
                )
                labels_P = np.array(
                    [
                        self._tree.decide(dict(zip(self._field_names, pt)))
                        for pt in coords_P
                    ]
                )
                return labels_P.reshape(self._fields.shape[1:])

        raise ValueError(
            "PGS: Must provide exactly one of `lithotypes` or `tree`"
        )

    def compute_lithotype(self, tree=None):
        """
        Compute lithotype from input SRFs using a decision tree.

        If `self._tree` is not set, a tree configuration must be provided via
        the `tree` argument. The method then builds or reuses the decision tree
        and applies it to the coordinates of the plurigaussian fields to assign
        a lithotype phase at each point.

        Parameters
        ----------
        tree : dict or None, optional
            Configuration for the decision tree. If None, `self._tree` must
            already be defined. Defaults to None.

        Returns
        -------
        lithotype : :class:`numpy.ndarray`
            Discrete label array of shape equal to the simulation domain,
            where each entry is the phase index determined by the tree.

        Raises
        ------
        ValueError
            If no decision tree is available or if `self._dim` does not equal
            the number of provided fields.
        """
        if self._tree is None and tree is None:
            raise ValueError(
                "PGS: Please provide a decision tree config or compute P to assemble"
            )
        if self._tree is None and tree is not None:
            self._tree = self.DecisionTree(tree)
            self._tree = self._tree.build_tree()

        if self._dim == len(self._fields):
            axes = [
                np.linspace(-3, 3, self._fields[0].shape[0])
                for _ in self._fields.shape[1:]
            ]  # works 2D 2 Fields
            mesh = np.meshgrid(*axes, indexing="ij")
            coords_L = np.stack([m.ravel() for m in mesh], axis=1)
            labels_L = np.array(
                [
                    self._tree.decide(dict(zip(self._field_names, pt)))
                    for pt in coords_L
                ]
            )
            L_shape = tuple(
                [self._fields.shape[1]] * len(self._fields.shape[1:])
            )
            L = labels_L.reshape(L_shape)
        else:
            raise ValueError("PGS: Only implemented for dim == len(fields)")

        return L

    def calc_lithotype_axes(self, lithotypes_shape):
        """Calculate the axes on which the lithorypes are defined.

        With the centroid of the correlations of the SRFs at the center,
        the axes are calculated, which hold all correlations.
        These axes are used for the lithotypes field.

        Parameters
        ----------
        lithotypes_shape : :class:`tuple`
            The shape of the lithotypes field.

        Returns
        -------
        pos_lith : :class:`numpy.ndarray`
            the axes holding all field correlations
        """
        pos_lith = []
        try:
            # structured grid
            centroid = self._fields.mean(axis=tuple(range(1, self._dim + 1)))
        except np.AxisError:
            # unstructured grid
            centroid = self._fields.mean(axis=1)
        for d in range(self._dim):
            l = np.floor(self._fields[d].min()) - 1
            h = np.ceil(self._fields[d].max()) + 1
            m = (h + l) / 2.0
            dist = max(np.abs(h - m), np.abs(l - m))
            pos_lith.append(
                np.linspace(
                    centroid[d] - dist,
                    centroid[d] + dist,
                    lithotypes_shape[d],
                )
            )
        return pos_lith

    def transform_coords(self, lithotypes_shape, pos):
        """Transform position from correlation coords to L indices.

        This is a helper method to get the lithoty pes indices for given
        correlated field values.

        Parameters
        ----------
        lithotypes_shape : :class:`tuple`
            The shape of the lithotypes field.
        pos : :class:`numpy.ndarray`
            The position in field coordinates, which will be transformed.

        Returns
        -------
        pos_trans : :class:`list`
            the transformed position tuple
        """
        pos_trans = []
        pos_lith = self.calc_lithotype_axes(lithotypes_shape)
        for d in range(self._dim):
            pos_trans.append(
                int(
                    (pos[d] - pos_lith[d][0])
                    / (pos_lith[d][-1] - pos_lith[d][0])
                    * lithotypes_shape[d]
                )
            )
        return pos_trans

    class DecisionTree:
        """
        Build and traverse a decision tree for assigning lithotype labels.

        This class constructs a tree of DecisionNode and LeafNode instances
        from a configuration mapping. Once built, it can classify input data
        by following the decision branches to a leaf action.

        Parameters
        ----------
        config : dict
            Mapping of node IDs to node specifications. Each entry must include:
            - 'type': 'decision' or 'leaf'
            - For decision nodes:
            • 'func' (callable) and 'args' (dict)
            • Optional 'yes_branch' and 'no_branch' keys naming other nodes
            - For leaf nodes:
        Notes
        -----
        - Call `build_tree()` to link nodes and obtain the root before using
        `decide()`.
        - The tree is immutable once built; rebuild to apply a new config.
        """

        def __init__(self, config):
            self._config = config
            self._tree = None

        def build_tree(self):
            """
            Construct the decision tree structure from the configuration.

            Iterates through the config to create DecisionNode and LeafNode
            instances, then links decision nodes to their yes/no branches.

            Returns
            -------
            root : DecisionNode or LeafNode
                The root node of the constructed decision tree.

            Raises
            ------
            KeyError
                If 'root' is not defined in the configuration.
            """
            nodes = {}
            for node_id, details in self._config.items():
                if details["type"] == "decision":
                    nodes[node_id] = self.DecisionNode(
                        func=details["func"], args=details["args"]
                    )
                elif details["type"] == "leaf":
                    nodes[node_id] = self.LeafNode(details["action"])
            for node_id, details in self._config.items():
                if details["type"] == "decision":
                    nodes[node_id].yes_branch = nodes.get(
                        details.get("yes_branch")
                    )
                    nodes[node_id].no_branch = nodes.get(
                        details.get("no_branch")
                    )

            return nodes["root"]

        def decide(self, data):
            """
            Traverse the built tree to make a decision for the given data.

            Parameters
            ----------
            data : dict
                A mapping of feature names to values, passed to decision functions
                in each DecisionNode.

            Returns
            -------
            result
                The action value from the reached LeafNode, or None if a branch
                is missing.

            Raises
            ------
            ValueError
                If the tree has not been built (i.e., `build_tree` not called).
            """
            if self._tree:
                return self._tree.decide(data)
            raise ValueError("The decision tree has not been built yet.")

        class DecisionNode:  # pylint: disable=too-few-public-methods
            """
            Internal node that evaluates a condition and routes to child branches.

            A DecisionNode wraps a boolean function and two optional branches
            (yes_branch and no_branch), which may be further DecisionNode or
            LeafNode instances.

            Parameters
            ----------
            func : callable
                A function that evaluates a condition on the input data.
                Must accept `data` as first argument and keyword args.
            args : dict
                Keyword arguments to pass to `func` when called.
            yes_branch : DecisionNode or LeafNode, optional
                Node to traverse if `func(data, **args)` returns True.
            no_branch : DecisionNode or LeafNode, optional
                Node to traverse if `func(data, **args)` returns False.
            """

            def __init__(self, func, args, yes_branch=None, no_branch=None):
                self.func = func
                self.args = args
                self.yes_branch = yes_branch
                self.no_branch = no_branch

            def decide(self, data):
                """
                Evaluate the decision function and traverse to the next node.

                Parameters
                ----------
                data : dict
                    Feature mapping passed to `func`.

                Returns
                -------
                result
                    The outcome of the subsequent node's `decide` method, or None
                    if the respective branch is not set.
                """
                if self.func(data, **self.args):
                    return (
                        self.yes_branch.decide(data)
                        if self.yes_branch
                        else None
                    )
                return self.no_branch.decide(data) if self.no_branch else None

        class LeafNode:  # pylint: disable=too-few-public-methods
            """
            Terminal node that returns a stored action when reached.

            A LeafNode represents the outcome of a decision path. Its `action`
            value is returned directly by `decide()`

            Parameters
            ----------
            action : any
                The value to return when this leaf is reached.
            """

            def __init__(self, action):
                self.action = action

            def decide(self, _data):
                """
                Return the leaf action, terminating the traversal.

                Parameters
                ----------
                data : dict
                    Ignored input data.

                Returns
                -------
                action : any
                    The action value stored in this leaf.
                """
                return self.action
