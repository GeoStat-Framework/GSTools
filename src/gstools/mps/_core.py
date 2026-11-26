"""Central gstools_core import + MPS capability check.

MPS Direct Sampling is Rust-only: it requires ``gstools_core`` (>= 1.4.0, the
release that first shipped the MPS kernels). This module hard-imports the core,
validates its version *and* that every required MPS export is present, then
re-exports the kernels. A single, explicit error is raised for both failure
modes — core missing, or an older core without the MPS exports.

The package declares Core as a mandatory dependency. The MPS interface gate
additionally rejects stale or incompatible local binaries even when their
package metadata reports a suitable version.
"""

from packaging.version import InvalidVersion, Version

_REQUIRED_EXPORTS = (
    "mps_simulate",
    "mps_dist_block_cat",
    "mps_dist_block_cat_masked",
    "mps_dist_block_cat_rayon",
    "mps_dist_block_l1",
    "mps_dist_block_l1_masked",
    "mps_dist_block_l2",
    "mps_dist_block_l2_masked",
    "mps_dist_block_lp",
    "mps_dist_block_lp_masked",
    "mps_dist_block_variation",
    "mps_dist_block_variation_masked",
    "mps_scan_node",
    "mps_scan_node_cat",
    "mps_cleanup_pass",
    "mps_api_version",
)
_MIN_VERSION = Version("1.4.0")
# MPS native ABI revision this Python layer requires. The crate version can
# stay 1.4.0 across an ABI change, so version alone is insufficient; the Core
# exports ``mps_api_version()`` and we gate on it. Bump in lockstep with the
# Rust ``mps_api_version_py`` whenever an MPS entry-point signature changes.
_REQUIRED_MPS_API = 8
_MSG = "GSTools MPS requires gstools-core>=1.4.0 with MPS support"


def _supported_version(value):
    """Require a PEP-440 release at least 1.4.0 (including post/local builds)."""
    try:
        return Version(str(value)) >= _MIN_VERSION
    except InvalidVersion:
        return False


try:
    import gstools_core as _gstools_core
except ImportError as exc:  # core not installed at all
    raise ImportError(f"{_MSG}; gstools_core is not installed.") from exc

_missing = [
    name for name in _REQUIRED_EXPORTS if not hasattr(_gstools_core, name)
]
_version = getattr(_gstools_core, "__version__", "0")
if _missing or not _supported_version(_version):
    _detail = (
        f" (found gstools_core {_version}, missing MPS exports: "
        f"{', '.join(_missing)})"
        if _missing
        else f" (found gstools_core {_version})"
    )
    raise ImportError(f"{_MSG}{_detail}.")

# Interface-revision gate (later revisions are not assumed callable): a same-versioned Core built before the current MPS
# entry-point signatures must be rejected with an actionable message rather
# than failing later with an opaque argument/type error.
try:
    _found_api = _gstools_core.mps_api_version()
except (TypeError, ValueError, OverflowError) as exc:
    raise ImportError(f"{_MSG}: invalid MPS interface revision") from exc
if type(_found_api) is not int or _found_api != _REQUIRED_MPS_API:
    raise ImportError(
        f"{_MSG}: this GSTools needs MPS native ABI revision "
        f"exactly {_REQUIRED_MPS_API}, but the installed gstools_core {_version} "
        f"reports revision {_found_api}. Rebuild gstools-core from the matching "
        "source (maturin build --release) and reinstall it."
    )

# Re-export the validated kernels so the MPS modules import them from here.
mps_simulate = _gstools_core.mps_simulate
mps_dist_block_cat = _gstools_core.mps_dist_block_cat
mps_dist_block_cat_masked = _gstools_core.mps_dist_block_cat_masked
mps_dist_block_cat_rayon = _gstools_core.mps_dist_block_cat_rayon
mps_dist_block_l1 = _gstools_core.mps_dist_block_l1
mps_dist_block_l1_masked = _gstools_core.mps_dist_block_l1_masked
mps_dist_block_l2 = _gstools_core.mps_dist_block_l2
mps_dist_block_l2_masked = _gstools_core.mps_dist_block_l2_masked
mps_dist_block_lp = _gstools_core.mps_dist_block_lp
mps_dist_block_lp_masked = _gstools_core.mps_dist_block_lp_masked
mps_dist_block_variation = _gstools_core.mps_dist_block_variation
mps_dist_block_variation_masked = _gstools_core.mps_dist_block_variation_masked
mps_scan_node = _gstools_core.mps_scan_node
mps_scan_node_cat = _gstools_core.mps_scan_node_cat
mps_cleanup_pass = _gstools_core.mps_cleanup_pass
mps_api_version = _gstools_core.mps_api_version
