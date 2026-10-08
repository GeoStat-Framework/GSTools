"""
Run an EDK check case with the multi-field prototype and count solves.

Uses the EDK readers and the day-by-day GSTools run from
``edk_gstools_comparison`` (sibling of the GSTools repo):

* day-by-day: one ``LocalExtDrift`` call per day (Rust), EDK special cases
  for < 3 stations / all-zero days in numpy
* multi: one ``local_krige_multi`` call for all days (this prototype),
  same special cases taken from the day-by-day run

Both are compared with each other and with the EDK reference output.

Run with:
    uv run python dev/edk_case_multi_check.py ../../EDK/check/case_01
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import gstools as gs
from local_kriging_multi_numpy import CACHE_MODES, local_krige_multi

DEFAULT_COMPARISON = Path(__file__).resolve().parents[2] / "edk_gstools_comparison"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="EDK case directory (contains edk.nml)")
    parser.add_argument("--comparison-dir", default=str(DEFAULT_COMPARISON),
                        help="path to edk_gstools_comparison (default: %(default)s)")
    parser.add_argument("--ref", default="output_save/pre.nc")
    parser.add_argument("--vario", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.comparison_dir)
    import edk_io
    from gstools_edk import BRANCH_KRIGE, build_model, interpolate
    from run_comparison import compare

    case = edk_io.read_case(args.case, vario_path=args.vario)
    opt = case.options
    print(f"case: {case.path}")

    # --- day by day (current approach) ---------------------------------
    t0 = time.perf_counter()
    field_day, branch, timing = interpolate(case)
    t_day = time.perf_counter() - t0
    krige_mask = branch == BRANCH_KRIGE
    print(f"\nday-by-day : {timing['kriging_systems']} systems solved, "
          f"{timing['gstools_krige']:.3f} s in GSTools (Rust), {t_day:.3f} s total")

    # --- multi prototype -----------------------------------------------
    model = build_model(case.variogram)
    valid = np.isfinite(case.cell_h)
    targets = (case.cell_x[valid], case.cell_y[valid])
    sta_pos = (case.sta_x, case.sta_y)
    zeros = np.zeros(len(case.sta_id))
    if opt["inter_mth"] == 2:
        template = gs.krige.LocalExtDrift(model, sta_pos, zeros,
                                          local_radius=opt["max_dist"],
                                          ext_drift=case.sta_h)
        drift_kw = {"ext_drift": case.cell_h[valid]}
    else:
        template = gs.krige.LocalOrdinary(model, sta_pos, zeros,
                                          local_radius=opt["max_dist"])
        drift_kw = {}

    results = {}
    print(f"\n{'cache':<6} {'solves':>8} {'hits last':>10} {'hits hash':>10} "
          f"{'skipped':>8} {'masks/target':>13} {'time [s]':>9}")
    for cache in CACHE_MODES:
        t0 = time.perf_counter()
        field_m, info = local_krige_multi(
            template, targets, case.sta_val, return_var=False,
            min_neighbors=3, cache=cache, return_info=True, **drift_kw,
        )
        dt = time.perf_counter() - t0
        results[cache] = field_m
        print(f"{cache:<6} {info['n_solves']:>8} {info['n_hits_last']:>10} "
              f"{info['n_hits_hash']:>10} {info['n_skipped']:>8} "
              f"{info['max_masks_per_target']:>13} {dt:>9.3f}")
    print("(multi times are pure NumPy, the day-by-day kriging runs in Rust)")

    # cache modes must not change the result
    for cache in ("last", "hash"):
        np.testing.assert_array_equal(results[cache], results["none"])

    # --- combine with EDK special cases and compare ----------------------
    field_multi = np.full_like(field_day, np.nan)
    field_multi[:, valid] = results["hash"]
    if opt["correct_neg"]:
        field_multi[field_multi < 0] = 0.0
    diff = np.abs(field_multi - field_day)[krige_mask]
    print(f"\nmulti vs day-by-day on kriged values: max |diff| = {diff.max():.3e} "
          f"({krige_mask.sum()} values)")
    combined = np.where(krige_mask, field_multi, field_day)

    ref_path = case.path / args.ref
    if not ref_path.exists():
        print(f"no reference at {ref_path}")
        return
    out = Path(__file__).resolve().parent / "output" / f"{case.path.name}_multi.nc"
    edk_io.write_netcdf(out, case, combined)
    print(f"\nmulti vs EDK reference {ref_path}:")
    ok = compare(edk_io.read_output(out, opt["variable_name"]),
                 edk_io.read_output(ref_path, opt["variable_name"]),
                 branch)
    print(f"\nRESULT: {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
