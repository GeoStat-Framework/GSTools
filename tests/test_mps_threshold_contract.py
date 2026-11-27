"""Threshold aliases preserve streams and fields, including nonzero matches."""

import numpy as np
import pytest

from gstools.mps import DirectSampling, MPSModel, TrainingImage


@pytest.mark.parametrize("scan", ["sequential", "random"])
@pytest.mark.parametrize("cleanup", [0, 1])
def test_threshold_alias_fields_and_remaining_streams(scan, cleanup):
    results, streams = [], []
    ti = TrainingImage(np.indices((10, 11)).sum(axis=0) % 3, n_neighbors=3)
    for settings in (
        {},
        {"threshold": 0},
        {"threshold": -0.0},
        {"threshold": None},
    ):
        ds = DirectSampling(
            MPSModel(
                ti,
                scan_path=scan,
                scan_fraction=0.2,
                post_processing=cleanup,
                **settings,
            ),
            seed=7,
        )
        # Hard value outside TI makes nonzero mismatches possible.
        ds.set_condition([[1], [1]], [4.0])
        results.append(
            ds(
                [np.arange(24)] * 2,
                num_threads=4,
                post_process=False,
                store=False,
            )
        )
        streams.append(ds.rng.random.uniform(size=32))
    for result, stream in zip(results[1:], streams[1:]):
        np.testing.assert_array_equal(result, results[0])
        np.testing.assert_array_equal(stream, streams[0])
    assert results[0][1, 1] == 4.0
