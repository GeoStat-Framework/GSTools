"""Small shared validators for DS counts and float64 representation."""

import numbers

import numpy as np


def exact_count(value, *, minimum, context):
    """Return ``value`` as an int (bools count as 0/1).

    Raises ``ValueError`` for fractional, non-finite or text input and for
    ``value < minimum``.
    """
    if isinstance(value, (numbers.Integral, np.bool_)):
        result = int(value)
    elif isinstance(value, numbers.Real) and np.isfinite(value):
        result = int(value)
        if value != result:
            raise ValueError(f"{context} must be an integer, got {value!r}")
    else:
        raise ValueError(f"{context} must be an integer, got {value!r}")
    if result < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value!r}")
    return result


def check_float64_integers(values, categorical, context):
    """Raise ``ValueError`` for integer-dtype values that change under float64 conversion."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or not array.size:
        return
    if int(array.min()) >= -(2**53) and int(array.max()) <= 2**53:
        return
    for value in np.unique(array):
        integer = int(value)
        if int(float(integer)) != integer:
            remedy = (
                "recode categorical labels to small integer codes"
                if categorical
                else "center/rescale continuous values before float64 conversion"
            )
            raise ValueError(
                f"{context}: integer value {integer} is not exactly representable "
                f"in the float64 DS engine; {remedy}."
            )
