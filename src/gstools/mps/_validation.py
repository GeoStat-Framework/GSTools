"""Small shared validators for DS counts and float64 representation."""

import numbers

import numpy as np


def exact_count(value, *, minimum, context):
    """Accept integers, bools and integral real floats without truncation.

    Integer types never pass through float64. Integral floats and bools remain
    supported for compatibility; strings and non-integral/nonfinite reals do not.
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


def _validate_real_data(values, categorical, context):
    """Require real numeric arrays with values representable in float64.

    Python integers make the comparison safe at signed/unsigned endpoints.
    Exactly representable large powers of two remain supported. This checks
    representation, not whether a value is a valid penalty matrix index.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(
            f"{context}: expected real numeric data, got dtype {array.dtype}."
        )
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        with np.errstate(over="ignore", under="ignore"):
            finite = array[np.isfinite(array)]
            if np.any(finite != finite.astype(np.float64).astype(array.dtype)):
                raise ValueError(
                    f"{context}: values must be exactly representable in float64."
                )
    if array.dtype.kind not in "iu" or not array.size:
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
