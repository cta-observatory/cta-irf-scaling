import scipy

# ==================================
# === Error function definitions ===
# ==================================


def gradient(x, x_min, x_max):
    """
    Gradient scaling function. The gradient is computed
    to result in +/-1 scales at x_max and x_min correspondingly.

    Parameters
    ----------
    x: ndarray
        An input array, for which to compute the scalings.
    x_min: float
        A point, that corresponds to -1 output value.
    x_max: float
        A point, that corresponds to +1 output value.

    Returns
    -------
    ndarray:
        An array of scales, ranging [-1;1] in [x_min; x_max] range.
    """

    res = (2*x - (x_min + x_max)) / (x_max - x_min)

    return res


def step(x, break_points):
    """
    Step-like scaling function.

    Parameters
    ----------
    x: ndarray
        An input array, for which to compute the scalings.
    break_points: tuple
        A list of the break points. Each entry should be a tuple of
        (break_position, break_width).

    Returns
    -------
    ndarray
        Array of computed scales in the [-1; 1] range.
    """

    # Applying the first break point
    break_point = break_points[0]
    break_x = break_point[0]
    break_width = break_point[1]

    res = scipy.tanh((x - break_x) / break_width)
    sign = 1

    # If there are more break points given, applying them as well
    for break_point in break_points[1:]:
        # First recalling the previous break point position
        break_x_old = break_x

        # New break point data
        break_x = break_point[0]
        break_width = break_point[1]

        # Will fill only points above the transition position
        trans_x = (break_x + break_x_old) / 2.0
        above_trans_x = scipy.where(x >= trans_x)

        # Flip the sign - above the transition position function behaviour is reversed
        sign *= -1
        res[above_trans_x] = sign * scipy.tanh((x[above_trans_x] - break_x) / break_width)

    return res

