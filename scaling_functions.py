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


def f_step_energy(log_en, break_points, step_trans_width=1.31):
    """
    Energy-dependent step error-function for both North and South IRFs.

    Parameters
    ----------
    log_en: ndarray
        Array with log energies, at which to evaluate the function.
    break_points: tuple
        A list of the break points. Each entry should be a tuple of
        (break_position, break_width).
    step_trans_width: float
        Transition width.

    Returns
    -------
    ndarray
        Array of scaling coefficients in the [-1; 1] range.
    """

    # Applying the first break point
    break_point = break_points[0]
    break_log_en = break_point[0]
    break_width = break_point[1]

    res = scipy.tanh((log_en - break_log_en) / (step_trans_width * break_width))
    sign = 1

    # If there are more break points given, applying them as well
    for break_point in break_points[1:]:
        # First recalling the previous break point position
        break_log_en_old = break_log_en

        # New break point data
        break_log_en = break_point[0]
        break_width = break_point[1]

        # Will fill only points above the transition position
        trans_log_energy = (break_log_en + break_log_en_old) / 2.0
        above_trans_en = scipy.where(log_en >= trans_log_energy)

        # Flip the sign - above the transition position function behaviour is reversed
        sign *= -1
        res[above_trans_en] = sign * scipy.tanh((log_en[above_trans_en] - break_log_en) / (step_trans_width * break_width))

    return res


def f_step_arr_dir(theta, theta1, sigma_theta1, theta2, sigma_theta2, hem):
    """
    Arrival-direction-dependent step error-function for both North and South IRFs.
    Parameters ----------
        theta: vector
            Arrival direction.
        theta1: float
            First transition point.
        sigma_theta1: float
            Angular resolution at the first transition point.
        theta2: float
            Second transition point.
        sigma_theta2: float
            Angular resolution at the second transition point.
        hem: string
            Hemisphere to which the IRFs are referred.
        step_trans_width: float
            Transition width.
    """

    # TODO: add the return value and params units to the docstring

    step_trans_width = 1.31
    if hem == "North" or (hem == "South" and theta < (theta1 + theta2) / 2):
        res = scipy.tanh((theta - theta1) / (step_trans_width * sigma_theta1))
    else:
        res = scipy.tanh((theta - theta2) / (step_trans_width * sigma_theta2))
    return res
