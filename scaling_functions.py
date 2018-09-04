import scipy

# ==================================
# === Error function definitions ===
# ==================================


def f_gradient_energy(en, en_min, en_max_north, en_max_south, hem):
    """
    Energy-dependent gradient error-function for both North and South IRFs.

    Parameters
    ----------
    en: ndarray
        Energy.
    en_min: float
        Minimum energy where the analysis is performed.
    en_max_north: float
        Maximum energy where the analysis is performed (for North site).
    en_max_south: float
        Maximum energy where the analysis is performed (for South site).
    hem: string
        Hemisphere to which the IRFs are referred.
    """

    # TODO: add the return value and params units to the docstring

    if hem == "North":
        res = (scipy.log10(en / en_min) + scipy.log10(en / en_max_north)) / scipy.log10(en_max_north / en_min)
    else:
        res = (scipy.log10(en / en_min) + scipy.log10(en / en_max_south)) / scipy.log10(en_max_south / en_min)
    return res


def f_gradient_arr_dir(theta, theta_max_north, theta_max_south, hem):
    """
    Arrival-direction-dependent gradient error-function for both North and South IRFs.

    Parameters
    ----------
    theta: vector
        Arrival direction.
    theta_max_north: float
        Maximum angle (for North site).
    theta_max_south: float
        Maximum angle (for South site).
    hem: string
        Hemisphere to which the IRFs are referred.
    """

    # TODO: add the return value and params units to the docstring

    if hem == "North":
        res = (2 * theta - theta_max_north) / theta_max_north
    else:
        res = (2 * theta - theta_max_south) / theta_max_south
    return res


def f_step_energy(log_en, log_en1, res1, log_en2, res2, hem):
    """
    Energy-dependent step error-function for both North and South IRFs.

    Parameters
    ----------
    log_en: vector
        log of the energy
    log_en1: float
        log of the energy at the first transition point.
    res1: float
        Energy resolution at the first transition point.
    log_en2: float
        log of the energy at the second transition point.
    res2: float
        Energy resolution at the second transition point.
    hem: string
        Hemisphere to which the IRFs are referred.
    step_trans_width: float
        Transition width.
    """

    # TODO: add the return value and params units to the docstring

    step_trans_width = 1.31

    res = scipy.zeros_like(log_en)

    if hem == "North":
        res = scipy.tanh((log_en - log_en1) / (step_trans_width * res1))
    elif hem == "South":
        trans_log_energy = (log_en1 + log_en2) / 2.0
        above_trans_en = scipy.where(log_en >= trans_log_energy)
        below_trans_en = scipy.where(log_en < trans_log_energy)

        res[below_trans_en] = scipy.tanh((log_en[below_trans_en] - log_en1) / (step_trans_width * res1))
        res[above_trans_en] = -1 * scipy.tanh((log_en[above_trans_en] - log_en2) / (step_trans_width * res2))

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
