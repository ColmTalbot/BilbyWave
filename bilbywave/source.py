import numpy as np


def morlet_gabor_wavelet(
    frequency_array,
    amplitude,
    q_factor,
    centre_frequency,
    ellipticity,
    delta_time,
    phase,
):
    """
    See equation (5) of https://arxiv.org/pdf/1410.3835.pdf

    Parameters
    ----------
    frequency_array: array-like
    amplitude: float
    q_factor: float
    centre_frequency: float
    ellipticity: float
    delta_time: float
    phase: float

    Returns
    -------
    dict: dictionary of plus and cross waveforms
    """
    tau = q_factor / (2 * np.pi * centre_frequency)
    delta_f_plus = frequency_array + centre_frequency
    delta_f_minus = frequency_array - centre_frequency

    h_plus = (
        amplitude
        * np.pi ** 0.5
        * tau
        / 2
        * np.exp(-np.pi ** 2 * tau ** 2 * delta_f_minus ** 2)
        * (
            np.exp(1j * (phase + 2 * np.pi * delta_f_minus * delta_time))
            + np.exp(
                -1j * (phase + 2 * np.pi * delta_f_plus * delta_time)
                - q_factor ** 2 * frequency_array / centre_frequency
            )
        )
    )
    h_cross = -1j * ellipticity * h_plus

    return {"plus": h_plus, "cross": h_cross}


def chirplet(
    frequency_array,
    amplitude,
    q_factor,
    centre_frequency,
    ellipticity,
    delta_time,
    phase,
    beta,
):
    """
    See equation (2) of https://arxiv.org/pdf/1804.03239.pdf

    Parameters
    ----------
    frequency_array: array-like
    amplitude: float
    q_factor: float
    centre_frequency: float
    ellipticity: float
    delta_time: float
    phase: float
    beta: float

    Returns
    -------
    dict: dictionary of plus and cross waveforms
    """
    tau = q_factor / (2 * np.pi * centre_frequency)
    delta = np.arctan(np.pi * beta) / 2
    delta_f = frequency_array - centre_frequency
    pi_beta = np.pi * beta

    h_plus = (
        amplitude
        * np.pi ** 0.5
        * tau
        / 2
        / (1 + pi_beta ** 2) ** 0.25
        * np.exp(
            -np.pi ** 2 * tau ** 2 * delta_f ** 2 / (1 + pi_beta ** 2)
            - 2j * np.pi * frequency_array * delta_time
        )
        * (
            np.exp(
                1j
                * (phase + delta - np.pi ** 3 * beta * tau ** 2 * delta_f ** 2)
                / (1 + pi_beta ** 2)
            )
            + np.exp(
                -1j
                * (phase + delta - np.pi ** 3 * beta ** 2 * delta_f ** 2)
                / (1 + pi_beta ** 2)
            )
            * np.exp(-q_factor ** 2 * frequency_array / centre_frequency)
        )
    )
    h_cross = -1j * ellipticity * h_plus

    return {"plus": h_plus, "cross": h_cross}
