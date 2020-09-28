import numpy as np


def morlet_gabor_wavelet(
    frequency_array,
    amplitude,
    q_factor,
    centre_frequency,
    ellipticity,
    delta_time,
    phase,
    **waveform_kwargs
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
    **waveform_kwargs
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
    minimum_frequency = waveform_kwargs.get("minimum_frequency", 0)
    maximum_frequency = waveform_kwargs.get(
        "minimum_frequency", frequency_array[-1]
    )

    tau = q_factor / (2 * np.pi * centre_frequency)
    delta = np.arctan(np.pi * beta) / 2
    pi_beta = np.pi * beta
    # minimum_frequency = max(minimum_frequency, centre_frequency - 6 / tau)
    # maximum_frequency = min(maximum_frequency, centre_frequency + 6 / tau)
    band = (
        (frequency_array >= minimum_frequency)
        & (frequency_array <= maximum_frequency)
    )

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    frequency_array = frequency_array[band]
    delta_f = frequency_array - centre_frequency
    h_plus[band] = (
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
            * np.exp(-(q_factor ** 2) * frequency_array / centre_frequency)
        )
    )
    h_cross = -1j * ellipticity * h_plus

    return {"plus": h_plus, "cross": h_cross}

def complex_ellipticity_chirplet(
    frequency_array,
    amplitude,
    q_factor,
    centre_frequency,
    ellipticity_mag,
    ellipticity_phase,
    delta_time,
    phase,
    beta,
    **waveform_kwargs
):
    minimum_frequency = waveform_kwargs.get("minimum_frequency", 0)
    maximum_frequency = waveform_kwargs.get(
        "minimum_frequency", frequency_array[-1]
    )

    tau = q_factor / (2 * np.pi * centre_frequency)
    delta = np.arctan(np.pi * beta) / 2
    pi_beta = np.pi * beta
    # minimum_frequency = max(minimum_frequency, centre_frequency - 6 / tau)
    # maximum_frequency = min(maximum_frequency, centre_frequency + 6 / tau)
    band = (
        (frequency_array >= minimum_frequency)
        & (frequency_array <= maximum_frequency)
    )

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    frequency_array = frequency_array[band]
    delta_f = frequency_array - centre_frequency
    h_plus[band] = (
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
            * np.exp(-(q_factor ** 2) * frequency_array / centre_frequency)
        )
    )
    h_cross = ellipticity_mag * np.exp(1j * ellipticity_phase) * h_plus

    return {"plus": h_plus, "cross": h_cross}

def complex_ellipticity_amp_freq_chirplet(
    frequency_array,
    amplitude,
    q_factor,
    centre_frequency,
    ellipticity_grad,
    ellipticity_ref,
    ellipticity_phase,
    delta_time,
    phase,
    beta,
    **waveform_kwargs
):
    minimum_frequency = waveform_kwargs.get("minimum_frequency", 0)
    maximum_frequency = waveform_kwargs.get(
        "minimum_frequency", frequency_array[-1]
    )

    tau = q_factor / (2 * np.pi * centre_frequency)
    delta = np.arctan(np.pi * beta) / 2
    pi_beta = np.pi * beta
    # minimum_frequency = max(minimum_frequency, centre_frequency - 6 / tau)
    # maximum_frequency = min(maximum_frequency, centre_frequency + 6 / tau)
    band = (
        (frequency_array >= minimum_frequency)
        & (frequency_array <= maximum_frequency)
    )

    h_plus = np.zeros_like(frequency_array, dtype=complex)
    frequency_array = frequency_array[band]
    delta_f = frequency_array - centre_frequency
    h_plus[band] = (
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
            * np.exp(-(q_factor ** 2) * frequency_array / centre_frequency)
        )
    )
    amp = ellipticity_ref + ellipticity_grad * np.log10(frequency_array)
    h_cross = amp * np.exp(1j * ellipticity_phase) * h_plus

    return {"plus": h_plus, "cross": h_cross}
