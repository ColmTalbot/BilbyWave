#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is WMAP7.
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import bilby
import bilbywave


# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.
sampling_frequency = 2048.

n_wavelets = 1

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'single_chirplet_{}'.format(n_wavelets)
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=100., mass_2=95., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=3000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator_1 = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - 3)
ifos.inject_signal(waveform_generator=waveform_generator_1,
                   parameters=injection_parameters)

test_like = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator_1)
test_like.parameters.update(injection_parameters.copy())


# Set up prior, which is a dictionary
priors = bilby.core.prior.ConditionalPriorDict()
# priors['luminosity_distance'] = bilby.core.prior.LogUniform(
#     minimum=1, maximum=10000, name='luminosity_distance')
priors['luminosity_distance'] = bilby.core.prior.DeltaFunction(1)
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1, latex_label='$t_c$')
priors['ra'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, latex_label='$\\alpha$', boundary='periodic')
priors['dec'] = bilby.core.prior.Cosine(
    latex_label='$\\delta$', boundary='reflective')
priors['psi'] = bilby.core.prior.Uniform(
    0, np.pi / 2, latex_label='$\\psi$', boundary='periodic')
priors['phase'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, latex_label='$\\phi$', boundary='periodic')
priors['ellipticity'] = bilby.core.prior.Uniform(
    -1, 1, latex_label='$\\epsilon$', boundary='reflective')

priors['n_wavelets'] = bilby.core.prior.DeltaFunction(n_wavelets)

for ii in range(n_wavelets):
    priors['q_factor_{}'.format(ii)] = bilby.core.prior.Uniform(
        0.01, 40, latex_label='$Q_{}$'.format(ii), boundary='reflective')
    priors['centre_frequency_{}'.format(ii)] = bilby.core.prior.Uniform(
        20, 100, latex_label='$f_{}$'.format(ii), unit='Hz')
    if ii == 0:
        priors['amplitude_{}'.format(ii)] = bilby.core.prior.LogUniform(
            minimum=1e-25, maximum=1e-21)
    else:
        priors['amplitude_{}'.format(ii)] = bilbywave.prior.OrderedAmplitude(
            minimum=1e-25, maximum=1e-21, latex_label='$A$',
            order=ii, base_name="amplitude"
        )
    priors['delta_time_{}'.format(ii)] = bilby.core.prior.Uniform(
        -1, 1, latex_label='$\\delta t_{}$'.format(ii))
    priors['phase_{}'.format(ii)] = bilby.core.prior.Uniform(
        0, 2 * np.pi, latex_label='$\\phi_{}$'.format(ii), boundary='periodic')
    priors['beta_{}'.format(ii)] = bilby.core.prior.Uniform(
        - (1 - 1 / np.pi**2)**0.5, (1 - 1 / np.pi**2)**0.5,
        latex_label='$\\beta_{}$'.format(ii), boundary='reflective')
priors['delta_time_0'] = 0.0
priors['phase_0'] = 0.0

source_model = bilbywave.source.chirplet

wfg = bilbywave.waveform_generator.MultiWavelet(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=source_model,
    start_time=ifos.start_time)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wfg, priors=priors,
    time_marginalization=False, phase_marginalization=True,
    distance_marginalization=False, jitter_time=False)

from time import time
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=500,
    outdir=outdir, label=label, walks=50, n_check_point=2000, poolsize=500,
    seed=int(time()))

non_marg_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=wfg,
    time_marginalization=False, phase_marginalization=False,
    distance_marginalization=False, jitter_time=False)
for key in priors:
    if isinstance(priors[key], bilby.core.prior.DeltaFunction):
        non_marg_likelihood.parameters[key] = priors[key].peak


non_marg_likelihood.parameters.update(dict(result.posterior.iloc[0]))

new_values = {
    key: list() for key in
    likelihood.marginalized_parameters + ['log_likelihood']}
for ii in tqdm.tqdm(range(len(result.posterior))):
    likelihood.parameters.update(dict(result.posterior.iloc[ii]))
    likelihood.parameters["phase"] = 0.0
    non_marg_likelihood.parameters.update(dict(result.posterior.iloc[ii]))
    new_params = likelihood.generate_posterior_sample_from_marginalized_likelihood()
    for key in likelihood.marginalized_parameters:
        new_values[key].append(new_params[key])
        non_marg_likelihood.parameters[key] = new_params[key]
    new_values["log_likelihood"].append(non_marg_likelihood.log_likelihood_ratio())

for key in priors:
    if key not in result.posterior and isinstance(likelihood.priors[key], bilby.core.prior.DeltaFunction):
        result.posterior[key] = priors[key].peak
for key in new_values:
    result.posterior[key] = new_values[key]

n_samples = min(len(result.posterior), 2000)
red_post = result.posterior.sample(n_samples)
fd_waveforms = {ifo.name: list() for ifo in ifos}
td_waveforms = {ifo.name: list() for ifo in ifos}
overlaps = {ifo.name: list() for ifo in ifos}
hf_inj = waveform_generator_1.frequency_domain_strain(injection_parameters)
for ii in tqdm.tqdm(range(n_samples)):
    parameters = dict(red_post.iloc[ii])
    wf_pols = wfg.frequency_domain_strain(parameters)
    for ifo in ifos:
        hf_inj_det = ifo.get_detector_response(hf_inj, injection_parameters)
        hf_det = ifo.get_detector_response(wf_pols, parameters)
        fd_waveforms[ifo.name].append(hf_det)
        td_waveforms[ifo.name].append(
            bilby.core.utils.infft(
                hf_det / ifo.amplitude_spectral_density_array,
                sampling_frequency))
        mask = ifo.frequency_mask
        overlap = np.real(
            bilby.gw.utils.noise_weighted_inner_product(
                hf_inj_det[mask], hf_det[mask],
                power_spectral_density=ifo.power_spectral_density_array[mask],
                duration=ifo.strain_data.duration) /
            ifo.optimal_snr_squared(hf_inj_det)**0.5 /
            ifo.optimal_snr_squared(hf_det)**0.5)
        overlaps[ifo.name].append(overlap)
for ifo in ifos:
    plt.hist(overlaps[ifo.name], bins=40, histtype='step', label=ifo.name)
plt.legend()
plt.savefig(f'outdir/{label}_overlaps.png')
plt.close()
fd_waveforms = {ifo.name: np.array(fd_waveforms[ifo.name]) for ifo in ifos}
td_waveforms = {ifo.name: np.array(td_waveforms[ifo.name]) for ifo in ifos}
for ifo in ifos:
    fig, axs = plt.subplots(2, 1)
    hf_inj_det = ifo.get_detector_response(hf_inj, injection_parameters)
    ht_inj_det = bilby.utils.infft(
        hf_inj_det / ifo.amplitude_spectral_density_array,
        sampling_frequency=sampling_frequency)
    name = ifo.name
    axs[0].fill_between(
        ifo.frequency_array, np.percentile(abs(fd_waveforms[name]), 0.5, axis=0),
        np.percentile(abs(fd_waveforms[name]), 99.5, axis=0), color='r', alpha=0.5)
    axs[1].fill_between(
        ifo.time_array, np.percentile(td_waveforms[name], 0.5, axis=0),
        np.percentile(td_waveforms[name], 99.5, axis=0), color='r', alpha=0.5)
    axs[0].loglog(ifo.frequency_array, abs(ifo.frequency_domain_strain),
                  alpha=0.5, color='b')
    axs[1].plot(ifo.time_array, bilby.utils.infft(
        ifo.whitened_frequency_domain_strain,
        sampling_frequency=sampling_frequency),
                  alpha=0.5, color='b')
    axs[0].loglog(ifo.frequency_array, abs(hf_inj_det), color='k')
    axs[1].plot(ifo.time_array, ht_inj_det, color='k')
    axs[0].set_xlim(ifo.minimum_frequency, ifo.maximum_frequency)
    axs[0].set_ylim(1e-27, 1e-21)
    axs[1].set_xlim(
        injection_parameters['geocent_time'] - 0.2,
        injection_parameters['geocent_time'] + 0.2)
    plt.savefig(f'outdir/{label}_{name}_reconstruct.png')
    plt.close()

wavelet_params = ['q_factor', 'centre_frequency', 'amplitude', 'delta_time',
                  'phase', 'beta']
for ii in range(n_wavelets):
    fig = result.plot_corner(
        parameters=[f'{par}_{ii}' for par in wavelet_params],
        filename=f'outdir/{label}_corner_{ii}.png')
    plt.close()
result.plot_corner(
    parameters=[
        'geocent_time', 'n_wavelets', 'luminosity_distance', 'phase', 'psi',
        'ra', 'dec', 'ellipticity', 'log_likelihood'],
    filename=f'outdir/{label}_corner_common.png')

