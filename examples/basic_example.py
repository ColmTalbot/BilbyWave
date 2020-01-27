#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal with a single wavelet.
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import bilby
import bilbywave


# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.
sampling_frequency = 2048.

n_wavelets = 10

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
#label = 'single_chirplet_{}'.format(n_wavelets)
#label = 'single_chirplet_multimodality_{}'.format(n_wavelets)
#label = 'wavelet_{}'.format(n_wavelets)
label = 'single_chirplet_multimodality_75M_{}'.format(n_wavelets)
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=75., mass_2=70., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=750., theta_jn=0.4, psi=2.659,
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
# TODO: make nice prior class which builds this automatically
prior_file = "single_wavelet.prior"
try:
    priors = bilby.core.prior.ConditionalPriorDict(prior_file)
except AttributeError:
    priors = bilby.core.prior.PriorDict(prior_file)
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1, latex_label='$t_c$')

source_model = bilbywave.source.chirplet
#source_model = bilbywave.source.morlet_gabor_wavelet

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

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=350,
    outdir=outdir, label=label, walks=35,
    seed=int(time()), result_class=bilby.gw.result.CBCResult)

# reconstruct the marginalised phase posterior
# TODO: make this automatic, also allow for distance/time
new_phases = list()
for ii in tqdm.tqdm(range(len(result.posterior))):
    parameters = dict(result.posterior.iloc[ii])
    likelihood.parameters.update(parameters)
    new_phases.append(likelihood.generate_phase_sample_from_marginalized_likelihood())
result.posterior["phase"] = new_phases

# do some plotting, plot the waveform and a few corner plots
result.plot_waveform_posterior(interferometers=ifos, format="html", n_samples=1000)

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