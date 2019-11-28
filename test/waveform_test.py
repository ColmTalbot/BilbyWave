import unittest

import numpy as np

from bilbywave import source


class TestSource(unittest.TestCase):

    def setUp(self) -> None:
        self.frequency_array = np.arange(1, 1025)
        self.parameters = dict(
            amplitude=1e-23,
            q_factor=5,
            centre_frequency=100,
            ellipticity=1,
            delta_time=0,
            phase=0
        )

    def tearDown(self) -> None:
        pass

    def test_morlet_gabor_runs(self):
        waveform_polarisations = source.morlet_gabor_wavelet(
            self.frequency_array, **self.parameters)
        self.assertIsInstance(waveform_polarisations, dict)

    def test_chirplet_runs(self):
        self.parameters["beta"] = 0
        waveform_polarisations = source.chirplet(
            self.frequency_array, **self.parameters)
        self.assertIsInstance(waveform_polarisations, dict)
