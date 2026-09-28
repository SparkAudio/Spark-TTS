import unittest

import numpy as np

from sparktts.utils.audio import audio_volume_normalize


class AudioVolumeNormalizeTests(unittest.TestCase):
    def test_quiet_audio_reaches_the_target_level(self):
        normalized = audio_volume_normalize(np.full(1000, 0.02))
        self.assertTrue(np.allclose(normalized, 0.2))

    def test_louder_audio_still_reaches_the_target_level(self):
        normalized = audio_volume_normalize(np.full(1000, 0.5))
        self.assertTrue(np.allclose(normalized, 0.2))

    def test_short_quiet_clip_stays_at_the_lift(self):
        normalized = audio_volume_normalize(np.full(5, 0.02))
        self.assertTrue(np.allclose(normalized, 0.1))


if __name__ == "__main__":
    unittest.main()
