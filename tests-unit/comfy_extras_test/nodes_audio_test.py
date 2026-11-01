from unittest.mock import MagicMock, patch

import pytest
import torch

# comfy_extras.nodes_audio imports torchaudio at module scope. AudioDuration does not
# need it, and importing it crashes the macOS CI runner.
with patch.dict("sys.modules", {"torchaudio": MagicMock()}):
    from comfy_extras.nodes_audio import AudioDuration


class TestAudioDurationExecute:
    @staticmethod
    def _exec(audio) -> object:
        return AudioDuration.execute(audio)

    def test_stereo_duration(self):
        sample_rate = 44100
        num_samples = 44100 * 3  # 3 seconds
        audio = {"waveform": torch.zeros((1, 2, num_samples)), "sample_rate": sample_rate}
        result = self._exec(audio)
        assert result[0] == pytest.approx(3.0)
        assert result[1] == sample_rate

    def test_mono_duration(self):
        sample_rate = 48000
        num_samples = 48000 * 2 + 4800  # 2.1 seconds
        audio = {"waveform": torch.zeros((1, 1, num_samples)), "sample_rate": sample_rate}
        result = self._exec(audio)
        assert result[0] == pytest.approx(2.1)
        assert result[1] == sample_rate

    def test_none_audio(self):
        result = self._exec(None)
        assert result[0] == 0.0
        assert result[1] == 0

    def test_zero_sample_rate_guard(self):
        audio = {"waveform": torch.zeros((1, 2, 100)), "sample_rate": 0}
        result = self._exec(audio)
        assert result[0] == 0.0
        assert result[1] == 0

    def test_empty_waveform(self):
        audio = {"waveform": torch.zeros((1, 2, 0)), "sample_rate": 44100}
        result = self._exec(audio)
        assert result[0] == 0.0
        assert result[1] == 44100
