"""Tests for spatial/enhancement.py — DAS, MVDR, WPE beamforming."""

from __future__ import annotations

import numpy as np
import pytest

from spatial.enhancement import AudioEnhancer
from spatial.models import ArrayGeometry, SpatialFrame, SpeakerLocation


@pytest.fixture
def square_geometry():
    positions = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    return ArrayGeometry(mic_positions=positions, is_calibrated=True, source="test")


@pytest.fixture
def enhancer(square_geometry):
    return AudioEnhancer(geometry=square_geometry, sample_rate=16000)


@pytest.fixture
def sample_frame():
    return SpatialFrame(
        timestamp=0.0, sample_rate=16000, fft_size=1024, hop_size=256,
        doa_map=np.zeros((10, 513), dtype=np.float32),
        confidence_map=np.ones((10, 513), dtype=np.float32) * 0.8,
    )


class TestDASBeamforming:
    def test_enhance_live_raises_not_implemented(self, enhancer, sample_frame):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            target = SpeakerLocation(speaker_id=0, azimuth_deg=45.0, confidence=0.9)
            enhancer.enhance_live(audio, sample_frame, [target])

    def test_delay_and_sum_raises_not_implemented(self, enhancer):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            target = SpeakerLocation(speaker_id=0, azimuth_deg=45.0, confidence=0.9)
            enhancer._delay_and_sum(audio, target)


class TestMVDR:
    def test_enhance_enrichment_raises_not_implemented(self, enhancer, sample_frame):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            target = SpeakerLocation(speaker_id=0, azimuth_deg=45.0, confidence=0.9)
            enhancer.enhance_enrichment(audio, sample_frame, [target])


class TestWPE:
    def test_wpe_dereverberate_raises_not_implemented(self, enhancer):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            enhancer._wpe_dereverberate(audio)
