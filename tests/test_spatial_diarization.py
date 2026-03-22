"""Tests for spatial/diarization.py — dual-path fusion, degradation modes."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from spatial.diarization import SpatialDiarizer, MIN_DEVICES_FOR_SPATIAL
from spatial.models import SpatialFrame


@pytest.fixture
def mock_proxy():
    proxy = MagicMock()
    proxy.identify.return_value = ("Speaker 1", 1)
    return proxy


@pytest.fixture
def diarizer(mock_proxy):
    return SpatialDiarizer(diarization_proxy=mock_proxy)


@pytest.fixture
def sample_frame():
    return SpatialFrame(
        timestamp=0.0, sample_rate=16000, fft_size=1024, hop_size=256,
        doa_map=np.zeros((10, 513), dtype=np.float32),
        confidence_map=np.ones((10, 513), dtype=np.float32) * 0.8,
    )


class TestDualPathFusion:
    def test_fuse_raises_not_implemented(self, diarizer, sample_frame):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(16000).astype(np.float32)
            diarizer.fuse(sample_frame, audio)

    def test_spatial_path_raises_not_implemented(self, diarizer, sample_frame):
        with pytest.raises(NotImplementedError):
            diarizer._spatial_path(sample_frame)

    def test_embedding_path_raises_not_implemented(self, diarizer):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(16000).astype(np.float32)
            diarizer._embedding_path(audio, 16000)


class TestConfidenceWeighting:
    def test_fusion_weights_raises_not_implemented(self, diarizer):
        with pytest.raises(NotImplementedError):
            diarizer._compute_fusion_weights(0.9, 0.3)


class TestDegradedMode:
    def test_spatial_not_available_below_min_devices(self, diarizer):
        diarizer.set_device_count(2)
        assert not diarizer.is_spatial_available

    def test_spatial_available_at_min_devices(self, diarizer):
        diarizer.set_device_count(MIN_DEVICES_FOR_SPATIAL)
        assert diarizer.is_spatial_available

    def test_degraded_mode_raises_not_implemented(self, diarizer):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(16000).astype(np.float32)
            diarizer._handle_degraded_mode(2, None, audio)

    def test_reset_session_clears_state(self, diarizer):
        diarizer._spatial_clusters[0] = MagicMock()
        diarizer.reset_session()
        assert len(diarizer._spatial_clusters) == 0
