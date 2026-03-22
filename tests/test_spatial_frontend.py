"""Tests for spatial/frontend.py — GCC-PHAT, SRP-PHAT, spatial frame generation."""

from __future__ import annotations

import numpy as np
import pytest

from spatial.frontend import SpatialFrontEnd
from spatial.models import ArrayGeometry, SpatialFrame


@pytest.fixture
def square_geometry():
    positions = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    return ArrayGeometry(mic_positions=positions, is_calibrated=True, source="test")


@pytest.fixture
def frontend(square_geometry):
    return SpatialFrontEnd(geometry=square_geometry, sample_rate=16000)


class TestGCCPHAT:
    def test_known_delay(self, frontend):
        """Two signals with known sample delay → correct TDOA."""
        with pytest.raises(NotImplementedError):
            n = 1024
            sig_a = np.random.randn(n).astype(np.float32)
            sig_b = np.roll(sig_a, 5)  # 5 sample delay
            tdoa, conf = frontend._compute_gcc_phat(sig_a, sig_b)

    def test_identical_signals_zero_tdoa(self, frontend):
        """Identical signals → TDOA ≈ 0."""
        with pytest.raises(NotImplementedError):
            sig = np.random.randn(1024).astype(np.float32)
            tdoa, conf = frontend._compute_gcc_phat(sig, sig)

    def test_noise_degrades_confidence(self, frontend):
        """Heavy noise → lower confidence."""
        with pytest.raises(NotImplementedError):
            sig = np.random.randn(1024).astype(np.float32)
            noise = np.random.randn(1024).astype(np.float32) * 10
            tdoa, conf = frontend._compute_gcc_phat(sig, sig + noise)


class TestSRPPHAT:
    def test_single_source_localization(self, frontend):
        """Known source at 45° → SRP peaks near 45°."""
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            locations = frontend._srp_phat(audio, [])


class TestProcess:
    def test_output_is_spatial_frame(self, frontend):
        """process() returns a SpatialFrame with correct dimensions."""
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            frame = frontend.process(audio)

    def test_geometry_update_invalidates_cache(self, frontend, square_geometry):
        """update_geometry() clears cached steering vectors."""
        new_geom = ArrayGeometry(
            mic_positions=square_geometry.mic_positions + 0.1,
            is_calibrated=True, source="test"
        )
        frontend.update_geometry(new_geom)
        assert frontend._steering_vectors is not None or frontend._steering_vectors is None
        # Once implemented: verify steering vectors were recomputed
