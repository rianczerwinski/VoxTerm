"""Tests for spatial/models.py — dataclass construction and computed properties."""

from __future__ import annotations

import numpy as np
import pytest

from spatial.models import (
    ArrayGeometry,
    CalibrationResult,
    EnhancedAudio,
    FusedSegment,
    SpatialDescriptor,
    SpatialFrame,
    SpeakerLocation,
    TDOAPair,
    RetentionArtifact,
)


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def square_geometry():
    """4-mic square array, 2m sides."""
    positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float32)
    return ArrayGeometry(mic_positions=positions, source="test")


@pytest.fixture
def sample_spatial_frame():
    """Minimal SpatialFrame for testing."""
    return SpatialFrame(
        timestamp=0.0,
        sample_rate=16000,
        fft_size=1024,
        hop_size=256,
        doa_map=np.zeros((10, 513), dtype=np.float32),
        confidence_map=np.ones((10, 513), dtype=np.float32) * 0.5,
    )


# ── ArrayGeometry ─────────────────────────────────────────────────────

class TestArrayGeometry:
    def test_construction_and_auto_fields(self, square_geometry):
        """mic_count and aperture_m auto-computed from positions."""
        assert square_geometry.mic_count == 4
        assert square_geometry.aperture_m == pytest.approx(2 * np.sqrt(2), abs=0.01)

    def test_pairwise_distances(self, square_geometry):
        dists = square_geometry.pairwise_distances()
        assert dists.shape == (4, 4)
        assert dists[0, 0] == pytest.approx(0.0)
        assert dists[0, 1] == pytest.approx(2.0)  # adjacent side
        assert dists[0, 2] == pytest.approx(2 * np.sqrt(2), abs=0.01)  # diagonal

    def test_centroid(self, square_geometry):
        c = square_geometry.centroid()
        assert c == pytest.approx([1.0, 1.0])

    def test_angular_resolution_at_freq(self, square_geometry):
        res_1k = square_geometry.angular_resolution_at_freq(1000)
        res_4k = square_geometry.angular_resolution_at_freq(4000)
        assert res_4k < res_1k  # higher freq → better resolution


# ── SpatialFrame ──────────────────────────────────────────────────────

class TestSpatialFrame:
    def test_n_time_bins(self, sample_spatial_frame):
        assert sample_spatial_frame.n_time_bins() == 10

    def test_n_freq_bins(self, sample_spatial_frame):
        assert sample_spatial_frame.n_freq_bins() == 513

    def test_freq_to_bin(self, sample_spatial_frame):
        # At 16kHz with fft_size=1024, bin width = 16000/1024 ≈ 15.625 Hz
        bin_1k = sample_spatial_frame.freq_to_bin(1000)
        assert bin_1k == 64  # 1000 / 15.625 = 64


# ── FusedSegment ──────────────────────────────────────────────────────

class TestFusedSegment:
    def test_construction_and_duration(self):
        seg = FusedSegment(
            start_sample=0, end_sample=16000,
            speaker_id=1, spatial_cluster_id=0, embedding_speaker_id=1,
            fused_confidence=0.8, spatial_confidence=0.7, embedding_confidence=0.9,
        )
        assert seg.duration_samples == 16000
        assert seg.duration_seconds(16000) == pytest.approx(1.0)


# ── Other dataclasses ─────────────────────────────────────────────────

class TestTDOAPair:
    def test_construction(self):
        pair = TDOAPair(mic_a=0, mic_b=1, tdoa_samples=5.3, tdoa_seconds=5.3/16000, confidence=0.9)
        assert pair.tdoa_seconds == pytest.approx(5.3 / 16000)


class TestCalibrationResult:
    def test_construction(self, square_geometry):
        result = CalibrationResult(
            geometry=square_geometry, residual_error=0.001, chirp_count=3
        )
        assert result.chirp_count == 3


class TestSpeakerLocation:
    def test_construction(self):
        loc = SpeakerLocation(speaker_id=0, azimuth_deg=45.0, confidence=0.8)
        assert loc.elevation_deg is None


class TestEnhancedAudio:
    def test_construction(self):
        audio = np.zeros(16000, dtype=np.float32)
        enhanced = EnhancedAudio(audio=audio, sample_rate=16000, method="delay_and_sum")
        assert enhanced.dereverberated is False


class TestRetentionArtifact:
    def test_construction(self):
        art = RetentionArtifact(
            artifact_type="raw_array", session_id="test",
            created_at="20260322T120000", access_tier="sensitive",
            storage_path="raw/test.npy",
        )
        assert art.compressed is False
