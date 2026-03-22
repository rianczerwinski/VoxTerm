"""Tests for spatial/retention.py — artifact persistence, retention enforcement, access tiers."""

from __future__ import annotations

import numpy as np
import pytest

from spatial.retention import (
    RetentionManager,
    RAW_ARRAY_TIER,
    ENHANCED_AUDIO_TIER,
)
from spatial.models import ArrayGeometry, EnhancedAudio, FusedSegment


@pytest.fixture
def retention_manager(tmp_path):
    return RetentionManager(storage_dir=tmp_path)


@pytest.fixture
def square_geometry():
    positions = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    return ArrayGeometry(mic_positions=positions, is_calibrated=True, source="test")


class TestStorageRoundTrip:
    def test_store_raw_array_raises_not_implemented(self, retention_manager, square_geometry):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            retention_manager.store_raw_array(audio, square_geometry, "test-session", 0.0)

    def test_store_enhanced_audio_raises_not_implemented(self, retention_manager):
        with pytest.raises(NotImplementedError):
            enhanced = EnhancedAudio(
                audio=np.zeros(16000, dtype=np.float32),
                sample_rate=16000, method="delay_and_sum"
            )
            retention_manager.store_enhanced_audio(enhanced, "test-session", 0.0)


class TestRetentionEnforcement:
    def test_enforce_retention_raises_not_implemented(self, retention_manager):
        with pytest.raises(NotImplementedError):
            retention_manager.enforce_retention()


class TestAccessTiers:
    def test_raw_array_tier(self):
        assert RAW_ARRAY_TIER == "sensitive"

    def test_enhanced_audio_tier(self):
        assert ENHANCED_AUDIO_TIER == "standard"


class TestOpusCompression:
    def test_compress_opus_raises_not_implemented(self, retention_manager):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(16000).astype(np.float32)
            retention_manager._compress_opus(audio, 16000)


class TestAuditLogging:
    def test_audit_access_raises_not_implemented(self, retention_manager):
        from spatial.models import RetentionArtifact
        art = RetentionArtifact(
            artifact_type="raw_array", session_id="test",
            created_at="20260322T120000", access_tier="sensitive",
            storage_path="raw/test.npy"
        )
        with pytest.raises(NotImplementedError):
            retention_manager.audit_access(art, "test-user", "read")


class TestLifecycle:
    def test_open_raises_not_implemented(self, retention_manager):
        with pytest.raises(NotImplementedError):
            retention_manager.open()
