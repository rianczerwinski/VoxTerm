"""Tests for spatial/calibration.py — chirp, TDOA, MDS geometry, Kalman."""

from __future__ import annotations

import numpy as np
import pytest

from spatial.calibration import CalibrationManager, CHIRP_DURATION_SEC


@pytest.fixture
def calibrator():
    return CalibrationManager(sample_rate=16000)


class TestChirpGeneration:
    def test_generate_chirp_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            chirp = calibrator.generate_chirp()

    def test_chirp_would_have_correct_length(self, calibrator):
        """Once implemented: chirp length = CHIRP_DURATION_SEC * sample_rate."""
        expected_samples = int(CHIRP_DURATION_SEC * 16000)
        assert expected_samples == 8000


class TestTDOAExtraction:
    def test_compute_tdoas_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            arrivals = {0: 0.0, 1: 0.001, 2: 0.002, 3: 0.0015}
            calibrator._compute_tdoas_from_arrivals(arrivals, reference=0)


class TestMDSGeometry:
    def test_calibrate_from_chirp_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            arrivals = {0: 0.0, 1: 0.001, 2: 0.002, 3: 0.0015}
            calibrator.calibrate_from_chirp(arrivals)

    def test_mds_geometry_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            from spatial.models import TDOAPair
            pairs = [TDOAPair(0, 1, 5.0, 5.0/16000, 0.9)]
            calibrator._mds_geometry(pairs)


class TestDegeneracyDetection:
    def test_check_degeneracy_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            from spatial.models import ArrayGeometry
            positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
            geom = ArrayGeometry(mic_positions=positions)
            calibrator._check_degeneracy(geom)


class TestKalmanRefinement:
    def test_update_from_speech_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            audio = np.random.randn(4, 16000).astype(np.float32)
            calibrator.update_from_speech(audio, vad_active=True)


class TestRecalibrationTrigger:
    def test_detect_significant_change_raises_not_implemented(self, calibrator):
        with pytest.raises(NotImplementedError):
            from spatial.models import TDOAPair
            pairs = [TDOAPair(0, 1, 50.0, 50.0/16000, 0.9)]
            calibrator._detect_significant_change(pairs)


class TestLifecycle:
    def test_not_calibrated_initially(self, calibrator):
        assert not calibrator.is_calibrated

    def test_geometry_none_initially(self, calibrator):
        assert calibrator.geometry is None
