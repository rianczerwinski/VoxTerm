"""Dual-path fusion diarizer — spatial + embedding analysis running simultaneously.

Within-session speaker diarization that fuses two independent analysis paths:
1. Spatial path: DOA clustering from SpatialFrontEnd descriptors
2. Embedding path: CAM++ voice-identity clustering from existing DiarizationProxy

Neither path is architecturally subordinated to the other. When spatial
discrimination is strong (well-separated speakers, low reverberation,
calibrated geometry with 4+ devices), spatial evidence dominates. When
spatial degrades (angular convergence, high reverb, movement, <4 devices),
embedding evidence carries the load. The transition is continuous — no
binary mode switch.

This preserves the full possibility space: future improvements to either
path propagate without restructuring. A better embedding model or a better
spatial algorithm slots in without changing the fusion architecture.

The module wraps the existing DiarizationProxy — it does not replace it.
The proxy continues to run its CAM++ pipeline. This module adds spatial
evidence alongside it and fuses the two.

Integration:
  from spatial import SpatialDiarizer
  diarizer = SpatialDiarizer(diarization_proxy=self._diarizer)
  segments = diarizer.fuse(spatial_frame, audio)

Reference: research/specs/spatial-architecture.md §2.2
Reference: research/deliverables/02-spatial-acoustics.md RQ3 (resolution bounds)
"""

from __future__ import annotations

import threading

import numpy as np

from config import SAMPLE_RATE
from spatial.models import (
    FusedSegment,
    SpeakerLocation,
    SpatialFrame,
)

# Import type only — actual proxy comes via constructor injection
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from diarization.proxy import DiarizationProxy

# ── constants ─────────────────────────────────────────────────────────

SPATIAL_WEIGHT_DEFAULT = 0.5
EMBEDDING_WEIGHT_DEFAULT = 0.5
SPATIAL_DOMINANCE_THRESHOLD = 0.8  # spatial confidence above this → spatial dominates
EMBEDDING_DOMINANCE_THRESHOLD = 0.8  # embedding confidence above this → embedding dominates
MIN_DEVICES_FOR_SPATIAL = 4  # minimum for full spatial diarization
DEGRADED_DEVICE_THRESHOLD = 3  # 3 devices → DOA hints only


class SpatialDiarizer:
    """Dual-path fusion diarizer: spatial + embedding evidence combined.

    Runs spatial DOA clustering and embedding-based clustering in parallel,
    then fuses their outputs with confidence-weighted combination. The
    fusion weights adapt dynamically based on each path's confidence in
    current conditions.

    Graceful degradation by device count:
    - 4+ devices: full spatial + embedding fusion
    - 3 devices: DOA hints only (spatial confidence low, embedding dominates)
    - 2 devices: dual-channel noise reduction (no spatial diarization)
    - 1 device: passthrough (embedding-only, same as current VoxTerm)

    Thread safety: internal state (spatial clusters, session mappings) is
    lock-serialized. The DiarizationProxy is accessed through its own
    thread-safe interface.

    Args:
        diarization_proxy: Existing CAM++ diarization proxy instance.
        spatial_weight: Base weight for spatial path (0-1).
        embedding_weight: Base weight for embedding path (0-1).
    """

    def __init__(
        self,
        diarization_proxy: DiarizationProxy,
        spatial_weight: float = SPATIAL_WEIGHT_DEFAULT,
        embedding_weight: float = EMBEDDING_WEIGHT_DEFAULT,
    ) -> None:
        self._proxy = diarization_proxy
        self._spatial_weight = spatial_weight
        self._embedding_weight = embedding_weight
        self._lock = threading.Lock()

        # Per-session state
        self._spatial_clusters: dict[int, SpeakerLocation] = {}
        self._n_devices: int = 0

    # ── lifecycle ─────────────────────────────────────────────────────

    @property
    def is_spatial_available(self) -> bool:
        """True when geometry is calibrated and >= MIN_DEVICES mics."""
        return self._n_devices >= MIN_DEVICES_FOR_SPATIAL

    def set_device_count(self, n: int) -> None:
        """Update the number of active capture devices."""
        with self._lock:
            self._n_devices = n

    def reset_session(self) -> None:
        """Clear spatial clusters and session state. Call on new session."""
        with self._lock:
            self._spatial_clusters.clear()

    # ── dual-path fusion ──────────────────────────────────────────────

    def fuse(
        self,
        spatial_frame: SpatialFrame | None,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> list[FusedSegment]:
        """Main entry point: run both paths, combine results.

        If spatial_frame is None or device count is insufficient, falls
        back to degraded mode (embedding-only with spatial hints where
        available).

        Args:
            spatial_frame: Output of SpatialFrontEnd.process(), or None.
            audio: Single-channel audio for embedding extraction, float32.
            sample_rate: Sample rate in Hz.

        Returns:
            List of FusedSegment with both spatial and embedding evidence.
        """
        raise NotImplementedError(
            "Implement: check device count → if degraded, call "
            "_handle_degraded_mode. Otherwise run _spatial_path and "
            "_embedding_path in sequence (or parallel with threading), "
            "then _combine_paths."
        )

    def _spatial_path(
        self, spatial_frame: SpatialFrame
    ) -> list[tuple[int, int, int, float]]:
        """Spatial-only segmentation using DOA clustering.

        Clusters TF bins by DOA angle, then converts clusters to
        time-domain speaker segments.

        Args:
            spatial_frame: SpatialFrame from front-end.

        Returns:
            List of (spatial_cluster_id, start_sample, end_sample, confidence).
        """
        raise NotImplementedError(
            "Implement: extract DOA map and confidence from spatial_frame, "
            "cluster by angular proximity (threshold based on array resolution), "
            "convert TF-bin clusters to sample-domain segments."
        )

    def _embedding_path(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[str, int, int, int, float]]:
        """Embedding-based segmentation via DiarizationProxy.

        Delegates to the existing CAM++ pipeline. This path is unchanged
        from current VoxTerm — the proxy handles all embedding extraction,
        clustering, and labeling.

        Args:
            audio: Single-channel float32 audio.
            sample_rate: Sample rate in Hz.

        Returns:
            List of (label, speaker_id, start_sample, end_sample, confidence).
        """
        raise NotImplementedError(
            "Implement: delegate to self._proxy.identify() or equivalent "
            "method that returns speaker segments with confidence scores."
        )

    def _combine_paths(
        self,
        spatial_segments: list[tuple[int, int, int, float]],
        embedding_segments: list[tuple[str, int, int, int, float]],
        spatial_confidence: float,
        embedding_confidence: float,
    ) -> list[FusedSegment]:
        """Confidence-weighted fusion of spatial and embedding segments.

        Aligns segments temporally, then for each time region, combines
        the spatial cluster assignment with the embedding speaker assignment
        using dynamic weights based on per-path confidence.

        The fusion is continuous — no binary mode switch. When spatial
        confidence is high, the spatial cluster ID determines the speaker.
        When embedding confidence is high, the embedding speaker ID
        determines it. In between, both contribute.

        Args:
            spatial_segments: From _spatial_path.
            embedding_segments: From _embedding_path.
            spatial_confidence: Overall spatial path confidence (0-1).
            embedding_confidence: Overall embedding path confidence (0-1).

        Returns:
            List of FusedSegment with all confidence scores populated.

        Reference: research/specs/spatial-architecture.md §5
            (open question: fusion weighting strategy)
        """
        raise NotImplementedError(
            "Implement: align segments temporally, compute dynamic weights "
            "via _compute_fusion_weights, assign fused speaker ID based on "
            "dominant path, populate all confidence fields in FusedSegment."
        )

    # ── confidence estimation ─────────────────────────────────────────

    def _estimate_spatial_confidence(self, spatial_frame: SpatialFrame) -> float:
        """Overall spatial confidence for this frame.

        Based on: geometry quality (calibration recency, number of mics),
        mean per-TF-bin confidence, array aperture relative to speech
        wavelength.

        Returns: float 0-1.
        """
        raise NotImplementedError(
            "Implement: combine geometry quality metrics with mean confidence "
            "from spatial_frame.confidence_map."
        )

    def _estimate_embedding_confidence(
        self, segments: list[tuple[str, int, int, int, float]]
    ) -> float:
        """Overall embedding confidence from DiarizationProxy output.

        Based on the proxy's match scores and cluster stability.

        Returns: float 0-1.
        """
        raise NotImplementedError(
            "Implement: aggregate confidence scores from embedding segments."
        )

    def _compute_fusion_weights(
        self, spatial_conf: float, embedding_conf: float
    ) -> tuple[float, float]:
        """Dynamic fusion weight adjustment.

        When one path has very high confidence (> DOMINANCE_THRESHOLD),
        it gets a disproportionate share of the weight. When both are
        moderate, they share roughly equally. When both are low, the
        result has low overall confidence.

        Args:
            spatial_conf: Spatial path confidence (0-1).
            embedding_conf: Embedding path confidence (0-1).

        Returns:
            (spatial_weight, embedding_weight) summing to 1.0.
        """
        raise NotImplementedError(
            "Implement: if spatial_conf > DOMINANCE_THRESHOLD, spatial gets "
            "~0.8-0.9 weight. If embedding_conf > DOMINANCE_THRESHOLD, "
            "embedding gets ~0.8-0.9. Otherwise, use base weights normalized. "
            "Exact strategy is an open question — start simple, tune empirically."
        )

    # ── spatial clustering ────────────────────────────────────────────

    def _cluster_by_doa(
        self,
        doa_map: np.ndarray,
        confidence_map: np.ndarray,
    ) -> dict[int, list[tuple[int, int]]]:
        """Cluster TF bins by DOA angle.

        Groups bins whose DOA estimates are within the array's angular
        resolution of each other. Only bins with confidence above threshold
        are clustered; low-confidence bins are unassigned.

        Args:
            doa_map: (time_bins, freq_bins) azimuth in degrees.
            confidence_map: (time_bins, freq_bins) 0-1.

        Returns:
            Dict mapping spatial_cluster_id → list of (time_bin, freq_bin).
        """
        raise NotImplementedError(
            "Implement: threshold confidence_map, extract reliable bins, "
            "cluster DOA values (e.g., simple angular histogram with peaks, "
            "or DBSCAN on circular coordinates)."
        )

    def _doa_to_speaker_segments(
        self,
        clusters: dict[int, list[tuple[int, int]]],
        n_samples: int,
    ) -> list[tuple[int, int, int, float]]:
        """Convert DOA clusters to time-domain speaker segments.

        Each spatial cluster maps to a contiguous time range where that
        cluster's DOA dominates.

        Args:
            clusters: From _cluster_by_doa.
            n_samples: Total audio length in samples.

        Returns:
            List of (spatial_cluster_id, start_sample, end_sample, confidence).
        """
        raise NotImplementedError(
            "Implement: for each cluster, find the time bins where it's "
            "active, convert bin indices to sample indices, merge adjacent "
            "active regions, compute per-segment confidence."
        )

    # ── degradation ───────────────────────────────────────────────────

    def _handle_degraded_mode(
        self,
        n_devices: int,
        spatial_frame: SpatialFrame | None,
        audio: np.ndarray,
    ) -> list[FusedSegment]:
        """Graceful fallback for insufficient devices.

        4+ devices: full spatial + embedding fusion (not called here)
        3 devices:  DOA hints — spatial confidence set low, embedding dominates
        2 devices:  dual-channel noise reduction, no spatial diarization
        1 device:   passthrough — embedding-only (current VoxTerm behavior)

        Reference: research/deliverables/02-spatial-acoustics.md Risk Register
        """
        raise NotImplementedError(
            "Implement: check n_devices, run embedding path as baseline, "
            "optionally annotate with DOA hints (3 devices) or skip spatial "
            "entirely (1-2 devices). Return FusedSegments with appropriate "
            "confidence scores reflecting the degraded state."
        )
