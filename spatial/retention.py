"""Storage and retention policy — cross-cutting artifact persistence.

Not a processing module. Defines the boundary between "live processing
window" and "durable artifact." Materializes privacy and governance
decisions as engineering constraints.

Raw array data enables both selective speaker isolation and selective
speaker erasure — these are mathematically identical (complementary
subspace projections). Access to raw array data must be controlled
as a sensitive capability, distinct from access to enhanced mono output.
This is managed as a data governance concern, not mitigated at the
representation level. (Regime 3: total fidelity capture.)

Retained artifacts:
  - Raw N-channel array data (full fidelity, enables future reprocessing)
  - Enhanced audio output from AudioEnhancer
  - Diarization metadata from SpatialDiarizer
  - Identity labels from SpatialIdentityBridge
  - Array geometry configuration
  - Spatial front-end parameters / version

Reference: research/specs/spatial-architecture.md §2.5
Reference: app/docs/spatial-erasure-analysis.md (erasure-isolation duality)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from spatial.models import (
    ArrayGeometry,
    EnhancedAudio,
    FusedSegment,
    RetentionArtifact,
)

# ── constants ─────────────────────────────────────────────────────────

RAW_ARRAY_TIER = "sensitive"
ENHANCED_AUDIO_TIER = "standard"
METADATA_TIER = "standard"

DEFAULT_RAW_RETENTION_HOURS = 168  # 7 days
DEFAULT_ENHANCED_RETENTION_HOURS = 720  # 30 days
DEFAULT_METADATA_RETENTION_HOURS = 8760  # 1 year

OPUS_BITRATE_KBPS = 16  # for long-term compressed storage
RAW_BYTES_PER_HOUR_PER_CHANNEL = 115_200_000  # 16kHz × 4 bytes × 3600s


class RetentionManager:
    """Cross-cutting artifact persistence with tiered access control.

    Manages storage of all spatial processing outputs. Each artifact
    has an access tier ("sensitive" for raw array data, "standard" for
    everything else) and a retention window (after which it's eligible
    for eviction).

    Raw array data is stored at full fidelity — no lossy compression
    at capture time. This enables reprocessing with improved algorithms
    without re-capture (Regime 3). Enhanced audio may be Opus-compressed
    for long-term storage.

    Governance: access to sensitive artifacts is logged via audit_access().
    The detail spec defines access tier semantics, deletion procedures,
    and audit requirements.

    Args:
        storage_dir: Base directory for retained artifacts.
        raw_retention_hours: How long to keep raw array data.
        enhanced_retention_hours: How long to keep enhanced audio.
        metadata_retention_hours: How long to keep diarization/identity metadata.
    """

    def __init__(
        self,
        storage_dir: Path,
        raw_retention_hours: float = DEFAULT_RAW_RETENTION_HOURS,
        enhanced_retention_hours: float = DEFAULT_ENHANCED_RETENTION_HOURS,
        metadata_retention_hours: float = DEFAULT_METADATA_RETENTION_HOURS,
    ) -> None:
        self._storage_dir = storage_dir
        self._raw_retention_hours = raw_retention_hours
        self._enhanced_retention_hours = enhanced_retention_hours
        self._metadata_retention_hours = metadata_retention_hours
        self._lock = threading.Lock()
        self._artifacts: list[RetentionArtifact] = []

    # ── lifecycle ─────────────────────────────────────────────────────

    def open(self) -> None:
        """Create storage directories, verify permissions."""
        raise NotImplementedError(
            "Implement: create storage_dir / 'raw', 'enhanced', 'metadata' "
            "subdirs. Set permissions (chmod 700 for sensitive). Load existing "
            "artifact index if any."
        )

    def close(self) -> None:
        """Flush pending writes, release locks."""
        raise NotImplementedError(
            "Implement: ensure all pending writes are flushed, save artifact "
            "index to disk."
        )

    # ── artifact persistence ──────────────────────────────────────────

    def store_raw_array(
        self,
        multichannel_audio: np.ndarray,
        geometry: ArrayGeometry,
        session_id: str,
        timestamp: float,
    ) -> RetentionArtifact:
        """Persist raw N-channel audio with geometry.

        Full fidelity — no compression, no downsampling. Enables future
        reprocessing with improved algorithms without re-capture.

        Storage: ~115 MB/hour per channel at 16kHz float32.
        6 channels, 2 hours = ~1.4 GB.

        Args:
            multichannel_audio: (N_channels, N_samples) float32.
            geometry: Array geometry at time of capture.
            session_id: Session identifier.
            timestamp: Capture timestamp (monotonic).

        Returns:
            RetentionArtifact with access_tier="sensitive".
        """
        raise NotImplementedError(
            "Implement: save multichannel_audio as .npy, save geometry "
            "alongside as .json, create RetentionArtifact with sensitive tier."
        )

    def store_enhanced_audio(
        self,
        enhanced: EnhancedAudio,
        session_id: str,
        timestamp: float,
    ) -> RetentionArtifact:
        """Persist enhanced mono output.

        May be Opus-compressed for long-term storage via _compress_opus().

        Args:
            enhanced: EnhancedAudio from AudioEnhancer.
            session_id: Session identifier.
            timestamp: Processing timestamp.

        Returns:
            RetentionArtifact with access_tier="standard".
        """
        raise NotImplementedError(
            "Implement: optionally compress with _compress_opus, save to "
            "'enhanced' subdir, create RetentionArtifact."
        )

    def store_diarization_metadata(
        self,
        segments: list[FusedSegment],
        session_id: str,
    ) -> RetentionArtifact:
        """Persist diarization segment boundaries, IDs, confidence scores."""
        raise NotImplementedError(
            "Implement: serialize segments to JSON, save to 'metadata' subdir."
        )

    def store_identity_labels(
        self,
        segments: list[FusedSegment],
        session_id: str,
    ) -> RetentionArtifact:
        """Persist persistent speaker identity labels."""
        raise NotImplementedError(
            "Implement: extract identity labels from segments, serialize, save."
        )

    def store_geometry(
        self,
        geometry: ArrayGeometry,
        session_id: str,
    ) -> RetentionArtifact:
        """Persist array geometry configuration.

        Required for any future reprocessing of raw array data.
        """
        raise NotImplementedError(
            "Implement: serialize geometry (mic_positions, metadata) to JSON."
        )

    def store_processing_params(
        self,
        params: dict[str, Any],
        session_id: str,
    ) -> RetentionArtifact:
        """Persist spatial front-end version and processing parameters.

        Required to reproduce or improve upon processing. Without these,
        raw array data can still be reprocessed but the original processing
        cannot be exactly reproduced.
        """
        raise NotImplementedError(
            "Implement: serialize params dict to JSON, save to metadata subdir."
        )

    # ── compression ───────────────────────────────────────────────────

    def _compress_opus(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Opus compression for long-term audio storage.

        16kHz mono at 16 kbps = ~6 KB/min. Quality loss is negligible
        for speech at this bitrate. Encoder state is ~11 KB with
        zero-allocation encoding (safe for real-time threads).

        Reference: research/deliverables/01-android-audio.md §Codec Analysis
        """
        raise NotImplementedError(
            "Implement: encode audio to Opus at OPUS_BITRATE_KBPS. "
            "Handle import failure gracefully (return raw bytes if Opus "
            "not available)."
        )

    def _decompress_opus(self, data: bytes, sample_rate: int) -> np.ndarray:
        """Decode Opus back to PCM float32."""
        raise NotImplementedError("Implement Opus decode.")

    # ── retention enforcement ─────────────────────────────────────────

    def enforce_retention(self) -> int:
        """Evict artifacts past their retention window.

        Returns the count of evicted artifacts. Should be called
        periodically (e.g., once per hour or on session end).
        """
        raise NotImplementedError(
            "Implement: iterate artifacts, check created_at + retention "
            "window against current time, delete expired artifacts, "
            "return count."
        )

    def _evict_by_tier(self, tier: str, max_age_hours: float) -> int:
        """Evict artifacts of a given tier older than max_age_hours."""
        raise NotImplementedError(
            "Implement: filter artifacts by tier, check age, delete file, "
            "remove from index."
        )

    def get_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[RetentionArtifact]:
        """Retrieve stored artifacts for a session.

        Args:
            session_id: Filter by session.
            artifact_type: Optional filter by type (e.g., "raw_array").

        Returns:
            List of matching RetentionArtifact metadata.
        """
        raise NotImplementedError(
            "Implement: filter self._artifacts by session_id and optionally "
            "artifact_type."
        )

    # ── governance ────────────────────────────────────────────────────

    def get_access_tier(self, artifact: RetentionArtifact) -> str:
        """Return the access tier for an artifact."""
        return artifact.access_tier

    def audit_access(
        self,
        artifact: RetentionArtifact,
        accessor: str,
        operation: str,
    ) -> None:
        """Log access to an artifact. Required for sensitive tier.

        Args:
            artifact: The artifact being accessed.
            accessor: Identifier of the accessing entity.
            operation: "read", "delete", "export", etc.
        """
        raise NotImplementedError(
            "Implement: append to audit log with timestamp, artifact ID, "
            "accessor, operation. For sensitive tier, log is mandatory."
        )
