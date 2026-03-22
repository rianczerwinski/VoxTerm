"""Cross-session identity bridge — links spatial clusters to persistent speakers.

Bridges ephemeral spatial-cluster IDs (per-session, from SpatialDiarizer §2.2)
to persistent speaker identities (cross-session, from SpeakerStore). The
existing embedding pipeline (CAM++, cosine similarity, adaptive thresholds)
is architecturally unchanged — this module feeds it better-segmented input.

The audio source for embedding extraction is deliberately bracketed as a
configurable parameter. The interface accepts any audio source; the detail
spec evaluates options (enhanced mono, beamformed-toward-speaker, raw
single-mic) without committing architecturally. The module boundary is
the same regardless of which audio feeds it.

Integration:
  bridge = SpatialIdentityBridge(speaker_store=self._speaker_store)
  labeled_segments = bridge.bridge(fused_segments, audio_sources)

Reference: research/specs/spatial-architecture.md §2.4
"""

from __future__ import annotations

import threading

import numpy as np

from config import SAMPLE_RATE
from spatial.models import FusedSegment

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from speakers.store import SpeakerStore

# ── constants ─────────────────────────────────────────────────────────

AUDIO_SOURCE_ENHANCED_MONO = "enhanced_mono"
AUDIO_SOURCE_BEAMFORMED = "beamformed"
AUDIO_SOURCE_RAW_SINGLE_MIC = "raw_single_mic"
DEFAULT_AUDIO_SOURCE = AUDIO_SOURCE_ENHANCED_MONO


class SpatialIdentityBridge:
    """Links ephemeral spatial-cluster IDs to persistent speaker identities.

    Per-session, the spatial diarizer assigns cluster IDs based on spatial
    position (cluster 0, 1, 2...). These have no meaning across sessions.
    This module takes those clusters, extracts audio for each, runs the
    existing embedding + matching pipeline (SpeakerStore), and attaches
    persistent identity labels.

    The existing embedding model, similarity metrics, clustering logic,
    and identity database are unchanged. This module receives
    better-segmented input; its own processing is the same.

    Args:
        speaker_store: Existing SpeakerStore instance for cross-session matching.
        audio_source: Which audio to use for embedding extraction.
            One of: "enhanced_mono", "beamformed", "raw_single_mic".
            This is a bracketed decision — configurable, not committed.
    """

    def __init__(
        self,
        speaker_store: SpeakerStore,
        audio_source: str = DEFAULT_AUDIO_SOURCE,
    ) -> None:
        self._store = speaker_store
        self._audio_source = audio_source
        self._lock = threading.Lock()

        # Per-session mapping: spatial_cluster_id → persistent speaker UUID
        self._cluster_to_identity: dict[int, str] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    @property
    def audio_source(self) -> str:
        """Current audio source selection for embedding extraction."""
        return self._audio_source

    def clear_session_mappings(self) -> None:
        """Reset cluster→identity mappings on new session."""
        with self._lock:
            self._cluster_to_identity.clear()

    # ── identity bridging ─────────────────────────────────────────────

    def bridge(
        self,
        fused_segments: list[FusedSegment],
        audio_sources: dict[str, np.ndarray],
        sample_rate: int = SAMPLE_RATE,
    ) -> list[FusedSegment]:
        """Attach persistent speaker identities to fused segments.

        For each segment, extracts audio from the configured source,
        computes a speaker embedding, matches against SpeakerStore,
        and updates the segment's speaker_id with the persistent identity.

        If a spatial cluster has already been mapped to a persistent
        identity in this session, reuses the mapping without re-extracting
        (cache hit). This avoids redundant embedding computation for
        speakers who remain in the same spatial position.

        Args:
            fused_segments: From SpatialDiarizer.fuse().
            audio_sources: Dict mapping source name to audio array.
                Keys: "enhanced_mono", "beamformed", "raw_single_mic".
                Not all keys are required — only the configured source.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Same segments with speaker_id updated to persistent identities
            where matches were found. Unmatched segments retain their
            original (ephemeral) speaker_id.
        """
        raise NotImplementedError(
            "Implement: for each segment, check cache (_cluster_to_identity). "
            "If miss, call _select_audio_for_embedding → _extract_and_match. "
            "Update segment.speaker_id with persistent identity."
        )

    def _select_audio_for_embedding(
        self,
        audio_sources: dict[str, np.ndarray],
        segment: FusedSegment,
    ) -> np.ndarray:
        """Extract the audio slice for embedding computation.

        Selects the configured audio source, then slices to the segment's
        time boundaries.

        Args:
            audio_sources: Available audio by source name.
            segment: Segment defining the time range.

        Returns:
            Audio slice, float32, mono.

        Raises:
            KeyError: If the configured audio source is not in audio_sources.
        """
        raise NotImplementedError(
            "Implement: select audio_sources[self._audio_source], slice "
            "from segment.start_sample to segment.end_sample."
        )

    def _extract_and_match(
        self,
        audio: np.ndarray,
        segment: FusedSegment,
    ) -> FusedSegment:
        """Compute embedding and match against SpeakerStore.

        Delegates to the existing SpeakerStore.classify_match() pipeline.
        If a match is found, updates the segment and caches the mapping.

        Args:
            audio: Mono audio slice for this segment.
            segment: The segment to update.

        Returns:
            Updated segment with persistent speaker_id (if matched).
        """
        raise NotImplementedError(
            "Implement: extract embedding from audio (delegate to proxy or "
            "direct CAM++ call), call self._store.classify_match(embedding), "
            "if match tier is 'high', update segment.speaker_id and cache "
            "the mapping in _cluster_to_identity."
        )

    # ── mapping cache ─────────────────────────────────────────────────

    def _spatial_cluster_to_identity(self, spatial_cluster_id: int) -> str | None:
        """Cached mapping from ephemeral cluster ID to persistent speaker UUID.

        Returns None if no mapping exists for this cluster in the current session.
        """
        with self._lock:
            return self._cluster_to_identity.get(spatial_cluster_id)

    def _update_mapping(self, spatial_cluster_id: int, persistent_id: str) -> None:
        """Cache a new cluster→identity mapping for the current session."""
        with self._lock:
            self._cluster_to_identity[spatial_cluster_id] = persistent_id
