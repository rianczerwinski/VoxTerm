"""Spatial audio data contracts — shared between all spatial modules.

Dataclasses defining the inter-module interfaces for spatial processing.
These are the structural contracts: if a module produces a SpatialFrame,
every consumer agrees on what that means. Changes here cascade everywhere;
stability here decouples everything else.

Follows the pattern of speakers/models.py: immutable-ish dataclasses with
computed properties, numpy fields, and type hints throughout.

Reference: research/specs/spatial-architecture.md §4 Interface Contracts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── array geometry ────────────────────────────────────────────────────

@dataclass
class ArrayGeometry:
    """Physical microphone array configuration.

    Defines the spatial layout of capture devices. Consumed by the spatial
    front-end (§2.1) to compute steering vectors and by calibration to
    estimate/refine positions.

    Positions are 2D (x, y) in meters, relative to an arbitrary origin.
    3D (x, y, z) supported but not required — most phone deployments are
    coplanar (on a table). The origin and rotation are arbitrary; only
    relative positions matter for beamforming and TDOA.

    Reference: research/specs/spatial-architecture.md §3
    """

    mic_positions: np.ndarray  # shape (N, 2) or (N, 3), meters
    mic_count: int = 0
    aperture_m: float = 0.0  # max pairwise distance
    is_calibrated: bool = False
    calibrated_at: str = ""  # YYYYMMDDTHHMMSS
    source: str = ""  # "chirp", "opportunistic", "manual", "clap"

    def __post_init__(self) -> None:
        if self.mic_count == 0:
            self.mic_count = self.mic_positions.shape[0]
        if self.aperture_m == 0.0 and self.mic_count >= 2:
            dists = self.pairwise_distances()
            self.aperture_m = float(np.max(dists))

    def pairwise_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances between all mics.

        Returns: (N, N) symmetric distance matrix in meters.
        """
        diff = self.mic_positions[:, np.newaxis, :] - self.mic_positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def centroid(self) -> np.ndarray:
        """Geometric center of the array. Shape: (2,) or (3,)."""
        return np.mean(self.mic_positions, axis=0)

    def angular_resolution_at_freq(self, freq_hz: float, speed_of_sound: float = 343.0) -> float:
        """Estimate angular resolution at a given frequency.

        Based on the Rayleigh criterion: θ ≈ λ / D where λ is wavelength
        and D is array aperture. Returns resolution in degrees.

        At 1kHz with 2m aperture: ~10° resolution.
        At 4kHz with 2m aperture: ~2.5° resolution.
        Below 500Hz with <2m aperture: effectively omnidirectional.

        Reference: research/deliverables/02-spatial-acoustics.md RQ3
        """
        if self.aperture_m < 1e-6 or freq_hz < 1e-6:
            return 180.0  # no spatial discrimination
        wavelength = speed_of_sound / freq_hz
        return float(np.degrees(wavelength / self.aperture_m))


# ── TDOA ──────────────────────────────────────────────────────────────

@dataclass
class TDOAPair:
    """Time-difference-of-arrival estimate for one microphone pair.

    At 16kHz, 1 sample = 62.5μs ≈ 2.1cm at speed of sound.
    Sub-sample interpolation (parabolic or sinc) can improve to ~1cm.

    Reference: research/deliverables/02-spatial-acoustics.md RQ1
    """

    mic_a: int  # index into ArrayGeometry.mic_positions
    mic_b: int
    tdoa_samples: float  # fractional samples (sub-sample interpolation)
    tdoa_seconds: float  # tdoa_samples / sample_rate
    confidence: float  # 0-1, based on GCC-PHAT peak sharpness


# ── calibration ───────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Output of a calibration procedure (chirp or clap).

    Contains the estimated geometry plus quality metrics.
    The residual_error indicates how well the TDOA pairs fit the
    estimated geometry — lower is better. Units: seconds.

    Reference: research/deliverables/02-spatial-acoustics.md §Calibration
    """

    geometry: ArrayGeometry
    tdoa_pairs: list[TDOAPair] = field(default_factory=list)
    residual_error: float = 0.0  # RMS TDOA residual in seconds
    chirp_count: int = 0  # number of chirps averaged
    timestamp: str = ""  # YYYYMMDDTHHMMSS


# ── spatial descriptors ───────────────────────────────────────────────

@dataclass
class SpatialDescriptor:
    """Per-time-frequency-bin spatial descriptor.

    The extensible atom of spatial information. Currently carries DOA
    and confidence; future algorithms may add direct-to-reverberant ratio,
    source width estimates, coherence measures, etc.

    The format is deliberately open to expansion — downstream consumers
    should access fields by name, not position, so new fields don't break
    existing code.

    Reference: research/specs/spatial-architecture.md §2.1 + §5
    """

    doa_azimuth: float  # degrees, 0-360
    doa_elevation: float | None = None  # degrees, -90 to 90 (None if 2D only)
    confidence: float = 0.0  # 0-1
    direct_to_reverberant_ratio: float | None = None  # dB (future)


@dataclass
class SpatialFrame:
    """Primary output of the Spatial Front-End (§2.1).

    Consumed by both SpatialDiarizer (§2.2) and AudioEnhancer (§2.3).
    This is the critical shared contract — changes here cascade everywhere.

    Contains per-time-frequency-bin spatial information: DOA estimates,
    spatial covariance matrices, and confidence maps. The covariance
    matrices enable MVDR beamforming in the enhancement path; the DOA
    map enables spatial clustering in the diarization path.

    Reference: research/specs/spatial-architecture.md §2.1, §4
    """

    timestamp: float  # monotonic time of frame start
    sample_rate: int  # Hz (typically 16000)
    fft_size: int  # STFT window size (typically 1024)
    hop_size: int  # STFT hop size (typically 256)

    # Per-TF-bin spatial data
    doa_map: np.ndarray  # (time_bins, freq_bins) — azimuth in degrees
    confidence_map: np.ndarray  # (time_bins, freq_bins) — 0-1
    covariance: np.ndarray | None = None  # (freq_bins, N_mics, N_mics) — complex

    # Detected sources (from SRP-PHAT)
    source_locations: list[SpeakerLocation] = field(default_factory=list)

    # Extensible descriptor array (future: additional per-bin features)
    descriptors: np.ndarray | None = None  # (time_bins, freq_bins, descriptor_dim)

    def n_time_bins(self) -> int:
        """Number of time bins in this frame."""
        return self.doa_map.shape[0]

    def n_freq_bins(self) -> int:
        """Number of frequency bins in this frame."""
        return self.doa_map.shape[1]

    def freq_to_bin(self, freq_hz: float) -> int:
        """Convert a frequency in Hz to the nearest bin index."""
        bin_width = self.sample_rate / self.fft_size
        return int(round(freq_hz / bin_width))


# ── speaker location ─────────────────────────────────────────────────

@dataclass
class SpeakerLocation:
    """Estimated position of a detected speaker.

    Produced by SRP-PHAT localization in the front-end, consumed by
    the diarization fusion layer and the UI (DOA arrows).

    Reference: research/deliverables/02-spatial-acoustics.md RQ3
    """

    speaker_id: int
    azimuth_deg: float  # 0-360
    elevation_deg: float | None = None  # -90 to 90 (None if 2D)
    confidence: float = 0.0  # 0-1
    last_updated: float = 0.0  # monotonic timestamp


# ── fused diarization output ──────────────────────────────────────────

@dataclass
class FusedSegment:
    """Speaker segment from dual-path fusion (§2.2).

    Carries evidence from both spatial and embedding-based diarization,
    plus the fused result. Consumed by identity bridging (§2.4) and
    retention (§2.5).

    The three confidence scores allow downstream consumers to understand
    which analysis path dominated for this segment — useful for quality
    assessment and debugging.

    Reference: research/specs/spatial-architecture.md §2.2
    """

    start_sample: int
    end_sample: int
    speaker_id: int  # fused speaker ID (may be updated by identity bridge)
    spatial_cluster_id: int  # ephemeral, per-session
    embedding_speaker_id: int  # from CAM++ clustering

    fused_confidence: float  # 0-1, combined
    spatial_confidence: float  # 0-1, spatial path contribution
    embedding_confidence: float  # 0-1, embedding path contribution

    doa: SpeakerLocation | None = None  # spatial location if available

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration_seconds(self, sample_rate: int = 16000) -> float:
        return self.duration_samples / sample_rate


# ── enhanced audio output ─────────────────────────────────────────────

@dataclass
class EnhancedAudio:
    """Output of the audio enhancement module (§2.3).

    Contains the beamformed/dereverberated audio plus metadata about
    how it was produced — enabling future reprocessing decisions.

    Reference: research/specs/spatial-architecture.md §2.3
    """

    audio: np.ndarray  # float32, mono, sample_rate Hz
    sample_rate: int
    method: str  # "delay_and_sum", "mvdr", "mvdr_wpe"
    snr_estimate: float | None = None  # dB
    spatial_scene: list[SpeakerLocation] = field(default_factory=list)
    dereverberated: bool = False
    enhancement_params: dict[str, Any] = field(default_factory=dict)


# ── retention artifacts ───────────────────────────────────────────────

@dataclass
class RetentionArtifact:
    """Metadata for a persisted artifact (§2.5).

    Does not contain the data itself — the data is on disk. This tracks
    what was stored, when, under what access controls, and whether it's
    been compressed.

    Reference: research/specs/spatial-architecture.md §2.5
    """

    artifact_type: str  # "raw_array", "enhanced_audio", "diarization_meta", etc.
    session_id: str
    created_at: str  # YYYYMMDDTHHMMSS
    access_tier: str  # "sensitive", "standard"
    storage_path: str  # relative path within retention directory
    compressed: bool = False
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
