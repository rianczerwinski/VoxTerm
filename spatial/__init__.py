"""Spatial audio processing for VoxTerm.

Five-module architecture for distributed microphone array processing:
  §2.1 SpatialFrontEnd   — GCC-PHAT TDOA, SRP-PHAT localization, TF-bin descriptors
  §2.2 SpatialDiarizer   — Dual-path fusion (spatial + embedding diarization)
  §2.3 AudioEnhancer     — Delay-and-sum (live) / MVDR+WPE (enrichment)
  §2.4 SpatialIdentityBridge — Cross-session speaker identity linking
  §2.5 RetentionManager  — Artifact persistence with tiered access control

Plus CalibrationManager for chirp-based geometry estimation.

Reference: research/specs/spatial-architecture.md
"""

from spatial.models import (
    ArrayGeometry,
    CalibrationResult,
    EnhancedAudio,
    FusedSegment,
    RetentionArtifact,
    SpatialDescriptor,
    SpatialFrame,
    SpeakerLocation,
    TDOAPair,
)
from spatial.frontend import SpatialFrontEnd
from spatial.diarization import SpatialDiarizer
from spatial.enhancement import AudioEnhancer
from spatial.identity import SpatialIdentityBridge
from spatial.retention import RetentionManager
from spatial.calibration import CalibrationManager

__all__ = [
    # Models
    "ArrayGeometry",
    "CalibrationResult",
    "EnhancedAudio",
    "FusedSegment",
    "RetentionArtifact",
    "SpatialDescriptor",
    "SpatialFrame",
    "SpeakerLocation",
    "TDOAPair",
    # Modules
    "SpatialFrontEnd",
    "SpatialDiarizer",
    "AudioEnhancer",
    "SpatialIdentityBridge",
    "RetentionManager",
    "CalibrationManager",
]
