# Detail Spec: Spatial Front-End (§2.1)

> From head spec: "Transforms raw multichannel audio into spatial descriptors consumable by downstream modules. Pure signal processing; no learned parameters; deterministic given array geometry configuration."

## Open Questions to Resolve

- **Temporal resolution:** What hop size / frame rate for spatial descriptors? 16ms (256 samples at 16kHz) is the current default — is this sufficient for real-time diarization and enhancement?
- **Frequency resolution:** FFT size 1024 gives ~15.6 Hz bins. Is this fine enough for spatial processing at speech frequencies?
- **Descriptor format extensibility:** Currently DOA + covariance + confidence. Future: direct-to-reverberant ratio, source width, coherence. How should the SpatialFrame accommodate additions without breaking consumers?
- **Confidence semantics:** Eigenvalue ratio is the default diffuseness measure. Are there better options for the ad-hoc distributed array use case?
- **Latency budget:** What is the maximum acceptable latency for the live path? The front-end feeds both diarization and enhancement — it's on the critical path.

## Interface Contracts to Define

**Produces:** `SpatialFrame` (see `spatial/models.py`)
- Consumed by: §2.2 SpatialDiarizer, §2.3 AudioEnhancer

**Consumes:** `ArrayGeometry` (from §3 via CalibrationManager)
- Raw N-channel audio stream (from audio capture pipeline)

## Sections

### Algorithm Pipeline
<!-- GCC-PHAT → SRP-PHAT → covariance → confidence → frame assembly -->

### Input Format
<!-- N-channel audio: shape, dtype, sample rate, chunking -->

### Output Format (SpatialFrame)
<!-- Field-by-field specification with shapes, dtypes, semantics -->

### Confidence Semantics
<!-- Eigenvalue ratio definition, calibration, edge cases -->

### Latency Budget
<!-- Per-stage timing, total budget, profiling methodology -->

### Extensibility
<!-- How to add new per-TF-bin descriptors without breaking consumers -->

### Dependencies
<!-- pyroomacoustics, numpy, existing VoxTerm audio pipeline -->
