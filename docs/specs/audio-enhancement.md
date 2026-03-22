# Detail Spec: Audio Enhancement (§2.3)

> From head spec: "Produces the retained audio artifact from raw array data, informed by spatial descriptors. Beamforming, dereverberation, noise suppression. Parallel, not serial — runs alongside diarization."

## Open Questions to Resolve

- **Output metadata:** What metadata accompanies the enhanced audio? SNR estimates, spatial scene descriptors, enhancement parameters used — to support future reprocessing decisions.
- **Live vs enrichment boundary:** When does the live path run and when does enrichment kick in? Is enrichment always deferred, or can it run opportunistically during speech pauses?
- **Per-speaker isolation:** The current spec produces enhanced mono (mixture of all targets). Should per-speaker isolated streams be a first-class output? This would benefit both transcription and cross-session identity.
- **Beamforming fallback chain:** If MVDR fails (ill-conditioned covariance), fall back to DAS? Or signal failure?

## Interface Contracts to Define

**Consumes:**
- Raw N-channel audio
- `SpatialFrame` from §2.1

**Produces:**
- `EnhancedAudio` with method, SNR, spatial scene metadata
- Consumed by: §2.4 SpatialIdentityBridge (for embedding extraction), §2.5 RetentionManager

## Sections

### Live Path (Delay-and-Sum)
<!-- Algorithm, compute budget, steering, fractional delay implementation -->

### Enrichment Path (MVDR + WPE)
<!-- MVDR derivation, regularization, WPE configuration, pipeline order -->

### Dereverberation
<!-- nara_wpe integration, taps/delay configuration, early vs late reflection boundary -->

### Noise Suppression
<!-- Post-beamforming spectral subtraction or Wiener filtering -->

### Output Format (EnhancedAudio)
<!-- Field specification, metadata requirements -->

### Metadata
<!-- What's stored alongside enhanced audio for future reprocessing -->
