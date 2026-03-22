# Detail Spec: Within-Session Diarization (§2.2)

> From head spec: "Fuses spatial descriptors with embedding-based clustering. Neither path subordinated. Continuous confidence-weighted fusion."

## Open Questions to Resolve

- **Fusion weighting strategy:** How to combine spatial and embedding confidence? Options range from simple confidence-weighted averaging to Bayesian fusion to learned combination. Start simple, leave room for novel approaches.
- **Dominance conditions:** When should one path effectively override the other? The threshold (0.8 default) is a starting point — empirical tuning required.
- **Spatial clustering algorithm:** Simple angular histogram with peak finding? DBSCAN on circular coordinates? Something else?
- **Temporal alignment:** Spatial and embedding paths may produce segments at different granularity. How to align them for fusion?

## Interface Contracts to Define

**Consumes:**
- `SpatialFrame` from §2.1 SpatialFrontEnd
- Embedding clustering output from existing DiarizationProxy (CAM++)

**Produces:**
- `list[FusedSegment]` with spatial, embedding, and fused confidence per segment
- Consumed by: §2.4 SpatialIdentityBridge, §2.5 RetentionManager

## Sections

### Dual-Path Architecture
<!-- How both paths run simultaneously, data flow diagram -->

### Spatial Path
<!-- DOA clustering algorithm, angular resolution constraints -->

### Embedding Path
<!-- DiarizationProxy integration, what API surface is consumed -->

### Fusion Strategy
<!-- Confidence combination function, weight dynamics, edge cases -->

### Confidence Combination
<!-- How spatial_confidence and embedding_confidence produce fused_confidence -->

### Degradation Modes
<!-- 4+ devices, 3 devices, 2 devices, 1 device behavior -->

### Integration with DiarizationProxy
<!-- Constructor injection, threading model, lifecycle coordination -->
