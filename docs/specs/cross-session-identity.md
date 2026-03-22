# Detail Spec: Cross-Session Identity (§2.4)

> From head spec: "Links spatial-cluster IDs to persistent speaker identities. Audio source is configurable (bracketed decision). Embedding model and identity database architecturally unchanged."

## Open Questions to Resolve

- **Audio source evaluation:** Compare enhanced mono, beamformed-toward-speaker, and raw single-mic for embedding quality. This is an empirical question — each option has tradeoffs (enhancement artifacts vs SNR vs training distribution match). The interface accepts any source; this spec evaluates which works best.
- **Cache invalidation:** When should the spatial-cluster → persistent-identity mapping be invalidated? Speaker movement changes spatial cluster assignment; the identity should follow.
- **Confidence propagation:** How does the fused confidence from §2.2 affect identity matching thresholds? Higher confidence → more aggressive matching?

## Interface Contracts to Define

**Consumes:**
- `list[FusedSegment]` from §2.2
- Audio from configurable source (dict[str, ndarray])

**Produces:**
- `list[FusedSegment]` with persistent speaker_id attached
- Speaker model updates to SpeakerStore
- Consumed by: §2.5 RetentionManager

## Sections

### Bridging Interface
<!-- How ephemeral cluster IDs map to persistent UUIDs -->

### Audio Source Selection
<!-- Evaluation of enhanced mono vs beamformed vs raw single-mic -->

### Mapping Lifecycle
<!-- Creation, caching, invalidation, session boundaries -->

### Integration with SpeakerStore
<!-- Which SpeakerStore methods are called, threading model -->
