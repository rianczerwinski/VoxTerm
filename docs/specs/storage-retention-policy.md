# Detail Spec: Storage / Retention Policy (§2.5)

> From head spec: "Cross-cutting architectural boundary. Specifies which outputs are persisted, in what form, under what access controls. Raw array data enables both isolation and erasure — complementary subspace projections. Managed as data governance, not signal-level mitigation."

## Open Questions to Resolve

- **Access tier definitions:** What does "sensitive" mean in practice? Who can access raw array data? Is it the app user only, or are there multi-user scenarios?
- **Retention durations:** Current defaults (7d raw, 30d enhanced, 1y metadata) — are these appropriate? Should they be user-configurable?
- **Deletion semantics:** When an artifact is "deleted," is it cryptographically erased or just unlinked? For sensitive tier, secure deletion may be required.
- **Audit requirements:** What audit trail is needed for sensitive artifact access? Logging granularity, retention of audit logs themselves.
- **Compression strategy:** When to compress (immediately after capture? after enrichment? on retention window transition)?

## Interface Contracts to Define

**Consumes:** Outputs from all other modules
- Raw N-channel audio + ArrayGeometry
- EnhancedAudio from §2.3
- FusedSegments from §2.2/§2.4
- Processing parameters from §2.1

**Produces:** RetentionArtifact metadata + persisted files on disk

## Sections

### Retained Artifacts
<!-- Complete list with format, size estimates, access tier -->

### Access Tiers
<!-- "sensitive" vs "standard" — definitions, permissions, enforcement -->

### Retention Windows
<!-- Per-tier defaults, user configurability, enforcement schedule -->

### Deletion Semantics
<!-- Unlink vs secure erase, cascade (delete raw → keep enhanced?), audit -->

### Compression Strategy
<!-- When and how to compress, Opus for audio, gzip for metadata -->

### Governance Surface
<!-- Erasure-isolation duality, what raw array access enables, risk model -->

### Audit Requirements
<!-- What's logged, retention of audit logs, access patterns to monitor -->
