# Detail Spec: Array Geometry (§3)

> From head spec: "Defines the physical mic array configuration. Pins the spatial resolution ceiling, which constrains what §2.1 can deliver, which constrains what §2.2 and §2.3 can achieve."

Separated from module specs because it has a different change cadence (hardware/deployment decisions vs software pipeline updates) and a different audience.

## Open Questions to Resolve

- **Target deployment environments:** Meeting rooms (3-15 people), home offices, classrooms, conference halls? Each has different spatial separation characteristics and RT60 profiles.
- **Per-form-factor angular resolution:** What resolution is achievable with 3/4/5/6/7 phones at typical table spacing? At what point do diminishing returns set in?
- **Minimum speaker separation assumptions:** How close can two speakers be before spatial discrimination fails? Research says >25-50cm at 1-2m from array center, but this needs validation.
- **3D vs 2D:** Are phones always coplanar (on a table)? What about handheld or mounted on stands? Does elevation matter for speech separation?

## Interface Contracts to Define

**Produces:** `ArrayGeometry` dataclass
- Consumed by: §2.1 SpatialFrontEnd (steering vectors, resolution bounds)

**Configured by:** CalibrationManager (chirp procedure estimates geometry)

## Sections

### Target Environments
<!-- Meeting room types, expected device counts, typical spacing -->

### Mic Configurations
<!-- Per-device-count optimal layouts (if controllable) vs random placement -->

### Angular Resolution Bounds
<!-- Theoretical limits and empirical expectations per configuration -->

### Aperture Calculations
<!-- How aperture relates to resolution at speech frequencies -->

### Calibration Requirements
<!-- What the calibration procedure needs from the geometry spec -->
