# Spatial Audio Erasure: Feasibility Analysis

What would it take to identify an arbitrary speaker and irrecoverably erase them from the audio stream, assuming VoxTerm captured via a 3D spatial mic array as its native representation?

## Why spatial makes identification easy

VoxTerm's current diarization pipeline (CAM++ embeddings → cosine similarity → clustering) exists *because* a single mic destroys all spatial information. The entire embedding-based approach compensates for the absence of a dimension that would make the problem trivial. With a mic array giving DOA (direction of arrival) per time-frequency bin, "which speaker is this?" reduces to "which spatial position is this energy coming from?" — a lookup, not a classification. Embeddings would still be needed for *cross-session identity* (is this the same person who was here Tuesday?), but within a session, spatial position is a nearly perfect discriminator for any speakers more than ~15° apart.

## The erasure problem — three layers

### Layer 1: The direct path

If the array has sufficient spatial resolution (enough mics for the angular separation between sources), a spatial null steered at the target speaker's position genuinely destroys the signal — not masking or attenuation, but a linear projection that removes the subspace. The target's direct-path energy is gone from the output in a way that's mathematically irrecoverable from the nulled signal alone. This part is clean.

### Layer 2: Reverberation bleeds the spatial footprint

In a real room, a speaker's voice doesn't arrive from one point — it arrives from everywhere, delayed and attenuated by wall reflections. The direct path might be 30°, but reverberant copies arrive from 150°, 210°, 340°... A null steered at 30° kills the direct sound but leaves the reflections, which carry the same speech content at lower SNR. For irrecoverable erasure, the speaker's full spatial footprint must be estimated — direct path plus all significant reflection paths — and all of them nulled.

This is where the fundamental **erasure-fidelity tradeoff** emerges: the more of the spatial field that gets nulled, the more collateral degradation to retained speakers whose reflections overlap with the target's. It's not a bug in the algorithm; it's a consequence of the physics. In a reverberant room, speakers' spatial footprints *overlap* in the reflection domain even when their direct paths are well-separated.

### Layer 3: Time-frequency coincidence

Even with perfect spatial processing, two speakers talking simultaneously at similar pitches create time-frequency bins where their energy is physically superimposed. A spatial filter can suppress one, but in bins where both speakers have significant energy at similar magnitudes, the suppression distorts the retained speaker's signal. The distortion *is* the trace of the erasure — an adversary with the original room impulse response and knowledge that erasure was performed could potentially estimate the erased content from the artifacts.

## What "irrecoverable" actually requires

For a cryptographic-grade guarantee (no adversary can reconstruct):

1. **Full spatial footprint estimation** — direct path + all reflections above some energy threshold. Requires either knowing the room impulse response (measurable but environment-specific) or estimating it online (hard, especially with multiple concurrent sources).

2. **Aggressive nulling with collateral acceptance** — null everything in the target's footprint, accept the quality degradation to other speakers. The output is provably free of the target but audibly impaired.

3. **T-F bin replacement, not just suppression** — in bins where the target is nulled, replace with synthesized room noise at the appropriate spatial position and spectral shape. This prevents an adversary from detecting *where* the erasure happened by looking for suspiciously quiet T-F regions (the "inpainting" approach, analogous to image forensics countermeasures).

Without (3), the erasure is irrecoverable in terms of *content* but detectable in terms of *having occurred*. Whether that matters depends on the threat model.

## Hardware constraints

All of this requires a mic array. Angular resolution scales with `lambda / D` where D is the array aperture — a 10cm array (phone-sized) gives ~30° resolution at speech frequencies, which separates speakers across a table but can't resolve two people on the same couch. A 1m array gives ~3° resolution, which handles most room geometries. The number of mics determines how many independent spatial nulls can be steered simultaneously — for N mics, there are N-1 degrees of freedom, so erasing one speaker in a room with K speakers requires N > K+1 at minimum (and more for reverb suppression).

For VoxTerm's current single-mic setup, none of this applies — spatial information can't be recovered after the fact from a mono recording.

## Privacy implication

If VoxTerm *did* capture spatially, the raw array data would need careful handling because it *enables* selective erasure — which means it also enables selective *surveillance* of individual speakers in a group conversation, even retroactively. The capability to erase and the capability to isolate are the same capability.
