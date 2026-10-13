# Generator Boolean layer: design

Status: proposal. The boolean strategy (displacement, voxel, or hybrid) is decided
by the measurements in `spike_results.md`. Everything else here is independent of it.

## Goal

A layer type that spawns instances of a primitive at generated points, orients and
scales them from data on the underlying surface, and applies a boolean
(add / subtract / intersect) against the layers beneath it. Edge damage is the
motivating case: place cutters along sharp edges, aligned to the edge normal, sized
from edge length, and subtract them.

It must be interactive. Dragging a parameter must not trigger a full rebuild.

## Pipeline and caching

Four stages. Each stage caches its output and is invalidated only by its own
parameters or by the stage before it. A slider edit re-runs its stage and the stages
after it, never the ones before.

```
 Primitive ─┐
            ├─▶ Distribution ─▶ Attributes ─▶ Instances ─▶ Boolean ─▶ delta
 Mesh data ─┘     (points)      (rotation,     (transforms)  (per leaf)
  masks, bakes                    scale, ...)
```

| Stage | Output | Invalidated by | Cost target |
|---|---|---|---|
| Distribution | points with surface frame (position, normal, tangent, source attributes) | seed, count, mode, density mask, mesh topology | rebuilt rarely, may take ~100 ms |
| Attributes | instance transforms | any mapping rule, primitive size | per-instance, microseconds, parallel |
| Spatial index | uniform grid over instance bounds | any transform change | milliseconds |
| Boolean | per-vertex delta, only in leaves touched by an instance | transform or lower-layer change in that leaf | per dirty leaf |

Because the engine already works in spatial leaves, the boolean runs only on leaves
that overlap an instance and only on leaves whose inputs changed. A change in a lower
layer dirties the same leaves, so the generator stays correct under sculpting.

## Distribution

Points are drawn from one of these sources:

- **Surface**: area-weighted samples over faces.
- **Vertices**, **face centres**: one point each, optionally thinned to N.
- **Edges**: sharp edges (dihedral-angle threshold) and curvature ridges, sampled by
  arc length. Each point carries the edge tangent, the edge normal (bisector of the
  two face normals) and the length of the edge chain it belongs to.
- **Grid / ordered**: lattice in a chosen frame, or fixed spacing along edge chains.
- **Chaotic**: seeded random with an optional minimum spacing (Poisson-disk style).

### Density mask

A per-vertex 0..1 field (any existing mask stack: painted, noise, curvature, cavity,
AO, thickness, the new Edge bake). It redistributes a fixed count. White pulls all
instances in, black gets none. It is importance sampling (per-face CDF), not
rejection, so the count does not fall when the mask is mostly black. Users paint the
mask themselves to force instances into a region.

This is separate from the layer mask, which scales how strongly the boolean affects
the surface.

## Attribute mapping

A short ordered table of rules. Each rule reads one source attribute, optionally
remaps it (range, curve, unit value), and writes one instance property.

Sources: surface normal, edge normal, edge tangent, edge-chain length, local edge
length, curvature, cavity, AO, thickness, density-mask value, noise, per-instance
random, instance index.

Targets: align an axis to a direction, roll around that axis, scale X / Y / Z or
uniform, height, offset along the normal, density weight.

Example: edge-chain length divided by a unit length, written to instance height. A
fixed set of rows in the Properties panel, not a node graph. A graph can be added
later behind the same data structure.

## Boolean

See `spike_results.md` for the measured trade-off and `spike5_gpu_sdf.md` for the GPU version of B (commit-only, not live). The options:

- **Displacement boolean (A).** Evaluate each instance's signed distance at every
  vertex near it and move vertices onto the primitive surface. Same mesh, same
  topology, so the layer, mask and undo model keep working. Edge crispness is limited
  by mesh density, and union cannot make overhangs.
- **Voxel boolean (B).** Combine the surface and the instances as signed distance
  fields in sparse bricks and extract a new surface. True boolean, crisp edges,
  arbitrary shape. Changes topology, so per-vertex layers, masks and undo do not map
  across without a transfer step.
- **Hybrid.** Live preview with A, then an explicit Apply that bakes with B and
  resamples layers onto the new surface.

Whichever is chosen, the layer stores parameters only. The result is procedural and
recomputed after subdivision, so detail follows the mesh density.

## Layer model integration

- The generator layer is a function layer: its delta depends on the positions below
  it, so it is evaluated in stack order, not summed independently.
- It supports the existing layer mask (strength per vertex), opacity, visibility,
  lock (editing parameters), folders and ordering from the layer UI pass.
- Project format: a new layer kind with a parameter block in `project.json`; no blob
  unless a custom OBJ primitive is embedded (stored as a blob). Old projects load
  unchanged.
- Undo: parameter edits are small commands. No geometry snapshot is needed because
  the result is procedural.

## UI

- Row badge and icon for generator layers; same row anatomy as other layers.
- Properties sections: Primitive, Distribution, Attributes (mapping rows), Boolean.
- A viewport overlay draws the instance gizmos while the layer is selected, so the
  user sees placement and orientation before the boolean runs.
- Presets: "Edge damage", "Surface pitting", "Rivets" (add), "Panel cuts".

## Risks

1. Boolean cost at 6M faces. Answered by the spike.
2. Edge detection on organic sculpts: dihedral thresholds give noisy edges. Use the
   curvature-ridge variant with hysteresis, and expose the threshold.
3. Union and intersect on a displacement-only boolean are approximations.
4. Merge conflicts: this touches `document.rs` composite and `layers.rs`, which the
   layer UI pass also changes (folders). Keep the generator in its own module and
   add one hook for function layers.
