//! `sculpt-cli rock-tiers [out.png] [seed] [voxel]`: a rock as one field with three tiers of carving.
//!
//! A lumpy ellipsoid is cut by big tilted boxes, then by medium boxes, then by small chips. Each tier is scattered on the
//! surface the tiers before it produced (a coarse extraction supplies that surface), so the small cuts land on the big
//! faces instead of where the sphere used to be. Everything blends with a smooth maximum; grain is noise in the field.

use std::path::Path;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::adf::uniform_surface_nets;
use sculpt_core::generator::remesh::{RemeshParams, non_manifold_edges, remesh};
use sculpt_core::{Document, mesh::PolyMesh};
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::{BoolOp, InstanceSet, Primitive};
use sculpt_core::geom::Aabb;
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};

use crate::render::{View, render};
use crate::rock_field::{bad_edges, nearest, scatter, smax, to_document};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Tier {
    set: InstanceSet,
    count: usize,
    blend: f32,
}

/// The field after the first `upto` tiers: the lumpy ellipsoid with each tier's cutters subtracted in turn.
fn field(base: &Noise, tiers: &[Tier], upto: usize, p: Vec3) -> f32 {
    let q = p / Vec3::new(1.0, 0.72, 0.88);
    let mut f = (q.length() - 1.0) * 0.72 + (base.sample(p) - 0.5) * 0.18;
    for t in &tiers[..upto] {
        f = smax(f, -nearest(&t.set, t.count, p), t.blend);
    }
    f
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("rock_tiers.png", |s| s.as_str()));
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let voxel: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.015);
    let bounds = Aabb { min: Vec3::splat(-1.5), max: Vec3::splat(1.5) };
    let base = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 1.1, octaves: 3, seed: seed as u32, ..Default::default() });
    let grain = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 9.0, octaves: 4, seed: seed as u32 + 9, ..Default::default() });

    let spec = |primitive, count, size, lift, spacing, tilt, k: u64| GeneratorSpec {
        primitive,
        op: BoolOp::Subtract,
        count,
        seed: seed + k,
        size,
        lift,
        min_spacing: spacing,
        size_var: 0.35,
        stretch_var: 0.25,
        tilt,
        lift_var: 0.12,
        ..GeneratorSpec::default()
    };
    // (spec, smooth-blend width)
    let plan = [
        (spec(Primitive::Box, 14, 0.6, 0.8, 0.75, 55.0, 0), 0.07),
        (spec(Primitive::Box, 28, 0.2, 0.9, 0.35, 50.0, 1), 0.05),
        (spec(Primitive::Sphere, 30, 0.06, 0.65, 0.15, 0.0, 2), 0.03),
    ];

    let mut tiers: Vec<Tier> = Vec::new();
    let views = [View { yaw: 0.6, pitch: 0.35, overlay: None }, View { yaw: 2.4, pitch: 0.2, overlay: None }];
    for (i, (s, blend)) in plan.iter().enumerate() {
        // The surface so far, coarsely, is what this tier scatters on.
        let coarse = uniform_surface_nets(&|p| field(&base, &tiers, tiers.len(), p), bounds, 0.04);
        let mut surface = to_document(&coarse)?;
        let instances = scatter(&mut surface, "tier", s.clone())?;
        println!("tier {i}: {} cutters scattered on a {}-triangle surface", instances.len(), coarse.tris.len());
        let count = instances.len();
        tiers.push(Tier { set: InstanceSet::bake_only(instances, 0.0), count, blend: *blend });
    }

    let full = |p: Vec3| field(&base, &tiers, tiers.len(), p) + (grain.sample(p) - 0.5) * 0.03;
    // The old extractor, for comparison.
    let t = Instant::now();
    let old = uniform_surface_nets(&full, bounds, voxel);
    println!("old surface nets: {} tris, {:.0} ms, {} non-manifold edges", old.tris.len(), t.elapsed().as_secs_f64() * 1e3, bad_edges(&old));
    // The remesher: manifold quads, sparse evaluation. The grain adds gradient, hence the Lipschitz bound.
    let t = Instant::now();
    let m = remesh(&full, bounds, &RemeshParams { voxel, lipschitz: 1.3 });
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("remesh: {} quads, {ms:.0} ms, {} non-manifold edges, {} of {} samples evaluated", m.quads.len(), non_manifold_edges(&m.quads), m.evaluated, m.total);
    render(&Document::from_mesh(m.into_poly_mesh())?, &views, 520, out)?;
    Ok(())
}
