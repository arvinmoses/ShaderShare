//! `sculpt-cli rock-layers [out.png] [seed] [voxel]`: the rock as real generator layers, solidified.
//!
//! A lumpy ellipsoid mesh carries three generator layers (big cuts, medium cuts, chips). `Document::solidify` turns the
//! stack into one field, scattering each layer on the surface below it, and extracts a clean manifold quad mesh.

use std::path::Path;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::document::{Grain, SolidSettings};
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::remesh::non_manifold_edges;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

use crate::render::{View, render};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub(crate) fn run(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("rock_layers.png", |s| s.as_str()));
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let voxel: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.015);

    // The clay: a lumpy ellipsoid.
    let lumps = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 1.1, octaves: 3, seed: seed as u32, ..Default::default() });
    let mut mesh = quad_sphere(6, 1.0);
    for p in &mut mesh.positions {
        let dir = *p;
        *p = dir * Vec3::new(1.0, 0.72, 0.88) * (1.0 + (lumps.sample(dir) - 0.5) * 0.25);
    }
    let mut doc = Document::from_mesh(mesh)?;

    let spec = |primitive, count, size, lift, spacing, tilt, k: u64, blend| GeneratorSpec {
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
        blend,
        ..GeneratorSpec::default()
    };
    doc.insert_generator("Big cuts", Placement::Top, spec(Primitive::Box, 14, 0.6, 0.8, 0.75, 55.0, 0, 0.12))?;
    doc.insert_generator("Medium cuts", Placement::Top, spec(Primitive::Box, 28, 0.2, 0.9, 0.35, 50.0, 1, 0.25))?;
    doc.insert_generator("Chips", Placement::Top, spec(Primitive::Sphere, 30, 0.06, 0.65, 0.15, 0.0, 2, 0.5))?;

    let t = Instant::now();
    let (mesh, stats) = doc.solidify(&SolidSettings { voxel, grain: Some(Grain { amplitude: 0.03, scale: 9.0, seed: seed as u32 + 9 }) })?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let bad = non_manifold_edges(&mesh.faces);
    println!(
        "solidify: {} layers, {} quads in {ms:.0} ms (scatter {:.0}, extract {:.0}); {} of {} samples evaluated; {bad} non-manifold edges",
        stats.layers, stats.quads, stats.scatter_ms, stats.extract_ms, stats.evaluated, stats.total
    );
    let views = [View { yaw: 0.6, pitch: 0.35, overlay: None }, View { yaw: 2.4, pitch: 0.2, overlay: None }];
    render(&Document::from_mesh(mesh)?, &views, 520, out)?;
    Ok(())
}
