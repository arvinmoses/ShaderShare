//! `sculpt-cli rock [out.png] [seed]`: build a rock from generator layers and render each stage.
//!
//! A sphere is the clay. Tilted boxes pile up the main masses, spheres round and chip them, and a last noise pass
//! roughens the surface. The stages are rendered side by side so each layer's contribution is visible.

use std::path::Path;

use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::mesh::PolyMesh;
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

use crate::render::{View, render};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Push every vertex along its normal by a noise value, around zero.
fn roughen(doc: &Document, amplitude: f32, scale: f32, seed: u32) -> PolyMesh {
    let noise = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale, octaves: 5, seed, ..Default::default() });
    let positions = doc.positions().iter().zip(doc.normals()).map(|(&p, &n)| p + n * (noise.sample(p) - 0.5) * 2.0 * amplitude).collect();
    PolyMesh { positions, faces: doc.faces().to_vec() }
}

/// Add the generator, then bake its result into a fresh document, so the next layer scatters on this surface.
fn bake_into(doc: &mut Document, name: &str, spec: GeneratorSpec) -> Res<Document> {
    doc.insert_generator(name, Placement::Top, spec)?;
    stage(doc)
}

/// Taubin smoothing: alternating shrink and inflate passes round sharp edges without shrinking the whole shape.
fn taubin(doc: &Document, iterations: usize) -> PolyMesh {
    let topo = doc.topology();
    let mut p = doc.positions().to_vec();
    let mut next = p.clone();
    for _ in 0..iterations {
        for factor in [0.5_f32, -0.53] {
            for (v, out) in next.iter_mut().enumerate() {
                let ring = topo.vert_verts.row(v);
                let mean = ring.iter().map(|&n| p[n as usize]).sum::<glam::Vec3>() / ring.len().max(1) as f32;
                *out = p[v] + (mean - p[v]) * factor;
            }
            std::mem::swap(&mut p, &mut next);
        }
    }
    PolyMesh { positions: p, faces: doc.faces().to_vec() }
}

fn stage_from(mesh: PolyMesh) -> Res<Document> {
    Ok(Document::from_mesh(mesh)?)
}

fn stage(doc: &Document) -> Res<Document> {
    Ok(Document::from_mesh(PolyMesh { positions: doc.positions().to_vec(), faces: doc.faces().to_vec() })?)
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("rock.png", |s| s.as_str()));
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let level: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    // Which layers to build, e.g. "slabs,smooth,chips,grain".
    let only = args.get(3).map_or("slabs,smooth,chips,grain", |s| s.as_str()).to_string();
    let on = |name: &str| only.split(',').any(|n| n == name);
    let mut doc = Document::from_mesh(quad_sphere(level, 1.0))?;
    let mut shots = vec![stage(&doc)?];

    // Facets: big boxes tilted at random cut planar faces into the sphere. Sunk 0.3 of their size deep, each leaves a
    // flat face; spacing keeps them from piling up in one place.
    let carve = std::env::var("ROCK_ADD").is_err();
    let slabs = GeneratorSpec {
        primitive: Primitive::Box,
        op: if carve { BoolOp::Subtract } else { BoolOp::Add },
        count: 16,
        seed,
        size: 0.6,
        lift: if carve { 0.6 } else { -0.1 },
        random_spin: true,
        min_spacing: 0.75,
        size_var: 0.3,
        stretch_var: 0.3,
        tilt: 89.0,
        lift_var: 0.15,
        ..GeneratorSpec::default()
    };
    if on("slabs") {
        doc = bake_into(&mut doc, "Slabs", slabs)?;
        shots.push(stage(&doc)?);
    }

    // Take the edges off: a volume-keeping smooth rounds the box edges into worn ridges.
    if on("smooth") {
        doc = stage_from(taubin(&doc, 40))?;
        shots.push(stage(&doc)?);
    }

    // Chips: small spheres bitten out.
    let chips = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 120, seed: seed + 2, size: 0.1, lift: 0.6, size_var: 0.6, stretch_var: 0.4, ..GeneratorSpec::default() };
    if on("chips") {
        doc = bake_into(&mut doc, "Chips", chips)?;
        shots.push(stage(&doc)?);
    }

    // Grain: noise pushed along the normals.
    if on("grain") {
        shots.push(Document::from_mesh(roughen(&doc, 0.025, 6.0, seed as u32))?);
    }

    let views = [View { yaw: 0.6, pitch: 0.35, overlay: None }];
    // One row per stage: render each as its own strip and join them vertically by writing separate files.
    for (i, d) in shots.iter().enumerate() {
        let path = out.with_file_name(format!("{}_{i}.png", out.file_stem().and_then(|s| s.to_str()).unwrap_or("rock")));
        render(d, &views, 420, &path)?;
        println!("stage {i}: {} faces -> {}", d.face_count(), path.display());
    }
    Ok(())
}
