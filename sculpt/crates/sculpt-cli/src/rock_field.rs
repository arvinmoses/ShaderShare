//! `sculpt-cli rock-field [out.png] [seed] [voxel]`: the rock again, as a field instead of stacked displacements.
//!
//! The whole stack (sphere, carving boxes, chips, grain) is one signed distance field with smooth blends, extracted
//! once with surface nets. There is no inside geometry and nothing to fold, so layers can stack freely. This is the
//! measurement spike for a Remesh checkpoint layer: it reports how long each extraction takes.

use std::path::Path;
use std::time::Instant;

use glam::Vec3;
use sculpt_core::generator::adf::{AdfMesh, AdfParams, BooleanAdf, Frame, uniform_surface_nets};
use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::{BoolOp, Instance, InstanceSet, Primitive};
use sculpt_core::geom::Aabb;
use sculpt_core::mesh::{NO_VERT, PolyMesh};
use sculpt_core::noise::{Noise, NoiseKind, NoiseParams};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

use crate::render::{View, render};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Smooth maximum: like `max(a, b)` but blended over a band of width `k` (polynomial, C1).
pub(crate) fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Distance to the nearest of `set`'s instances (negative inside one). Brute force: a spike, not the feature.
pub(crate) fn nearest(set: &InstanceSet, count: usize, p: Vec3) -> f32 {
    (0..count as u32).map(|i| set.sdf(i, p)).fold(f32::INFINITY, f32::min)
}

pub(crate) fn scatter(doc: &mut Document, name: &str, spec: GeneratorSpec) -> Res<Vec<Instance>> {
    let id = doc.insert_generator(name, Placement::Top, spec)?;
    let instances: Vec<Instance> = doc.layer(id).and_then(|l| l.generator_cache()).map(|c| c.instances().to_vec()).unwrap_or_default();
    Ok(instances)
}

/// Edges used by other than two triangles: 0 means the surface is a closed manifold.
pub(crate) fn bad_edges(mesh: &AdfMesh) -> usize {
    let mut uses = std::collections::HashMap::new();
    for t in &mesh.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *uses.entry((a.min(b), a.max(b))).or_insert(0u32) += 1;
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

pub(crate) fn to_document(mesh: &AdfMesh) -> Res<Document> {
    let faces = mesh.tris.iter().map(|t| [t[0], t[1], t[2], NO_VERT]).collect();
    Ok(Document::from_mesh(PolyMesh { positions: mesh.positions.clone(), faces })?)
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let out = Path::new(args.first().map_or("rock_field.png", |s| s.as_str()));
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let voxel: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let bounds = Aabb { min: Vec3::splat(-1.3), max: Vec3::splat(1.3) };
    let radius = 1.0_f32;

    // Facets: tilted box cutters scattered on the sphere.
    let mut sphere = Document::from_mesh(quad_sphere(6, radius))?;
    let facets = GeneratorSpec { primitive: Primitive::Box, op: BoolOp::Subtract, count: 22, seed, size: 0.5, lift: 0.75, min_spacing: 0.55, size_var: 0.3, stretch_var: 0.2, tilt: 60.0, lift_var: 0.1, ..GeneratorSpec::default() };
    let facet_instances = scatter(&mut sphere, "Facets", facets)?;
    let facet_set = InstanceSet::bake_only(facet_instances.clone(), 0.0);
    let facet_count = facet_set.len();
    let carved = |p: Vec3| smax(p.length() - radius, -nearest(&facet_set, facet_count, p), 0.10);

    // Chips scatter on the carved surface, which is one coarse extraction away, so they land on the new faces.
    let t = Instant::now();
    let coarse = uniform_surface_nets(&carved, bounds, 0.05);
    println!("carve, coarse preview: {} tris in {:.0} ms", coarse.tris.len(), t.elapsed().as_secs_f64() * 1e3);
    let mut surface = to_document(&coarse)?;
    let chips = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 50, seed: seed + 2, size: 0.09, lift: 0.6, size_var: 0.6, stretch_var: 0.4, ..GeneratorSpec::default() };
    let chip_instances = scatter(&mut surface, "Chips", chips)?;
    let chip_set = InstanceSet::bake_only(chip_instances.clone(), 0.0);
    let chip_count = chip_set.len();

    let grain = Noise::new(&NoiseParams { kind: NoiseKind::Fbm, scale: 6.0, octaves: 5, seed: seed as u32, ..Default::default() });
    let field = |p: Vec3| smax(carved(p), -nearest(&chip_set, chip_count, p), 0.04) + (grain.sample(p) - 0.5) * 0.05;

    let views = [View { yaw: 0.6, pitch: 0.35, overlay: None }];
    adf_rock(&facet_instances, &chip_instances, &grain, radius, bounds, voxel, &views, out)?;
    render(&to_document(&coarse)?, &views, 420, &out.with_file_name("rf_carved.png"))?;
    for vs in [voxel * 2.0, voxel] {
        let t = Instant::now();
        let mesh = uniform_surface_nets(&field, bounds, vs);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("full rock, voxel {vs}: {} tris, {} verts, {ms:.0} ms, {} non-manifold edges ({facet_count} facets, {chip_count} chips)", mesh.tris.len(), mesh.positions.len(), bad_edges(&mesh));
        let name = format!("rf_{}.png", (vs * 1000.0) as u32);
        render(&to_document(&mesh)?, &views, 420, &out.with_file_name(name))?;
    }
    Ok(())
}

/// The same rock through the adaptive field: sphere minus every cutter (hard min, as the ADF has no smooth blend yet),
/// extracted with dual contouring. Grain cannot live in an adaptive field, so it is pushed along the normals afterwards.
#[allow(clippy::too_many_arguments)]
fn adf_rock(facets: &[Instance], chips: &[Instance], grain: &Noise, radius: f32, bounds: Aabb, edge: f32, views: &[View], out: &Path) -> Res<()> {
    let cutters: Vec<Instance> = facets.iter().chain(chips).copied().collect();
    let mean_extent = cutters.iter().map(|i| i.aabb().extent().max_element()).sum::<f32>() / cutters.len() as f32;
    for tol in [1.0_f32, 0.5] {
        let params = AdfParams::new(edge, tol, BoolOp::Subtract);
        let frame = Frame::new(bounds, &params, mean_extent);
        let (mut adf, _) = BooleanAdf::new(frame, params, cutters.clone());
        let t = Instant::now();
        let times = adf.build(&|p: Vec3, _bound: f32| p.length() - radius);
        let built = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mesh = adf.extract(None);
        let extracted = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "ADF tol {tol}e (edge {edge}): {} tris, build {built:.0} ms (base {:.0}, adapt {:.0}), extract {extracted:.0} ms, {} non-manifold edges, {} quads missing a vertex of {}",
            mesh.tris.len(), times.base_ms, times.adapt_ms, bad_edges(&mesh), mesh.missing, mesh.quads
        );
        let plain = to_document(&mesh)?;
        let name = format!("adf_{}.png", (tol * 10.0) as u32);
        render(&plain, views, 420, &out.with_file_name(&name))?;
        // Grain, then the same view.
        let positions = plain.positions().iter().zip(plain.normals()).map(|(&p, &n)| p + n * (grain.sample(p) - 0.5) * 0.05).collect();
        let grained = Document::from_mesh(PolyMesh { positions, faces: plain.faces().to_vec() })?;
        render(&grained, views, 420, &out.with_file_name(format!("adf_grain_{}.png", (tol * 10.0) as u32)))?;
    }
    Ok(())
}
