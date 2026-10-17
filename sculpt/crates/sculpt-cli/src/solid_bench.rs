//! `sculpt-cli solid-bench [level] [divisions] [threads]`: where a solid preview's time goes while a slider is dragged.
//!
//! Builds the test scene of the app (a sphere, 14 tilted box cuts, a noise layer), then solidifies it repeatedly while
//! growing the cuts a little each time, as a Size drag does, inside a pool of `threads` like the preview's.

use std::time::Instant;

use sculpt_core::document::SolidSettings;
use sculpt_core::generator::layer::{GeneratorKind, GeneratorSpec, NoiseSpec};
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, Placement};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub(crate) fn run(args: &[String]) -> Res<()> {
    let level: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let divisions: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(75.0);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    // `stroke`: each step is a brush stroke on the surface instead of a slider change.
    let strokes = args.get(3).is_some_and(|a| a == "stroke");
    let mut doc = Document::from_mesh(quad_sphere(level, 1.0))?;
    let cuts = doc.insert_generator("cuts", Placement::Top, GeneratorSpec { primitive: Primitive::Box, op: BoolOp::Subtract, count: 14, size: 0.45, lift: 0.7, seed: 5, ..GeneratorSpec::default() })?;
    doc.insert_generator("noise", Placement::Top, GeneratorSpec { kind: GeneratorKind::Noise, noise: NoiseSpec { amplitude: 0.03, scale: 10.9, octaves: 4, seed: 1 }, ..GeneratorSpec::default() })?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let voxel = doc.bounds().extent().max_element() / divisions;
    println!("level {level} ({} faces), {divisions} divisions (voxel {voxel:.4}), {threads} threads", doc.face_count());
    println!("| step | snapshot ms | prep ms | scatter ms | extract ms | to document ms | total ms | quads | base reused/computed |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for step in 0..8 {
        if strokes {
            use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
            let a = step as f32 * 0.7;
            let dir = sculpt_core::glam::Vec3::new(a.cos(), 0.6, a.sin()).normalize();
            let side = dir.cross(sculpt_core::glam::Vec3::Y).normalize();
            let path: Vec<_> = (0..=20).map(|i| ((dir + side * (i as f32 / 20.0 - 0.5) * 0.5).normalize(), 1.0)).collect();
            brush::stroke(&mut doc, &mut ClayBuildup::default(), &BrushSettings { radius: 0.15, strength: 0.6, ..Default::default() }, &path)?;
        } else {
            let mut meta = doc.layer(cuts).expect("layer").meta();
            let mut g = meta.generator.clone().expect("generator");
            g.size = 0.45 + 0.01 * step as f32;
            meta.generator = Some(g);
            doc.set_layer_meta(cuts, meta, true)?;
        }
        let t = Instant::now();
        let input = doc.solid_input();
        let snapshot = t.elapsed().as_secs_f64() * 1e3;
        let (stats, to_doc, total) = pool.install(|| -> Result<_, String> {
            let t = Instant::now();
            let (mesh, stats) = input.run(&SolidSettings { voxel }).map_err(|e| e.to_string())?;
            let t2 = Instant::now();
            let _preview = Document::from_mesh(mesh).map_err(|e| e.to_string())?;
            Ok((stats, t2.elapsed().as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3))
        })?;
        println!(
            "| {step} | {snapshot:.1} | {:.0} | {:.0} | {:.0} | {to_doc:.0} | {total:.0} | {} | {}/{} |",
            stats.prep_ms, stats.scatter_ms, stats.extract_ms, stats.quads, stats.base_hits, stats.base_misses
        );
    }
    Ok(())
}
