//! `sculpt-cli generator-edit-bench`: how long each kind of generator parameter edit takes to reach the surface.
//!
//! ```text
//! sculpt-cli generator-edit-bench [levels=8,10] [count=60] [size=0.10] [--lazy]
//! ```
//! Each edit goes through `Document::set_layer_meta` with coalescing on, exactly as a slider drag does in the app, and
//! the time is the whole call (placement, indexing and recompositing the leaves the cutters reach). Medians of 9 edits.
//! `--lazy` leaves vertex normals out of date after each edit, as the app does with flat shading.

use std::time::Instant;

use sculpt_core::generator::layer::GeneratorSpec;
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, LayerId, Placement};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn edit(doc: &mut Document, id: LayerId, spec: &GeneratorSpec) -> f64 {
    let mut meta = doc.layer(id).expect("generator layer").meta();
    meta.generator = Some(spec.clone());
    let t = Instant::now();
    doc.set_layer_meta(id, meta, true).expect("edit");
    t.elapsed().as_secs_f64() * 1e3
}

/// Alternate between `a` and `b` and report the median edit time.
fn alternate(doc: &mut Document, id: LayerId, a: &GeneratorSpec, b: &GeneratorSpec) -> f64 {
    let mut times: Vec<f64> = (0..9).map(|i| edit(doc, id, if i % 2 == 0 { b } else { a })).collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

pub(crate) fn run(args: &[String]) -> Res<()> {
    let levels: Vec<u32> = args.first().map_or(vec![8, 10], |s| s.split(',').filter_map(|v| v.parse().ok()).collect());
    let lazy = args.iter().any(|a| a == "--lazy");
    let count: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let size: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.10);
    let base = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count, seed: 7, size, lift: 0.6, random_spin: true };
    println!("| faces | create ms | count ms | seed ms | size ms | lift ms | primitive ms | operation ms |");
    println!("|---|---|---|---|---|---|---|---|");
    for level in levels {
        let t = Instant::now();
        let mut doc = Document::from_mesh(quad_sphere(level, 1.0))?;
        let build = t.elapsed().as_secs_f64() * 1e3;
        doc.set_lazy_normals(lazy);
        let faces = doc.face_count();
        let t = Instant::now();
        let id = doc.insert_generator("g", Placement::Top, base.clone())?;
        let create = t.elapsed().as_secs_f64() * 1e3;
        let row = [
            alternate(&mut doc, id, &base, &GeneratorSpec { count: count + 30, ..base.clone() }),
            alternate(&mut doc, id, &base, &GeneratorSpec { seed: 8, ..base.clone() }),
            alternate(&mut doc, id, &base, &GeneratorSpec { size: size * 1.2, ..base.clone() }),
            alternate(&mut doc, id, &base, &GeneratorSpec { lift: 0.5, ..base.clone() }),
            alternate(&mut doc, id, &base, &GeneratorSpec { primitive: Primitive::Box, ..base.clone() }),
            alternate(&mut doc, id, &base, &GeneratorSpec { op: BoolOp::Add, ..base.clone() }),
        ];
        println!(
            "| {faces} | {create:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |   (mesh built in {build:.0} ms)",
            row[0], row[1], row[2], row[3], row[4], row[5]
        );
    }
    Ok(())
}
