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
    // `stroke`: each step is a brush stroke on the surface instead of a slider change. `check`: inspect one result.
    let strokes = args.get(3).is_some_and(|a| a == "stroke");
    if args.get(3).is_some_and(|a| a == "check") {
        return check(level, divisions);
    }
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
            let (mesh, stats) = input.run(&SolidSettings::new(voxel)).map_err(|e| e.to_string())?;
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

/// Topology health of the remeshed test scene: what a brush trips over.
fn check(level: u32, divisions: f32) -> Res<()> {
    use sculpt_core::generator::remesh::non_manifold_edges;
    let mut doc = Document::from_mesh(quad_sphere(level, 1.0))?;
    let op = match std::env::var("SOLID_OP").as_deref() { Ok("intersect") => BoolOp::Intersect, Ok("add") => BoolOp::Add, _ => BoolOp::Subtract };
    doc.insert_generator("cuts", Placement::Top, GeneratorSpec { primitive: Primitive::Box, op, count: 14, size: 0.45, lift: 0.7, seed: 5, ..GeneratorSpec::default() })?;
    doc.insert_generator("noise", Placement::Top, GeneratorSpec { kind: GeneratorKind::Noise, noise: NoiseSpec { amplitude: 0.03, scale: 10.9, octaves: 4, seed: 1 }, ..GeneratorSpec::default() })?;
    let voxel = doc.solid_input().surface_bounds().extent().max_element() / divisions;
    let (mesh, _) = doc.solidify(&SolidSettings::new(voxel))?;
    let p = &mesh.positions;
    // Degenerate quads: tiny area or a repeated corner.
    let area = |f: &[u32; 4]| ((p[f[2] as usize] - p[f[0] as usize]).cross(p[f[3] as usize] - p[f[1] as usize])).length() * 0.5;
    let tiny = mesh.faces.iter().filter(|f| area(f) < (voxel * voxel) * 1e-4).count();
    let repeated = mesh.faces.iter().filter(|f| (0..4).any(|i| (i + 1..4).any(|j| f[i] == f[j]))).count();
    // Vertices whose faces form more than one fan (a pinch where two sheets touch).
    let mut faces_of = vec![Vec::new(); p.len()];
    for (fi, f) in mesh.faces.iter().enumerate() {
        for &v in f {
            faces_of[v as usize].push(fi);
        }
    }
    let mut pinched = 0;
    for (v, fs) in faces_of.iter().enumerate() {
        if fs.is_empty() {
            continue;
        }
        // Walk the fan: faces are joined when they share an edge through v.
        let mut seen = vec![false; fs.len()];
        let mut stack = vec![0];
        seen[0] = true;
        while let Some(i) = stack.pop() {
            let f = &mesh.faces[fs[i]];
            let k = f.iter().position(|&x| x as usize == v).unwrap();
            let (a, b) = (f[(k + 1) % 4], f[(k + 3) % 4]);
            for (j, &g) in fs.iter().enumerate() {
                if !seen[j] && mesh.faces[g].contains(&a) || !seen[j] && mesh.faces[g].contains(&b) {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        if seen.iter().any(|s| !s) {
            pinched += 1;
        }
    }
    let unused = faces_of.iter().filter(|f| f.is_empty()).count();
    let shortest = mesh.faces.iter().flat_map(|f| (0..4).map(move |k| (f[k], f[(k + 1) % 4]))).map(|(a, b)| p[a as usize].distance(p[b as usize])).fold(f32::MAX, f32::min);
    println!("{} quads, {} verts: non-manifold edges {}, pinched vertices {pinched}, tiny quads {tiny}, quads with a repeated corner {repeated}, unused vertices {unused}, shortest edge {:.2e} (voxel {voxel:.4})", mesh.faces.len(), p.len(), non_manifold_edges(&mesh.faces), shortest);

    // Every brush on the remeshed mesh, then the stack edited and the sculpt carried across: report the longest edge,
    // which is where a spike shows.
    use sculpt_core::brush::{self, BrushSettings, ClayBuildup, MoveBrush, Smooth, TrimDynamic};
    let longest = |d: &Document| d.faces().iter().flat_map(|f| (0..sculpt_core::mesh::face_len(f)).map(move |k| (f[k], f[(k + 1) % sculpt_core::mesh::face_len(f)]))).map(|(a, b)| d.positions()[a as usize].distance(d.positions()[b as usize])).fold(0.0f32, f32::max);
    // Quad quality: how thin the thinnest quads are (shortest over longest edge).
    let thinness = |positions: &[sculpt_core::glam::Vec3], faces: &[[u32; 4]]| -> (usize, usize) {
        let mut slivers = (0, 0);
        for f in faces {
            let e: Vec<f32> = (0..4).map(|k| positions[f[k] as usize].distance(positions[f[(k + 1) % 4] as usize])).collect();
            let r = e.iter().cloned().fold(f32::MAX, f32::min) / e.iter().cloned().fold(0.0, f32::max).max(1e-12);
            if r < 0.05 { slivers.0 += 1; }
            if r < 0.01 { slivers.1 += 1; }
        }
        slivers
    };
    // Winding: across a consistently oriented surface, the two quads on an edge walk it in opposite directions.
    let misoriented = |faces: &[[u32; 4]]| {
        let mut dirs = std::collections::HashMap::new();
        for f in faces {
            for k in 0..4 {
                *dirs.entry((f[k], f[(k + 1) % 4])).or_insert(0u32) += 1;
            }
        }
        dirs.iter().filter(|(k, n)| k.0 < k.1 && (**n > 1 || dirs.get(&(k.1, k.0)).copied().unwrap_or(0) > 1)).count()
    };
    println!("edges walked the same way by both quads (wrong winding): {}", misoriented(&mesh.faces));
    let (s5, s1) = thinness(&mesh.positions, &mesh.faces);
    println!("quads thinner than 1:20 {s5}, thinner than 1:100 {s1} (of {})", mesh.faces.len());
    let mut top = Document::remeshed(mesh, doc, voxel)?;
    let rest_normals: Vec<sculpt_core::glam::Vec3> = top.faces().iter().map(|f| sculpt_core::mesh::face_normal(top.positions(), f)).collect();
    let flipped = |d: &Document| d.faces().iter().zip(&rest_normals).filter(|(f, n)| sculpt_core::mesh::face_normal(d.positions(), f).dot(**n) < 0.0).count();
    println!("remeshed: longest edge {:.4} ({:.1} voxels)", longest(&top), longest(&top) / voxel);
    let s = BrushSettings { radius: 0.35, strength: 1.0, ..Default::default() };
    let path = |a: f32| -> Vec<(sculpt_core::glam::Vec3, f32)> {
        let dir = sculpt_core::glam::Vec3::new(a.cos(), 0.5, a.sin()).normalize();
        let side = dir.cross(sculpt_core::glam::Vec3::Y).normalize();
        (0..=30).map(|i| ((dir + side * (i as f32 / 30.0 - 0.5) * 0.8).normalize(), 1.0)).collect()
    };
    for (i, name) in ["clay", "clay inverted", "trim", "smooth", "move"].iter().enumerate() {
        let a = i as f32 * 1.3;
        match *name {
            "clay" => drop(brush::stroke(&mut top, &mut ClayBuildup::default(), &s, &path(a))?),
            "clay inverted" => drop(brush::stroke(&mut top, &mut ClayBuildup::default(), &BrushSettings { invert: true, ..s.clone() }, &path(a))?),
            "trim" => drop(brush::stroke(&mut top, &mut TrimDynamic::default(), &s, &path(a))?),
            "smooth" => drop(brush::stroke(&mut top, &mut Smooth::default(), &s, &path(a))?),
            _ => {
                let from = path(a)[15].0 * 2.0;
                let hit = top.raycast(&sculpt_core::geom::Ray::new(from, -from)).expect("hit");
                top.begin_stroke("Move");
                let mut mv = MoveBrush::new(false);
                mv.begin(&top, &hit, &s);
                mv.drag(&mut top, hit.normal * 0.15)?;
                top.end_stroke();
            }
        }
        println!("after {name}: longest edge {:.4} ({:.1} voxels), quads turned inside out {}", longest(&top), longest(&top) / voxel, flipped(&top));
    }
    let mut source = top.take_remesh_source().expect("source");
    let old_stack = source.solid_input().field(&SolidSettings::new(voxel))?;
    let id = source.layers().iter().find(|l| l.is_generator()).expect("generator").id;
    let mut meta = source.layer(id).expect("layer").meta();
    let mut g = meta.generator.clone().expect("spec");
    g.seed = 11;
    g.size *= 1.4;
    meta.generator = Some(g);
    source.set_layer_meta(id, meta, false)?;
    let as_volume = std::env::var("SOLID_CARRY").as_deref() == Ok("volume");
    let rebuilt = if as_volume { top.rebuilt_as_volume(&old_stack, source, &SolidSettings::new(voxel))? } else { top.rebuilt_from_source(source)? };
    println!("carried as {}", if as_volume { "volume" } else { "layers" });
    println!("rebuilt: edges walked the same way by both quads (wrong winding): {}", misoriented(rebuilt.faces()));
    let (s5, s1) = thinness(rebuilt.base_positions(), rebuilt.faces());
    println!("rebuilt: quads thinner than 1:20 {s5}, thinner than 1:100 {s1} (of {})", rebuilt.faces().len());
    let base_n: Vec<sculpt_core::glam::Vec3> = rebuilt.faces().iter().map(|f| sculpt_core::mesh::face_normal(rebuilt.base_positions(), f)).collect();
    let folded_by_carry = rebuilt.faces().iter().zip(&base_n).filter(|(f, n)| sculpt_core::mesh::face_normal(rebuilt.positions(), f).dot(**n) < 0.0).count();
    println!("rebuilt: quads the carried strokes turned inside out {folded_by_carry}");
    let mut rebuilt = rebuilt;
    let rest_after: Vec<sculpt_core::glam::Vec3> = rebuilt.faces().iter().map(|f| sculpt_core::mesh::face_normal(rebuilt.positions(), f)).collect();
    if let Some(l) = rebuilt.layers().first().map(|l| l.id) {
        rebuilt.set_active_layer(Some(l))?;
    }
    for (i, name) in ["clay", "smooth", "trim", "clay inverted"].iter().enumerate() {
        let a = 0.4 + i as f32 * 1.1;
        match *name {
            "clay" => drop(brush::stroke(&mut rebuilt, &mut ClayBuildup::default(), &s, &path(a))?),
            "clay inverted" => drop(brush::stroke(&mut rebuilt, &mut ClayBuildup::default(), &BrushSettings { invert: true, ..s.clone() }, &path(a))?),
            "trim" => drop(brush::stroke(&mut rebuilt, &mut TrimDynamic::default(), &s, &path(a))?),
            _ => drop(brush::stroke(&mut rebuilt, &mut Smooth::default(), &s, &path(a))?),
        }
        let turned = rebuilt.faces().iter().zip(&rest_after).filter(|(f, n)| sculpt_core::mesh::face_normal(rebuilt.positions(), f).dot(**n) < 0.0).count();
        println!("after {name} on the rebuilt mesh: quads turned inside out {turned}, longest edge {:.1} voxels", longest(&rebuilt) / voxel);
    }
    println!("after editing the stack below: longest edge {:.4} ({:.1} voxels), non-manifold edges {}", longest(&rebuilt), longest(&rebuilt) / voxel, non_manifold_edges(rebuilt.faces()));
    // Spikes: a vertex carried far off the solid it sits on. Distance of each sculpted vertex from the unsculpted surface.
    let edges: Vec<f32> = rebuilt.faces().iter().flat_map(|f| (0..4).map(move |k| (f[k], f[(k + 1) % 4]))).map(|(a, b)| rebuilt.positions()[a as usize].distance(rebuilt.positions()[b as usize]) / voxel).collect();
    println!("edges longer than 4 voxels: {}, longer than 8: {}", edges.iter().filter(|&&e| e > 4.0).count(), edges.iter().filter(|&&e| e > 8.0).count());
    if let Ok(path) = std::env::var("SOLID_CHECK_RENDER") {
        let views: Vec<crate::render::View> = [0.3f32, 1.9, 3.5].iter().map(|&yaw| crate::render::View { yaw, pitch: 0.3, overlay: None }).collect();
        crate::render::render(&top, &views, 400, std::path::Path::new(&format!("{path}_before.png")))?;
        crate::render::render(&rebuilt, &views, 400, std::path::Path::new(&format!("{path}_after.png")))?;
    }
    Ok(())
}
