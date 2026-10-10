//! Headless driver for the sculpt engine.
//!
//! ```text
//! sculpt-cli demo  [out_dir]           scripted session -> project + OBJs
//! sculpt-cli bench [level]             brush latency on a dense sphere
//! sculpt-cli info  <project_dir>       summarize a project
//! sculpt-cli import-channel <project_dir> <name> <file>
//! sculpt-cli render <project_dir> <out.png>
//! ```

mod render;

use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;
use sculpt_core::bake::{BakeSettings, MeshAttribute};
use sculpt_core::brush::{self, Brush, BrushSettings, ClayBuildup, Dab, MoveBrush, PaintBrush, Smooth, TrimDynamic};
use sculpt_core::geom::Ray;
use sculpt_core::io::{obj, project, read_channel_file};
use sculpt_core::mask::{BlendMode, Levels, MaskLayer, MaskSource, MaskStack};
use sculpt_core::noise::{NoiseKind, NoiseParams};
use sculpt_core::pose::PoseTransform;
use sculpt_core::primitives::{quad_sphere, quad_sphere_res};
use sculpt_core::{Document, PaintTarget};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("demo") => demo(Path::new(args.get(1).map_or("out", |s| s.as_str()))),
        Some("lod-bench") => lod_bench(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000), args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0)),
        Some("bench") => bench(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(9)),
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("import-channel") if args.len() == 4 => import_channel(Path::new(&args[1]), &args[2], Path::new(&args[3])),
        Some("render") if args.len() == 3 => render_project(Path::new(&args[1]), Path::new(&args[2])),
        _ => {
            eprintln!(
                "usage: sculpt-cli demo [out] | bench [level] | info <project> | import-channel <project> <name> <file> | render <project> <png>"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn hit_from(doc: &Document, from: Vec3) -> sculpt_core::SurfaceHit {
    doc.raycast(&Ray::new(from, -from)).expect("ray toward origin hits the sphere")
}

/// A stroke as the user would draw it: directions swept from `a` to `b`,
/// each picked onto the surface by a ray toward the origin (like a mouse pick).
fn arc(doc: &Document, a: Vec3, b: Vec3, n: usize) -> Vec<(Vec3, f32)> {
    (0..=n)
        .filter_map(|i| {
            let t = i as f32 / n as f32;
            // Pen pressure ramps in and out like a real stroke.
            let pressure = (std::f32::consts::PI * t).sin().max(0.3);
            let dir = a.normalize().slerp(b.normalize(), t);
            doc.raycast(&Ray::new(dir * 4.0, -dir)).map(|h| (h.point, pressure))
        })
        .collect()
}

fn step(label: &str, t: Instant) {
    println!("  {:<44} {:>8.1} ms", label, t.elapsed().as_secs_f64() * 1e3);
}

fn demo(out: &Path) -> Res<()> {
    std::fs::create_dir_all(out)?;
    println!("Building level-7 quad sphere...");
    let t = Instant::now();
    let mut doc = Document::from_mesh(quad_sphere(5, 1.0))?;
    doc.subdivide()?;
    doc.subdivide()?;
    step(&format!("{} faces, {} leaves", doc.face_count(), doc.bvh().leaves.len()), t);

    // --- Base form: Move brush pulls out a "head" bump, clay builds a brow.
    println!("Sculpting base:");
    let t = Instant::now();
    let s = BrushSettings { radius: 0.6, strength: 1.0, front_faces_only: false, ..Default::default() };
    let mut mv = MoveBrush::new(true);
    let hit = hit_from(&doc, Vec3::new(0.0, 0.0, 3.0));
    doc.begin_stroke("Move");
    mv.begin(&doc, &hit, &s);
    for i in 1..=10 {
        mv.drag(&mut doc, Vec3::new(0.0, 0.0, 0.04 * i as f32))?;
    }
    mv.end();
    doc.end_stroke();
    step("topological move (10 drag updates)", t);

    let t = Instant::now();
    let clay = BrushSettings { radius: 0.12, strength: 0.7, ..Default::default() };
    for k in 0..3 {
        let y = 0.25 + k as f32 * 0.04;
        let path = arc(&doc, Vec3::new(-0.5, y, 1.0), Vec3::new(0.5, y, 1.0), 60);
        brush::stroke(&mut doc, &mut ClayBuildup::default(), &clay, &path)?;
    }
    step("3 clay buildup strokes", t);

    let t = Instant::now();
    let trim = BrushSettings { radius: 0.15, strength: 0.8, ..Default::default() };
    let path = arc(&doc, Vec3::new(0.6, -0.2, 0.8), Vec3::new(0.6, 0.4, 0.8), 40);
    brush::stroke(&mut doc, &mut TrimDynamic::default(), &trim, &path)?;
    step("trim dynamic plane cut", t);

    let t = Instant::now();
    let path = arc(&doc, Vec3::new(-0.5, 0.3, 1.0), Vec3::new(0.0, 0.3, 1.0), 20);
    brush::stroke(
        &mut doc,
        &mut Smooth::default(),
        &BrushSettings { radius: 0.1, strength: 0.5, ..Default::default() },
        &path,
    )?;
    step("smooth stroke", t);

    // --- Detail layer with a procedural + hand-painted mask.
    println!("Detail layer:");
    let detail = doc.add_layer("Pores and folds");
    let t = Instant::now();
    let fine = BrushSettings { radius: 0.06, strength: 0.8, ..Default::default() };
    for k in 0..6 {
        let a = Vec3::new(-0.8 + k as f32 * 0.3, -0.5, 0.6);
        let path = arc(&doc, a, a + Vec3::new(0.1, 0.9, 0.2), 40);
        brush::stroke(&mut doc, &mut ClayBuildup::default(), &fine, &path)?;
    }
    step("6 detail strokes on layer", t);

    let t = Instant::now();
    let paint = BrushSettings { radius: 0.5, strength: 1.0, ..Default::default() };
    let path = arc(&doc, Vec3::new(-0.4, 0.0, 1.0), Vec3::new(0.4, 0.0, 1.0), 10);
    brush::stroke(
        &mut doc,
        &mut PaintBrush { target: PaintTarget::Channel("paint.detail".into()), value: 1.0 },
        &paint,
        &path,
    )?;
    step("hand-paint mask channel", t);

    let t = Instant::now();
    let mask = MaskStack::new(0.0)
        .with(
            MaskLayer::new(
                "breakup noise",
                MaskSource::Noise(NoiseParams { kind: NoiseKind::Fbm, scale: 5.0, seed: 3, ..Default::default() }),
            )
            .levels(Levels::range(0.35, 0.65)),
        )
        .with(MaskLayer::new("painted", MaskSource::Channel { name: "paint.detail".into() }).blend(BlendMode::Screen).opacity(0.8))
        .with(
            MaskLayer::new("keep out of cavities", MaskSource::Mesh { attribute: MeshAttribute::Cavity })
                .blend(BlendMode::Subtract)
                .opacity(0.5),
        );
    doc.set_layer_mask(detail, Some(mask))?;
    step("layer mask (noise + paint - cavity bake)", t);

    let t = Instant::now();
    doc.set_layer_opacity(detail, 0.6)?;
    step("layer strength slider -> 60%", t);

    // --- Mesh bakes for downstream masks.
    println!("Bakes:");
    for attr in [MeshAttribute::Curvature, MeshAttribute::AmbientOcclusion, MeshAttribute::Thickness] {
        let t = Instant::now();
        doc.bake(attr, &BakeSettings { ao_rays: 16, thickness_rays: 8, ..Default::default() });
        step(&format!("{attr:?}"), t);
    }

    // --- Pose: rotate the bump with a topological mask, detail follows.
    println!("Pose:");
    let t = Instant::now();
    let tip = hit_from(&doc, Vec3::new(0.0, 0.0, 3.0));
    let weights = doc.topological_weights(tip.vertex, 0.9, 0.6, 2);
    doc.pose(&weights, PoseTransform::Rotate { pivot: Vec3::new(0.0, 0.0, 0.6), axis: Vec3::X, angle: -0.5 })?;
    step("topological mask + rotate", t);

    // --- Save everything.
    let t = Instant::now();
    let proj = out.join("demo.sculpt");
    project::save(&doc, &proj)?;
    obj::write(&out.join("demo_composite.obj"), &doc.export_mesh(true))?;
    obj::write(&out.join("demo_base.obj"), &doc.export_mesh(false))?;
    step("save project + OBJs", t);
    render_project(&proj, &out.join("demo.png"))?;

    let t = Instant::now();
    let back = project::load(&proj)?;
    step("reload project", t);
    let err = back
        .export_mesh(true)
        .positions
        .iter()
        .zip(doc.export_mesh(true).positions.iter())
        .map(|(a, b)| a.distance(*b))
        .fold(0.0f32, f32::max);
    println!("  round-trip max error: {err:e}");
    println!("Undo history: {:?}", doc.undo_labels());
    println!("Wrote {}", out.display());
    Ok(())
}

fn bench(level: u32) -> Res<()> {
    println!("threads: {}", rayon::current_num_threads());
    let t = Instant::now();
    let mut doc = Document::from_mesh(quad_sphere(level.min(6), 1.0))?;
    while doc.level() + level.min(6) < level {
        doc.subdivide()?;
    }
    step(&format!("build level {level}: {} faces / {} verts", doc.face_count(), doc.vertex_count()), t);
    doc.add_layer("bench");

    let mut run = |name: &str, brush: &mut dyn Brush, radius: f32| -> Res<()> {
        let s = BrushSettings { radius, strength: 0.5, ..Default::default() };
        let hit = hit_from(&doc, Vec3::new(0.3, 0.4, 3.0));
        let affected = doc.compute_displacements(hit.point, radius, |_, _, _| Some(Vec3::X)).vertex_count();
        doc.begin_stroke(name);
        let n = 40;
        let t = Instant::now();
        for i in 0..n {
            let off = Vec3::new(i as f32 * radius * 0.1, 0.0, 0.0);
            let h = doc.project_to_surface(hit.point + off, hit.normal, radius).unwrap_or(hit);
            brush.dab(&mut doc, &Dab { center: h.point, normal: h.normal, direction: Vec3::X, pressure: 1.0 }, &s)?;
        }
        let per = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        doc.end_stroke();
        println!("  {name:<16} r={radius:<5} ~{affected:>8} verts/dab  {per:>7.2} ms/dab  ({:>5.0} dabs/s)", 1e3 / per);
        Ok(())
    };
    for r in [0.05, 0.15, 0.4] {
        run("clay buildup", &mut ClayBuildup::default(), r)?;
        run("trim dynamic", &mut TrimDynamic::default(), r)?;
        run("smooth", &mut Smooth::default(), r)?;
    }

    let s = BrushSettings { radius: 0.4, strength: 1.0, ..Default::default() };
    let hit = hit_from(&doc, Vec3::new(0.0, 0.0, 3.0));
    let mut mv = MoveBrush::new(false);
    mv.begin(&doc, &hit, &s);
    let t = Instant::now();
    for i in 0..20 {
        mv.drag(&mut doc, Vec3::new(0.0, 0.01 * i as f32, 0.0))?;
    }
    println!("  move drag        r=0.4   {:>8} verts       {:>7.2} ms/update", mv.grabbed_vertices(), t.elapsed().as_secs_f64() * 1e3 / 20.0);

    let id = doc.layers()[0].id;
    let t = Instant::now();
    doc.set_layer_opacity(id, 0.5)?;
    step("layer slider change", t);
    let t = Instant::now();
    let m = MaskStack::new(0.0).with(MaskLayer::new("n", MaskSource::Noise(NoiseParams::default())));
    doc.set_layer_mask(id, Some(m))?;
    step("evaluate fbm layer mask (whole mesh)", t);
    let t = Instant::now();
    doc.undo();
    step("undo", t);
    Ok(())
}

fn info(dir: &Path) -> Res<()> {
    let doc = project::load(dir)?;
    println!("{}: level {}, {} verts, {} faces", dir.display(), doc.level(), doc.vertex_count(), doc.face_count());
    for l in doc.layers() {
        println!(
            "  layer {:>2} '{}' opacity {:.2}{}{} mask:{} storage {:.1} MB",
            l.id.0,
            l.name,
            l.opacity,
            if l.visible { "" } else { " hidden" },
            if l.locked { " locked" } else { "" },
            l.mask.as_ref().map_or(0, |m| m.layers.len()),
            l.delta_bytes() as f64 / 1e6
        );
    }
    for name in doc.channels().keys() {
        println!("  channel {name}");
    }
    Ok(())
}

fn import_channel(dir: &Path, name: &str, file: &Path) -> Res<()> {
    let mut doc = project::load(dir)?;
    let values = read_channel_file(file, doc.vertex_count())?;
    doc.import_channel(name, &values)?;
    project::save(&doc, &PathBuf::from(dir))?;
    println!("imported {} values into channel '{name}'", values.len());
    Ok(())
}

/// Front, three-quarter and side views, plus the first layer's mask in blue.
fn render_project(dir: &Path, png: &Path) -> Res<()> {
    let doc = project::load(dir)?;
    let mask = doc.layers().first().and_then(|l| l.mask_values().map(|m| m.to_vec()));
    let mut views = vec![
        render::View { yaw: 0.0, pitch: 0.15, overlay: None },
        render::View { yaw: -0.8, pitch: 0.15, overlay: None },
        render::View { yaw: -1.57, pitch: 0.0, overlay: None },
    ];
    if mask.is_some() {
        views.push(render::View { yaw: 0.0, pitch: 0.15, overlay: mask });
    }
    render::render(&doc, &views, 512, png)?;
    println!("rendered {}", png.display());
    Ok(())
}

/// Level-of-detail cost at scale: build the tree for a sphere of about `quads` quads and report what a
/// frame would draw from several distances. Triangle counts are what a GPU-independent budget needs.
fn lod_bench(quads: u32, detail: f32) -> Res<()> {
    use sculpt_core::lod::{LodParams, View};
    println!("threads: {}", rayon::current_num_threads());
    let res = ((quads as f64 / 6.0).sqrt().round() as u32).max(2);
    let t = Instant::now();
    let mut mesh = quad_sphere_res(res, 1.0);
    if detail > 0.0 {
        // Fine fbm relief, the kind of detail a sculpt actually has (about 40 to 640 cycles per unit).
        let noise = sculpt_core::noise::Noise::new(&NoiseParams { scale: 40.0, octaves: 5, ..Default::default() });
        for p in &mut mesh.positions {
            let n = p.normalize();
            *p = n * (1.0 + detail * (noise.sample(n) - 0.5));
        }
    }
    step(&format!("generate {} quads (relief {detail})", mesh.faces.len()), t);
    let t = Instant::now();
    let mut doc = Document::from_mesh(mesh)?;
    step(&format!("document + BVH: {} leaves", doc.bvh().leaves.len()), t);
    let t = Instant::now();
    let mut params = LodParams::default();
    if let Some(w) = std::env::var("SCULPT_LOD_NORMAL_WEIGHT").ok().and_then(|v| v.parse().ok()) {
        params.normal_weight = w;
    }
    doc.build_lod(params);
    step(&format!("LOD tree build (normal weight {})", params.normal_weight), t);
    let tree = doc.lod().expect("built").clone();
    println!(
        "  full {:>10} tris   pool {:>10} tris ({:.2}x)   {:.0} MB of indices   {} nodes",
        tree.full_triangles,
        tree.pool_triangles(),
        tree.pool_triangles() as f64 / tree.full_triangles as f64,
        tree.indices.len() as f64 * 4.0 / 1e6,
        tree.nodes.len()
    );
    // 1080p vertical field of view 35 degrees, like the app camera.
    let (h, fov) = (1080.0f32, 35f32.to_radians());
    let focal = h / (2.0 * (fov / 2.0).tan());
    println!("  view                      tau 1px: tris   nodes  culled  select ms | tau 2px: tris | budget 3M: tris  tau");
    for (name, dist) in [("whole model", 4.0f32), ("close", 2.0), ("very close", 1.3), ("zoomed in", 1.08)] {
        let eye = Vec3::new(0.0, 0.0, dist);
        let proj = glam::camera::rh::proj::directx::perspective(fov, 16.0 / 9.0, dist * 0.01, dist * 100.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let v = View::from_view_proj(proj * view, eye, focal);
        let t = Instant::now();
        let a = tree.select(doc.bvh(), &v, 1.0, u64::MAX);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let b = tree.select(doc.bvh(), &v, 2.0, u64::MAX);
        let c = tree.select(doc.bvh(), &v, 1.0, 3_000_000);
        println!("  {name:<22} {:>14} {:>7} {:>7} {:>8.3}    | {:>14}  | {:>14} {:>5.1}", a.triangles, a.nodes, a.culled, ms, b.triangles, c.triangles, c.tau_px);
    }
    Ok(())
}
