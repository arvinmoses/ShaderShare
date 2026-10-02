use glam::Vec3;
use sculpt_core::bake::MeshAttribute;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup, MoveBrush, PaintBrush, Smooth, TrimDynamic};
use sculpt_core::geom::Ray;
use sculpt_core::io::{obj, project};
use sculpt_core::mask::{BlendMode, Levels, MaskLayer, MaskSource, MaskStack};
use sculpt_core::mesh::{self, PolyMesh};
use sculpt_core::noise::{NoiseKind, NoiseParams};
use sculpt_core::pose::PoseTransform;
use sculpt_core::primitives::{cube, grid, quad_sphere};
use sculpt_core::subdiv::Subdivider;
use sculpt_core::{Document, PaintTarget};

fn sphere_doc(level: u32) -> Document {
    Document::from_mesh(quad_sphere(level, 1.0)).unwrap()
}

fn top_hit(doc: &Document) -> sculpt_core::SurfaceHit {
    doc.raycast(&Ray::new(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y)).expect("hit top of sphere")
}

/// Short horizontal path across the top of the unit sphere.
fn top_path(len: f32) -> Vec<(Vec3, f32)> {
    (0..=20).map(|i| (Vec3::new(-len / 2.0 + len * i as f32 / 20.0, 1.0, 0.0), 1.0)).collect()
}

fn max_radius(doc: &Document) -> f32 {
    doc.positions().iter().map(|p| p.length()).fold(0.0, f32::max)
}

fn min_radius(doc: &Document) -> f32 {
    doc.positions().iter().map(|p| p.length()).fold(f32::MAX, f32::min)
}

#[test]
fn catmull_clark_counts_and_linearity() {
    let c = cube();
    let s = Subdivider::new(c.positions.len(), &c.faces);
    assert_eq!(s.new_faces().len(), 24);
    assert_eq!(s.new_vertex_count(), 8 + 12 + 6);

    // Linear: subdividing a sum equals summing subdivisions.
    let a: Vec<Vec3> = c.positions.clone();
    let b: Vec<Vec3> = c.positions.iter().map(|p| Vec3::new(p.y, p.z * 2.0, -p.x)).collect();
    let sum: Vec<Vec3> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();
    let (sa, sb, ss) = (s.apply(&a), s.apply(&b), s.apply(&sum));
    for i in 0..ss.len() {
        assert!((sa[i] + sb[i] - ss[i]).length() < 1e-5);
    }
    assert_eq!(quad_sphere(3, 1.0).faces.len(), 6 * 64);
}

#[test]
fn bvh_renumbering_is_a_valid_permutation_and_raycasts() {
    let doc = sphere_doc(5);
    let bvh = doc.bvh();
    let mut covered = vec![false; doc.vertex_count()];
    let mut expected_start = 0;
    for leaf in &bvh.leaves {
        assert_eq!(leaf.owned.start, expected_start, "owned ranges must be contiguous");
        expected_start = leaf.owned.end;
        for v in leaf.owned_range() {
            assert!(!covered[v]);
            covered[v] = true;
        }
    }
    assert!(covered.iter().all(|c| *c));
    // Canonical round trip.
    let canon = doc.export_mesh(true);
    let src = quad_sphere(5, 1.0);
    for (a, b) in canon.positions.iter().zip(&src.positions) {
        assert!((*a - *b).length() < 1e-6);
    }
    let hit = top_hit(&doc);
    assert!((hit.point.y - 1.0).abs() < 0.01);
    assert!(hit.normal.y > 0.99);
}

#[test]
fn clay_buildup_adds_material_and_plateaus() {
    let mut doc = sphere_doc(6);
    let s = BrushSettings { radius: 0.25, strength: 0.6, ..Default::default() };
    let before = max_radius(&doc);
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.6)).unwrap();
    let after = max_radius(&doc);
    assert!(after > before + 0.01, "clay should raise the surface: {before} -> {after}");
    // It is plane limited: never more than ~one plane height per pass.
    assert!(after < before + s.radius * 0.6);
    // Inverted digs.
    let mut dig = sphere_doc(6);
    brush::stroke(&mut dig, &mut ClayBuildup::default(), &BrushSettings { invert: true, ..s.clone() }, &top_path(0.6)).unwrap();
    assert!(min_radius(&dig) < 0.99);
}

#[test]
fn trim_dynamic_only_removes_material() {
    let mut doc = sphere_doc(6);
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    let before: Vec<f32> = doc.positions().iter().map(|p| p.length()).collect();
    brush::stroke(&mut doc, &mut TrimDynamic { smooth_border: 0.0, ..Default::default() }, &s, &top_path(0.4)).unwrap();
    let grew = doc.positions().iter().zip(&before).filter(|(p, b)| p.length() > **b + 1e-5).count();
    let cut = doc.positions().iter().zip(&before).filter(|(p, b)| p.length() < **b - 1e-3).count();
    assert_eq!(grew, 0, "trim must not add material");
    assert!(cut > 10, "trim should flatten the dome");
    // The top becomes planar: heights near the pole cluster tightly.
    let ys: Vec<f32> = doc.positions().iter().filter(|p| p.y > 0.0 && p.x.abs() < 0.05 && p.z.abs() < 0.05).map(|p| p.y).collect();
    let spread = ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread < 0.004, "top should be flattened, spread {spread}");
}

#[test]
fn smooth_reduces_noise() {
    let mut doc = Document::from_mesh(grid(40, 2.0)).unwrap();
    let noisy: Vec<f32> = (0..doc.vertex_count()).map(|i| ((i * 7919) % 13) as f32 / 13.0 * 0.05).collect();
    let mut pts = doc.export_mesh(false);
    for (p, n) in pts.positions.iter_mut().zip(&noisy) {
        p.y = *n;
    }
    doc = Document::from_mesh(pts).unwrap();
    let rough = |d: &Document| {
        let t = d.topology();
        (0..d.vertex_count())
            .map(|v| {
                let r = t.vert_verts.row(v);
                let avg = r.iter().map(|&u| d.positions()[u as usize].y).sum::<f32>() / r.len() as f32;
                (avg - d.positions()[v].y).abs()
            })
            .sum::<f32>()
    };
    let before = rough(&doc);
    let s = BrushSettings { radius: 2.0, strength: 1.0, spacing: 0.5, front_faces_only: false, ..Default::default() };
    brush::stroke(&mut doc, &mut Smooth::default(), &s, &[(Vec3::new(0.0, 0.02, 0.0), 1.0)]).unwrap();
    assert!(rough(&doc) < before * 0.6);
}

#[test]
fn move_brush_and_topological_move() {
    // Two spheres close together: plain move drags both, topological only one.
    let mut m = quad_sphere(4, 0.5);
    let mut other = quad_sphere(4, 0.5);
    other.translate(Vec3::new(1.05, 0.0, 0.0));
    m.merge(&other);
    let s = BrushSettings { radius: 0.5, strength: 1.0, front_faces_only: false, ..Default::default() };

    for topological in [false, true] {
        let mut doc = Document::from_mesh(m.clone()).unwrap();
        let hit = doc.raycast(&Ray::new(Vec3::new(0.45, 0.0, 3.0), -Vec3::Z)).unwrap();
        let before = doc.positions()[hit.vertex as usize];
        let mut mv = MoveBrush::new(topological);
        mv.begin(&doc, &hit, &s);
        mv.drag(&mut doc, Vec3::new(0.0, 0.3, 0.0)).unwrap();
        let half = m.positions.len() / 2;
        let other_moved = doc.export_mesh(true).positions[half..]
            .iter()
            .zip(&m.positions[half..])
            .any(|(a, b)| (*a - *b).length() > 1e-4);
        assert_eq!(other_moved, !topological, "topological={topological}");
        // The grabbed vertex follows the cursor.
        let lift = doc.positions()[hit.vertex as usize].y - before.y;
        assert!(lift > 0.25, "lift {lift}");
        mv.end();
    }
}

#[test]
fn layers_opacity_mask_and_undo() {
    let mut doc = sphere_doc(5);
    let original = doc.positions().to_vec();
    let layer = doc.add_layer("Detail");
    let s = BrushSettings { radius: 0.3, strength: 0.8, ..Default::default() };
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).unwrap();
    assert!(doc.base_positions().iter().zip(&original).all(|(a, b)| a == b), "base untouched when sculpting a layer");
    let full = doc.positions().to_vec();
    let moved: Vec<usize> = (0..full.len()).filter(|&i| (full[i] - original[i]).length() > 1e-4).collect();
    assert!(!moved.is_empty());
    let sparse = doc.layer(layer).unwrap().allocated_leaves().len();
    assert!(sparse < doc.bvh().leaves.len(), "layer storage is sparse");

    // Strength slider scales the layer linearly.
    doc.set_layer_opacity(layer, 0.5).unwrap();
    for &i in &moved {
        let expect = original[i] + (full[i] - original[i]) * 0.5;
        assert!((doc.positions()[i] - expect).length() < 1e-5);
    }
    doc.set_layer_opacity(layer, 1.0).unwrap();

    // A gradient mask hides the layer on the -X side only.
    let mask = MaskStack::new(0.0).with(MaskLayer::new("ramp", MaskSource::Gradient { axis: Vec3::X, from: -0.01, to: 0.01 }));
    doc.set_layer_mask(layer, Some(mask)).unwrap();
    for &i in &moved {
        let p = doc.positions()[i];
        if original[i].x < -0.02 {
            assert!((p - original[i]).length() < 1e-6, "masked side hidden");
        } else if original[i].x > 0.02 {
            assert!((p - full[i]).length() < 1e-5, "unmasked side shown");
        }
    }
    doc.set_layer_mask(layer, None).unwrap();

    // Undo/redo of the stroke.
    assert!(doc.undo());
    assert!(doc.positions().iter().zip(&original).all(|(a, b)| (*a - *b).length() < 1e-6));
    assert!(doc.redo());
    assert!(doc.positions().iter().zip(&full).all(|(a, b)| (*a - *b).length() < 1e-6));

    // Locked layers refuse strokes.
    doc.set_layer_locked(layer, true).unwrap();
    assert!(brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).is_err());
}

#[test]
fn freeze_protects_vertices() {
    let mut doc = sphere_doc(5);
    let frozen = doc.positions().to_vec();
    doc.set_freeze(vec![1.0; doc.vertex_count()]).unwrap();
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).unwrap();
    assert_eq!(doc.positions(), &frozen[..]);

    // Paint freeze off in one spot, sculpt only lands there.
    let hit = top_hit(&doc);
    let mut unfreeze = PaintBrush { target: PaintTarget::Freeze, value: 0.0 };
    brush::stroke(&mut doc, &mut unfreeze, &BrushSettings { radius: 0.2, strength: 1.0, ..Default::default() }, &[(hit.point, 1.0)]).unwrap();
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).unwrap();
    let changed: Vec<Vec3> = (0..doc.vertex_count()).filter(|&i| doc.positions()[i] != frozen[i]).map(|i| frozen[i]).collect();
    assert!(!changed.is_empty());
    assert!(changed.iter().all(|p| p.distance(hit.point) < 0.2));
}

#[test]
fn mask_stack_blends_noise_with_painting() {
    let mut doc = sphere_doc(5);
    // Hand paint a channel at the top.
    let hit = top_hit(&doc);
    let mut paint = PaintBrush { target: PaintTarget::Channel("paint.wear".into()), value: 1.0 };
    brush::stroke(&mut doc, &mut paint, &BrushSettings { radius: 0.4, strength: 1.0, ..Default::default() }, &[(hit.point, 1.0)]).unwrap();
    // Undo removes the channel that the stroke created.
    assert!(doc.channel("paint.wear").is_some());
    doc.undo();
    assert!(doc.channel("paint.wear").is_none());
    doc.redo();

    let noise = NoiseParams { kind: NoiseKind::Fbm, scale: 3.0, seed: 7, ..Default::default() };
    let stack = MaskStack::new(0.0)
        .with(MaskLayer::new("noise", MaskSource::Noise(noise)).levels(Levels::range(0.3, 0.7)))
        .with(
            MaskLayer::new("paint", MaskSource::Channel { name: "paint.wear".into() })
                .blend(BlendMode::Multiply),
        );
    let values = doc.evaluate_mask(&stack).unwrap();
    let painted = doc.channel("paint.wear").unwrap();
    for v in 0..doc.vertex_count() {
        assert!((0.0..=1.0).contains(&values[v]));
        if painted[v] == 0.0 {
            assert_eq!(values[v], 0.0, "multiply by unpainted = 0");
        }
    }
    assert!(values.iter().any(|&x| x > 0.2) && values.iter().any(|&x| x > 0.0 && x < 0.9));

    // JSON round trip: stacks are data.
    let json = serde_json::to_string(&stack).unwrap();
    assert_eq!(serde_json::from_str::<MaskStack>(&json).unwrap(), stack);
    let from_other_tool: MaskStack = serde_json::from_str(
        r#"{"base":0,"layers":[{"source":{"type":"mesh","attribute":"curvature"}},
           {"source":{"type":"noise","kind":"cellular","scale":8},"blend":"screen","opacity":0.5}]}"#,
    )
    .unwrap();
    assert!(doc.evaluate_mask(&from_other_tool).is_ok());
}

#[test]
fn mesh_bakes_behave() {
    // A sphere: convex everywhere, unoccluded, thick.
    let mut doc = sphere_doc(4);
    doc.bake(MeshAttribute::AmbientOcclusion, &Default::default());
    doc.bake(MeshAttribute::Thickness, &Default::default());
    let ao = doc.channel("mesh.ao").unwrap();
    assert!(ao.iter().all(|&a| a > 0.95), "convex sphere has no occlusion");
    let th = doc.channel("mesh.thickness").unwrap();
    assert!(th.iter().all(|&t| t > 0.9), "sphere is thick relative to bake distance");

    // Dig a groove: its floor is concave and more occluded.
    let s = BrushSettings { radius: 0.15, strength: 1.0, invert: true, ..Default::default() };
    for _ in 0..3 {
        brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.8)).unwrap();
    }
    doc.bake(MeshAttribute::Curvature, &Default::default());
    doc.bake(MeshAttribute::AmbientOcclusion, &Default::default());
    let low = doc.positions().iter().enumerate().filter(|(_, p)| p.y > 0.5).min_by(|a, b| a.1.length().total_cmp(&b.1.length())).unwrap().0;
    assert!(doc.channel("mesh.curvature").unwrap()[low] < 0.5, "groove floor is concave");
    assert!(doc.channel("mesh.ao").unwrap()[low] < 0.95, "groove floor is occluded");
}

#[test]
fn subdivide_keeps_layers() {
    let mut doc = sphere_doc(3);
    doc.add_layer("L");
    let s = BrushSettings { radius: 0.4, strength: 1.0, ..Default::default() };
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).unwrap();
    let peak = max_radius(&doc);
    let faces = doc.face_count();
    doc.subdivide().unwrap();
    assert_eq!(doc.face_count(), faces * 4);
    assert_eq!(doc.level(), 1);
    assert!((max_radius(&doc) - peak).abs() < 0.05, "detail survives subdivision");
    // Turning the layer off returns to the (subdivided) base.
    let id = doc.layers()[0].id;
    doc.set_layer_visible(id, false).unwrap();
    assert!(max_radius(&doc) < 1.0001);
}

#[test]
fn pose_rotates_and_layers_follow() {
    let mut doc = sphere_doc(4);
    let id = doc.add_layer("detail");
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.3)).unwrap();
    let peak_before = doc.positions().iter().cloned().max_by(|a, b| a.y.total_cmp(&b.y)).unwrap();

    // Weight everything (rigid rotate) by 90 degrees about Z.
    let w = vec![1.0; doc.vertex_count()];
    doc.pose(&w, PoseTransform::Rotate { pivot: Vec3::ZERO, axis: Vec3::Z, angle: std::f32::consts::FRAC_PI_2 }).unwrap();
    let peak_after = doc.positions().iter().cloned().max_by(|a, b| (-a.x).total_cmp(&-b.x)).unwrap();
    assert!((peak_after.length() - peak_before.length()).abs() < 1e-3, "layer detail rotated with the base");
    assert!(peak_after.x < -1.0);
    assert!(doc.layer(id).is_some());

    // Topological weights fall off with geodesic distance; undo restores pose.
    let tw = doc.topological_weights(top_hit(&doc).vertex, 0.8, 0.5, 1);
    assert!(tw.iter().any(|&x| x > 0.9) && tw.iter().any(|&x| x == 0.0));
    assert!(doc.undo());
    let restored = doc.positions().iter().cloned().max_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
    assert!((restored - peak_before).length() < 1e-4);
}

#[test]
fn project_round_trip() {
    let mut doc = sphere_doc(4);
    let id = doc.add_layer("Wrinkles");
    let s = BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() };
    brush::stroke(&mut doc, &mut ClayBuildup::default(), &s, &top_path(0.5)).unwrap();
    let hit = top_hit(&doc);
    brush::stroke(
        &mut doc,
        &mut PaintBrush { target: PaintTarget::Channel("paint.a".into()), value: 1.0 },
        &BrushSettings { radius: 0.3, strength: 1.0, ..Default::default() },
        &[(hit.point, 1.0)],
    )
    .unwrap();
    let mask = MaskStack::new(0.2).with(MaskLayer::new("p", MaskSource::Channel { name: "paint.a".into() }).blend(BlendMode::Add));
    doc.set_layer_mask(id, Some(mask)).unwrap();
    doc.set_layer_opacity(id, 0.75).unwrap();
    doc.metadata.insert("ui.camera".into(), serde_json::json!({"fov": 40}));

    let dir = std::env::temp_dir().join(format!("sculpt-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    project::save(&doc, &dir).unwrap();
    let back = project::load(&dir).unwrap();
    let (a, b) = (doc.export_mesh(true), back.export_mesh(true));
    assert_eq!(a.faces, b.faces);
    for (p, q) in a.positions.iter().zip(&b.positions) {
        assert!((*p - *q).length() < 1e-6);
    }
    assert_eq!(back.layers()[0].name, "Wrinkles");
    assert_eq!(back.layers()[0].opacity, 0.75);
    assert_eq!(back.active_layer(), Some(id));
    assert_eq!(back.metadata["ui.camera"]["fov"], 40);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn obj_and_channel_import_use_source_order() {
    let src = quad_sphere(3, 1.0);
    let text = obj::to_string(&src);
    let parsed = obj::parse(&text).unwrap();
    assert_eq!(parsed.faces, src.faces);
    let mut doc = Document::from_mesh(parsed).unwrap();
    // External data: value = source index parity.
    let ext: Vec<f32> = (0..src.positions.len()).map(|i| (i % 2) as f32).collect();
    doc.import_channel("ext", &ext).unwrap();
    assert_eq!(doc.to_canonical(doc.channel("ext").unwrap()), ext);
    for c in [0usize, 1, 17] {
        let i = doc.canonical_index(c);
        assert_eq!(doc.positions()[i], src.positions[c]);
    }
    assert_eq!(mesh::permute(&(0..3).collect::<Vec<u32>>(), &[1, 2, 3]), vec![1, 2, 3]);
    let _ = PolyMesh::default();
}

