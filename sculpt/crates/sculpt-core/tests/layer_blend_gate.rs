//! Max and Min keep the parts of a layer that raise or lower the base surface, judged against the base's normals: a
//! tall column's walls face sideways in the composite, and judging there dropped parts of the column (spikes).

use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::geom::Ray;
use sculpt_core::mesh::face_normal;
use sculpt_core::primitives::quad_sphere;
use sculpt_core::{Document, LayerBlend, LayerId};

const R: f32 = 0.5;

fn top_z(doc: &Document, x: f32, y: f32) -> f32 {
    doc.raycast(&Ray::new(Vec3::new(x, y, 3.0), -Vec3::Z)).map_or(f32::NAN, |h| h.point.z)
}

fn inverted(doc: &Document) -> usize {
    let (pos, base) = (doc.positions(), doc.base_positions());
    doc.faces().iter().filter(|f| face_normal(pos, f).dot(face_normal(base, f)) < 0.0).count()
}

fn stroke_at(doc: &mut Document, brush: &mut ClayBuildup, s: &BrushSettings, cx: f32, half: f32) {
    let path: Vec<(Vec3, f32)> = (0..=16)
        .filter_map(|k| {
            let t = k as f32 / 16.0;
            let x = cx - half + 2.0 * half * if k <= 8 { t * 2.0 } else { 2.0 - t * 2.0 };
            doc.raycast(&Ray::new(Vec3::new(x, 0.0, 3.0), -Vec3::Z)).map(|h| (h.point, 1.0))
        })
        .collect();
    brush::stroke(doc, brush, s, &path).unwrap();
}

fn set_blend(doc: &mut Document, id: LayerId, b: LayerBlend) {
    let mut m = doc.layer(id).unwrap().meta();
    m.blend = b;
    doc.set_layer_meta(id, m, false).unwrap();
}

#[test]
fn max_keeps_a_tall_column_and_min_keeps_a_dent() {
    let mut doc = Document::from_mesh(quad_sphere(6, R)).unwrap();
    let id = doc.add_layer("clay");
    let s = BrushSettings { radius: 0.12, strength: 0.5, ..Default::default() };
    // A column at x = -0.15 twice the radius high (its walls face sideways and bulge) and a dent at x = +0.2.
    let mut tall = ClayBuildup { height: 0.32, ..Default::default() };
    for _ in 0..25 {
        stroke_at(&mut doc, &mut tall, &s, -0.15, 0.04);
    }
    let dent = BrushSettings { invert: true, radius: 0.08, ..s };
    for _ in 0..3 {
        stroke_at(&mut doc, &mut ClayBuildup::default(), &dent, 0.2, 0.03);
    }
    let base_top = |x: f32| (R * R - x * x).sqrt();
    let col = top_z(&doc, -0.15, 0.0) - base_top(-0.15);
    let pit = top_z(&doc, 0.2, 0.0) - base_top(0.2);
    eprintln!("column {col:.4}, dent {pit:.4}");
    assert!(col > 0.25 && pit < -0.005, "the setup builds a tall column and a dent");
    let add = doc.positions().to_vec();
    let add_inverted = inverted(&doc);

    set_blend(&mut doc, id, LayerBlend::Max);
    let (mcol, mpit) = (top_z(&doc, -0.15, 0.0) - base_top(-0.15), top_z(&doc, 0.2, 0.0) - base_top(0.2));
    eprintln!("Max: column {mcol:.4}, dent {mpit:.4}, inverted {} (Add {add_inverted})", inverted(&doc));
    assert!((mcol - col).abs() < 1e-5, "Max keeps the column whole: {mcol} vs {col}");
    assert!(mpit.abs() < 2e-3, "Max leaves the dent out: {mpit}");
    assert!(inverted(&doc) <= add_inverted, "Max folds no faces Add does not");
    // Around the column (away from the dent) Max is Add exactly.
    let pos = doc.positions();
    let near_col = doc.base_positions().iter().enumerate().filter(|(_, p)| p.x < 0.0 && p.z > 0.2).map(|(v, _)| v);
    let worst = near_col.map(|v| pos[v].distance(add[v])).fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "Max matches Add around the column, worst {worst}");

    set_blend(&mut doc, id, LayerBlend::Min);
    let (ncol, npit) = (top_z(&doc, -0.15, 0.0) - base_top(-0.15), top_z(&doc, 0.2, 0.0) - base_top(0.2));
    eprintln!("Min: column {ncol:.4}, dent {npit:.4}, inverted {}", inverted(&doc));
    assert!(ncol.abs() < 2e-3, "Min leaves the column out: {ncol}");
    assert!((npit - pit).abs() < 1e-5, "Min keeps the dent: {npit} vs {pit}");
    assert_eq!(inverted(&doc), 0, "Min folds no faces");

    set_blend(&mut doc, id, LayerBlend::Add);
    let back = doc.positions().iter().zip(&add).map(|(a, b)| a.distance(*b)).fold(0.0f32, f32::max);
    assert!(back < 1e-5, "back at Add is the surface as sculpted (to rounding: strokes add up dab by dab), {back}");
}
