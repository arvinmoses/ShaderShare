//! Attribute mapping: the surface decides how each instance looks.

use std::sync::Arc;

use glam::Vec3;
use sculpt_core::bake::MeshAttribute;
use sculpt_core::generator::layer::{GeneratorCache, GeneratorSpec};
use sculpt_core::generator::mapping::{AttributeRow, AttributeSource, AttributeTarget, Remap};
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere_res;
use sculpt_core::{Document, LayerId, Placement};

const SIZE: f32 = 0.05;

fn sphere() -> Document {
    Document::from_mesh(quad_sphere_res(60, 1.0)).unwrap()
}

/// Cutters centred on the surface with no random variation, so every difference comes from the rows.
fn spec(mapping: Vec<AttributeRow>) -> GeneratorSpec {
    GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 400, seed: 5, size: SIZE, lift: 0.0, random_spin: true, mapping, ..GeneratorSpec::default() }
}

fn place(doc: &mut Document, spec: GeneratorSpec) -> (LayerId, Arc<GeneratorCache>) {
    let id = doc.insert_generator("g", Placement::Top, spec).unwrap();
    let cache = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    (id, cache)
}

fn up() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

fn row(source: AttributeSource, target: AttributeTarget, strength: f32) -> AttributeRow {
    AttributeRow::new(source, target, strength)
}

#[test]
fn slope_makes_instances_bigger_where_the_surface_faces_up() {
    let mut doc = sphere();
    let (_, cache) = place(&mut doc, spec(vec![row(AttributeSource::Slope { axis: up() }, AttributeTarget::Size, 1.0)]));
    let on_top: Vec<f32> = cache.instances().iter().filter(|i| i.translation.y > 0.9).map(|i| i.scale.x).collect();
    let on_side: Vec<f32> = cache.instances().iter().filter(|i| i.translation.y.abs() < 0.1).map(|i| i.scale.x).collect();
    assert!(!on_top.is_empty() && !on_side.is_empty());
    assert!(on_top.iter().all(|&s| s > SIZE * 1.8), "straight up doubles the size");
    assert!(on_side.iter().all(|&s| s < SIZE * 1.15), "edge-on leaves it alone");
    // Between the two the size grows steadily with how far up it is.
    let mut by_height: Vec<(f32, f32)> = cache.instances().iter().filter(|i| i.translation.y > 0.0).map(|i| (i.translation.y, i.scale.x)).collect();
    by_height.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(by_height.windows(2).all(|w| w[1].1 >= w[0].1 - 1e-4), "size never falls as the surface turns upward");
}

#[test]
fn height_lifts_instances_off_the_surface() {
    let mut doc = sphere();
    let rows = vec![row(AttributeSource::Height { axis: up(), from: -1.0, to: 1.0 }, AttributeTarget::Lift, 1.0)];
    let (_, cache) = place(&mut doc, spec(rows));
    for i in cache.instances().iter().step_by(10) {
        let dir = i.translation.normalize();
        let t = (dir.y + 1.0) * 0.5;
        // The centre sits `lift * size` above the surface, and lift = t.
        let lifted = i.translation.length() - 1.0;
        assert!((lifted - t * i.scale.x).abs() < 0.01, "lift is {lifted}, expected {}", t * i.scale.x);
    }
}

#[test]
fn a_painted_channel_is_read_with_the_surface_interpolation() {
    let mut doc = sphere();
    let values: Vec<f32> = doc.positions().iter().map(|p| p.y.max(0.0)).collect();
    doc.set_channel("paint.mask", values).unwrap();
    let rows = vec![row(AttributeSource::Channel { name: "paint.mask".into() }, AttributeTarget::Size, 1.0)];
    let (_, cache) = place(&mut doc, spec(rows));
    for i in cache.instances() {
        let want = SIZE * (1.0 + i.translation.normalize().y.max(0.0));
        assert!((i.scale.x - want).abs() < SIZE * 0.06, "size {} against {want}", i.scale.x);
    }
}

#[test]
fn a_mesh_attribute_is_baked_when_a_row_needs_it() {
    let mut doc = sphere();
    assert!(doc.channel(MeshAttribute::Cavity.channel_name()).is_none());
    let rows = vec![row(AttributeSource::Mesh { attribute: MeshAttribute::Cavity }, AttributeTarget::Size, 1.0)];
    let (_, cache) = place(&mut doc, spec(rows));
    assert!(doc.channel(MeshAttribute::Cavity.channel_name()).is_some(), "the cavity bake exists after placing");
    assert!(cache.instances().iter().all(|i| i.scale.x >= SIZE - 1e-6 && i.scale.x <= SIZE * 2.0 + 1e-6), "sizes stay within the row's range");
}

#[test]
fn order_noise_and_random_each_do_their_own_thing() {
    let mut doc = sphere();
    let (_, ordered) = place(&mut doc, spec(vec![row(AttributeSource::Index, AttributeTarget::Size, 1.0)]));
    let sizes: Vec<f32> = ordered.instances().iter().map(|i| i.scale.x).collect();
    assert!(sizes.windows(2).all(|w| w[1] > w[0]), "each instance is a little bigger than the one before");
    assert!((sizes[0] - SIZE).abs() < 1e-6 && (sizes[sizes.len() - 1] - SIZE * 2.0).abs() < 1e-5, "from the base size to double");

    let noisy = |doc: &mut Document| place(doc, spec(vec![row(AttributeSource::Noise { scale: 3.0, seed: 2 }, AttributeTarget::Size, 1.0)])).1;
    let a = noisy(&mut doc);
    let b = noisy(&mut doc);
    let sizes: Vec<f32> = a.instances().iter().map(|i| i.scale.x).collect();
    assert!(sizes.iter().all(|&s| (SIZE - 1e-6..=SIZE * 2.0 + 1e-6).contains(&s)));
    assert!(sizes.iter().cloned().fold(0.0, f32::max) - sizes.iter().cloned().fold(f32::MAX, f32::min) > SIZE * 0.3, "the noise really spreads the sizes");
    assert!(a.instances().iter().zip(b.instances()).all(|(x, y)| x.scale == y.scale), "and the same noise gives the same sizes");

    let (_, leaning) = place(&mut doc, spec(vec![row(AttributeSource::Random { seed: 1 }, AttributeTarget::Tilt, 30.0)]));
    let mut worst = 0.0f32;
    for i in leaning.instances() {
        worst = worst.max((i.rotation * Vec3::Y).angle_between(i.translation.normalize()).to_degrees());
    }
    assert!(worst <= 30.5 && worst > 5.0, "instances lean up to the mapped 30 degrees, worst {worst}");
    assert!(leaning.instances().iter().all(|i| (i.scale.x - SIZE).abs() < 1e-6), "and the tilt row leaves sizes alone");
}

#[test]
fn a_height_row_stretches_only_along_the_normal() {
    let mut doc = sphere();
    let (_, cache) = place(&mut doc, spec(vec![row(AttributeSource::Slope { axis: up() }, AttributeTarget::Stretch, 1.5)]));
    let top = cache.instances().iter().find(|i| i.translation.y > 0.95).unwrap();
    assert!((top.scale.x - SIZE).abs() < 1e-6 && (top.scale.z - SIZE).abs() < 1e-6, "the width is unchanged");
    assert!(top.scale.y > SIZE * 2.3, "the height grows by 1 + 1.5 where the surface faces up, got {}", top.scale.y);
}

#[test]
fn the_remap_can_threshold_and_invert() {
    let mut doc = sphere();
    let mut r = row(AttributeSource::Slope { axis: up() }, AttributeTarget::Size, 1.0);
    r.remap = Remap { in_min: 0.5, in_max: 0.5, gamma: 1.0, invert: false };
    let (_, hard) = place(&mut doc, spec(vec![r.clone()]));
    for i in hard.instances() {
        let slope = i.translation.normalize().y.max(0.0);
        let want = if slope >= 0.5 { SIZE * 2.0 } else { SIZE };
        assert!((i.scale.x - want).abs() < SIZE * 0.05, "slope {slope} gave {}", i.scale.x);
    }
    r.remap.invert = true;
    let (_, flipped) = place(&mut doc, spec(vec![r]));
    for (a, b) in hard.instances().iter().zip(flipped.instances()) {
        assert!((a.scale.x + b.scale.x - SIZE * 3.0).abs() < SIZE * 0.05, "inverting swaps big and small");
    }
}

#[test]
fn disabled_or_zero_strength_rows_change_nothing() {
    let mut doc = sphere();
    let (_, plain) = place(&mut doc, spec(vec![]));
    let mut off = row(AttributeSource::Slope { axis: up() }, AttributeTarget::Size, 1.0);
    off.enabled = false;
    let (_, with_off) = place(&mut doc, spec(vec![off, row(AttributeSource::Index, AttributeTarget::Lift, 0.0)]));
    assert!(plain.instances().iter().zip(with_off.instances()).all(|(a, b)| a.scale == b.scale && a.translation == b.translation && a.rotation == b.rotation));
}

#[test]
fn editing_a_row_rebuilds_the_instances_but_not_the_scatter() {
    let mut doc = sphere();
    let (id, first) = place(&mut doc, spec(vec![row(AttributeSource::Slope { axis: up() }, AttributeTarget::Size, 0.5)]));
    doc.set_generator(id, spec(vec![row(AttributeSource::Slope { axis: up() }, AttributeTarget::Size, 1.0)])).unwrap();
    let second = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(second.shares_samples_with(&first), "changing a strength reuses the scattered points");
    let top_first = first.instances().iter().find(|i| i.translation.y > 0.95).unwrap().scale.x;
    let top_second = second.instances().iter().find(|i| i.translation.y > 0.95).unwrap().scale.x;
    assert!(top_second > top_first * 1.2, "but the sizes moved ({top_first} to {top_second})");
    doc.set_generator(id, spec(vec![row(AttributeSource::Index, AttributeTarget::Size, 1.0)])).unwrap();
    let third = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(third.shares_samples_with(&second), "a different source still does not scatter again");
}

#[test]
fn editing_a_channel_a_row_reads_updates_the_instances() {
    let mut doc = sphere();
    let paint = |doc: &mut Document, f: fn(Vec3) -> f32| {
        let values: Vec<f32> = doc.positions().iter().map(|p| f(*p)).collect();
        doc.set_channel("paint.mask", values).unwrap();
    };
    paint(&mut doc, |_| 0.0);
    let rows = vec![row(AttributeSource::Channel { name: "paint.mask".into() }, AttributeTarget::Size, 1.0)];
    let (id, before) = place(&mut doc, spec(rows));
    assert!(before.instances().iter().all(|i| (i.scale.x - SIZE).abs() < 1e-6), "an empty channel leaves sizes alone");
    paint(&mut doc, |_| 1.0);
    doc.end_stroke();
    let after = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(after.instances().iter().all(|i| (i.scale.x - SIZE * 2.0).abs() < 1e-5), "painting the channel updates the layer");
}

#[test]
fn rows_survive_save_and_old_specs_have_none() {
    let rows = vec![
        row(AttributeSource::Mesh { attribute: MeshAttribute::Cavity }, AttributeTarget::Size, 1.2),
        AttributeRow { remap: Remap { in_min: 0.2, in_max: 0.8, gamma: 1.5, invert: true }, ..row(AttributeSource::Height { axis: up(), from: -1.0, to: 1.0 }, AttributeTarget::Lift, -0.4) },
    ];
    let spec = spec(rows);
    let back: GeneratorSpec = serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(back, spec);
    let old = r#"{"primitive":"Sphere","op":"subtract","count":10,"seed":1,"size":0.1,"lift":0.6,"random_spin":true}"#;
    assert!(serde_json::from_str::<GeneratorSpec>(old).unwrap().mapping.is_empty());
}
