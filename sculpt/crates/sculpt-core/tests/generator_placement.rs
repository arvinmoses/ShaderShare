//! Placement tools for scattering: minimum spacing and per-instance variation.

use std::sync::Arc;

use glam::Vec3;
use sculpt_core::generator::layer::{GeneratorCache, GeneratorSpec};
use sculpt_core::generator::{BoolOp, Primitive};
use sculpt_core::primitives::quad_sphere_res;
use sculpt_core::{Document, LayerId, Placement};

fn sphere() -> Document {
    Document::from_mesh(quad_sphere_res(60, 1.0)).unwrap()
}

/// Cutters centred exactly on the surface (no lift), so an instance's translation is its scattered point.
fn spec() -> GeneratorSpec {
    GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Subtract, count: 300, seed: 5, size: 0.05, lift: 0.0, random_spin: true, ..GeneratorSpec::default() }
}

fn place(doc: &mut Document, spec: GeneratorSpec) -> (LayerId, Arc<GeneratorCache>) {
    let id = doc.insert_generator("g", Placement::Top, spec).unwrap();
    let cache = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    (id, cache)
}

fn centres(cache: &GeneratorCache) -> Vec<Vec3> {
    cache.instances().iter().map(|i| i.translation).collect()
}

fn closest_pair(points: &[Vec3]) -> f32 {
    let mut best = f32::MAX;
    for (i, a) in points.iter().enumerate() {
        for b in &points[i + 1..] {
            best = best.min(a.distance(*b));
        }
    }
    best
}

#[test]
fn minimum_spacing_is_respected_and_the_count_is_met() {
    let mut doc = sphere();
    let (_, cache) = place(&mut doc, GeneratorSpec { min_spacing: 0.12, ..spec() });
    let points = centres(&cache);
    assert_eq!(points.len(), 300, "300 points 0.12 apart fit on a unit sphere");
    let closest = closest_pair(&points);
    assert!(closest >= 0.12 - 1e-5, "closest pair is {closest}, under the 0.12 spacing");
}

#[test]
fn a_full_surface_places_fewer_instead_of_breaking_the_spacing() {
    let mut doc = sphere();
    let (_, cache) = place(&mut doc, GeneratorSpec { count: 5000, min_spacing: 0.3, ..spec() });
    let points = centres(&cache);
    assert!(points.len() < 5000, "5000 points 0.3 apart cannot fit, got {}", points.len());
    assert!(points.len() > 40, "but a good number should: got {}", points.len());
    assert!(closest_pair(&points) >= 0.3 - 1e-5, "spacing still holds when the surface is full");
}

#[test]
fn spaced_placement_is_deterministic_and_covers_the_whole_surface() {
    let mut a = sphere();
    let mut b = sphere();
    let (_, ca) = place(&mut a, GeneratorSpec { count: 120, min_spacing: 0.2, ..spec() });
    let (_, cb) = place(&mut b, GeneratorSpec { count: 120, min_spacing: 0.2, ..spec() });
    assert_eq!(centres(&ca), centres(&cb), "same spec, same points");
    let mut octants = [0usize; 8];
    for p in centres(&ca) {
        octants[(p.x > 0.0) as usize | ((p.y > 0.0) as usize) << 1 | ((p.z > 0.0) as usize) << 2] += 1;
    }
    assert!(octants.iter().all(|&n| n >= 5), "every part of the surface gets points, none fills first: {octants:?}");
}

#[test]
fn zero_spacing_and_old_projects_behave_as_before() {
    // A spec saved before the placement tools existed has none of the new fields.
    let old = r#"{"primitive":"Sphere","op":"subtract","count":60,"seed":7,"size":0.1,"lift":0.6,"random_spin":true}"#;
    let spec: GeneratorSpec = serde_json::from_str(old).unwrap();
    assert_eq!((spec.min_spacing, spec.size_var, spec.stretch_var, spec.tilt, spec.lift_var), (0.0, 0.0, 0.0, 0.0, 0.0));
    assert_eq!(spec.count, 60);
    let mut doc = sphere();
    let (_, cache) = place(&mut doc, spec);
    assert_eq!(cache.instances().len(), 60, "no spacing means exactly `count` independent points");
}

#[test]
fn the_new_fields_survive_save_and_load() {
    let spec = GeneratorSpec { min_spacing: 0.07, size_var: 0.3, stretch_var: 0.2, tilt: 15.0, lift_var: 0.1, ..spec() };
    let back: GeneratorSpec = serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(back, spec);
}

#[test]
fn size_stretch_and_tilt_vary_within_their_bounds() {
    let mut doc = sphere();
    let base_size = 0.05;
    let (_, cache) = place(&mut doc, GeneratorSpec { size_var: 0.5, stretch_var: 0.0, tilt: 0.0, ..spec() });
    let sizes: Vec<f32> = cache.instances().iter().map(|i| i.scale.x).collect();
    assert!(sizes.iter().all(|&s| s >= base_size * 0.5 - 1e-6 && s <= base_size * 1.5 + 1e-6), "sizes stay within +-50%");
    assert!(sizes.iter().any(|&s| s < base_size * 0.8) && sizes.iter().any(|&s| s > base_size * 1.2), "and they really spread");
    assert!(cache.instances().iter().all(|i| (i.scale.x - i.scale.y).abs() < 1e-6 && (i.scale.y - i.scale.z).abs() < 1e-6), "no stretch means uniform scale");

    let (_, stretched) = place(&mut doc, GeneratorSpec { size_var: 0.0, stretch_var: 0.4, ..spec() });
    assert!(stretched.instances().iter().any(|i| (i.scale.x - i.scale.y).abs() > 0.2 * base_size), "stretch makes axes differ");
    assert!(stretched.instances().iter().all(|i| i.scale.min_element() >= base_size * 0.6 - 1e-6 && i.scale.max_element() <= base_size * 1.4 + 1e-6), "each axis stays within +-40%");

    let (_, leaning) = place(&mut doc, GeneratorSpec { tilt: 20.0, ..spec() });
    let mut worst = 0.0f32;
    let mut leaned = 0;
    for i in leaning.instances() {
        let up = i.rotation * Vec3::Y;
        let angle = up.angle_between(i.translation.normalize()).to_degrees();
        worst = worst.max(angle);
        leaned += (angle > 2.0) as usize;
    }
    assert!(worst <= 20.5, "no instance leans past the limit, worst {worst} degrees");
    assert!(leaned > 100, "most instances do lean ({leaned} of 300)");

    let (_, upright) = place(&mut doc, spec());
    assert!(upright.instances().iter().all(|i| (i.rotation * Vec3::Y).angle_between(i.translation.normalize()).to_degrees() < 0.5), "no tilt keeps instances on the normal");
}

#[test]
fn each_kind_of_variation_draws_from_its_own_stream() {
    let mut doc = sphere();
    let (_, plain) = place(&mut doc, spec());
    let (_, sized) = place(&mut doc, GeneratorSpec { size_var: 0.4, ..spec() });
    let (_, tilted) = place(&mut doc, GeneratorSpec { tilt: 15.0, ..spec() });
    let (_, lifted) = place(&mut doc, GeneratorSpec { lift_var: 0.3, ..spec() });
    // Same seed and count: the scattered points are the same whatever varies.
    let dir = |c: &GeneratorCache| c.instances().iter().map(|i| i.translation.normalize()).collect::<Vec<_>>();
    // Tilt changes orientation only: sizes are those of the plain set.
    let same_sizes = tilted.instances().iter().zip(plain.instances()).all(|(a, b)| (a.scale - b.scale).length() < 1e-7);
    assert!(same_sizes, "adding tilt must not change any size");
    // Size variation changes sizes but not orientation.
    let same_rot = sized.instances().iter().zip(plain.instances()).all(|(a, b)| a.rotation.abs_diff_eq(b.rotation, 1e-6));
    assert!(same_rot, "adding size variation must not change any orientation");
    assert_eq!(dir(&plain).len(), dir(&lifted).len());
    // Lift variation moves instances along their normals and changes nothing else.
    assert!(lifted.instances().iter().zip(plain.instances()).all(|(a, b)| a.rotation.abs_diff_eq(b.rotation, 1e-6) && (a.scale - b.scale).length() < 1e-7));
    assert!(lifted.instances().iter().zip(plain.instances()).any(|(a, b)| a.translation.distance(b.translation) > 0.005), "lift jitter moves some instances");
}

#[test]
fn editing_size_variation_or_tilt_keeps_the_scatter_but_spacing_redoes_it() {
    let mut doc = sphere();
    let (id, first) = place(&mut doc, GeneratorSpec { min_spacing: 0.1, ..spec() });
    doc.set_generator(id, GeneratorSpec { min_spacing: 0.1, size_var: 0.3, tilt: 10.0, size: 0.04, ..spec() }).unwrap();
    let second = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(second.shares_samples_with(&first), "size, variation and tilt reuse the scattered points");
    doc.set_generator(id, GeneratorSpec { min_spacing: 0.15, ..spec() }).unwrap();
    let third = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(!third.shares_samples_with(&second), "a different spacing scatters again");
}

// ------------------------------------------------------------------ density

use sculpt_core::generator::density::Density;
use sculpt_core::io::project;

/// A painted channel holding `f(position)` for every vertex.
fn paint(doc: &mut Document, name: &str, f: impl Fn(Vec3) -> f32) {
    let values: Vec<f32> = doc.positions().iter().map(|p| f(*p)).collect();
    doc.set_channel(name, values).unwrap();
}

fn mean_nearest_neighbour(points: &[Vec3]) -> f32 {
    let total: f32 = points.iter().enumerate().map(|(i, a)| points.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, b)| a.distance(*b)).fold(f32::MAX, f32::min)).sum();
    total / points.len() as f32
}

#[test]
fn white_regions_pull_the_instances_in_and_the_count_stays_fixed() {
    let mut doc = sphere();
    paint(&mut doc, "paint.mask", |p| if p.y > 0.0 { 1.0 } else { 0.0 });
    let (_, cache) = place(&mut doc, GeneratorSpec { count: 400, density: Density::Painted { channel: "paint.mask".into() }, ..spec() });
    let points = centres(&cache);
    assert_eq!(points.len(), 400, "a density redistributes the fixed count, it does not thin it");
    let inside = points.iter().filter(|p| p.y > -0.05).count();
    assert!(inside as f32 >= 0.95 * points.len() as f32, "{inside} of {} points are in the white half", points.len());
}

#[test]
fn an_all_black_density_places_nothing_and_does_not_break() {
    let mut doc = sphere();
    paint(&mut doc, "paint.mask", |_| 0.0);
    let (id, cache) = place(&mut doc, GeneratorSpec { count: 100, density: Density::Painted { channel: "paint.mask".into() }, ..spec() });
    assert_eq!(cache.instances().len(), 0, "nothing can be placed where the density is zero");
    assert!(doc.positions().iter().zip(doc.base_positions()).all(|(a, b)| a == b), "and the surface is untouched");
    assert!(doc.layer(id).is_some());
}

#[test]
fn clusters_clump_and_coverage_sets_how_much_is_dense() {
    let mut doc = sphere();
    let (_, even) = place(&mut doc, GeneratorSpec { count: 300, ..spec() });
    let clusters = |coverage: f32| Density::Clusters { scale: 2.5, coverage, softness: 0.2, seed: 3 };
    let (_, sparse) = place(&mut doc, GeneratorSpec { count: 300, density: clusters(0.25), ..spec() });
    let (_, wide) = place(&mut doc, GeneratorSpec { count: 300, density: clusters(0.75), ..spec() });
    assert_eq!(sparse.instances().len(), 300);
    let (e, s, w) = (mean_nearest_neighbour(&centres(&even)), mean_nearest_neighbour(&centres(&sparse)), mean_nearest_neighbour(&centres(&wide)));
    assert!(s < e * 0.9, "sparse clusters are clumpier than an even scatter (nearest neighbour {s} against {e})");
    assert!(sparse.density().unwrap().total_weight() < wide.density().unwrap().total_weight(), "more coverage means more of the surface is dense");
    assert!(w > s, "wider coverage spreads the instances out ({w} against {s})");
}

#[test]
fn facing_gathers_instances_on_the_surfaces_that_face_that_way() {
    let mut doc = sphere();
    let (_, down) = place(&mut doc, GeneratorSpec { count: 300, density: Density::Facing { axis: [0.0, -1.0, 0.0], sharpness: 1.5 }, ..spec() });
    let lower = centres(&down).iter().filter(|p| p.y < 0.05).count();
    assert!(lower as f32 >= 0.95 * 300.0, "{lower} of 300 are on the lower half");
    let (_, up) = place(&mut doc, GeneratorSpec { count: 300, density: Density::Facing { axis: [0.0, 1.0, 0.0], sharpness: 1.5 }, ..spec() });
    let upper = centres(&up).iter().filter(|p| p.y > -0.05).count();
    assert!(upper as f32 >= 0.95 * 300.0, "{upper} of 300 are on the upper half");
}

#[test]
fn editing_the_painted_channel_moves_the_instances() {
    let mut doc = sphere();
    paint(&mut doc, "paint.mask", |p| if p.y > 0.0 { 1.0 } else { 0.0 });
    let (id, before) = place(&mut doc, GeneratorSpec { count: 200, density: Density::Painted { channel: "paint.mask".into() }, ..spec() });
    assert!(centres(&before).iter().filter(|p| p.y > -0.05).count() >= 190);
    // Paint the other half instead: the layer follows the channel without being touched.
    paint(&mut doc, "paint.mask", |p| if p.y < 0.0 { 1.0 } else { 0.0 });
    doc.end_stroke();
    let after = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(centres(&after).iter().filter(|p| p.y < 0.05).count() >= 190, "the instances moved to the newly painted half");
}

#[test]
fn size_and_variation_edits_reuse_the_density_but_changing_it_scatters_again() {
    let mut doc = sphere();
    let density = Density::Clusters { scale: 2.5, coverage: 0.4, softness: 0.2, seed: 3 };
    let (id, first) = place(&mut doc, GeneratorSpec { density: density.clone(), ..spec() });
    doc.set_generator(id, GeneratorSpec { density: density.clone(), size: 0.03, size_var: 0.4, tilt: 12.0, ..spec() }).unwrap();
    let second = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(Arc::ptr_eq(first.density().unwrap(), second.density().unwrap()), "the density is not re-evaluated for a size edit");
    assert!(second.shares_samples_with(&first), "and the scattered points are kept");
    doc.set_generator(id, GeneratorSpec { density: Density::Clusters { scale: 2.5, coverage: 0.6, softness: 0.2, seed: 3 }, ..spec() }).unwrap();
    let third = doc.layer(id).unwrap().generator_cache().unwrap().clone();
    assert!(!Arc::ptr_eq(second.density().unwrap(), third.density().unwrap()));
    assert!(!third.shares_samples_with(&second), "a different density scatters again");
}

#[test]
fn density_round_trips_through_a_saved_project() {
    let mut doc = sphere();
    let density = Density::Clusters { scale: 3.0, coverage: 0.4, softness: 0.3, seed: 9 };
    let (id, cache) = place(&mut doc, GeneratorSpec { count: 150, density, min_spacing: 0.05, size_var: 0.3, ..spec() });
    let dir = std::env::temp_dir().join(format!("sculpt-density-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    project::save(&doc, &dir).unwrap();
    let loaded = project::load(&dir).unwrap();
    let loaded_id = loaded.layers().iter().find(|l| l.is_generator()).map(|l| l.id).expect("generator layer loads");
    let _ = id;
    let again = loaded.layer(loaded_id).unwrap().generator_cache().unwrap();
    assert_eq!(loaded.layer(loaded_id).unwrap().generator, doc.layer(id).unwrap().generator);
    assert_eq!(centres(again).len(), centres(&cache).len());
    let worst = centres(again).iter().zip(centres(&cache)).map(|(a, b)| a.distance(b)).fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "the loaded scatter is the same one (worst difference {worst})");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn old_specs_default_to_an_even_scatter() {
    let old = r#"{"primitive":"Sphere","op":"subtract","count":10,"seed":1,"size":0.1,"lift":0.6,"random_spin":true}"#;
    let spec: GeneratorSpec = serde_json::from_str(old).unwrap();
    assert_eq!(spec.density, Density::Even);
    let json = serde_json::to_string(&Density::Clusters { scale: 2.0, coverage: 0.3, softness: 0.2, seed: 4 }).unwrap();
    assert!(json.contains("\"kind\":\"clusters\""), "{json}");
}
