//! Volume brushes behave like the plane brushes of mesh sculpting: clay fills cavities before it builds and keeps
//! building as you scrub (a few lifts a pass), trim flattens bumps down to a plane and does not tunnel, and the lattice grows
//! instead of clipping the solid flat.

use glam::Vec3;
use sculpt_core::generator::volume::Volume;

const R: f32 = 0.3;
const H: f32 = 0.02;

/// A stroke of dabs at `at`, as the app makes them (each at the surface under the cursor).
fn stroke_at(v: &mut Volume, at: &[Vec3], h: f32, trim: bool) {
    v.begin_stroke();
    for &p in at {
        let dir = p.normalize();
        let (hit, _) = v.raycast(dir * 8.0, -dir).unwrap();
        v.plane_dab(hit, dir, R, h, 0.5, 0.3, trim).unwrap();
    }
    v.end_stroke();
}

fn spot(n: usize) -> Vec<Vec3> {
    vec![Vec3::Z; n]
}

/// A pass back and forth across the top.
fn scrub() -> Vec<Vec3> {
    (0..60).map(|i| Vec3::new(0.25 * (i as f32 * 0.21).sin(), 0.0, 1.0)).collect()
}

/// Height of the surface, looking down at `x` along z.
fn height(v: &Volume, x: f32) -> f32 {
    v.raycast(Vec3::new(x, 0.0, 8.0), Vec3::NEG_Z).unwrap().0.z
}

#[test]
fn clay_fills_a_cavity_before_it_builds() {
    // A ball with a dimple in the top.
    let mut v = Volume::from_field(&|p: Vec3| (p.length() - 1.0).max(0.1 - (p - Vec3::Z).length()), 1.6, 0.01);
    let (pit, side) = (height(&v, 0.0), height(&v, 0.2));
    stroke_at(&mut v, &spot(6), H, false);
    let (pit2, side2) = (height(&v, 0.0), height(&v, 0.2));
    eprintln!("clay over a dimple: the dimple rose {:.3}, the surface beside it {:.3}", pit2 - pit, side2 - side);
    assert!(pit2 - pit > 0.06, "the cavity did not fill");
    assert!(side2 - side < 0.4 * (pit2 - pit), "it built up beside the cavity as much as in it");
}

#[test]
fn scrubbing_keeps_building_without_running_away() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    let mut rise = Vec::new();
    for _ in 0..4 {
        stroke_at(&mut v, &scrub(), H, false);
        rise.push(height(&v, 0.0) - 1.0);
    }
    eprintln!("clay: the top after each scrubbing pass {rise:.3?} (plane lift {H})");
    assert!(rise[0] > 0.5 * H && rise[0] < 8.0 * H, "{}", rise[0]);
    assert!(rise.windows(2).all(|w| w[1] > w[0] + 0.3 * H), "passes stopped building");
    // Each pass builds a few lifts' worth, not dozens (a pass is about a dozen overlapping dabs at any point). A cursor
    // held still makes no dabs in the app (they are spaced by distance), so scrubbing is the case that can run away.
    assert!(rise.windows(2).all(|w| w[1] - w[0] < 6.0 * H), "a pass built too much: {rise:?}");
}

#[test]
fn trim_flattens_a_bump_and_does_not_tunnel() {
    // A ball with a bump on top.
    let mut v = Volume::from_field(&|p: Vec3| (p.length() - 1.0).min((p - Vec3::Z).length() - 0.08), 1.6, 0.01);
    let (bump, side) = (height(&v, 0.0), height(&v, 0.2));
    stroke_at(&mut v, &spot(20), 0.01, true);
    let (bump2, side2) = (height(&v, 0.0), height(&v, 0.2));
    eprintln!("trim over a bump: the bump went {bump:.3} -> {bump2:.3}, beside it {side:.3} -> {side2:.3}");
    assert!(bump - bump2 > 0.05, "the bump was not cut down");
    assert!(side - side2 < 0.03, "it dug beside the bump");
    // In one place, many dabs settle instead of tunnelling.
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    stroke_at(&mut v, &spot(100), H, true);
    let depth = 1.0 - height(&v, 0.0);
    eprintln!("trim: 100 dabs in one place dug {depth:.3}");
    assert!(depth < 6.0 * H, "{depth}");
}

#[test]
fn the_lattice_grows_instead_of_clipping() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, 0.02);
    let (_, hi) = v.bounds();
    for _ in 0..40 {
        stroke_at(&mut v, &spot(10), 0.05, false);
    }
    let reached = height(&v, 0.0);
    let (_, hi2) = v.bounds();
    eprintln!("clay reached z = {reached:.2}; the lattice top went {:.2} -> {:.2}", hi.z, hi2.z);
    assert!(reached > hi.z + 0.2, "clay stopped at the old lattice edge {}", hi.z);
    assert!(reached < hi2.z - 0.1);
    let mesh = v.mesh();
    let open = sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces);
    assert_eq!(open, 0, "non-manifold edges after growing");
}

#[test]
fn the_surface_stays_manifold() {
    use sculpt_core::generator::remesh::non_manifold_edges;
    // Heavy, overlapping clay and carving of every size.
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.012);
    v.extract_all();
    v.begin_stroke();
    for i in 0..300 {
        let a = i as f32 * 0.37;
        let dir = Vec3::new(a.sin() * (i as f32 * 0.11).cos(), (i as f32 * 0.23).sin(), a.cos()).normalize();
        let r = 0.03 + 0.25 * ((i * 7 % 11) as f32 / 10.0);
        v.dab(dir * (1.0 - 0.3 * r), r, 0.2 * r, i % 3 == 0);
    }
    v.end_stroke();
    let mesh = v.mesh();
    let open = non_manifold_edges(&mesh.faces);
    eprintln!("after 300 dabs: {} quads, {open} non-manifold edges", mesh.faces.len());
    assert_eq!(open, 0);
    // A field that is noise at the scale of the lattice: every ambiguous case there is.
    let noise = |p: Vec3| (p.length() - 1.0) + 0.02 * ((p.x * 211.0).sin() * (p.y * 197.0).sin() * (p.z * 223.0).sin());
    let v = Volume::from_field(&noise, 1.4, 0.01);
    let mut v = v;
    v.extract_all();
    let mesh = v.mesh();
    let open = non_manifold_edges(&mesh.faces);
    eprintln!("lattice-scale noise: {} quads, {open} non-manifold edges", mesh.faces.len());
    assert_eq!(open, 0);
}

#[test]
#[ignore]
fn time_dabs() {
    let voxel = (4.0 * std::f32::consts::PI / 5e5f32).sqrt();
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.8, voxel);
    v.extract_all();
    for (r, label) in [(0.05, "small"), (0.15, "medium"), (0.3, "large")] {
        let (mut f, mut e, mut n) = (0.0, 0.0, 0);
        for i in 0..40 {
            let a = i as f32 * 0.05;
            let dir = Vec3::new(a.sin() * 0.5, 0.2, 1.0).normalize();
            let s = v.dab(dir * (1.0 - 0.5 * r), r, 0.2 * r, i % 2 == 0);
            f += s.field_ms;
            e += s.extract_ms;
            n += 1;
        }
        eprintln!("{label} r {r}: field (with cleanup) {:.2} ms, extract {:.2} ms per dab", f / n as f64, e / n as f64);
    }
}

fn bytes(v: &Volume) -> Vec<u8> {
    let mut out = Vec::new();
    v.write_to(&mut out).unwrap();
    out
}

#[test]
fn undo_and_redo_restore_exactly() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    v.extract_all();
    let start = (bytes(&v), v.mesh().faces.len());
    stroke_at(&mut v, &scrub(), H, false);
    let one = (bytes(&v), v.mesh().faces.len());
    stroke_at(&mut v, &spot(20), H, true);
    let two = (bytes(&v), v.mesh().faces.len());
    assert_ne!(one.0, two.0);
    assert!(v.undo());
    assert_eq!((bytes(&v), v.mesh().faces.len()), one, "undoing the trim");
    assert!(v.undo());
    assert_eq!((bytes(&v), v.mesh().faces.len()), start, "undoing the clay");
    assert!(!v.undo());
    assert!(v.redo() && v.redo());
    assert_eq!((bytes(&v), v.mesh().faces.len()), two, "redoing both");
    // A new stroke after an undo drops what could have been redone.
    assert!(v.undo());
    stroke_at(&mut v, &spot(5), H, false);
    assert!(!v.redo());
}

#[test]
fn undo_survives_the_lattice_growing() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, 0.02);
    v.extract_all();
    let start = bytes(&v);
    let (_, hi) = v.bounds();
    let mut strokes = 0;
    while v.bounds().1 == hi {
        stroke_at(&mut v, &spot(10), 0.05, false);
        strokes += 1;
    }
    for _ in 0..strokes {
        assert!(v.undo());
    }
    // The lattice stays grown; the solid inside it is as it was.
    let mut w = Volume::read_from(&mut bytes(&v).as_slice()).unwrap();
    let mut u = Volume::read_from(&mut start.as_slice()).unwrap();
    let (a, b) = (w.mesh(), u.mesh());
    assert_eq!(a.faces.len(), b.faces.len());
    let _ = (&mut w, &mut u);
}

#[test]
fn specks_are_removed_and_real_bits_kept() {
    let voxel = 0.01;
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, voxel);
    v.extract_all();
    let plain = v.mesh().faces.len();
    // A crumb a couple of voxels across, floating just off the surface, and a pebble well clear of it.
    v.begin_stroke();
    v.dab(Vec3::new(0.0, 0.0, 1.05), 1.5 * voxel, 0.3 * voxel, false);
    v.dab(Vec3::new(0.0, 1.1, 0.0), 8.0 * voxel, 2.0 * voxel, false);
    let removed = v.end_stroke();
    let hit_crumb = v.raycast(Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z).unwrap().0.z;
    let hit_pebble = v.raycast(Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y).unwrap().0.y;
    let mesh = v.mesh();
    eprintln!("specks removed {removed}; looking down at the crumb hits z {hit_crumb:.3}; at the pebble y {hit_pebble:.3}; quads {} (ball alone {plain})", mesh.faces.len());
    assert_eq!(removed, 1);
    assert!((hit_crumb - 1.0).abs() < voxel, "the crumb is still there");
    assert!(hit_pebble > 1.15, "the pebble went");
    assert_eq!(sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces), 0);
    // Undo brings the stroke's change back out, crumb and all, and redo takes it again.
    assert!(v.undo());
    assert_eq!(v.mesh().faces.len(), plain);
}

#[test]
fn a_volume_survives_saving() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    v.extract_all();
    stroke_at(&mut v, &scrub(), H, false);
    stroke_at(&mut v, &spot(20), H, true);
    let data = bytes(&v);
    let w = Volume::read_from(&mut data.as_slice()).unwrap();
    let (a, b) = (v.mesh(), w.mesh());
    eprintln!("saved {} KB; quads {} -> {}", data.len() / 1024, a.faces.len(), b.faces.len());
    assert_eq!(a.faces.len(), b.faces.len());
    let (sa, sb): (Vec3, Vec3) = (a.positions.iter().sum(), b.positions.iter().sum());
    assert!((sa - sb).length() / (a.positions.len() as f32) < 1e-4);
    assert!(Volume::read_from(&mut &data[..data.len() / 2]).is_err());
}
