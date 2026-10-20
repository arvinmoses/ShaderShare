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
    assert!(open < 10, "{open} non-manifold edges after growing");
}
