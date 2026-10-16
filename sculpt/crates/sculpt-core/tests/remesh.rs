use glam::Vec3;
use sculpt_core::generator::remesh::{RemeshParams, non_manifold_edges, remesh};
use sculpt_core::geom::Aabb;

fn bounds(r: f32) -> Aabb {
    Aabb { min: Vec3::splat(-r), max: Vec3::splat(r) }
}

#[test]
fn sphere_is_closed_and_round() {
    let m = remesh(&|p: Vec3| p.length() - 1.0, bounds(1.4), &RemeshParams::new(0.05));
    assert_eq!(non_manifold_edges(&m.quads), 0);
    let worst = m.positions.iter().map(|p| (p.length() - 1.0).abs()).fold(0.0, f32::max);
    assert!(worst < 0.03, "vertices stay on the sphere, worst {worst}");
}

#[test]
fn sparse_evaluation_skips_most_of_the_lattice() {
    let m = remesh(&|p: Vec3| p.length() - 1.0, bounds(1.4), &RemeshParams { voxel: 0.02, lipschitz: 1.0 });
    assert!(m.evaluated * 3 < m.total, "evaluated {} of {}", m.evaluated, m.total);
}

#[test]
fn two_surfaces_through_one_voxel_stay_manifold() {
    // A wall thinner than a voxel: two surfaces pass through the same cells.
    let wall = |p: Vec3| (p.x.abs() - 0.02).max(p.y.abs() - 0.6).max(p.z.abs() - 0.6);
    let m = remesh(&wall, bounds(1.0), &RemeshParams { voxel: 0.1, lipschitz: 1.0 });
    assert_eq!(non_manifold_edges(&m.quads), 0);
}

#[test]
fn spheres_meeting_at_a_point_stay_manifold() {
    // The diagonal ambiguity: two balls just touching.
    let balls = |p: Vec3| (p - Vec3::new(0.3, 0.3, 0.3)).length().min((p + Vec3::new(0.3, 0.3, 0.3)).length()) - 0.3 * 3f32.sqrt();
    let m = remesh(&balls, bounds(1.2), &RemeshParams::new(0.07));
    assert_eq!(non_manifold_edges(&m.quads), 0);
}

#[test]
fn noisy_field_stays_manifold() {
    // A deliberately hostile field: a ball with a high-frequency ripple, so many cells are ambiguous.
    let rippled = |p: Vec3| p.length() - 0.8 + 0.08 * (p.x * 23.0).sin() * (p.y * 19.0).sin() * (p.z * 29.0).sin();
    let m = remesh(&rippled, bounds(1.2), &RemeshParams { voxel: 0.04, lipschitz: 3.0 });
    assert!(!m.quads.is_empty());
    assert_eq!(non_manifold_edges(&m.quads), 0);
}
