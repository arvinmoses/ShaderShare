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

/// A deterministic value in [-1, 1] for each lattice point: every cell and face is as ambiguous as it can be.
fn lattice_noise(p: Vec3, voxel: f32, seed: u32) -> f32 {
    let q = |v: f32| (v / voxel).round() as i32;
    let mut h = seed ^ (q(p.x) as u32).wrapping_mul(0x8da6_b343) ^ (q(p.y) as u32).wrapping_mul(0xd816_3841) ^ (q(p.z) as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[test]
fn white_noise_on_the_lattice_stays_manifold() {
    for seed in 0..6 {
        let m = remesh(&|p: Vec3| lattice_noise(p, 0.1, seed), bounds(1.0), &RemeshParams { voxel: 0.1, lipschitz: 1e6 });
        assert!(m.quads.len() > 500, "seed {seed}: a lot of surface, got {}", m.quads.len());
        let bad = non_manifold_edges(&m.quads);
        // The field is noise right up to the bounds, so surface that touches the bounds is open there; only count edges inside.
        let inner: Vec<_> = sculpt_core::generator::remesh::non_manifold_edge_list(&m.quads)
            .into_iter()
            .filter(|&((a, b), _)| [a, b].iter().all(|&v| m.positions[v as usize].abs().max_element() < 0.8))
            .collect();
        assert!(inner.is_empty(), "seed {seed}: {} bad edges inside the bounds of {bad} total, first {:?}", inner.len(), inner.first());
    }
}
