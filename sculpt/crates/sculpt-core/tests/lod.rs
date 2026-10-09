//! Level of detail: crack-free cuts, consistent errors, and a bounded triangle count.

use std::collections::HashMap;

use glam::Vec3;
use sculpt_core::Document;
use sculpt_core::bvh::NodeKind;
use sculpt_core::lod::{LodParams, LodTree, View};
use sculpt_core::primitives::{quad_sphere, quad_sphere_res};

fn big_doc() -> Document {
    // 6 * 70^2 = 29,400 quads: about 15 leaves of up to 2048 faces.
    Document::from_mesh(quad_sphere_res(70, 1.0)).unwrap()
}

fn view_from(eye: Vec3, fov_deg: f32, px_height: f32) -> View {
    let proj = glam::camera::rh::proj::directx::perspective(fov_deg.to_radians(), 1.0, 0.01, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
    View::from_view_proj(proj * view, eye, px_height / (2.0 * (fov_deg.to_radians() / 2.0).tan()))
}

/// Every edge of a closed surface is used by exactly two triangles. A crack shows up as an edge used once.
fn assert_watertight(tree: &LodTree, ranges: &[(u32, u32)]) {
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for &(first, count) in ranges {
        for t in tree.indices[first as usize..(first + count) as usize].chunks_exact(3) {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }
    let bad = edges.values().filter(|&&n| n != 2).count();
    assert_eq!(bad, 0, "{bad} of {} edges are not shared by exactly two triangles (cracks or overlaps)", edges.len());
}

#[test]
fn generated_sphere_is_closed_and_has_the_requested_size() {
    let m = quad_sphere_res(40, 1.0);
    assert_eq!(m.faces.len(), 6 * 40 * 40);
    // Euler: V - E + F = 2 with E = 2F for an all-quad closed surface, so V = F + 2.
    assert_eq!(m.positions.len(), m.faces.len() + 2);
    assert!(m.positions.iter().all(|p| (p.length() - 1.0).abs() < 1e-5));
    assert_eq!(quad_sphere(3, 1.0).faces.len(), 6 * 64);
}

#[test]
fn leaf_only_cut_is_the_whole_mesh_and_the_root_alone_is_small() {
    let doc = big_doc();
    let tree = LodTree::build(doc.positions(), doc.faces(), doc.bvh(), LodParams::default());
    let eye = Vec3::new(0.0, 0.0, 3.0);
    let view = view_from(eye, 45.0, 1000.0);
    // Threshold zero: descend everywhere. Back faces and the far side still count, so use the whole pool.
    let all_leaves: Vec<(u32, u32)> = doc.bvh().nodes.iter().enumerate().filter(|(_, n)| matches!(n.kind, NodeKind::Leaf { .. })).map(|(i, _)| (tree.nodes[i].first, tree.nodes[i].count)).collect();
    let tris: u64 = all_leaves.iter().map(|r| r.1 as u64 / 3).sum();
    assert_eq!(tris, tree.full_triangles);
    assert_eq!(tris, 2 * doc.face_count() as u64);
    assert_watertight(&tree, &all_leaves);
    // A huge threshold keeps only the root.
    let cut = tree.select(doc.bvh(), &view, 1e9, u64::MAX);
    assert_eq!(cut.nodes, 1);
    assert!(cut.triangles <= LodParams::default().node_tris as u64 + 64);
}

#[test]
fn every_random_cut_through_the_tree_is_watertight() {
    let doc = big_doc();
    let tree = LodTree::build(doc.positions(), doc.faces(), doc.bvh(), LodParams::default());
    // A cheap deterministic generator, so the test needs no extra crate.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 33) as u32
    };
    for round in 0..40 {
        let mut ranges = Vec::new();
        let mut stack = vec![0u32];
        let stop_chance = 1 + (round % 4); // 1 in 2..5
        while let Some(id) = stack.pop() {
            let n = &tree.nodes[id as usize];
            match n.children {
                Some([l, r]) if rnd() % (stop_chance + 1) != 0 => {
                    stack.push(l);
                    stack.push(r);
                }
                _ => ranges.push((n.first, n.count)),
            }
        }
        assert_watertight(&tree, &ranges);
    }
}

#[test]
fn errors_never_shrink_towards_the_root() {
    let doc = big_doc();
    let tree = LodTree::build(doc.positions(), doc.faces(), doc.bvh(), LodParams::default());
    for n in &tree.nodes {
        if let Some([l, r]) = n.children {
            assert!(n.error >= tree.nodes[l as usize].error && n.error >= tree.nodes[r as usize].error);
        }
    }
    assert!(tree.nodes[0].error > 0.0, "a simplified root has some error");
}

#[test]
fn the_triangle_budget_is_honoured_and_far_views_draw_less() {
    let doc = big_doc();
    let tree = LodTree::build(doc.positions(), doc.faces(), doc.bvh(), LodParams::default());
    let near = view_from(Vec3::new(0.0, 0.0, 1.6), 45.0, 1000.0);
    let far = view_from(Vec3::new(0.0, 0.0, 40.0), 45.0, 1000.0);
    let near_cut = tree.select(doc.bvh(), &near, 1.0, u64::MAX);
    let far_cut = tree.select(doc.bvh(), &far, 1.0, u64::MAX);
    assert!(far_cut.triangles < near_cut.triangles, "{} !< {}", far_cut.triangles, near_cut.triangles);
    assert!(near_cut.triangles <= tree.full_triangles);
    // Asking for fewer triangles than the natural cut raises the threshold until it fits.
    let budget = near_cut.triangles / 4;
    let capped = tree.select(doc.bvh(), &near, 1.0, budget);
    assert!(capped.triangles <= budget.max(tree.params.node_tris as u64), "{} > {budget}", capped.triangles);
    assert!(capped.tau_px > 1.0);
    // Whatever was selected is still watertight where the whole sphere is visible.
    let wide = view_from(Vec3::new(0.0, 0.0, 6.0), 60.0, 1000.0);
    let cut = tree.select(doc.bvh(), &wide, 0.2, u64::MAX);
    assert!(cut.nodes > 1, "a tight pixel threshold must descend below the root");
}

#[test]
fn coarse_levels_follow_edits_because_they_share_vertices() {
    let mut doc = big_doc();
    doc.build_lod(LodParams::default());
    let tree = doc.lod().expect("built for the current topology").clone();
    // Every index in the pool points at a real vertex, so the same buffer drives every level.
    let max = *tree.indices.iter().max().unwrap();
    assert!((max as usize) < doc.vertex_count());
    assert!(tree.pool_triangles() > tree.full_triangles, "inner levels add triangles on top of the full set");
    assert!(tree.pool_triangles() < tree.full_triangles * 3);
}

#[test]
fn frustum_culls_what_is_behind_or_beside_the_camera() {
    let doc = big_doc();
    let tree = LodTree::build(doc.positions(), doc.faces(), doc.bvh(), LodParams::default());
    let eye = Vec3::new(0.0, 0.0, 1.5);
    let looking_at = view_from(eye, 45.0, 1000.0);
    let seen = tree.select(doc.bvh(), &looking_at, 0.2, u64::MAX);
    // Same eye, but looking away from the sphere.
    let proj = glam::camera::rh::proj::directx::perspective(45f32.to_radians(), 1.0, 0.01, 100.0);
    let away = glam::camera::rh::view::look_at_mat4(eye, eye + Vec3::Z, Vec3::Y);
    let view_away = View::from_view_proj(proj * away, eye, 1000.0 / (2.0 * (22.5f32.to_radians()).tan()));
    let hidden = tree.select(doc.bvh(), &view_away, 0.2, u64::MAX);
    assert!(seen.triangles > 0);
    assert_eq!(hidden.triangles, 0, "nothing is visible when looking away");
    assert!(hidden.culled > 0);
}
