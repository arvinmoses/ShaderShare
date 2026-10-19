//! Volume brushes: a stroke lays down one even layer (or cuts one even groove) however much its dabs overlap, and the
//! lattice grows instead of clipping the solid flat.

use glam::Vec3;
use sculpt_core::generator::volume::Volume;

/// Dabs as the app places them: each at the surface under the cursor, moved onto the surface the stroke began on.
fn stroke(v: &mut Volume, dabs: usize, r: f32, h: f32, carve: bool) {
    v.begin_stroke();
    for _ in 0..dabs {
        let (hit, _) = v.raycast(Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z).unwrap();
        let on = v.onto_base(hit, Vec3::Z);
        let n = v.normal_at(on);
        v.dab_disc(on, n, r, h, h, carve);
    }
    v.end_stroke();
}

fn top(v: &Volume) -> f32 {
    v.raycast(Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z).unwrap().0.z
}

#[test]
fn overlapping_dabs_in_one_stroke_do_not_stack() {
    let (r, h) = (0.3, 0.08);
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    stroke(&mut v, 40, r, h, false);
    let once = top(&v) - 1.0;
    eprintln!("clay: 40 dabs in one place raised the surface {once:.3} (disc height {h})");
    assert!(once > 0.7 * h && once < 1.4 * h, "{once}");
    // The next stroke builds on the first.
    stroke(&mut v, 40, r, h, false);
    let twice = top(&v) - 1.0;
    eprintln!("clay: a second stroke took it to {twice:.3}");
    assert!(twice > once + 0.7 * h && twice < once + 1.4 * h, "{twice}");
}

#[test]
fn carving_digs_one_even_groove_per_stroke() {
    let (r, h) = (0.3, 0.08);
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.6, 0.01);
    stroke(&mut v, 40, r, h, true);
    let depth = 1.0 - top(&v);
    eprintln!("carve: 40 dabs in one place dug {depth:.3} (disc height {h})");
    assert!(depth > 0.7 * h && depth < 1.4 * h, "{depth}");
}

#[test]
fn the_lattice_grows_instead_of_clipping() {
    let mut v = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, 0.02);
    let (lo, hi) = v.bounds();
    for _ in 0..12 {
        stroke(&mut v, 10, 0.3, 0.1, false);
    }
    let reached = top(&v);
    let (lo2, hi2) = v.bounds();
    eprintln!("clay reached z = {reached:.2}; lattice {lo}..{hi} grew to {lo2}..{hi2}");
    assert!(reached > hi.z + 0.2, "clay stopped at the old lattice edge {}", hi.z);
    assert!(reached < hi2.z - 0.1);
    // Every block of surface still joins up.
    let mesh = v.mesh();
    let open = sculpt_core::generator::remesh::non_manifold_edges(&mesh.faces);
    assert!(open < 10, "{open} non-manifold edges after growing");
}
