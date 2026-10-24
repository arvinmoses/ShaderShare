//! Surface noise: its height stays within its amplitude and holds fewer octaves on a coarser lattice; as a layer in
//! the stack it changes nothing in the field, its mask paints along the surface, and it saves, opens and undoes.

use glam::Vec3;
use sculpt_core::generator::detail_layer::Layered;
use sculpt_core::generator::level_volume::LevelVolume;
use sculpt_core::surface_noise::SurfaceNoise;

#[test]
fn heights_stay_within_the_amplitude_and_octaves_follow_the_lattice() {
    for kind in [0u32, 1] {
        let n = SurfaceNoise { on: true, amplitude: 0.02, scale: 9.0, octaves: 6, kind, seed: 4 };
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..20000 {
            let t = i as f32;
            let p = Vec3::new((t * 0.71).sin(), (t * 0.37).cos(), (t * 0.13).sin()) * 1.3;
            let h = n.height(p, 6);
            lo = lo.min(h);
            hi = hi.max(h);
        }
        eprintln!("kind {kind}: heights {lo:.4} to {hi:.4} (amplitude 0.02)");
        assert!(lo >= -0.02 && hi <= 0.02);
        assert!(hi - lo > 0.02, "kind {kind}: hardly any noise");
    }
    let n = SurfaceNoise { on: true, amplitude: 0.02, scale: 4.0, octaves: 6, kind: 0, seed: 1 };
    // Wavelengths 0.25, 0.125, ... : four steps of 0.03 hold the first one only; of 0.002, five.
    assert_eq!(n.octaves_for(0.03, 4.0), 2);
    assert_eq!(n.octaves_for(0.002, 4.0), 5);
}

fn bytes(l: &Layered) -> Vec<u8> {
    let mut out = Vec::new();
    l.write_to(&mut out).unwrap();
    out
}

#[test]
fn a_noise_layer_masks_saves_and_undoes_and_leaves_the_field_alone() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    let mut l = Layered::new(&mut s, 2);
    let before: Vec<_> = s.keys().iter().map(|k| (*k, s.block(*k).unwrap().to_vec())).collect();
    let noise = SurfaceNoise { on: true, amplitude: 0.01, scale: 8.0, octaves: 5, kind: 1, seed: 2 };
    let e0 = l.noise_epoch();
    let k = l.add_noise("Surface noise", noise);
    assert!(l.noise_epoch() > e0);
    assert_eq!(l.noise(k), Some(&noise));
    assert!(l.layers()[k].noise);
    // Its mask painted away over the front: blocks along the surface there, and the field untouched.
    let e1 = l.noise_epoch();
    l.paint_mask(&mut s, Vec3::new(0.0, 0.0, 1.0), 0.3, 0.0);
    assert!(l.noise_epoch() > e1);
    let (shown, mask) = l.shown_noise().expect("shown");
    assert_eq!(shown, noise);
    assert!(!mask.is_empty(), "no mask painted");
    let after: Vec<_> = s.keys().iter().map(|k| (*k, s.block(*k).unwrap().to_vec())).collect();
    assert!(before == after, "surface noise changed the field");
    // Saved and opened: the same layers.
    let (_, opened) = Layered::read_from(&mut &bytes(&l)[..], 4).unwrap();
    let opened = opened.unwrap();
    assert_eq!(opened.noise(k), Some(&noise));
    assert_eq!(opened.shown_noise().unwrap().1.len(), mask.len());
    // Undo the mask, then the layer.
    l.undo(&mut s);
    assert!(l.shown_noise().unwrap().1.is_empty());
    l.undo(&mut s);
    assert!(l.shown_noise().is_none());
}

/// Baked into a detail layer: the surface stands out of the form by the noise (in the octaves the layer's level holds),
/// none where the mask was painted away; the noise layer is hidden; one undo takes it back.
#[test]
fn baking_noise_puts_it_in_the_field_within_its_mask() {
    let mut s = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 5);
    let mut l = Layered::new(&mut s, 3);
    let noise = SurfaceNoise { on: true, amplitude: 0.01, scale: 2.0, octaves: 4, kind: 0, seed: 7 };
    let k = l.add_noise("Surface noise", noise);
    // Masked away round +x (twice, so its middle is wholly off).
    l.paint_mask(&mut s, Vec3::new(1.0, 0.0, 0.0), 0.5, 0.0);
    l.paint_mask(&mut s, Vec3::new(1.0, 0.0, 0.0), 0.5, 0.0);
    let blocks = l.noise_blocks(k, 3);
    let baked = l.bake_noise(&mut s, k, 3).expect("baked");
    assert!(!l.layers()[k + usize::from(baked <= k)].visible, "the noise layer still shows");
    eprintln!("{} blocks at level 3; baked layer {}", blocks.len(), baked);
    let octaves = noise.octaves_for(s.voxel(3), 4.0).max(1);
    let out = |s: &LevelVolume, d: Vec3| s.raycast(d * 1.2, -d, s.levels() - 1, false).map_or(f32::NAN, |(q, _)| q.length() - 1.0);
    let (mut worst, mut n) = (0.0f32, 0);
    for i in 0..200 {
        let t = i as f32;
        let d = Vec3::new((t * 0.71).sin(), (t * 0.37).cos(), (t * 0.13).sin() + 0.3).normalize();
        // Off where the mask stroke's middle was (it fades over its radius), whole well away from it.
        let want = if d.x > 0.99 { 0.0 } else if d.x < 0.4 { noise.height(d, octaves) } else { continue };
        let got = out(&s, d);
        worst = worst.max((got - want).abs());
        n += 1;
    }
    eprintln!("baked: surface out of the ball within {worst:.5} of the noise at {n} points (level 3 step {:.5})", s.voxel(3));
    assert!(worst < 0.5 * s.voxel(3) + 0.001, "the baked surface is off by {worst}");
    l.undo(&mut s);
    assert!(l.layers()[k].visible);
    assert_eq!(l.layers().len(), 2, "undo left the baked layer");
    let back = (0..50).map(|i| out(&s, Vec3::new((i as f32).sin(), (i as f32 * 0.3).cos(), 0.5).normalize()).abs()).fold(0.0f32, f32::max);
    assert!(back < 1e-3, "undo left the noise in the field: {back}");
}
