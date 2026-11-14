//! `sculpt-app --gpu-layer-rows-check [--triangles M]`: mesh layers fed to the GPU rows ([`crate::layer_rows`]) through
//! volume strokes with refits and swaps, and layer dabs, as the app's frames run them (tick, volume sync, rows sync).
//!
//! 1. A ball with four mesh layers (Add, Subtract, Add at 200% opacity, Max): 20 volume strokes (some while a refit
//!    runs, some under the layers) and 50 layer dabs, a layer hidden and shown, an opacity change. After each step,
//!    every row read back from the GPU equals composite - base (offset) and composite normal - base normal (packed),
//!    for every block whose rows are shown, and the drawn vertex less its offset is the mesh vertex's base position
//!    (the GPU's vertex is the mesh's): 0 wrong. Blocks not shown are only those the volume changed after their mesh
//!    was taken; at the end, settled, none.
//! 2. The normal change: drawn normals against the mesh's composite normals, with and without it, and the GPU's own
//!    normal against the base mesh normal (what the change can be no better than).
//! 3. Per frame at about 2.5M triangles: a layer dab's rows < 0.5 ms, a swap's < 2 ms, every row < 10 ms.

use std::time::Instant;

use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
use sculpt_core::generator::level_volume::Key;
use sculpt_core::geom::Ray;
use sculpt_core::mesh_layers::{MeshLayerSettings, MeshLayers, SwapStats};
use sculpt_core::{LayerBlend, LayerId};

use super::check::{device, median, read};
use super::rows::{NONE, pack_normal_change};
use crate::layer_rows::{self, RowsDirty};
use crate::viewport::Viewport;
use crate::volume_mode::VolumeMode;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The app's frame loop, driven by hand: the volume, its mesh layers, the viewport.
struct Scene<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    mode: VolumeMode,
    ml: MeshLayers,
    vp: Viewport,
    eye: Vec3,
    /// Volume blocks changed since the last refit started.
    changed: Vec<Key>,
    /// What the next rows sync is told.
    dirty: RowsDirty,
    swaps: usize,
    dabs: usize,
    /// Per block, what last listed it (for reports of wrong rows).
    events: std::collections::HashMap<Key, String>,
}

impl Scene<'_> {
    fn gather(&mut self) {
        self.changed.extend(self.mode.vol.dirty_keys());
    }

    fn took(&mut self, r: Option<sculpt_core::Result<SwapStats>>) -> Res<()> {
        if let Some(r) = r {
            let st = r?;
            if !st.discarded {
                self.swaps += 1;
                for k in &st.blocks.changed {
                    self.events.insert(*k, format!("swap {}", self.swaps));
                }
                self.dirty.blocks.extend(st.blocks.changed);
                self.dirty.removed.extend(st.blocks.removed);
                // The new document marks everything: its rows are the swap's blocks.
                let _ = self.ml.doc.take_geometry_dirty();
            }
        }
        Ok(())
    }

    /// One frame: a refit that finished is swapped in (the tick), the volume synced, then the rows. True if the
    /// volume wants more frames.
    fn frame(&mut self) -> Res<bool> {
        let r = self.ml.poll(&self.mode.vol);
        self.took(r)?;
        self.gather();
        let more = self.vp.sync_volume_on(self.device, self.queue, &mut self.mode, self.eye, 1000.0, Some(1000.0));
        let d = std::mem::take(&mut self.dirty);
        self.vp.sync_layer_rows(Some(&self.ml), d);
        Ok(more)
    }

    /// The volume at rest: refit for what changed.
    fn refit(&mut self) {
        self.gather();
        let mut keys = std::mem::take(&mut self.changed);
        keys.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
        keys.dedup();
        if !keys.is_empty() {
            self.ml.refit_async(&self.mode.vol, &keys);
        }
    }

    fn wait(&mut self) -> Res<()> {
        while self.ml.busy() {
            let r = self.ml.wait(&self.mode.vol);
            self.took(r)?;
        }
        Ok(())
    }

    /// Frames until the volume and the refits have nothing left.
    fn settle(&mut self) -> Res<()> {
        for _ in 0..400 {
            let more = self.frame()?;
            if !more && self.mode.vol.dirty_keys().is_empty() && !self.ml.busy() && self.dirty.blocks.is_empty() && self.dirty.removed.is_empty() {
                // One more, for what the view changed.
                self.frame()?;
                return Ok(());
            }
            if !self.ml.busy() {
                self.refit();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Err("never settled".into())
    }

    /// A clay stroke on the volume at level `l` from direction `a` to `b`, a frame every `every` dabs (and refits
    /// finishing meanwhile swapped in).
    fn volume_stroke(&mut self, l: usize, a: Vec3, b: Vec3, r: f32, every: usize) -> Res<()> {
        let v = &mut self.mode.vol;
        v.begin_stroke();
        let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
        let mut along = Vec3::ZERO;
        for k in 0..=steps {
            let v = &mut self.mode.vol;
            let dir = a.lerp(b, k as f32 / steps as f32).normalize();
            if let Some((hit, n)) = v.raycast(dir * 3.0, -dir, l, true) {
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, _)) = v.plane_dab(l, hit, rough, r, r * 0.095, 0.5, 0.4, false) {
                    along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
                }
                v.carry_pending(None);
            }
            if k % every == every - 1 {
                self.frame()?;
            }
        }
        self.mode.vol.end_stroke();
        self.mode.vol.carry_pending(None);
        Ok(())
    }

    /// A layer dab on layer `id` at direction `dir` (waits for a refit under way first, as the app does).
    fn layer_dab(&mut self, id: LayerId, dir: Vec3, radius: f32) -> Res<usize> {
        self.wait()?;
        self.ml.doc.set_active_layer(Some(id))?;
        let d = dir.normalize();
        let Some(h) = self.ml.doc.raycast(&Ray::new(d * 3.0, -d)) else { return Ok(0) };
        let s = BrushSettings { radius, strength: 0.5, ..Default::default() };
        let mut clay = ClayBuildup { height: 0.1, ..Default::default() };
        let side = d.any_orthonormal_vector();
        brush::stroke(&mut self.ml.doc, &mut clay, &s, &[(h.point, 1.0), (h.point + side * radius * 0.3, 1.0)])?;
        // Told by its leaves, or (every third dab, when no swap is in this frame) by its blocks.
        self.dabs += 1;
        let by_blocks = self.dabs % 3 == 0 && self.dirty.blocks.is_empty();
        match self.ml.doc.take_geometry_dirty() {
            sculpt_core::DirtySet::Leaves(l) if !by_blocks => {
                let n = l.len();
                for k in layer_rows::dab_blocks(&self.ml, &sculpt_core::DirtySet::Leaves(l.clone())) {
                    self.events.insert(k, format!("dab {} by leaves after swap {}", self.dabs, self.swaps));
                }
                self.dirty.leaves.extend(l);
                Ok(n)
            }
            g => {
                let blocks = layer_rows::dab_blocks(&self.ml, &g);
                let n = blocks.len();
                for k in &blocks {
                    self.events.insert(*k, format!("dab {} by blocks after swap {}", self.dabs, self.swaps));
                }
                self.dirty.blocks.extend(blocks);
                Ok(n)
            }
        }
    }

    fn set_blend(&mut self, id: LayerId, blend: LayerBlend) -> Res<()> {
        let mut m = self.ml.doc.layer(id).ok_or("no layer")?.meta();
        m.blend = blend;
        self.ml.doc.set_layer_meta(id, m, false)?;
        self.dirty.all = true;
        Ok(())
    }
}

/// What [`verify`] found.
#[derive(Default, Debug)]
struct Found {
    blocks: usize,
    vertices: usize,
    /// Rows not equal to composite - base (bit for bit, normal change as packed), pairs of the wrong length or none.
    wrong: usize,
    /// Drawn vertices less their offset more than 0.05 steps from the mesh vertex's base position.
    misplaced: usize,
    worst_steps: f32,
    unshown: usize,
    /// Per vertex whose normal the layers change: (drawn normal, composite normal, base normal).
    normals: Vec<(Vec3, Vec3, Vec3)>,
    /// The first blocks with wrong rows, described.
    bad: Vec<String>,
}

/// Rows on the GPU against the mesh layers, and the drawn surface against the mesh (see the module).
fn verify(sc: &mut Scene) -> Res<Found> {
    let mut f = Found::default();
    let (device, queue) = (sc.device, sc.queue);
    let gpu = sc.vp.levels().ok_or("no volume on the GPU")?;
    gpu.flush_rows();
    let (starts, rows) = {
        let (a, b) = gpu.row_buffers();
        (read(device, queue, a), read(device, queue, b))
    };
    let ml = &sc.ml;
    let doc = &ml.doc;
    let (pos, base, nrm, base_n, vi) = (doc.positions(), doc.base_positions(), doc.normals(), ml.base_normals(), ml.vertex_vi());
    let mut shown: Vec<(usize, Key, u32)> = Vec::new();
    for (i, (k, _)) in ml.blocks().iter().enumerate() {
        let r = ml.block_vertices(i);
        if r.is_empty() {
            continue;
        }
        if !gpu.rows_shown(*k) {
            f.unshown += 1;
            continue;
        }
        let s = gpu.slot(*k).ok_or("rows shown on a block with no slot")?;
        let pair = [starts[2 * s as usize], starts[2 * s as usize + 1]];
        let n = vi[r.end - 1] as usize + 1;
        if pair[0] == NONE || pair[1] as usize != n {
            f.wrong += r.len();
            continue;
        }
        let mut want = vec![[0u32; 4]; n];
        for c in r.clone() {
            let v = doc.canonical_index(c);
            let off = pos[v] - base[v];
            want[vi[c] as usize] = [off.x.to_bits(), off.y.to_bits(), off.z.to_bits(), pack_normal_change(nrm[v] - base_n[v])];
        }
        let mut bad = (0, 0.0f32, 0.0f32);
        for (j, w) in want.iter().enumerate() {
            let at = 4 * (pair[0] as usize + j);
            if rows[at..at + 4] != w[..] {
                let g = |x: &[u32]| (Vec3::new(f32::from_bits(x[0]), f32::from_bits(x[1]), f32::from_bits(x[2])), super::rows::unpack_normal_change(x[3]));
                let (go, gn) = g(&rows[at..at + 4]);
                let (wo, wn) = g(&w[..]);
                bad = (bad.0 + 1, bad.1.max((go - wo).length()), bad.2.max((gn - wn).length()));
            }
        }
        if bad.0 > 0 {
            f.wrong += bad.0;
            if f.bad.len() < 6 {
                f.bad.push(format!("{k:?}: {} of {} rows, offset off by {:.2e}, normal change by {:.3}; last: {} (swaps {})", bad.0, n, bad.1, bad.2, sc.events.get(k).map_or("-", |s| s.as_str()), sc.swaps));
            }
        }
        f.blocks += 1;
        f.vertices += r.len();
        shown.push((i, *k, s));
    }
    // The drawn vertices: as many per block as the GPU has (its cells, as the CPU extracts them).
    let keys: Vec<Key> = shown.iter().map(|b| b.1).collect();
    let (surf, _) = sc.mode.vol.extract_in(&keys, &gpu.show());
    let slots: Vec<u32> = shown.iter().map(|b| b.2).collect();
    let counts: Vec<u32> = shown.iter().map(|b| surf[&b.1].verts.len() as u32).collect();
    let drawn = sc.vp.capture(device, queue, &slots, &counts);
    let mut at = 0usize;
    for ((i, k, _), n) in shown.iter().zip(&counts) {
        let step = sc.mode.vol.voxel(k.0 as usize);
        for c in ml.block_vertices(*i) {
            let v = doc.canonical_index(c);
            let (p, dn) = drawn[at + vi[c] as usize];
            let err = (p - (pos[v] - base[v]) - base[v]).length() / step;
            f.worst_steps = f.worst_steps.max(err);
            f.misplaced += (err > 0.05) as usize;
            if (nrm[v] - base_n[v]).length() > 0.02 {
                f.normals.push((dn, nrm[v].normalize_or_zero(), base_n[v].normalize_or_zero()));
            }
        }
        at += *n as usize;
    }
    Ok(f)
}

/// [`verify`], printed; a failure noted in `fails` (with `last`, a block not shown is one).
fn check(sc: &mut Scene, what: &str, last: bool, fails: &mut Vec<String>) -> Res<Found> {
    let f = verify(sc)?;
    println!(
        "{what}: {} blocks, {} vertices shown: {} rows wrong, {} drawn off the mesh (worst {:.4} steps); {} blocks not shown; rows sync {:?}",
        f.blocks, f.vertices, f.wrong, f.misplaced, f.worst_steps, f.unshown, sc.vp.layer_rows.stats
    );
    for b in &f.bad {
        println!("  wrong: {b}");
    }
    if f.wrong > 0 || f.misplaced > 0 || (last && f.unshown > 0) || f.vertices == 0 {
        fails.push(what.to_string());
    }
    Ok(f)
}

fn angle(a: Vec3, b: Vec3) -> f32 {
    a.normalize_or_zero().dot(b.normalize_or_zero()).clamp(-1.0, 1.0).acos().to_degrees()
}

fn pct(mut v: Vec<f32>, p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * p).round() as usize]
}

/// A spread of directions over the ball (a golden-angle spiral), from `n` of them.
fn spiral(i: usize, n: usize) -> Vec3 {
    let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
    let r = (1.0 - y * y).sqrt();
    let t = i as f32 * 2.399_963;
    Vec3::new(r * t.cos(), y, r * t.sin())
}

/// Four layers (Add, Subtract, Add at 200%, Max) with a few dabs each around +z.
fn layers(sc: &mut Scene, radius: f32) -> Res<[LayerId; 4]> {
    let ids = [sc.ml.doc.add_layer("add"), sc.ml.doc.add_layer("subtract"), sc.ml.doc.add_layer("double"), sc.ml.doc.add_layer("max")];
    for (n, id) in ids.iter().enumerate() {
        for k in 0..3 {
            sc.layer_dab(*id, Vec3::new(-0.3 + 0.2 * n as f32 + 0.05 * k as f32, 0.1 * k as f32, 1.0), radius)?;
        }
    }
    sc.set_blend(ids[1], LayerBlend::Subtract)?;
    sc.ml.doc.set_layer_opacity(ids[2], 2.0)?;
    sc.set_blend(ids[3], LayerBlend::Max)?;
    sc.dirty.all = true;
    Ok(ids)
}

pub fn run(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}");
    let cam = crate::camera::Camera { distance: 3.4, yaw: 0.0, pitch: 0.0, ..Default::default() };
    let eye = cam.eye();

    // ---- 1. strokes, swaps and dabs on a ball
    let mode = VolumeMode::ball(1e6);
    let ml = MeshLayers::from_volume(&mode.vol, MeshLayerSettings { cap: mode.cap, ..Default::default() })?;
    let vp = Viewport::with_device(&device, include_str!("../viewport.wgsl"));
    let mut sc = Scene { device: &device, queue: &queue, mode, ml, vp, eye, changed: Vec::new(), dirty: RowsDirty::default(), swaps: 0, dabs: 0, events: Default::default() };
    sc.mode.vol.drain_dirty();
    sc.settle()?;
    let ids = layers(&mut sc, 0.12)?;
    sc.settle()?;
    let mut fails: Vec<String> = Vec::new();
    check(&mut sc, "layers made", true, &mut fails)?;
    let mut dabs = 0usize;
    for i in 0..20 {
        // Every few strokes cross the layers (+z); the level alternates (new blocks at the finer one).
        let d = if i % 3 == 0 { Vec3::new(-0.25 + 0.05 * i as f32, 0.05, 1.0) } else { spiral(i, 20) };
        let side = d.any_orthonormal_vector() * 0.12;
        let l = 1 + (i % 2);
        sc.volume_stroke(l, d - side, d + side, if l == 1 { 0.08 } else { 0.04 }, 3)?;
        // Most strokes start their refit at once; every third one runs while the last refit is still under way.
        if i % 3 != 2 {
            sc.wait()?;
        }
        sc.refit();
        sc.frame()?;
        // Layer dabs between strokes (each waits for the refit), 50 in all.
        let n = if i < 10 { 2 } else { 3 };
        for k in 0..n {
            let id = ids[(i + k) % 4];
            let at = if k == 0 { d } else { spiral(i * 3 + k, 60) };
            sc.layer_dab(id, at, 0.06 + 0.02 * k as f32)?;
            dabs += 1;
            sc.frame()?;
        }
        if i == 7 {
            // Hide a layer and show it again; then 200% on another.
            sc.ml.doc.set_layer_visible(ids[0], false)?;
            sc.dirty.all = true;
            sc.frame()?;
            check(&mut sc, "a layer hidden", false, &mut fails)?;
            sc.ml.doc.set_layer_visible(ids[0], true)?;
            sc.dirty.all = true;
            sc.frame()?;
        }
        if i == 12 {
            // No visible layer (rows off), then back.
            sc.vp.sync_layer_rows(Some(&sc.ml), RowsDirty { none: true, ..Default::default() });
            if sc.vp.layers_shown || sc.vp.levels().unwrap().has_rows() {
                fails.push("rows not off with `none`".into());
            }
            sc.frame()?;
            sc.dirty.all = true;
            sc.frame()?;
            sc.frame()?;
        }
        if i == 15 {
            sc.ml.doc.set_layer_opacity(ids[0], 2.0)?;
            sc.dirty.all = true;
        }
        sc.frame()?;
        check(&mut sc, &format!("stroke {i}"), false, &mut fails)?;
    }
    dabs += 12;
    // A volume stroke under way while a swap lands: then all settled.
    sc.settle()?;
    let what = format!("settled after 20 strokes, {dabs} layer dabs, {} swaps", sc.swaps);
    let last = check(&mut sc, &what, true, &mut fails)?;

    // ---- 2. the normal change: the same vertices drawn with the layers' rows, with offsets only, and with no rows
    let with: Vec<f32> = last.normals.iter().map(|(d, c, _)| angle(*d, *c)).collect();
    let drawn_with = |sc: &mut Scene, normal: bool, offset: bool| -> Res<Vec<(Vec3, Vec3, Vec3)>> {
        let ml = &sc.ml;
        let idx: Vec<usize> = (0..ml.blocks().len()).filter(|&i| !ml.block_vertices(i).is_empty()).collect();
        let blocks: Vec<(Key, usize)> = idx.iter().map(|&i| (ml.blocks()[i].0, layer_rows::block_row_count(ml, i))).collect();
        let gpu = sc.vp.levels().ok_or("no volume")?;
        gpu.set_many_block_rows(&blocks, u64::MAX, |j, rows| {
            layer_rows::layer_rows(ml, idx[j], rows);
            for r in rows.iter_mut() {
                if !normal {
                    r.normal = 0;
                }
                if !offset {
                    r.offset = [0.0; 3];
                }
            }
        });
        // (The rows are not the layers' here: only the normals are wanted.)
        Ok(verify(sc)?.normals)
    };
    let plain: Vec<f32> = drawn_with(&mut sc, false, true)?.iter().map(|(d, c, _)| angle(*d, *c)).collect();
    let base_err: Vec<f32> = drawn_with(&mut sc, false, false)?.iter().map(|(d, _, b)| angle(*d, *b)).collect();
    println!(
        "gate 2: {} vertices whose normal the layers change; drawn normal against the composite's: median {:.2}, p95 {:.2} deg with the change, {:.2} / {:.2} with offsets only; the GPU's normal against the base mesh's (no rows): {:.2} / {:.2}",
        with.len(),
        pct(with.clone(), 0.5),
        pct(with.clone(), 0.95),
        pct(plain.clone(), 0.5),
        pct(plain.clone(), 0.95),
        pct(base_err.clone(), 0.5),
        pct(base_err.clone(), 0.95)
    );
    if with.is_empty() || with.len() != plain.len() || pct(with.clone(), 0.95) > pct(base_err.clone(), 0.95) + 1.0 || pct(with.clone(), 0.5) >= pct(plain.clone(), 0.5) {
        fails.push("gate 2: the normal change".into());
    }
    sc.dirty.all = true;
    sc.frame()?;
    check(&mut sc, "rows back on", true, &mut fails)?;

    // ---- 3. timing at about 2.5M triangles
    let tri_m: f32 = args.iter().position(|a| a == "--triangles").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2.5);
    let t = Instant::now();
    let mode = VolumeMode::ball(tri_m * 4e6 / 1.5);
    let ml = MeshLayers::from_volume(&mode.vol, MeshLayerSettings { cap: mode.cap, ..Default::default() })?;
    let tris: usize = ml.doc.faces().iter().map(|f| sculpt_core::mesh::face_len(f) - 2).sum();
    println!("timing: {tris} triangles, {} vertices, {} blocks (made in {:.1} s)", ml.doc.vertex_count(), ml.blocks().len(), t.elapsed().as_secs_f64());
    let vp = Viewport::with_device(&device, include_str!("../viewport.wgsl"));
    let mut sc = Scene { device: &device, queue: &queue, mode, ml, vp, eye, changed: Vec::new(), dirty: RowsDirty::default(), swaps: 0, dabs: 0, events: Default::default() };
    sc.mode.vol.drain_dirty();
    sc.settle()?;
    let ids = layers(&mut sc, 0.12)?;
    sc.settle()?;
    // Every row (an opacity drag): the call and sending the rows.
    let mut all_ms = Vec::new();
    for k in 0..7 {
        sc.ml.doc.set_layer_opacity(ids[0], 1.0 + 0.1 * k as f32)?;
        let t = Instant::now();
        sc.vp.sync_layer_rows(Some(&sc.ml), RowsDirty { all: true, ..Default::default() });
        sc.vp.levels().unwrap().flush_rows();
        all_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
    let st_all = sc.vp.layer_rows.stats;
    // A layer dab's rows (a brush over about 3k vertices): the call with the leaves it touched, and sending; every
    // other dab given as blocks instead (reported).
    let (mut dab_ms, mut dab_rows, mut dab_changed) = (Vec::new(), Vec::new(), Vec::new());
    let mut dab_parts: Vec<(f64, f64, f64)> = Vec::new();
    let (mut blk_ms, mut blk_rows) = (Vec::new(), Vec::new());
    for k in 0..60 {
        sc.ml.doc.set_active_layer(Some(ids[k % 2]))?;
        let d = Vec3::new(-0.3 + 0.01 * k as f32, 0.05, 1.0).normalize();
        let h = sc.ml.doc.raycast(&Ray::new(d * 3.0, -d)).ok_or("missed")?;
        let s = BrushSettings { radius: 0.1, strength: 0.5, ..Default::default() };
        let mut clay = ClayBuildup { height: 0.1, ..Default::default() };
        brush::stroke(&mut sc.ml.doc, &mut clay, &s, &[(h.point, 1.0)])?;
        let t = Instant::now();
        let g = sc.ml.doc.take_geometry_dirty();
        let leaves = match g {
            sculpt_core::DirtySet::Leaves(l) => l,
            _ => return Err("a dab marked everything".into()),
        };
        if k % 2 == 0 {
            sc.vp.sync_layer_rows(Some(&sc.ml), RowsDirty { leaves, ..Default::default() });
            let t_sync = t.elapsed().as_secs_f64() * 1e3;
            sc.vp.levels().unwrap().flush_rows();
            dab_ms.push(t.elapsed().as_secs_f64() * 1e3);
            dab_parts.push((sc.vp.layer_rows.stats.leaf_ms, sc.vp.layer_rows.stats.ms, t_sync));
            dab_rows.push(sc.vp.layer_rows.stats.leaf_rows as f64);
            dab_changed.push(sc.vp.layer_rows.stats.leaf_changed as f64);
        } else {
            let blocks = layer_rows::dab_blocks(&sc.ml, &sculpt_core::DirtySet::Leaves(leaves));
            sc.vp.sync_layer_rows(Some(&sc.ml), RowsDirty { blocks, ..Default::default() });
            sc.vp.levels().unwrap().flush_rows();
            blk_ms.push(t.elapsed().as_secs_f64() * 1e3);
            blk_rows.push(sc.vp.layer_rows.stats.rows as f64);
        }
        if k % 10 == 9 {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    // A swap's rows: a volume stroke, its refit waited for, then the frame's rows (timed alone).
    let mut swap_ms = Vec::new();
    let mut swap_blocks = Vec::new();
    for i in 0..6 {
        let d = spiral(i * 5 + 2, 30);
        let side = d.any_orthonormal_vector() * 0.1;
        sc.volume_stroke(1, d - side, d + side, 0.06, 1000)?;
        sc.refit();
        sc.wait()?;
        let r = sc.ml.poll(&sc.mode.vol);
        sc.took(r)?;
        sc.gather();
        sc.vp.sync_volume_on(&device, &queue, &mut sc.mode, eye, 1000.0, Some(1000.0));
        let d = std::mem::take(&mut sc.dirty);
        swap_blocks.push(d.blocks.len() as f64);
        let t = Instant::now();
        sc.vp.sync_layer_rows(Some(&sc.ml), d);
        sc.vp.levels().unwrap().flush_rows();
        swap_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        sc.settle()?;
    }
    let (dab, swap, all) = (median(dab_ms), median(swap_ms), median(all_ms));
    println!("gate 3: a layer dab's rows by its leaves (median {:.0} rows made, {:.0} changed): {dab:.3} ms (bar 0.5)", median(dab_rows), median(dab_changed));
    println!(
        "        (rows of the leaves {:.3} ms, the call {:.3}, with taking the leaves {:.3}, then sending; given as blocks instead: {:.0} rows made, {:.3} ms)",
        median(dab_parts.iter().map(|p| p.0).collect()),
        median(dab_parts.iter().map(|p| p.1).collect()),
        median(dab_parts.iter().map(|p| p.2).collect()),
        median(blk_rows),
        median(blk_ms)
    );
    println!("gate 3: a swap's rows (median {:.0} blocks): {swap:.3} ms (bar 2)", median(swap_blocks));
    println!("gate 3: every row ({} blocks, {} rows): {all:.2} ms (bar 10)", st_all.blocks, st_all.rows);
    let (bytes, used, held) = sc.vp.levels().unwrap().rows_memory();
    println!("gate 3: row memory on the GPU {:.1} MB (bar 26): {used} rows used of {held} held", bytes as f64 / 1048576.0);
    check(&mut sc, "2.5M settled", true, &mut fails)?;
    if dab > 0.5 || swap > 2.0 || all > 10.0 || bytes as f64 > 26.0 * 1048576.0 {
        fails.push("gate 3: over a bar".into());
    }
    if !fails.is_empty() {
        return Err(format!("failed: {}", fails.join("; ")).into());
    }
    println!("mesh layers fed to the GPU rows: every gate passes");
    Ok(())
}
