//! The volume playground's tests. (The diagnostics run by hand, on files or settings from the environment, are in
//! `diagnostics`; the mesh layers' in `mesh_tests`.)

use super::*;

/// At the demo's scale (a ball of 1M triangles at level 0, a layer of a few thousand blocks), a big brush on the
/// form: what a frame costs while the layer follows off the frame, and how long it takes to catch up.
#[test]
#[ignore]
fn big_form_strokes_under_a_layer_stay_cheap_a_frame() {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6));
    for j in 0..12 {
        for i in 0..60 {
            let p = Vec3::new(-0.06 + 0.002 * i as f32, -0.06 + 0.01 * j as f32, 1.0).normalize();
            mode.vol.begin_stroke();
            mode.vol.ball_dab(4, p, 0.0015, 0.0004, true);
            mode.vol.end_stroke();
        }
    }
    let layered = std::env::var_os("NO_LAYER").is_none();
    if layered {
        mode.make_layer(4);
        mode.target = Target::Form;
    }
    let blocks = mode.layer.as_ref().map_or(0, |l| l.blocks());
    let (mut dab_ms, mut tick_ms) = (0.0f64, 0.0f64);
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    let mut frames = 0;
    mode.begin_stroke();
    for k in 0..=24 {
        let d = Vec3::new(-0.1 + 0.2 * k as f32 / 24.0, 0.0, 1.0).normalize();
        let t = Instant::now();
        if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
            mode.dab(hit, 0.5, 0.05, 1.0, 0.5, false);
        }
        dab_ms += ms(t);
        let t2 = Instant::now();
        mode.tick();
        tick_ms += ms(t2);
        let ms = ms(t);
        worst = worst.max(ms);
        total += ms;
        frames += 1;
    }
    let t = Instant::now();
    mode.end_stroke();
    let pen_up = ms(t);
    let landed = mode.layer.as_ref().map_or(0, |l| l.follows);
    let t = Instant::now();
    let mut ticks = 0;
    while mode.busy() {
        mode.tick();
        ticks += 1;
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let catch_up = ms(t);
    let last = mode.layer.as_ref().map(|l| l.last).unwrap_or_default();
    eprintln!("dabs {:.1} ms, ticks {:.1} ms a frame; {landed} follows landed during the stroke", dab_ms / frames as f64, tick_ms / frames as f64);
    eprintln!("layer {blocks} blocks: a frame (dab and tick) {:.1} ms on average, {worst:.1} at most; pen up {pen_up:.1} ms; caught up {catch_up:.0} ms later ({ticks} ticks); last follow {last:?}", total / frames as f64);
    assert!(!mode.busy());
}

/// Fine work at level 5 over the front of the ball (rows of small dots), as the demo's patch.
fn fine_patch(mode: &mut VolumeMode) {
    for j in 0..12 {
        for i in 0..60 {
            let p = Vec3::new(-0.06 + 0.002 * i as f32, -0.06 + 0.01 * j as f32, 1.0).normalize();
            mode.vol.begin_stroke();
            mode.vol.ball_dab(5, p, 0.0015, 0.0004, true);
            mode.vol.end_stroke();
        }
    }
}

/// A stroke of `dabs` dabs from `a` to `b` (directions from the centre), a frame per dab as the app runs it.
fn frames_stroke(mode: &mut VolumeMode, a: Vec3, b: Vec3, r: f32, px: f32, dabs: usize) {
    mode.begin_stroke();
    for k in 0..=dabs {
        let d = a.lerp(b, k as f32 / dabs.max(1) as f32).normalize();
        if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
            mode.dab(hit, r, px, 1.0, 0.5, false);
        }
        mode.tick();
        mode.vol.drain_dirty();
    }
    mode.end_stroke();
}

/// The pen down again a frame after the last stroke's pen up: from the new stroke's begin to its first dab applied.
/// Then a stroke of 30 dabs, one a frame at 60 frames a second, and the frames until all is in: the first dab's
/// wait, the slowest frame of the stroke, and how long until the stroke was done for real.
fn next_stroke_wait(mode: &mut VolumeMode, at: Vec3, r: f32, px: f32) -> f64 {
    std::thread::sleep(std::time::Duration::from_millis(16));
    mode.tick();
    mode.vol.drain_dirty();
    let t = Instant::now();
    // As before: everything landed first.
    if std::env::var_os("SCULPT_WAIT_OLD").is_some() {
        mode.settle();
    }
    mode.begin_stroke();
    let mut first = 0.0;
    let mut worst = 0.0f64;
    let mut ahead_frames = 0;
    let side = at.normalize().any_orthonormal_vector();
    for k in 0..30 {
        let f = Instant::now();
        let d = (at.normalize() + side * 0.004 * k as f32).normalize();
        if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
            mode.dab(hit, r, px, 1.0, 0.5, false);
        }
        if k == 0 {
            first = ms(t);
        }
        mode.tick();
        mode.vol.drain_dirty();
        ahead_frames += mode.ahead.is_some() as usize;
        let ms = ms(f);
        if k > 0 {
            worst = worst.max(ms);
        }
        std::thread::sleep(std::time::Duration::from_secs_f64((16.0 - ms).max(1.0) / 1e3));
    }
    mode.end_stroke();
    let mut after = 0.0f64;
    while mode.busy() {
        let f = Instant::now();
        mode.tick();
        mode.vol.drain_dirty();
        ahead_frames += mode.ahead.is_some() as usize;
        after = after.max(ms(f));
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    eprintln!("   first dab {first:.1} ms; slowest frame of the stroke after it {worst:.0} ms, after the pen up {after:.0} ms; the stroke began for real {:.0} ms after, ran ahead {ahead_frames} frames, caught up {:.0} ms after", mode.lag_ms, mode.catch_up_ms);
    first
}

/// A fast drag's dabs, queued and done a frame's share at a time, leave exactly what doing each at once leaves,
/// and a frame takes no more than its share while a burst lasts.
#[test]
fn queued_dabs_land_as_done_at_once() {
    let dabs: Vec<Vec3> = (0..48).map(|k| Vec3::new(-0.3 + 0.0125 * k as f32, 0.02 * (k as f32 * 0.4).sin(), 1.0).normalize()).collect();
    let mut at_once = VolumeMode::ball(2e5);
    let mut queued = VolumeMode::ball(2e5);
    for (k, mode) in [&mut at_once, &mut queued].into_iter().enumerate() {
        mode.begin_stroke();
        for (i, d) in dabs.iter().enumerate() {
            let (hit, _) = mode.raycast(*d * 3.0, -*d).unwrap();
            if k == 0 {
                mode.brush_dab(DabInput { at: hit, radius: 0.08, px: 0.002, strength: 0.8, hardness: 0.3, brush: Brush::Trim });
            } else {
                mode.queue_dab(DabInput { at: hit, radius: 0.08, px: 0.002, strength: 0.8, hardness: 0.3, brush: Brush::Trim });
                // Frames that keep up with only some of what is queued: a dab, or a frame's share.
                if i % 4 == 3 {
                    assert_eq!(mode.run_dabs(0.0), 1);
                }
                if i == 40 {
                    let before = mode.queued.len();
                    let n = mode.run_frame_dabs();
                    assert!(n >= 1 && n <= before);
                }
            }
        }
        if k == 1 {
            assert!(mode.dabs_queued(), "a burst leaves dabs for the next frames");
        }
        mode.end_stroke();
        assert!(!mode.dabs_queued());
        mode.settle();
    }
    same_field(&at_once.vol, &queued.vol, 0.0).unwrap();
}

/// A file from before mesh layers, with the old volume layers' records after the volume (an `SCLY` chunk): it
/// opens as the form was shown (the volume held the layers' composite), with no layers, a note saying they were
/// merged, and flagged lossy (so Save becomes Save As). A chunk cut short opens the same way.
#[test]
fn old_volume_layers_open_merged() {
    let mut mode = VolumeMode::ball(2e5);
    mode.begin_stroke();
    for k in 0..12 {
        let d = Vec3::new(-0.2 + 0.03 * k as f32, 0.05, 1.0).normalize();
        let (hit, _) = mode.raycast(d * 3.0, -d).unwrap();
        mode.brush_dab(DabInput { at: hit, radius: 0.08, px: 0.002, strength: 0.8, hardness: 0.3, brush: Brush::Clay });
    }
    mode.end_stroke();
    mode.settle();
    let mut plain = Vec::new();
    mode.write_to(&mut plain).unwrap();
    let fresh = VolumeMode::read_from(&mut &plain[..]).unwrap();
    assert!(fresh.lossy_open.is_none(), "{:?}", fresh.lossy_open);
    // "SCLY", the records' length, the records (whatever they hold: they are not read).
    let body: Vec<u8> = (0..4000u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut old = plain.clone();
    old.extend_from_slice(b"SCLY");
    old.extend_from_slice(&(body.len() as u64).to_le_bytes());
    old.extend_from_slice(&body);
    for (what, bytes) in [("whole", &old[..]), ("cut short", &old[..old.len() - 1000])] {
        let back = VolumeMode::read_from(&mut &bytes[..]).unwrap_or_else(|e| panic!("{what}: refused: {e}"));
        same_field(&back.vol, &fresh.vol, 0.0).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(!back.vol.keys().is_empty(), "{what}: the volume is gone");
        assert!(back.layer.is_none() && back.mesh.is_none() && back.layer_rows().is_none(), "{what}: layers");
        let note = back.lossy_open.clone().unwrap_or_else(|| panic!("{what}: not flagged lossy"));
        assert!(note.contains("old volume layers were merged"), "{what}: {note}");
        assert_eq!(back.mesh_note.as_ref(), Some(&note), "{what}");
    }
}

/// At the app's scale (a ball of 1M triangles), how long a new stroke waits for the last one's work: after a big
/// form stroke over a detail layer, after a generator edit at levels 3 and 4, after a big plain stroke carrying
/// detail.
#[test]
#[ignore]
fn waits_before_a_new_stroke() {
    let only = std::env::var("SCULPT_WAIT_CASE").ok();
    let run = |name: &str| only.as_deref().is_none_or(|o| name.contains(o));
    if run("form") {
        let mut mode = VolumeMode::ball(1e6);
        fine_patch(&mut mode);
        mode.make_layer(5);
        mode.target = Target::Form;
        mode.vol.drain_dirty();
        frames_stroke(&mut mode, Vec3::new(-0.1, 0.0, 1.0), Vec3::new(0.1, 0.0, 1.0), 0.5, 0.05, 24);
        let ms = next_stroke_wait(&mut mode, Vec3::new(0.05, 0.05, 1.0), 0.3, 0.05);
        eprintln!("after a big form stroke over a detail layer: {ms:.0} ms to the first dab");
    }
    for level in [3usize, 4] {
        if !run(&format!("generator{level}")) {
            continue;
        }
        let mut mode = VolumeMode::ball(1e6);
        mode.add_generator();
        let k = mode.layer.as_ref().unwrap().active();
        mode.set_layer_level(k, level);
        mode.settle();
        mode.vol.drain_dirty();
        let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
        spec.size *= 1.2;
        mode.set_generator(k, &spec, false);
        let ms = next_stroke_wait(&mut mode, Vec3::new(0.3, 0.2, 1.0), 0.2, 0.02);
        eprintln!("after a generator edit at level {level}: {ms:.0} ms to the first dab");
    }
    if run("plain") {
        let mut mode = VolumeMode::ball(1e6);
        fine_patch(&mut mode);
        mode.vol.keep_detail = true;
        mode.vol.drain_dirty();
        frames_stroke(&mut mode, Vec3::new(-0.1, 0.0, 1.0), Vec3::new(0.1, 0.0, 1.0), 0.3, 0.05, 24);
        let ms = next_stroke_wait(&mut mode, Vec3::new(0.05, 0.05, 1.0), 0.3, 0.05);
        eprintln!("after a big plain stroke carrying detail: {ms:.0} ms to the first dab");
    }
}

/// What the GPU holds, as the viewport keeps it: every block's samples, sent again whenever the volume says it
/// changed.
struct Mirror(std::collections::HashMap<Key, Vec<u32>>);

impl Mirror {
    fn of(vol: &mut LevelVolume) -> Mirror {
        vol.drain_dirty();
        Mirror(vol.keys().into_iter().map(|k| (k, vol.block(k).unwrap().iter().map(|x| x.to_bits()).collect())).collect())
    }

    fn sync(&mut self, vol: &mut LevelVolume) {
        for k in vol.drain_dirty() {
            match vol.block(k) {
                Some(b) => {
                    self.0.insert(k, b.iter().map(|x| x.to_bits()).collect());
                }
                None => {
                    self.0.remove(&k);
                }
            }
        }
    }

    /// Whether it holds what the volume does, and if not, where not.
    fn check(&self, vol: &LevelVolume) -> Result<(), String> {
        let keys = vol.keys();
        let missing: Vec<&Key> = keys.iter().filter(|k| !self.0.contains_key(k)).take(4).collect();
        let extra = self.0.len() as i64 - keys.len() as i64 + missing.len() as i64;
        let stale: Vec<&Key> = keys.iter().filter(|k| self.0.get(k).is_some_and(|m| m.iter().zip(vol.block(**k).unwrap().iter()).any(|(a, b)| *a != b.to_bits()))).take(4).collect();
        if missing.is_empty() && extra == 0 && stale.is_empty() {
            Ok(())
        } else {
            Err(format!("the GPU's copy is off: missing {missing:?}, {extra} blocks too many, stale {stale:?}"))
        }
    }
}

/// Everything a mode holds, for exact comparisons: every sample shown and of the form, and the layers as saved.
/// A checkerboard fill, and a block held only as a parent, count as any value (which corner of a face is
/// filled depends on when follows landed, whoever waits).
struct Print {
    shown: Vec<(Key, Vec<u32>)>,
    form: Vec<(Key, Vec<u32>)>,
    layers: Vec<u8>,
}

const ANY: u32 = u32::MAX;

fn print(mode: &VolumeMode) -> Print {
    let samples = |v: &LevelVolume, parent_only: &dyn Fn(Key) -> bool| -> Vec<(Key, Vec<u32>)> {
        let fills: Vec<f32> = (0..v.levels()).map(|m| 1e-3 * v.voxel(m)).collect();
        v.keys()
            .into_iter()
            .map(|k| {
                let is_fill = |x: f32| fills[..=k.0 as usize].iter().any(|f| (x.abs() - f).abs() < 1e-12);
                (k, if parent_only(k) { vec![ANY; 512] } else { v.block(k).unwrap().iter().map(|x| if is_fill(*x) { ANY } else { x.to_bits() }).collect() })
            })
            .collect()
    };
    let l = mode.layer.as_ref();
    let parent_only = |k: Key| l.is_some_and(|l| !l.form().exists(k.0 as usize, k.1) && !(0..l.layers().len()).any(|j| l.layer_holds(j, k.0 as usize, k.1)));
    let mut layers = Vec::new();
    if let Some(l) = l {
        l.write_to(&mut layers).unwrap();
    }
    Print { shown: samples(&mode.vol, &parent_only), form: l.map_or(Vec::new(), |l| samples(l.form(), &|_| false)), layers }
}

fn same(a: &Print, b: &Print) -> Result<(), String> {
    let close = |x: &[(Key, Vec<u32>)], y: &[(Key, Vec<u32>)], what: &str| -> Result<(), String> {
        if x.len() != y.len() || x.iter().zip(y).any(|(p, q)| p.0 != q.0) {
            let (kx, ky): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (x.iter().map(|p| (p.0.0, p.0.1.to_array())).collect(), y.iter().map(|p| (p.0.0, p.0.1.to_array())).collect());
            return Err(format!("{what}: other blocks (only one side {:?}, only the other {:?})", kx.difference(&ky).take(4).collect::<Vec<_>>(), ky.difference(&kx).take(4).collect::<Vec<_>>()));
        }
        for (p, q) in x.iter().zip(y) {
            if let Some(i) = (0..p.1.len()).find(|&i| p.1[i] != q.1[i] && p.1[i] != ANY && q.1[i] != ANY) {
                return Err(format!("{what}: block {:?} differs at sample {i}: {} against {}", p.0, f32::from_bits(p.1[i]), f32::from_bits(q.1[i])));
            }
        }
        Ok(())
    };
    close(&a.shown, &b.shown, "shown")?;
    close(&a.form, &b.form, "form")?;
    if a.layers != b.layers {
        return Err("the layers differ".into());
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Step {
    /// A stroke on `target` from direction `a` to `b` (points on the unit ball).
    Stroke { target: Target, a: Vec3, b: Vec3, r: f32, px: f32, carve: bool, dabs: usize },
    /// The active generator's size scaled (an edit whose job lands later).
    Generator(f32),
}

/// Run `steps`: each stroke begun as soon as the last one's pen is up (`ahead`), or after everything landed;
/// a frame (a tick, the GPU's copy synced) after every dab. Returns which strokes ran ahead.
fn script(mode: &mut VolumeMode, steps: &[Step], ahead: bool, gpu: &mut Mirror) -> Vec<bool> {
    script_at(mode, steps, ahead, gpu, 1)
}

/// [`script`] at `per_frame` dabs a frame.
fn script_at(mode: &mut VolumeMode, steps: &[Step], ahead: bool, gpu: &mut Mirror, per_frame: usize) -> Vec<bool> {
    let mut ran = Vec::new();
    for step in steps {
        match *step {
            Step::Stroke { target, a, b, r, px, carve, dabs } => {
                if !ahead {
                    mode.settle();
                }
                mode.target = target;
                mode.begin_stroke();
                ran.push(mode.ahead.is_some());
                for k in 0..=dabs {
                    let at = a.lerp(b, k as f32 / dabs.max(1) as f32).normalize();
                    mode.dab(at, r, px, 1.0, 0.5, carve);
                    if k % per_frame == per_frame - 1 {
                        mode.tick();
                        gpu.sync(&mut mode.vol);
                    }
                }
                mode.end_stroke();
                mode.tick();
                gpu.sync(&mut mode.vol);
            }
            Step::Generator(scale) => {
                let layer = mode.layer.as_ref().unwrap();
                let k = (0..layer.layers().len()).rev().find(|&k| layer.generator(k).is_some()).unwrap();
                let mut spec = layer.generator(k).unwrap().clone();
                spec.size *= scale;
                mode.set_generator(k, &spec, false);
                // (No frame: the next stroke begins while its job is surely still under way.)
                gpu.sync(&mut mode.vol);
            }
        }
    }
    mode.settle();
    gpu.sync(&mut mode.vol);
    ran
}

/// `setup` then `steps`, once waiting before every stroke and once running ahead: both end the same, undo
/// stroke by stroke the same and redo the same, and what the GPU was sent always adds up to what is shown.
fn ahead_lands_as_waiting(name: &str, setup: &dyn Fn(&mut VolumeMode), steps: &[Step]) -> (Vec<bool>, usize) {
    ahead_lands_as_waiting_as(name, setup, steps, true)
}

/// The field two volumes show at every lattice point either stores, within `tol` of each level's step.
fn same_field(a: &LevelVolume, b: &LevelVolume, tol: f32) -> Result<(), String> {
    let mut keys = a.keys();
    keys.extend(b.keys());
    for (l, bc) in keys {
        let l = l as usize;
        for q in 0..512 {
            let i = bc * 8 + glam::IVec3::new(q & 7, (q >> 3) & 7, q >> 6);
            let (x, y) = (a.sample(l, i, false), b.sample(l, i, false));
            if (x - y).abs() > tol * a.voxel(l) {
                return Err(format!("level {l} block {bc:?} cell {}: {x} against {y} (stored on each side: {}, {})", q, a.exists(l, bc), b.exists(l, bc)));
            }
        }
    }
    Ok(())
}

/// [`ahead_lands_as_waiting`]; unless `exact`, only as many undo steps, undoing them all brings back the start
/// exactly and redoing them all the end: carrying in jobs ends as its jobs land, so as the frames fell, whoever
/// waits (`carrying_ends_as_its_jobs_land_whoever_waits`).
fn ahead_lands_as_waiting_as(name: &str, setup: &dyn Fn(&mut VolumeMode), steps: &[Step], exact: bool) -> (Vec<bool>, usize) {
    let new = || {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 6));
        setup(&mut mode);
        mode.settle();
        mode
    };
    let (mut waited, mut ahead) = (new(), new());
    let start = print(&ahead);
    let history = |m: &VolumeMode| m.layer.as_ref().map(|l| l.history_len());
    let before = history(&ahead);
    let (mut gw, mut ga) = (Mirror::of(&mut waited.vol), Mirror::of(&mut ahead.vol));
    script(&mut waited, steps, false, &mut gw);
    let ran = script(&mut ahead, steps, true, &mut ga);
    eprintln!("{name}: strokes run ahead {ran:?}");
    ga.check(&ahead.vol).unwrap_or_else(|e| panic!("{name}: {e}"));
    let end = print(&ahead);
    // Undo steps the steps made: as many as waiting made (one a stroke or edit that changed anything).
    let made = match (before, history(&ahead)) {
        (Some(a), Some(b)) => b - a,
        _ => steps.len(),
    };
    if let (Some(a), Some(b)) = (before, history(&waited)) {
        assert_eq!(made, b - a, "{name}: running ahead made {made} undo steps, waiting {}", b - a);
    }
    let same_as_waiting = |a: &VolumeMode, b: &VolumeMode| if exact { same(&print(a), &print(b)) } else { Ok(()) };
    same_as_waiting(&waited, &ahead).unwrap_or_else(|e| panic!("{name}: running ahead ended otherwise than waiting: {e}"));
    let mut undone = 0;
    loop {
        let (a, b) = (waited.undo(), ahead.undo());
        assert_eq!(a, b, "{name}: one could undo and the other not, after {undone} undos");
        if !a {
            break;
        }
        undone += 1;
        ga.sync(&mut ahead.vol);
        ga.check(&ahead.vol).unwrap_or_else(|e| panic!("{name}: after {undone} undos, {e}"));
        same_as_waiting(&waited, &ahead).unwrap_or_else(|e| panic!("{name}: after {undone} undos: {e}"));
        if undone == made {
            same(&start, &print(&ahead)).unwrap_or_else(|e| panic!("{name}: undoing every step did not bring back the start: {e}"));
        }
    }
    for redone in 1..=undone {
        assert!(waited.redo() && ahead.redo());
        ga.sync(&mut ahead.vol);
        ga.check(&ahead.vol).unwrap_or_else(|e| panic!("{name}: after {redone} redos, {e}"));
        same_as_waiting(&waited, &ahead).unwrap_or_else(|e| panic!("{name}: after {redone} redos: {e}"));
    }
    same(&end, &print(&ahead)).unwrap_or_else(|e| panic!("{name}: redoing everything did not bring back the end: {e}"));
    let (mesh, _) = ahead.vol.surface(ahead.vol.levels() - 1);
    assert_eq!(sculpt_core::mesh::edge_counts(&mesh.faces).0, 0, "{name}: the surface has open edges");
    eprintln!("{name}: the same after every one of {undone} undos and redos");
    (ran, ahead.adopted)
}

fn patch(mode: &mut VolumeMode) {
    for j in 0..4 {
        for i in 0..12 {
            let p = Vec3::new(-0.06 + 0.01 * i as f32, -0.03 + 0.02 * j as f32, 1.0).normalize();
            mode.vol.begin_stroke();
            mode.vol.ball_dab(4, p, 0.006, 0.002, true);
            mode.vol.end_stroke();
        }
    }
}

fn big(target: Target, a: Vec3, b: Vec3, r: f32) -> Step {
    Step::Stroke { target, a, b, r, px: 0.05, carve: false, dabs: 8 }
}

#[test]
fn strokes_run_ahead_of_a_plain_strokes_end_land_as_waiting() {
    let _one = timing_lock();
    let (ran, _) = ahead_lands_as_waiting(
        "plain",
        &|_| {},
        &[
            big(Target::Detail, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.0, 1.0), 0.3),
            big(Target::Detail, Vec3::new(0.0, -0.2, 1.0), Vec3::new(0.1, 0.2, 1.0), 0.25),
            Step::Stroke { target: Target::Detail, a: Vec3::new(0.0, 0.0, 1.0), b: Vec3::new(0.05, 0.02, 1.0), r: 0.02, px: 0.0005, carve: true, dabs: 10 },
            big(Target::Detail, Vec3::new(0.1, 0.1, 1.0), Vec3::new(-0.1, 0.0, 1.0), 0.4),
        ],
    );
    assert!(ran.iter().skip(1).all(|r| *r), "every stroke after the first should have run ahead: {ran:?}");
}

/// Carrying in jobs ends as its jobs happened to land: two sessions that both wait, one dab a frame and three a
/// frame, differ as much as running ahead does (so that is what carrying is compared within).
#[test]
#[ignore]
fn carrying_ends_as_its_jobs_land_whoever_waits() {
    let steps = [
        big(Target::Detail, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.0, 1.0), 0.3),
        big(Target::Detail, Vec3::new(0.0, -0.1, 1.0), Vec3::new(0.05, 0.1, 1.0), 0.25),
        big(Target::Detail, Vec3::new(0.1, 0.0, 1.0), Vec3::new(-0.1, 0.05, 1.0), 0.3),
    ];
    let new = || {
        let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 6));
        patch(&mut mode);
        mode.vol.keep_detail = true;
        mode.settle();
        mode
    };
    let (mut a, mut b) = (new(), new());
    let (mut ga, mut gb) = (Mirror::of(&mut a.vol), Mirror::of(&mut b.vol));
    script_at(&mut a, &steps, false, &mut ga, 1);
    script_at(&mut b, &steps, false, &mut gb, 3);
    eprintln!("exactly: {:?}", same(&print(&a), &print(&b)));
    eprintln!("within 0.02 of a step: {:?}", same_field(&a.vol, &b.vol, 0.02));
    eprintln!("within 0.1 of a step: {:?}", same_field(&a.vol, &b.vol, 0.1));
}

#[test]
fn strokes_run_ahead_of_a_kept_detail_stroke_land_as_waiting() {
    let _one = timing_lock();
    // Carrying detail under big strokes: queued, carried in jobs, then the end. Carrying in jobs ends as its jobs
    // happened to land (so as the frames fell), so it is compared as two waiting sessions at different frame
    // rates are: a few in-between blocks on one side only, and the field within a tenth of a step
    // (`carrying_ends_as_its_jobs_land_whoever_waits`).
    let (ran, _) = ahead_lands_as_waiting_as(
        "carrying",
        &|m| {
            patch(m);
            m.vol.keep_detail = true;
        },
        &[
            big(Target::Detail, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.0, 1.0), 0.3),
            big(Target::Detail, Vec3::new(0.0, -0.1, 1.0), Vec3::new(0.05, 0.1, 1.0), 0.25),
            big(Target::Detail, Vec3::new(0.1, 0.0, 1.0), Vec3::new(-0.1, 0.05, 1.0), 0.3),
        ],
        false,
    );
    assert!(ran[1], "the second stroke should have run ahead");
}

#[test]
fn strokes_run_ahead_over_layers_land_as_waiting() {
    let _one = timing_lock();
    let (ran, _) = ahead_lands_as_waiting(
        "layers",
        &|m| {
            patch(m);
            m.make_layer(3);
        },
        &[
            big(Target::Form, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.0, 1.0), 0.3),
            big(Target::Form, Vec3::new(0.0, -0.2, 1.0), Vec3::new(0.05, 0.2, 1.0), 0.25),
            Step::Stroke { target: Target::Detail, a: Vec3::new(-0.03, 0.0, 1.0), b: Vec3::new(0.03, 0.01, 1.0), r: 0.015, px: 0.0004, carve: false, dabs: 8 },
            big(Target::Form, Vec3::new(0.1, 0.0, 1.0), Vec3::new(-0.1, 0.05, 1.0), 0.35),
            Step::Stroke { target: Target::Mask { show: false }, a: Vec3::new(0.0, 0.0, 1.0), b: Vec3::new(0.02, 0.0, 1.0), r: 0.03, px: 0.001, carve: false, dabs: 4 },
            big(Target::Form, Vec3::new(0.0, 0.1, 1.0), Vec3::new(0.0, -0.1, 1.0), 0.3),
        ],
    );
    assert!(ran.iter().skip(1).all(|r| *r), "every stroke after the first should have run ahead: {ran:?}");
}

#[test]
fn strokes_after_a_generator_edit_land_as_waiting() {
    let _one = timing_lock();
    let rocks = |m: &mut VolumeMode| {
        patch(m);
        m.make_layer(3);
        m.add_generator();
        let k = m.layer.as_ref().unwrap().active();
        m.set_layer_level(k, 3);
    };
    // A form stroke takes the generator at its own level at once: it does not wait, nor run ahead.
    let (ran, adopted) = ahead_lands_as_waiting(
        "generator, then the form",
        &rocks,
        &[Step::Generator(1.3), big(Target::Form, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.1, 1.0), 0.3), Step::Generator(0.8), big(Target::Form, Vec3::new(0.0, -0.2, 1.0), Vec3::new(0.1, 0.2, 1.0), 0.4)],
    );
    assert!(ran.iter().all(|r| !r), "{ran:?}");
    assert_eq!(adopted, 2, "the form strokes did not take the generator from its job");
    // A stroke on a detail layer runs ahead until the generator's job is in.
    let detail = |m: &mut VolumeMode| {
        rocks(m);
        m.set_active_layer(0);
    };
    let (ran, _) = ahead_lands_as_waiting(
        "generator, then a detail layer",
        &detail,
        &[
            Step::Generator(1.3),
            Step::Stroke { target: Target::Detail, a: Vec3::new(-0.03, 0.0, 1.0), b: Vec3::new(0.03, 0.01, 1.0), r: 0.015, px: 0.0004, carve: false, dabs: 8 },
            big(Target::Form, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.1, 1.0), 0.3),
        ],
    );
    assert!(ran[0], "{ran:?}");
}

/// What a dab on the copy costs with nothing landing meanwhile, against one on the real state.
#[test]
#[ignore]
fn a_dab_on_the_copy_costs_what_a_dab_does() {
    let mut mode = VolumeMode::ball(1e6);
    fine_patch(&mut mode);
    mode.make_layer(5);
    mode.target = Target::Form;
    frames_stroke(&mut mode, Vec3::new(-0.1, 0.0, 1.0), Vec3::new(0.1, 0.0, 1.0), 0.5, 0.05, 24);
    mode.settle();
    mode.vol.drain_dirty();
    for ahead in [false, true, false, true] {
        if ahead {
            mode.run_ahead();
        }
        mode.begin_stroke();
        let mut ms = Vec::new();
        for k in 0..6 {
            let d = Vec3::new(0.05 + 0.004 * k as f32, 0.05, 1.0).normalize();
            let (hit, _) = mode.raycast(d * 3.0, -d).unwrap();
            let t = Instant::now();
            mode.dab(hit, 0.3, 0.05, 1.0, 0.5, false);
            ms.push(crate::ms(t));
            mode.vol.drain_dirty();
        }
        eprintln!("ahead {ahead}: dabs {ms:.1?} ms");
        mode.end_stroke();
        mode.settle();
        mode.vol.drain_dirty();
    }
}

/// A volume with two layers of different resolutions saves and opens again with both, showing the same.
#[test]
fn layers_save_and_open_with_the_volume() {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
    let dot = Vec3::new(0.0, 0.0, 1.0);
    mode.vol.begin_stroke();
    mode.vol.ball_dab(4, dot, 0.004, 0.001, false);
    mode.vol.end_stroke();
    mode.make_layer(3);
    mode.add_layer(3);
    mode.begin_stroke();
    let (hit, _) = mode.raycast(Vec3::new(0.03, 0.0, 3.0), -Vec3::Z).unwrap();
    mode.dab(hit, 0.012, 0.01, 1.0, 0.5, false);
    mode.end_stroke();
    mode.settle();
    let top = |m: &VolumeMode, x: f32| m.vol.raycast(Vec3::new(x, 0.0, 3.0), -Vec3::Z, 4, false).map_or(f32::NAN, |(p, _)| p.z);
    let mut file = Vec::new();
    mode.write_to(&mut file).unwrap();
    let opened = VolumeMode::read_from(&mut &file[..]).unwrap();
    let layers = opened.layer.as_ref().map(|l| l.layers()).unwrap_or_default();
    eprintln!("opened {} layers: {:?}", layers.len(), layers.iter().map(|l| (l.level, l.blocks)).collect::<Vec<_>>());
    assert_eq!(layers.len(), 2);
    assert!(layers[1].blocks > 0, "the stroke on the second layer was not kept");
    for x in [0.0, 0.03] {
        assert!((top(&opened, x) - top(&mode, x)).abs() < 2e-5, "opened, the surface is not where it was");
    }
}

/// Pieces of a surface: their sizes in vertices.
pub(super) fn pieces(mesh: &sculpt_core::mesh::PolyMesh) -> Vec<usize> {
    let n = mesh.positions.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for f in &mesh.faces {
        let k = if f[3] == sculpt_core::mesh::NO_VERT { 3 } else { 4 };
        for j in 1..k {
            let (a, b) = (root(&mut parent, f[0] as usize), root(&mut parent, f[j] as usize));
            parent[a.max(b)] = a.min(b);
        }
    }
    let mut used = vec![false; n];
    for f in &mesh.faces {
        for &v in f.iter().filter(|&&v| v != sculpt_core::mesh::NO_VERT) {
            used[v as usize] = true;
        }
    }
    let mut size = std::collections::HashMap::new();
    for i in (0..n).filter(|&i| used[i]) {
        *size.entry(root(&mut parent, i)).or_insert(0usize) += 1;
    }
    size.into_values().collect()
}

/// Spiky vertices: farther from the mean of their neighbours than `k` times the mean edge to them, with where
/// they are.
pub(super) fn spikes(mesh: &sculpt_core::mesh::PolyMesh, k: f32) -> Vec<Vec3> {
    let n = mesh.positions.len();
    let mut sum = vec![Vec3::ZERO; n];
    let mut len = vec![0.0f32; n];
    let mut cnt = vec![0u32; n];
    for f in &mesh.faces {
        let m = if f[3] == sculpt_core::mesh::NO_VERT { 3 } else { 4 };
        for j in 0..m {
            let (a, b) = (f[j] as usize, f[(j + 1) % m] as usize);
            if a == b {
                continue;
            }
            let (pa, pb) = (mesh.positions[a], mesh.positions[b]);
            sum[a] += pb;
            sum[b] += pa;
            len[a] += (pa - pb).length();
            len[b] += (pa - pb).length();
            cnt[a] += 1;
            cnt[b] += 1;
        }
    }
    (0..n).filter(|&i| cnt[i] > 0 && (mesh.positions[i] - sum[i] / cnt[i] as f32).length() > k * len[i] / cnt[i] as f32).map(|i| mesh.positions[i]).collect()
}

/// Points of a height map whose bending (|second differences| over both axes) passes `by`: pits and grain.
pub(super) fn bent(vol: &LevelVolume, c: Vec3, half: f32, px: usize, by: f32) -> (usize, usize) {
    use rayon::prelude::*;
    let n = c.normalize();
    let u = n.cross(Vec3::Y).normalize();
    let v = n.cross(u);
    let step = 2.0 * half / px as f32;
    let hs: Vec<f32> = (0..px * px)
        .into_par_iter()
        .map(|k| {
            let q = c + u * (-half + step * (k % px) as f32) + v * (-half + step * (k / px) as f32);
            vol.raycast(q + n * 0.3, -n, std::env::var("SCULPT_REPLAY_CAP").ok().and_then(|s| s.parse().ok()).unwrap_or(vol.levels() - 1), false).map_or(f32::NAN, |(hit, _)| (hit - c).dot(n))
        })
        .collect();
    let (mut cnt, mut out) = (0, 0);
    for j in 1..px - 1 {
        for i in 1..px - 1 {
            let h = hs[j * px + i];
            let d = (hs[j * px + i - 1] + hs[j * px + i + 1] - 2.0 * h).abs() + (hs[(j - 1) * px + i] + hs[(j + 1) * px + i] - 2.0 * h).abs();
            if d.is_finite() {
                cnt += 1;
                out += (d > by) as usize;
            }
        }
    }
    (out, cnt)
}

/// The demo's `trim` script on the app's fresh ball (clay at levels 2 and 3, grooves trimmed at level 4, then a wide
/// trim at level 2 cutting down through them), its strokes as straight lines between their first and last dabs: the
/// wide trim leaves no lattice of dents. A finer level's sample at its band holds only a bound and reads its distance
/// from the levels below; it took a level's reading whole up to that level's plateau and the deeper reading past it,
/// a jump where the reading crossed that line, so the finer blocks under the dab came out dented in a lattice, while
/// the stroke was drawn and after. Points whose height bends by more than a fifth of the finest step: over a window
/// across the trim 916 of 158404 with the jump, 614 now (what is left lies where the trim crossed the clay built at
/// level 3); over where it cut through the middle groove's start, 182 of 39204, 4 now.
#[test]
fn a_wide_trim_through_fine_grooves_leaves_no_lattice() {
    let _one = timing_lock();
    let mut mode = VolumeMode::ball(1e6);
    let strokes: [([f32; 3], [f32; 3], f32, f32, f32, bool, usize); 6] = [
        ([0.1607, 0.3789, 0.9114], [0.4199, 0.3604, 0.8329], 0.0661, 1.0, 0.2, false, 30),
        ([0.1684, 0.3160, 0.9338], [0.4311, 0.3028, 0.8503], 0.0330, 1.0, 0.2, false, 60),
        ([0.1954, 0.4023, 0.9012], [0.3797, 0.3977, 0.8413], 0.0109, 1.0, 0.3, true, 120),
        ([0.2012, 0.3615, 0.9175], [0.3839, 0.3572, 0.8591], 0.0108, 1.0, 0.3, true, 120),
        ([0.2052, 0.3193, 0.9302], [0.3886, 0.3153, 0.8719], 0.0109, 1.0, 0.3, true, 120),
        ([0.2035, 0.3406, 0.9243], [0.3957, 0.3321, 0.8634], 0.0650, 0.8, 0.3, true, 22),
    ];
    let mut before = 0;
    for (k, &(a, b, r, strength, hardness, carve, n)) in strokes.iter().enumerate() {
        let (a, b) = (Vec3::from(a), Vec3::from(b));
        if k == 5 {
            before = bent(&mode.vol, a, 0.06, 400, 0.2 * mode.vol.voxel(6)).0;
        }
        mode.begin_stroke();
        for i in 0..n {
            let d = a.lerp(b, i as f32 / (n - 1) as f32).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.dab(hit, r, 0.00055, strength, hardness, carve);
            }
            mode.tick();
        }
        mode.end_stroke();
        mode.settle();
    }
    let (after, cnt) = bent(&mode.vol, Vec3::from(strokes[5].0), 0.06, 400, 0.2 * mode.vol.voxel(6));
    // Where the trim cut through the middle groove's start.
    let c = Vec3::from(strokes[5].0);
    let n = c.normalize();
    let u = n.cross(Vec3::Y).normalize();
    let w = c + u * 0.005 - n.cross(u) * 0.023;
    let (dents, near) = bent(&mode.vol, w, 0.015, 200, 0.2 * mode.vol.voxel(6));
    eprintln!("{before} points bent before the wide trim, {after} of {cnt} after; over the groove's start {dents} of {near}");
    assert!(cnt > 100_000 && near > 30_000, "the window missed the surface: {cnt}, {near} points");
    assert!(dents <= 20, "the wide trim left a lattice of dents over the groove's start: {dents} points bent");
}

/// A big level 0 stroke over sculpted form and fine work, as the review's: what its surface looks like mid-stroke
/// (small loose pieces, open or tangled edges), dab by dab.
#[test]
#[ignore]
fn a_big_stroke_mid_stroke_surface() {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.005, 6));
    let stroke = |mode: &mut VolumeMode, a: Vec3, b: Vec3, r: f32, px: f32, carve: bool, dabs: usize, watch: bool| {
        mode.begin_stroke();
        for k in 0..=dabs {
            let d = a.lerp(b, k as f32 / dabs as f32).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.dab(hit, r, px, 1.0, 0.5, carve);
            }
            mode.tick();
            if watch {
                let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
                let (open, nm) = sculpt_core::mesh::edge_counts(&mesh.faces);
                let mut sizes = pieces(&mesh);
                sizes.sort_unstable();
                let small = sizes.iter().filter(|&&s| s < sculpt_core::generator::level_volume::SPECK_VERTS).count();
                let sp = spikes(&mesh, 0.6);
                let lv: Vec<usize> = (0..mode.vol.levels()).map(|l| sp.iter().filter(|p| mode.vol.level_of(**p) == l).count()).collect();
                eprintln!("dab {k} at level {}: {} pieces ({small} small), open edges {open}, tangled {nm}; {} spiky vertices, by the finest level there {lv:?}", mode.level, sizes.len(), sp.len());
            }
        }
        mode.end_stroke();
        mode.settle();
    };
    // Broad form, then zoomed-in clay, then fine trims.
    for (y, r) in [(0.05f32, 0.2f32), (-0.1, 0.18)] {
        stroke(&mut mode, Vec3::new(-0.2, y, 1.0), Vec3::new(0.2, y, 1.0), r, 0.02, false, 12, false);
    }
    stroke(&mut mode, Vec3::new(0.0, 0.2, 1.0), Vec3::new(0.05, -0.15, 1.0), 0.1, 0.02, true, 10, false);
    for y in [-0.02f32, 0.02] {
        stroke(&mut mode, Vec3::new(-0.08, y, 1.0), Vec3::new(0.08, y, 1.0), 0.03, 0.0006, false, 16, false);
    }
    for k in 0..5 {
        let y = -0.03 + 0.012 * k as f32;
        stroke(&mut mode, Vec3::new(-0.05, y, 1.0), Vec3::new(0.05, y + 0.003, 1.0), 0.004, 0.0001, true, 12, false);
    }
    let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
    eprintln!("before: {} pieces, {} spiky vertices; blocks per level {:?}", pieces(&mesh).len(), spikes(&mesh, 0.6).len(), mode.vol.memory().0);
    stroke(&mut mode, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.1, 0.02, 1.0), 0.45, 0.02, false, 8, true);
    let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
    eprintln!("after the stroke ended: {} pieces, {} spiky vertices", pieces(&mesh).len(), spikes(&mesh, 0.6).len());
}

/// A new ball's rocks at `size`, then a five-frame drag of their size: each frame's `set_generator` (ms), and the
/// release's. Between frames no job composes the rocks at their own level; once the frames after the release
/// have brought it in, what is shown is the stack composed afresh, and the drag undoes as one step.
fn drag_rocks(size: f32, level: Option<usize>) -> (Vec<f64>, f64, usize) {
    drag_rocks_by(size, level, |s, i| s.size = size * (1.0 + 0.01 * i as f32))
}

/// [`drag_rocks`], each frame's spec changed by `step` (the frame's number from 1).
fn drag_rocks_by(size: f32, level: Option<usize>, step: impl Fn(&mut GeneratorSpec, usize)) -> (Vec<f64>, f64, usize) {
    let _one = timing_lock();
    let mut mode = VolumeMode::ball(1e6);
    mode.add_generator();
    let k = mode.layer.as_ref().unwrap().active();
    if let Some(l) = level {
        mode.layer.as_mut().unwrap().set_level(&mut mode.vol, k, l);
    }
    let level = mode.layer.as_ref().unwrap().layers()[k].level;
    let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
    spec.size = size;
    mode.set_generator(k, &spec, false);
    mode.settle();
    // Sent to the GPU, as the viewport takes them a frame at a time.
    mode.vol.drain_dirty();
    let before = mode.layer.as_ref().unwrap().history_len();
    let (mut frames, mut on_gpu) = (Vec::new(), 0);
    for i in 1..=5 {
        // (A slider's drag moves it a little a frame.)
        step(&mut spec, i);
        let t = Instant::now();
        mode.set_generator(k, &spec, true);
        frames.push(ms(t));
        // (Taken by the viewport.)
        on_gpu += mode.gpu_plan.take().is_some() as usize;
        mode.tick();
        mode.vol.drain_dirty();
        assert!(mode.gen_job.is_none(), "a generator job runs between the drag's frames");
    }
    let t = Instant::now();
    mode.end_generator_edit();
    let release = ms(t);
    let start = Instant::now();
    while mode.busy() && start.elapsed().as_secs() < 120 {
        mode.tick();
        mode.vol.drain_dirty();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let landed = ms(start);
    assert!(!mode.busy(), "the rocks never came in at their own level");
    let layer = mode.layer.as_ref().unwrap();
    assert_eq!(layer.generator(k), Some(&spec));
    assert_eq!(layer.history_len(), before + 1, "the drag is not one undo step");
    // Shown as composed afresh, to the tolerance `wide_far_stroke_over_layers` holds the file to.
    let afresh = layer.composed_afresh();
    let (mut live, mut fresh) = (mode.vol.keys(), afresh.keys());
    live.sort_by_key(|k| (k.0, k.1.to_array()));
    fresh.sort_by_key(|k| (k.0, k.1.to_array()));
    assert!(live == fresh, "shown holds other blocks than composed afresh: {} against {}", live.len(), fresh.len());
    let round0 = 2.0 * mode.vol.band(0) / 32767.0;
    let mut apart = vec![0.0f32; mode.vol.levels()];
    for &key in &live {
        for (a, b) in mode.vol.block(key).unwrap().iter().zip(afresh.block(key).unwrap().iter()) {
            let e = &mut apart[key.0 as usize];
            *e = e.max((a - b).abs());
        }
    }
    for (l, a) in apart.iter().enumerate() {
        assert!(*a < (0.01 * mode.vol.voxel(l)).max(round0), "size {size}: at level {l} shown differs from composed afresh by {a}");
    }
    eprintln!("size {size} at level {level}: drag frames {frames:.1?} ms ({on_gpu} joined on the GPU), release {release:.1} ms, landed {landed:.0} ms later; {} blocks", live.len());
    (frames, release, on_gpu)
}

/// A drag of the size of rocks shown by the GPU at their own level (2) does not wait each frame on composing an
/// earlier frame's at their own level (it did: each frame about 3 s at size 0.2, against 0.2 s now).
#[test]
fn dragging_a_rock_slider_does_not_wait_each_frame() {
    let (frames, release, _) = drag_rocks(0.2, Some(2));
    let worst = frames.iter().cloned().fold(0.0, f64::max);
    assert!(worst < 600.0 && release < 200.0, "a drag frame took {worst:.0} ms, the release {release:.0}");
}

/// A new ball's rocks (at level 1) dragged big are shown by the GPU too: each frame was a full compose on the CPU
/// through every finer level, about 2.7 s at size 0.4, against about 80 ms now. (Every frame joined on the GPU;
/// the time's bar leaves room for the other tests running beside it.)
#[test]
fn dragging_big_new_rocks_stays_quick() {
    let (frames, release, on_gpu) = drag_rocks(0.4, None);
    assert_eq!(on_gpu, frames.len(), "not every drag frame was joined on the GPU");
    let worst = frames.iter().cloned().fold(0.0, f64::max);
    assert!(worst < 600.0 && release < 200.0, "a drag frame took {worst:.0} ms, the release {release:.0}");
}

/// [`drag_rocks`] at three sizes, for the times: at the level a new generator takes, and (`SCULPT_GEN_LEVEL`)
/// another.
#[test]
#[ignore]
fn rock_drag_times() {
    let level = std::env::var("SCULPT_GEN_LEVEL").ok().and_then(|s| s.parse().ok());
    for size in [0.05, 0.2, 0.4] {
        drag_rocks(size, level);
    }
}

/// [`drag_rocks_by`] dragging the count (by 5 a frame), for the times.
#[test]
#[ignore]
fn rock_count_drag_times() {
    for size in [0.05, 0.2] {
        drag_rocks_by(size, None, |s, i| s.count = 12 + 5 * i as u32);
    }
    for base in [100u32, 400, 1500] {
        eprintln!("count from {base}:");
        drag_rocks_by(0.05, None, move |s, i| s.count = base + base / 10 * i as u32);
    }
}

/// Blocks of jobs' results a frame may put into what is shown itself (the rest worked out off the frame).
const FRAME_PUTS: usize = 20_000;

/// Frames until nothing is busy, as the app runs them: the slowest tick, how long in all, and the most blocks of
/// jobs' results a tick put into what is shown itself.
fn frames_until_settled(mode: &mut VolumeMode) -> (f64, f64, usize) {
    let start = Instant::now();
    let (mut worst, mut most) = (0.0f64, 0usize);
    while mode.busy() && start.elapsed().as_secs() < 120 {
        let put = mode.layer.as_ref().map_or(0, |l| l.put_on_frame);
        let t = Instant::now();
        mode.tick();
        worst = worst.max(ms(t));
        most = most.max(mode.layer.as_ref().map_or(0, |l| l.put_on_frame) - put);
        mode.vol.drain_dirty();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(!mode.busy(), "never settled");
    (worst, ms(start), most)
}

/// What is shown is the stack composed afresh (to the tolerance `drag_rocks` holds it to).
fn shown_as_afresh(mode: &VolumeMode, what: &str) {
    let afresh = mode.layer.as_ref().unwrap().composed_afresh();
    let (mut live, mut fresh) = (mode.vol.keys(), afresh.keys());
    live.sort_by_key(|k| (k.0, k.1.to_array()));
    fresh.sort_by_key(|k| (k.0, k.1.to_array()));
    assert!(live == fresh, "{what}: shown holds other blocks than composed afresh: {} against {}", live.len(), fresh.len());
    let round0 = 2.0 * mode.vol.band(0) / 32767.0;
    for &key in &live {
        let a = mode.vol.block(key).unwrap().iter().zip(afresh.block(key).unwrap().iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(a < (0.01 * mode.vol.voxel(key.0 as usize)).max(round0), "{what}: block {key:?} differs from composed afresh by {a}");
    }
}

/// A new ball's rocks dragged to size 0.4 (a frame's drag, joined on the GPU) and released.
fn released_big_rocks() -> (VolumeMode, usize, GeneratorSpec) {
    let mut mode = VolumeMode::ball(1e6);
    mode.add_generator();
    mode.settle();
    mode.vol.drain_dirty();
    let k = mode.layer.as_ref().unwrap().active();
    let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
    spec.size = 0.4;
    mode.set_generator(k, &spec, true);
    assert!(mode.gpu_plan.take().is_some(), "the drag was not joined on the GPU");
    mode.tick();
    mode.vol.drain_dirty();
    mode.end_generator_edit();
    (mode, k, spec)
}

/// Big results land without a long frame: a rock slider released at size 0.4 (about 230 thousand blocks composed
/// off the frame) and a small clay stroke under those rocks (whose last follow composes them all again). Each is
/// put into what is shown off the frame, the frame only taking it (it took 300 to 600 ms a frame); shown is as
/// composed afresh after, and the release undoes as one step. Times are printed (`--nocapture`); the bar on them
/// leaves room for the other tests running beside it.
#[test]
fn big_results_land_without_a_long_frame() {
    let _one = timing_lock();
    let (mut mode, k, spec) = released_big_rocks();
    let before = mode.layer.as_ref().unwrap().history_len();
    let whole = mode.layer.as_ref().unwrap().taken_whole;
    let (worst, landed, most) = frames_until_settled(&mut mode);
    let layer = mode.layer.as_ref().unwrap();
    eprintln!("release at size 0.4: worst tick {worst:.1} ms, landed {landed:.0} ms after the release, at most {most} blocks put on a frame; {} blocks", mode.vol.keys().len());
    assert_eq!(layer.taken_whole, whole + 1, "the rocks were not taken whole");
    assert!(most <= FRAME_PUTS, "a frame put {most} blocks");
    assert!(worst < 150.0, "a frame took {worst:.0} ms");
    assert_eq!(layer.generator(k), Some(&spec));
    shown_as_afresh(&mode, "released");
    // A small clay stroke on the form under the rocks.
    mode.target = Target::Form;
    let whole = mode.layer.as_ref().unwrap().taken_whole;
    frames_stroke(&mut mode, Vec3::new(-0.03, 0.0, 1.0), Vec3::new(0.03, 0.01, 1.0), 0.05, 0.0002, 16);
    let (worst, settled, most) = frames_until_settled(&mut mode);
    eprintln!("clay stroke under them: worst tick {worst:.1} ms, settled {settled:.0} ms after the pen up, at most {most} blocks put on a frame");
    assert_eq!(mode.layer.as_ref().unwrap().taken_whole, whole + 1, "the stroke's last follow was not taken whole");
    assert!(most <= FRAME_PUTS, "a frame put {most} blocks");
    assert!(worst < 150.0, "a frame took {worst:.0} ms");
    shown_as_afresh(&mode, "stroked");
    // Undone: the stroke, then the release.
    assert!(mode.undo() && mode.undo());
    assert_eq!(mode.layer.as_ref().unwrap().history_len(), before - 1);
    assert_ne!(mode.layer.as_ref().unwrap().generator(k), Some(&spec));
    shown_as_afresh(&mode, "undone");
}

/// Edits arriving while a big release's result is still being worked out: a form stroke (which takes the rocks at
/// their own level and leaves the job's result stale), and another drag and release of the slider. What is shown
/// ends as composed afresh.
#[test]
fn edits_during_a_big_release_land_as_composed_afresh() {
    let _one = timing_lock();
    let (mut mode, k, mut spec) = released_big_rocks();
    assert!(mode.gen_job.is_some());
    mode.target = Target::Form;
    frames_stroke(&mut mode, Vec3::new(-0.03, 0.0, 1.0), Vec3::new(0.03, 0.01, 1.0), 0.05, 0.0002, 8);
    frames_until_settled(&mut mode);
    shown_as_afresh(&mode, "stroke during the release");
    mode.target = Target::Detail;
    mode.layer.as_mut().unwrap().set_active(k);
    spec.size = 0.35;
    mode.set_generator(k, &spec, true);
    mode.gpu_plan.take();
    mode.end_generator_edit();
    assert!(mode.gen_job.is_some());
    mode.tick();
    spec.size = 0.3;
    mode.set_generator(k, &spec, true);
    mode.gpu_plan.take();
    mode.end_generator_edit();
    frames_until_settled(&mut mode);
    assert_eq!(mode.layer.as_ref().unwrap().generator(k), Some(&spec));
    shown_as_afresh(&mode, "released twice");
}

/// With a detail layer, a mask stroke hides the layer where it passes, and undo brings it back.
#[test]
fn a_generator_edit_lands_at_its_own_level_off_the_frame() {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
    mode.add_generator();
    let k = mode.layer.as_ref().unwrap().active();
    let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
    spec.size *= 1.3;
    mode.set_generator(k, &spec, false);
    // Frames go on until the job is in.
    let start = Instant::now();
    while mode.busy() && start.elapsed().as_secs() < 20 {
        mode.tick();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(!mode.busy(), "the job never came in");
    let layer = mode.layer.as_mut().unwrap();
    assert_eq!(layer.generator(k), Some(&spec));
    assert!(layer.generator_take().is_none(), "still shown as a preview");
}

#[test]
fn surface_noise_saves_as_a_layer_and_older_files_open_with_it() {
    let n = SurfaceNoise { on: true, amplitude: 0.02, scale: 7.0, octaves: 4, kind: 1, seed: 9 };
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.05, 3));
    mode.add_noise_with(n);
    let mut buf = Vec::new();
    mode.write_to(&mut buf).unwrap();
    let back = VolumeMode::read_from(&mut &buf[..]).unwrap();
    assert_eq!(back.shown_noise().map(|s| s.0), Some(n));
    // A file of one volume and its surface noise after it, as before noise was a layer.
    let vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.05, 3);
    let mut old = Vec::new();
    vol.write_to(&mut old).unwrap();
    let json = serde_json::to_vec(&n).unwrap();
    old.extend_from_slice(b"SNOI");
    old.extend_from_slice(&(json.len() as u32).to_le_bytes());
    old.extend_from_slice(&json);
    let back = VolumeMode::read_from(&mut &old[..]).unwrap();
    assert_eq!(back.shown_noise().map(|s| s.0), Some(n));
}

#[test]
fn a_mask_stroke_hides_the_detail_layer() {
    let mut mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, 5));
    let dot = Vec3::new(0.0, 0.0, 1.0);
    mode.vol.begin_stroke();
    mode.vol.ball_dab(4, dot, 0.004, 0.001, false);
    mode.vol.end_stroke();
    let top = |m: &VolumeMode| m.vol.raycast(dot * 3.0, -dot, 4, false).map_or(f32::NAN, |(p, _)| p.z);
    let raised = top(&mode);
    mode.make_layer(3);
    assert!((top(&mode) - raised).abs() < 2e-4, "the layer does not show the dot");
    mode.target = Target::Mask { show: false };
    mode.begin_stroke();
    let (hit, _) = mode.raycast(dot * 3.0, -dot).unwrap();
    for _ in 0..4 {
        mode.dab(hit, 0.02, 0.001, 1.0, 0.5, false);
    }
    mode.end_stroke();
    let hidden = top(&mode);
    eprintln!("dot top {raised:.5}, hidden {hidden:.5}");
    assert!(hidden < raised - 0.002, "the mask did not hide the dot");
    assert!(mode.undo());
    assert!((top(&mode) - raised).abs() < 2e-4, "undo did not bring the dot back");
}

/// Big brushes work coarse, small brushes zoomed in work fine, and nothing finer than the volume has.
#[test]
fn brushes_pick_their_level_from_their_size_on_screen() {
    let mode = VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.02, LEVELS));
    // Zoomed out: a pixel is a level 0 step or more.
    assert_eq!(mode.level_for(0.3, 0.02), 0);
    // A large brush zoomed in still works coarse: its radius is many steps of any finer level.
    assert_eq!(mode.level_for(0.5, 0.0005), 0);
    // A small brush zoomed in: as fine as a pixel.
    assert_eq!(mode.level_for(0.02, 0.0012), 4);
    assert_eq!(mode.level_for(0.004, 0.00001), LEVELS - 1);
}

/// A new ball as the menu makes it (its coarse level 0 under the step of a million triangles): fine bumps taken
/// into a detail layer, a generator layer of rocks over them, then a wide clay stroke on the form seen from far,
/// which works at the new level 0. What is shown is what opening the saved file composes afresh; undo and redo
/// bring back each side exactly; saving what was opened writes the same bytes; the bumps ride the form; the
/// surface stays closed.
#[test]
fn a_wide_far_stroke_at_the_new_coarse_level_keeps_the_layers_whole() {
    let _one = timing_lock();
    wide_far_stroke_over_layers("new ball", VolumeMode::ball(1e6), 1, Brush::Clay);
}

/// [`a_wide_far_stroke_at_the_new_coarse_level_keeps_the_layers_whole`] with the Smooth brush: the form under
/// the layers is smoothed (not raised), and the same holds.
#[test]
fn a_wide_far_smooth_stroke_keeps_the_layers_whole() {
    let _one = timing_lock();
    wide_far_stroke_over_layers("new ball, smooth", VolumeMode::ball(1e6), 1, Brush::Smooth);
}

/// [`a_wide_far_stroke_at_the_new_coarse_level_keeps_the_layers_whole`] on a ball as made before the coarse level
/// 0 (level 0 at the step of a million triangles, six levels), as files from then open.
#[test]
fn a_wide_far_stroke_on_a_ball_as_before_keeps_the_layers_whole() {
    let _one = timing_lock();
    wide_far_stroke_over_layers("ball as before", VolumeMode::new(LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, ball_voxel0(1e6) / 2.0, OPEN_LEVELS)), 0, Brush::Clay);
}

/// The bumps at level `4 + shift`, the layer from level `3 + shift`: the same steps on either ball.
fn wide_far_stroke_over_layers(name: &str, mut mode: VolumeMode, shift: usize, brush: Brush) {
    let levels = mode.vol.levels();
    let xs: Vec<f32> = (0..5).map(|i| -0.04 + 0.02 * i as f32).collect();
    for &x in &xs {
        mode.vol.begin_stroke();
        mode.vol.ball_dab(4 + shift, Vec3::new(x, 0.0, 1.0).normalize(), 0.004, 0.0015, false);
        mode.vol.end_stroke();
    }
    mode.make_layer(3 + shift);
    mode.add_generator();
    mode.settle();
    // How far the shown surface stands out of the form near each bump (the most a little either side).
    let out_of = |m: &VolumeMode, x: f32| -> f32 {
        let p = Vec3::new(x, 0.0, 1.0).normalize();
        let f = m.layer.as_ref().unwrap().form();
        let Some((q, n)) = f.raycast(p * 3.0, -p, f.levels() - 1, false) else { return f32::NAN };
        m.vol.raycast(q + n * 0.02, -n, m.vol.levels() - 1, false).map_or(f32::NAN, |(r, _)| 0.02 - (q + n * 0.02 - r).length())
    };
    let relief = |m: &VolumeMode| -> f32 { xs.iter().map(|&x0| (0..9).map(|j| out_of(m, x0 - 0.002 + 0.0005 * j as f32)).fold(0.0f32, f32::max)).sum::<f32>() / xs.len() as f32 };
    let ground = |m: &VolumeMode| m.vol.raycast(Vec3::new(0.0, 0.06, 1.0).normalize() * 3.0, -Vec3::new(0.0, 0.06, 1.0).normalize(), m.vol.levels() - 1, false).map_or(f32::NAN, |(p, _)| p.length());
    let (bumps0, ground0) = (relief(&mode), ground(&mode));
    let start = print(&mode);
    let history = |m: &VolumeMode| m.layer.as_ref().map_or(0, |l| l.history_len());
    let before = history(&mode);
    let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
    let open0 = sculpt_core::mesh::edge_counts(&mesh.faces);

    // The stroke, seen from far (eye about 3 from the surface, focal 1100 px), a frame after each dab.
    mode.target = Target::Form;
    let (r, px) = (0.45f32, 3.0 / 1100.0);
    let t = Instant::now();
    mode.begin_stroke();
    let mut dab_ms = Vec::new();
    for k in 0..=12 {
        let d = Vec3::new(-0.08 + 0.16 * k as f32 / 12.0, 0.01, 1.0).normalize();
        let Some((hit, _)) = mode.raycast(d * 3.0, -d) else { continue };
        let td = Instant::now();
        mode.brush_dab(DabInput { at: hit, radius: r, px, strength: 1.0, hardness: 0.5, brush });
        dab_ms.push(ms(td));
        mode.tick();
    }
    let level = mode.level;
    let te = Instant::now();
    mode.end_stroke();
    mode.settle();
    let (end_ms, all_ms) = (ms(te), ms(t));
    dab_ms.sort_by(f64::total_cmp);
    eprintln!("{name}: the stroke at level {level} (step {:.4}): dabs {:.1} ms median, {:.1} at most; its end and the layers settling {end_ms:.0} ms; all {all_ms:.0} ms", mode.vol.voxel(level), dab_ms[dab_ms.len() / 2], dab_ms[dab_ms.len() - 1]);
    assert_eq!(level, 0, "a wide brush from far did not work at the coarse level 0");
    let end = print(&mode);
    let (bumps1, ground1) = (relief(&mode), ground(&mode));
    eprintln!("bumps out of the form {bumps0:.5} -> {bumps1:.5}; the form rose {:.4}", ground1 - ground0);
    if brush == Brush::Smooth {
        // A ball smoothed stays the ball (within a tenth of level 0's step).
        assert!((ground1 - ground0).abs() < 0.1 * mode.vol.voxel(0), "smoothing the ball moved it by {}", ground1 - ground0);
    } else {
        assert!(ground1 - ground0 > 0.01, "the stroke did not build");
    }
    assert!(bumps0 > 0.0008, "the layer does not hold the bumps");
    assert!(bumps1 > 0.9 * bumps0 && bumps1 < 1.1 * bumps0, "the bumps did not ride the form whole");
    let (mesh, _) = mode.vol.surface(mode.vol.levels() - 1);
    let open1 = sculpt_core::mesh::edge_counts(&mesh.faces);
    eprintln!("open and tangled edges {open0:?} -> {open1:?}; {} pieces", pieces(&mesh).len());
    assert_eq!(open1.0, 0, "the surface has open edges");

    // Shown as opening the file composes it afresh (to the file's rounding), and saved again the same.
    let mut file = Vec::new();
    mode.write_to(&mut file).unwrap();
    let mut opened = VolumeMode::read_from(&mut &file[..]).unwrap();
    assert_eq!(opened.vol.levels(), levels);
    let (mut live, mut afresh) = (mode.vol.keys(), opened.vol.keys());
    live.sort_by_key(|k| (k.0, k.1.to_array()));
    afresh.sort_by_key(|k| (k.0, k.1.to_array()));
    assert!(live == afresh, "shown live holds other blocks than composing the file afresh: {} against {}", live.len(), afresh.len());
    // Per level: the most a sample is apart (in steps), how many are apart by more than a thousandth of a step,
    // and the block most apart.
    let mut apart = vec![(0.0f32, 0usize, None); levels];
    for &k in &live {
        let vx = mode.vol.voxel(k.0 as usize);
        for (a, b) in mode.vol.block(k).unwrap().iter().zip(opened.vol.block(k).unwrap().iter()) {
            let d = (a - b).abs() / vx;
            let e = &mut apart[k.0 as usize];
            if d > 1e-3 {
                e.1 += 1;
            }
            if d > e.0 {
                *e = (d, e.1, Some((k, *a, *b)));
            }
        }
    }
    eprintln!("{name}: {} blocks shown, the same afresh; per level, values apart at most, samples apart by over 0.001 of a step, where: {apart:?}", live.len());
    // The file keeps each level's samples in 16 bits of its band: level 0's band is 16 of its steps, so its
    // rounding (a 65536th of 32 steps) carries into the finer levels composed from it, as a larger share of
    // their finer steps. Within a hundredth of a step, or two of level 0's roundings.
    let round0 = 2.0 * mode.vol.band(0) / 32767.0;
    for (l, a) in apart.iter().enumerate() {
        assert!(a.0 * mode.vol.voxel(l) < (0.01 * mode.vol.voxel(l)).max(round0), "{name}: at level {l} shown live differs from composing the file afresh by {} of a step", a.0);
    }
    let mut again = Vec::new();
    opened.write_to(&mut again).unwrap();
    assert!(again == file, "saving what was opened wrote other bytes");

    // Undo every step the stroke made, then redo them.
    let made = history(&mode) - before;
    assert!(made >= 1);
    for _ in 0..made {
        assert!(mode.undo());
    }
    same(&start, &print(&mode)).unwrap_or_else(|e| panic!("undo did not bring back the start: {e}"));
    for _ in 0..made {
        assert!(mode.redo());
    }
    same(&end, &print(&mode)).unwrap_or_else(|e| panic!("redo did not bring back the stroke: {e}"));
}

/// Scrubbing a big brush back and forth over bumpy clay: the brush turns a little per dab, far less than the
/// surface normal under it does.
#[test]
fn scrubbing_turns_the_brush_gently() {
    let mut mode = VolumeMode::ball(4e5);
    let px = 0.05;
    for i in 0..30 {
        let a = i as f32 * 2.4;
        let at = Vec3::new(0.35 * a.cos() * (i as f32 / 30.0), 0.35 * a.sin() * (i as f32 / 30.0), 1.0).normalize();
        mode.begin_stroke();
        if let Some((hit, _)) = mode.raycast(at * 3.0, -at) {
            for _ in 0..10 {
                mode.dab(hit, 0.08, px, 1.0, 0.9, false);
            }
        }
        mode.end_stroke();
        // (No frames run here: each stroke ends before the next, as frames between them would see to.)
        mode.settle();
    }
    let (mut turn, mut raw_turn, mut worst, mut n) = (0.0f32, 0.0f32, 0.0f32, 0);
    let (mut last, mut last_raw) = (Vec3::ZERO, Vec3::ZERO);
    mode.begin_stroke();
    for i in 0..240 {
        let x = 0.4 * ((i as f32) * 0.08).sin();
        let y = 0.05 * ((i as f32) * 0.31).sin();
        let at = Vec3::new(x, y, 1.0).normalize();
        let Some((hit, _)) = mode.raycast(at * 3.0, -at) else { continue };
        let raw = mode.vol.normal_upto(hit, mode.vol.levels() - 1, true);
        mode.dab(hit, 0.3, px, 0.7, 0.5, false);
        if last != Vec3::ZERO {
            let a = last.angle_between(mode.stroke_n).to_degrees();
            turn += a;
            worst = worst.max(a);
            raw_turn += last_raw.angle_between(raw).to_degrees();
            n += 1;
        }
        last = mode.stroke_n;
        last_raw = raw;
    }
    mode.end_stroke();
    let (mean, raw_mean) = (turn / n as f32, raw_turn / n as f32);
    eprintln!("{:.2} ms per dab of radius 0.3", mode.last_ms / mode.last_dabs as f64);
    eprintln!("per dab the brush turned {mean:.2} deg on average ({worst:.2} at most); the normal under it turned {raw_mean:.2}");
    assert!(mean < raw_mean / 3.0 && worst < 6.0);
}

/// A close trim over clay rows and a wider trim a level coarser leaves no stripes across it: where it cut past its
/// level's band (two and a half steps; a close trim cuts three), the samples it found at the band, only a bound on
/// their distance, read their distance from the surface under the brush. Blended from the bound, they moved less
/// than their neighbours, and the new surface rippled every few steps where the lattice's planes crossed the old.
#[test]
fn a_close_trim_leaves_no_stripes() {
    let _one = timing_lock();
    let mut mode = VolumeMode::ball(1e6);
    let centre = Vec3::new(0.3, -0.05, 0.95).normalize();
    let (t1, t2) = (Vec3::Y.cross(centre).normalize(), centre.cross(Vec3::Y.cross(centre).normalize()));
    let line = |mode: &mut VolumeMode, a: (f32, f32), b: (f32, f32), r: f32, px: f32, trim: bool| {
        let (pa, pb) = (centre + t1 * a.0 + t2 * a.1, centre + t1 * b.0 + t2 * b.1);
        let steps = ((pa - pb).length() / (0.15 * r)).ceil() as usize;
        mode.begin_stroke();
        for k in 0..=steps {
            let d = pa.lerp(pb, k as f32 / steps as f32).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.dab(hit, r, px, 0.8, 0.3, trim);
            }
            mode.tick();
        }
        mode.end_stroke();
        mode.settle();
    };
    for row in 0..5 {
        for pass in 0..2 {
            let y = -0.03 + 0.014 * row as f32 + 0.003 * pass as f32;
            line(&mut mode, (-0.06, y), (0.06, y + 0.004), 0.016, 0.0004, false);
        }
    }
    line(&mut mode, (-0.07, 0.0), (0.07, 0.01), 0.038, 0.000476, true);
    line(&mut mode, (-0.03, -0.05), (0.0, 0.06), 0.03, 0.000476, true);
    // Over the close trim's far end (past the rows), heights along the ball's normal less their mean over a
    // window four and a half of its steps across (the ripple's length), in steps.
    let vx = mode.vol.voxel(4);
    let (u1, u2) = centre.any_orthonormal_pair();
    let (g, k, half) = (0.0004, 3usize, 0.08);
    let top = mode.vol.levels() - 1;
    let (rows, cols) = (20..90usize, 90..190usize);
    let at = |i: usize, j: usize| {
        let p = centre * 4.0 + u1 * (-half + g * i as f32) + u2 * (-half + g * j as f32);
        mode.vol.raycast(p, -centre, top, false).map_or(f32::NAN, |(q, _)| q.dot(centre))
    };
    let (w, hgt) = (cols.len() + 2 * k, rows.len() + 2 * k);
    let h: Vec<f32> = (0..w * hgt).map(|q| at(cols.start - k + q % w, rows.start - k + q / w)).collect();
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for j in k..hgt - k {
        for i in k..w - k {
            let mut mean = 0.0;
            for b in j - k..=j + k {
                for a in i - k..=i + k {
                    mean += h[b * w + a];
                }
            }
            let d = (h[j * w + i] - mean / ((2 * k + 1) * (2 * k + 1)) as f32) / vx;
            if d.is_finite() {
                sum += (d * d) as f64;
                cnt += 1;
            }
        }
    }
    let rms = (sum / cnt.max(1) as f64).sqrt();
    eprintln!("ripple over the close trim's end: {rms:.4} of a step (rms, {cnt} points)");
    // 0.013 blended from the bound.
    assert!(rms < 0.009, "the close trim's surface ripples by {rms:.4} of a step");
}
