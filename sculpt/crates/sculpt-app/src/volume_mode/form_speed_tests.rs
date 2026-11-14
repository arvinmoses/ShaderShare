//! Back-to-back form strokes on a ball of about 5M triangles (all in level 0), headless: whole frames (dabs and the
//! rest) without and with a mesh layer, per brush radius. A timing gate, ignored by default; run it alone:
//!
//! `cargo test --release -p sculpt-app form_strokes_at_5m -- --ignored --nocapture --test-threads=1`

use std::time::{Duration, Instant};

use glam::Vec3;

use super::*;
use crate::ms;

/// One frame as the app runs it: a frame's share of the dabs, the tick, the GPU taking the volume's changes, the rows.
fn frame(mode: &mut VolumeMode) {
    mode.run_frame_dabs();
    mode.tick();
    mode.vol.drain_dirty();
    let _ = mode.take_rows_dirty();
}

/// Frames until nothing is left to land (strokes' ends, refits, swaps).
fn settle_frames(mode: &mut VolumeMode) {
    let mut n = 0;
    loop {
        frame(mode);
        n += 1;
        if !mode.busy() && !mode.dabs_queued() {
            return;
        }
        assert!(n < 20_000, "never settled");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// What one row of strokes measured.
struct Row {
    frame_p95: f64,
    frame_max: f64,
    dab_p95: f64,
}

/// Eight strokes back to back on the form at radius `r`, the pen crossing 0.6 of the ball's radius in 0.6 s (36
/// frames, dabs every 0.15 r), then `gap` frames of 14 ms with the pen up.
fn strokes(mode: &mut VolumeMode, what: &str, r: f32, gap: usize) -> Row {
    let (mut frames, mut dabs, mut busy_frames, mut aheads, mut ndabs) = (Vec::new(), Vec::new(), 0usize, 0usize, 0usize);
    // Per frame: (stroke, frame, begin, dabs, the rest, running ahead).
    let mut parts: Vec<(usize, usize, f64, f64, f64, bool)> = Vec::new();
    let t_all = Instant::now();
    for i in 0..8 {
        let d = Vec3::new((i as f32 * 1.7).sin() * 0.5, (i as f32 * 0.9).cos() * 0.4, 1.0).normalize();
        let side = d.any_orthonormal_vector();
        mode.target = Target::Form;
        let t = Instant::now();
        mode.begin_stroke();
        let begin = ms(t);
        aheads += mode.ahead.is_some() as usize;
        let (len, nf) = (0.6f32, 36usize);
        let mut done = 0usize;
        for f in 0..nf {
            let upto = ((f + 1) as f32 / nf as f32 * len / (0.15 * r)) as usize;
            while done < upto {
                let dd = (d + side * (0.15 * r * done as f32 - 0.5 * len)).normalize();
                if let Some((hit, _)) = mode.raycast(dd * 3.0, -dd) {
                    mode.queue_dab(DabInput { at: hit, radius: r, px: 0.003, strength: 0.6, hardness: 0.3, brush: Brush::Clay });
                }
                done += 1;
            }
            busy_frames += mode.mesh.as_ref().is_some_and(|m| m.layers.busy()) as usize;
            let t = Instant::now();
            let n = mode.run_frame_dabs();
            let td = ms(t);
            mode.tick();
            mode.vol.drain_dirty();
            let _ = mode.take_rows_dirty();
            let all = ms(t);
            let b = if f == 0 { begin } else { 0.0 };
            frames.push(all + b);
            parts.push((i, f, b, td, all - td, mode.ahead.is_some()));
            if n > 0 {
                dabs.push(td / n as f64);
                ndabs += n;
            }
            std::thread::sleep(Duration::from_millis(4));
        }
        mode.end_stroke();
        for g in 0..gap {
            busy_frames += mode.mesh.as_ref().is_some_and(|m| m.layers.busy()) as usize;
            let t = Instant::now();
            mode.tick();
            mode.vol.drain_dirty();
            let _ = mode.take_rows_dirty();
            let all = ms(t);
            frames.push(all);
            parts.push((i, 36 + g, 0.0, 0.0, all, mode.ahead.is_some()));
            std::thread::sleep(Duration::from_millis(14));
        }
    }
    let over = frames.iter().filter(|&&f| f > 16.0).count();
    let n = frames.len();
    let row = Row { frame_p95: pct(&mut frames, 0.95), frame_max: pct(&mut frames, 1.0), dab_p95: pct(&mut dabs, 0.95) };
    eprintln!(
        "{what}, r {r}, pen up {gap} frames: frame ms median {:.1} p95 {:.1} max {:.1}; over 16 ms {over}/{n}; dab ms median {:.2} p95 {:.2} ({ndabs} dabs); strokes run ahead {aheads}/8; frames with a refit running {busy_frames}; {:.1} s",
        pct(&mut frames, 0.5),
        row.frame_p95,
        row.frame_max,
        pct(&mut dabs, 0.5),
        row.dab_p95,
        t_all.elapsed().as_secs_f64()
    );
    parts.sort_by(|a, b| (b.2 + b.3 + b.4).partial_cmp(&(a.2 + a.3 + a.4)).unwrap());
    for p in parts.iter().take(3).filter(|p| p.2 + p.3 + p.4 > 10.0) {
        eprintln!("    slow frame: stroke {} frame {}: begin {:.1} dabs {:.1} rest {:.1} ms; running ahead {}", p.0, p.1, p.2, p.3, p.4, p.5);
    }
    let t = Instant::now();
    settle_frames(mode);
    if let Some(m) = mode.mesh.as_ref() {
        eprintln!("    settled in {:.0} ms; refits swapped {}", ms(t), m.times.swaps);
    }
    row
}

/// Gate: with a mesh layer, form strokes at r 0.03 and 0.1 have no frame over 16 ms and dabs (p95) within 1.3 times
/// the volume's alone; at r 0.3 on the volume alone, frames p95 under 20 ms. (`FORM_TRIS` sets the ball's size.)
#[test]
#[ignore]
fn form_strokes_at_5m() {
    let _one = timing_lock();
    let tris: f32 = std::env::var("FORM_TRIS").ok().and_then(|s| s.parse().ok()).unwrap_or(13.2e6);
    let mut alone = VolumeMode::ball(tris);
    let mut solo = Vec::new();
    for r in [0.03f32, 0.1, 0.3] {
        solo.push(strokes(&mut alone, "form, volume only", r, 10));
    }
    drop(alone);
    let mut mode = VolumeMode::ball(tris);
    mode.add_mesh_layer();
    settle_frames(&mut mode);
    assert!(mode.mesh.is_some(), "{:?}", mode.mesh_note);
    let mut with = Vec::new();
    for r in [0.03f32, 0.1] {
        with.push(strokes(&mut mode, "form, with a mesh layer", r, 10));
    }
    // (Reported, not a bar: the pen up long enough between strokes for a refit to start and run under the next.)
    strokes(&mut mode, "form, with a mesh layer", 0.03, 25);

    let mut fails = Vec::new();
    for (k, r) in [0.03f32, 0.1].into_iter().enumerate() {
        if with[k].frame_max > 16.0 {
            fails.push(format!("r {r} with a mesh layer: a frame of {:.1} ms (bar 16)", with[k].frame_max));
        }
        let ratio = with[k].dab_p95 / solo[k].dab_p95;
        if ratio > 1.3 {
            fails.push(format!("r {r} with a mesh layer: dab p95 {:.2} ms, {ratio:.2} times the volume's alone (bar 1.3)", with[k].dab_p95));
        }
    }
    if solo[2].frame_p95 >= 20.0 {
        fails.push(format!("r 0.3, volume only: frame p95 {:.1} ms (bar 20)", solo[2].frame_p95));
    }
    assert!(fails.is_empty(), "{}", fails.join("; "));
}

/// A volume stroke with a mesh layer: no refit starts until the pen has been up `REFIT_AFTER_MS`, then one does; a
/// layer stroke begun sooner refits first (nothing left waiting), and so does a save. (On a loaded machine a stroke's
/// end can take longer than `REFIT_AFTER_MS` to land, and the refit rightly begins meanwhile: then what is checked is
/// that it began no sooner.)
#[test]
fn refit_waits_for_the_pen_up() {
    let _one = timing_lock();
    // A small form stroke, frames going until its end lands. Returns how long after the pen up a refit was first seen
    // under way (or done), if one began meanwhile.
    let small = |mode: &mut VolumeMode| -> Option<f64> {
        mode.target = Target::Form;
        mode.begin_stroke();
        for k in 0..12 {
            let d = Vec3::new(0.1 * k as f32 / 11.0, 0.0, 1.0).normalize();
            if let Some((hit, _)) = mode.raycast(d * 3.0, -d) {
                mode.queue_dab(DabInput { at: hit, radius: 0.1, px: 0.002, strength: 0.8, hardness: 0.3, brush: Brush::Clay });
            }
        }
        mode.end_stroke();
        let swaps = mode.mesh.as_ref().unwrap().times.swaps;
        let mut refit = None;
        // The stroke's end lands.
        while mode.ending || mode.end_job.is_some() || mode.carrying() || mode.ahead.is_some() {
            frame(mode);
            let m = mode.mesh.as_ref().unwrap();
            if refit.is_none() && (m.layers.busy() || m.times.swaps > swaps) {
                refit = Some(ms(mode.pen_up));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        refit
    };
    // Right after `small`: no refit began before its time, and one that has not begun leaves the stroke's blocks
    // waiting. Returns whether one has begun.
    let begun = |mode: &VolumeMode, early: Option<f64>| -> bool {
        let m = mode.mesh.as_ref().unwrap();
        match early {
            Some(t) => {
                assert!(t >= REFIT_AFTER_MS, "a refit began {t:.0} ms after the pen up");
                true
            }
            None => {
                assert!(!m.dirty.is_empty(), "the volume stroke's blocks wait for the refit");
                false
            }
        }
    };
    let mut mode = VolumeMode::ball(2e5);
    mode.add_mesh_layer();
    settle_frames(&mut mode);
    assert!(mode.mesh.is_some(), "{:?}", mode.mesh_note);

    let early = small(&mut mode);
    let up = mode.pen_up;
    let swaps = mode.mesh.as_ref().unwrap().times.swaps;
    let after = match early {
        Some(t) => t,
        None => loop {
            frame(&mut mode);
            let m = mode.mesh.as_ref().unwrap();
            if m.layers.busy() || m.times.swaps > swaps {
                break ms(up);
            }
            assert!(up.elapsed().as_secs() < 10, "no refit started");
            std::thread::sleep(Duration::from_millis(2));
        },
    };
    assert!(after >= REFIT_AFTER_MS, "a refit started {after:.0} ms after the pen up");
    settle_frames(&mut mode);

    // A layer stroke right after a volume stroke: everything the volume did is in first.
    let early = small(&mut mode);
    begun(&mode, early);
    mode.target = Target::Detail;
    mode.begin_stroke();
    assert!(!mode.mesh.as_ref().unwrap().pending(), "a layer stroke began with the layers behind the volume");
    mode.end_stroke();

    // A save right after a volume stroke: the layers saved are the volume's.
    let early = small(&mut mode);
    if !begun(&mode, early) {
        assert!(mode.mesh.as_ref().unwrap().pending());
    }
    let mut bytes = Vec::new();
    mode.write_to(&mut bytes).unwrap();
    assert!(!mode.mesh.as_ref().unwrap().pending(), "saved with the layers behind the volume");
}
