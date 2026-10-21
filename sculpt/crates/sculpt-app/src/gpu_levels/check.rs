//! `sculpt-app --gpu-levels-check [--bench]`: the GPU extractor against the CPU's reference, exactly.
//!
//! A ball is sculpted with plane brushes at every level, undone and redone, and shown at every level cap; after each
//! step the GPU's surface is read back and compared with `LevelVolume::extract` block by block: the word per cell
//! (vertex numbering, pieces, corner signs, transition) and the quads (corner by corner, in order) must be identical,
//! and the vertices must agree to within the 16-bit samples' rounding. With `--bench`, a larger ball is timed: a full
//! extraction, and a sync per dab of strokes at each level.

use std::time::Instant;

use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use sculpt_core::generator::level_volume::{BlockSurface, Key, LevelVolume};

use super::{GpuLevels, PAGE_INDICES, SLOT_WORDS, VSTRIDE, packed_samples};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn device() -> Res<(wgpu::Device, wgpu::Queue, String)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: None, force_fallback_adapter: false, apply_limit_buckets: false }))?;
    let have = adapter.limits();
    let limits = wgpu::Limits {
        max_buffer_size: have.max_buffer_size.min(4 << 30),
        max_storage_buffer_binding_size: have.max_storage_buffer_binding_size.min(2 << 30),
        max_storage_buffers_per_shader_stage: have.max_storage_buffers_per_shader_stage.min(16),
        max_compute_workgroups_per_dimension: have.max_compute_workgroups_per_dimension,
        ..Default::default()
    };
    let features = adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("gpu_levels"), required_features: features, required_limits: limits, ..Default::default() }))?;
    device.set_device_lost_callback(|reason, msg| eprintln!("device lost: {reason:?} {msg}"));
    let info = adapter.get_info();
    Ok((device, queue, format!("{} ({:?})", info.name, info.backend)))
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buf.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, buf.size());
    queue.submit([enc.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |r| if let Err(e) = r { eprintln!("map failed: {e}") });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped")).to_vec();
    staging.unmap();
    out
}

/// A clay (or trim) stroke at level `l` between two directions, as the app lays dabs.
fn stroke(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool, mut each: impl FnMut(&mut LevelVolume)) {
    v.begin_stroke();
    let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
    let mut along = Vec3::ZERO;
    for k in 0..=steps {
        let dir = a.lerp(b, k as f32 / steps as f32).normalize();
        let Some((hit, n)) = v.raycast(dir * 3.0, -dir, l, true) else { continue };
        let h = r * if trim { 0.075 } else { 0.095 };
        let rough = if along == Vec3::ZERO { n } else { along };
        if let Some((n, _)) = v.plane_dab(l, hit, rough, r, h, 0.5, 0.4, trim) {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        each(v);
    }
    v.end_stroke();
}

/// Compare the GPU's surface with the reference, block by block. Returns (blocks, quads) compared.
fn compare(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &GpuLevels, vol: &LevelVolume, when: &str) -> Res<(usize, usize)> {
    let cap = gpu.cap;
    let keys: Vec<Key> = vol.keys().into_iter().filter(|k| k.0 as usize <= cap).collect();
    let (want, _) = vol.extract(&keys, cap);
    let cells = read(device, queue, &gpu.buf.cells);
    let verts = read(device, queue, &gpu.buf.verts);
    let slot_pages = read(device, queue, &gpu.buf.slot_pages);
    let pages = read(device, queue, &gpu.buf.pages_buf);
    let status = read(device, queue, &gpu.buf.status);
    let key_of = |s: u32| gpu.key_of[s as usize].expect("a quad points at a free slot");
    let mut quads_total = 0;
    let mut worst = 0.0f32;
    for k in &keys {
        let s = *gpu.slot_of.get(k).ok_or_else(|| format!("{when}: block {k:?} has no slot"))? as usize;
        let w: &BlockSurface = &want[k];
        let got_cells = &cells[s * 512..s * 512 + 512];
        if got_cells != w.cells.as_slice() {
            let i = (0..512).find(|&i| got_cells[i] != w.cells[i]).unwrap();
            return Err(format!("{when}: block {k:?} cell {i}: GPU {:#010x}, CPU {:#010x}", got_cells[i], w.cells[i]).into());
        }
        // Vertices: within the rounding of 16-bit samples and 10-bit positions.
        let vx = vol.voxel(k.0 as usize);
        let corner = vol.origin() + (k.1 * 8).as_vec3() * vx;
        for (i, p) in w.verts.iter().enumerate() {
            let q = verts[(s * VSTRIDE as usize + i) * 2];
            let local = Vec3::new((q & 1023) as f32, ((q >> 10) & 1023) as f32, ((q >> 20) & 1023) as f32) / 1023.0 * 8.0;
            let gp = corner + local * vx;
            let off = (gp - *p).length() / vx;
            if off > 0.05 && std::env::var("SCULPT_CHECK_DIAG").is_ok() {
                let samples = read(device, queue, &gpu.buf.samples);
                let want_s = packed_samples(vol, *k);
                let got_s = &samples[s * 365..s * 365 + 365];
                let diffs: Vec<String> = (0..729)
                    .filter(|&j| (got_s[j / 2] >> (16 * (j % 2))) & 0xffff != (want_s[j / 2] >> (16 * (j % 2))) & 0xffff)
                    .take(12)
                    .map(|j| format!("({},{},{}) gpu {} cpu {}", j % 9, (j / 9) % 9, j / 81, ((got_s[j / 2] >> (16 * (j % 2))) & 0xffff) as u16 as i16, ((want_s[j / 2] >> (16 * (j % 2))) & 0xffff) as u16 as i16))
                    .collect();
                eprintln!("{when}: block {k:?} vertex {i} off {off:.3}: samples differing {diffs:?}; gpu local {local:?} cpu local {:?}", (*p - corner) / vx);
            }
            worst = worst.max(off);
        }
        // Quads, in order.
        let n_pages = slot_pages[s * SLOT_WORDS as usize] as usize;
        let n_quads = slot_pages[s * SLOT_WORDS as usize + 1] as usize;
        if n_quads != w.quads.len() {
            return Err(format!("{when}: block {k:?}: GPU {n_quads} quads in {n_pages} pages, CPU {}", w.quads.len()).into());
        }
        for (qi, cq) in w.quads.iter().enumerate() {
            let page = slot_pages[s * SLOT_WORDS as usize + 2 + qi / 64] as usize;
            let at = page * PAGE_INDICES as usize + (qi % 64) * 6;
            let tri = &pages[at..at + 6];
            let decode = |x: u32| (key_of(x >> 11), (x & 511) as u16, ((x >> 9) & 3) as u8);
            let g = [decode(tri[0]), decode(tri[1]), decode(tri[2]), decode(tri[5])];
            if g != *cq || tri[3] != tri[0] || tri[4] != tri[2] {
                return Err(format!("{when}: block {k:?} quad {qi}: GPU {g:?}, CPU {cq:?}").into());
            }
        }
        quads_total += n_quads;
    }
    if status[3] != 0 {
        return Err(format!("{when}: the GPU dropped {} quads with a corner missing", status[3]).into());
    }
    println!("{when}: {} blocks, {quads_total} quads identical; vertices within {worst:.4} steps; {} pages free", keys.len(), status[0] as i32);
    if worst > 0.05 {
        return Err(format!("{when}: vertices differ by {worst} steps").into());
    }
    Ok((keys.len(), quads_total))
}

pub fn run(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}");
    if args.iter().any(|a| a == "--bench") {
        return bench(&device, &queue);
    }
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    // Carrying detail a little at a time, as the app does, so the GPU sees the queued blocks arrive late.
    vol.carry_budget = Some(0.5);
    let mut gpu = GpuLevels::new(&device, &queue, &vol, 3);
    gpu.sync(&mut vol);
    compare(&device, &queue, &gpu, &vol, "fresh ball")?;
    let plan: [(usize, Vec3, Vec3, f32, bool); 7] = [
        (0, Vec3::new(-0.6, 0.2, 1.0), Vec3::new(0.6, 0.3, 1.0), 0.35, false),
        (1, Vec3::new(-0.4, -0.1, 1.0), Vec3::new(0.3, -0.05, 1.0), 0.06, false),
        (2, Vec3::new(0.1, 0.0, 1.0), Vec3::new(0.25, 0.05, 1.0), 0.03, true),
        (3, Vec3::new(0.0, 0.12, 1.0), Vec3::new(0.05, 0.12, 1.0), 0.012, false),
        (0, Vec3::new(-0.5, -0.4, 1.0), Vec3::new(0.4, -0.3, 1.0), 0.3, true),
        (2, Vec3::new(-0.2, 0.1, 1.0), Vec3::new(0.0, 0.2, 1.0), 0.02, false),
        (0, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.1, 1.0), 0.4, false),
    ];
    for (i, (l, a, b, r, trim)) in plan.iter().enumerate() {
        // Synced after every dab, as the app does each frame.
        let every = std::env::var("SCULPT_CHECK_EVERY_DAB").is_ok();
        let mut dab = 0;
        let mut failed: Option<String> = None;
        stroke(&mut vol, *l, *a, *b, *r, *trim, |v| {
            v.carry_pending(Some(0.5));
            let s = gpu.sync(v);
            dab += 1;
            if every && failed.is_none() && let Err(e) = compare(&device, &queue, &gpu, v, &format!("stroke {} dab {dab} (full {}, released {}, uploaded {})", i + 1, s.full, s.released, s.uploaded)) {
                failed = Some(e.to_string());
                if std::env::var("SCULPT_CHECK_DIAG").is_ok() {
                    let c = gpu.cap;
                    gpu.set_cap(c);
                    gpu.sync(v);
                    eprintln!("after a full rebuild: {:?}", compare(&device, &queue, &gpu, v, "rebuilt").err());
                }
            }
        });
        if let Some(e) = failed {
            return Err(e.into());
        }
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("stroke {} (level {l}, {})", i + 1, if *trim { "trim" } else { "clay" }))?;
    }
    for i in 0..4 {
        vol.undo();
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("undo {}", i + 1))?;
    }
    vol.redo();
    gpu.sync(&mut vol);
    compare(&device, &queue, &gpu, &vol, "redo")?;
    for cap in [0, 1, 2, 3] {
        gpu.set_cap(cap);
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("levels up to {cap}"))?;
    }
    println!("the GPU surface matches the reference at every step");
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn bench(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    // Level 0 at about 1M triangles' step; five more levels for detail.
    let voxel0 = 0.0035;
    let t = Instant::now();
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, voxel0, 6);
    vol.carry_budget = Some(6.0);
    println!("ball at level 0 step {voxel0} ({} blocks) in {:.0} ms", vol.keys().len(), t.elapsed().as_secs_f64() * 1e3);
    let mut gpu = GpuLevels::new(device, queue, &vol, 5);
    let t = Instant::now();
    let s = gpu.sync(&mut vol);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    println!("first sync: {} blocks uploaded and extracted, {:.1} MB, CPU {:.1} ms, total {:.1} ms", s.uploaded, s.bytes as f64 / 1e6, s.cpu_ms, t.elapsed().as_secs_f64() * 1e3);
    let mut rows = Vec::new();
    for (l, r) in [(0usize, 0.25f32), (1, 0.08), (2, 0.04), (3, 0.015), (4, 0.008), (5, 0.004)] {
        let (mut dab_ms, mut sync_ms, mut gpu_ms) = (Vec::new(), Vec::new(), Vec::new());
        let mut most = (0, 0);
        let a = Vec3::new(-0.1, 0.05 * l as f32, 1.0);
        let b = a + Vec3::new(0.2 * r / 0.04, 0.0, 0.0).min(Vec3::splat(0.5));
        vol.begin_stroke();
        let steps = 24;
        let mut along = Vec3::ZERO;
        for k in 0..=steps {
            let dir = a.lerp(b, k as f32 / steps as f32).normalize();
            let Some((hit, n)) = vol.raycast(dir * 3.0, -dir, l, true) else { continue };
            let t = Instant::now();
            let rough = if along == Vec3::ZERO { n } else { along };
            if let Some((n, _)) = vol.plane_dab(l, hit, rough, r, r * 0.09, 0.5, 0.4, false) {
                along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
            }
            dab_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let s = gpu.sync(&mut vol);
            sync_ms.push(s.cpu_ms);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            gpu_ms.push(t.elapsed().as_secs_f64() * 1e3);
            most = (most.0.max(s.uploaded), most.1.max(s.extracted));
        }
        let t = Instant::now();
        let end = vol.end_stroke();
        let end_ms = t.elapsed().as_secs_f64() * 1e3;
        let s = gpu.sync(&mut vol);
        rows.push(format!(
            "| {l} | {:.4} | {:.0} | {:.2} | {:.2} | {:.2} | {} / {} | {end_ms:.1} ({} rebased) | {} |",
            vol.voxel(l),
            r / vol.voxel(l),
            median(dab_ms),
            median(sync_ms),
            median(gpu_ms),
            most.0,
            most.1,
            end.rebased,
            s.extracted
        ));
    }
    // A big level 0 stroke right over the fine work: every finer level's detail under it rides along. Frame by frame
    // as the app does it: the frame's dabs (two, the app's spacing apart, a brisk stroke) only queue the carrying,
    // then 6 ms of it, then the sync.
    {
        let (mut frame_ms, mut dab_ms, mut carry_ms, mut sync_ms, mut queued) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0);
        vol.carry_budget = Some(0.0);
        vol.begin_stroke();
        let mut along = Vec3::ZERO;
        let r = 0.3;
        for frame in 0..6 {
            let t = Instant::now();
            for d in 0..2 {
                let x = -0.25 + 0.15 * r * (2 * frame + d) as f32;
                let dir = Vec3::new(x, 0.1, 1.0).normalize();
                let Some((hit, n)) = vol.raycast(dir * 3.0, -dir, 0, true) else { continue };
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, _)) = vol.plane_dab(0, hit, rough, r, r * 0.09, 0.5, 0.4, false) {
                    along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
                }
            }
            dab_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let tc = Instant::now();
            queued = queued.max(vol.carry_pending(Some(6.0)));
            carry_ms.push(tc.elapsed().as_secs_f64() * 1e3);
            let s = gpu.sync(&mut vol);
            sync_ms.push(s.cpu_ms);
            frame_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let t = Instant::now();
        let mut frames = 0;
        while vol.carry_busy() {
            vol.carry_pending(Some(6.0));
            gpu.sync(&mut vol);
            frames += 1;
        }
        let catch_up = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        vol.end_stroke();
        let end_ms = t.elapsed().as_secs_f64() * 1e3;
        gpu.sync(&mut vol);
        vol.carry_budget = Some(6.0);
        println!("a level 0 stroke (radius 0.3) over the fine work, two dabs a frame: frame {:.1} ms CPU (median: dabs {:.1}, carrying {:.1}, up to {queued} finer blocks a frame, sync {:.1}); the queued detail caught up in {frames} more frames ({catch_up:.0} ms); stroke end {end_ms:.0} ms", median(frame_ms), median(dab_ms), median(carry_ms), median(sync_ms));
    }
    println!();
    println!("| level | step | brush radius (steps) | dab ms CPU | sync ms CPU | sync to GPU done ms | blocks uploaded / extracted (most) | stroke end ms | extracted after |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("{r}");
    }
    let (counts, bytes) = vol.memory();
    println!("blocks per level {counts:?}, {:.1} MB of samples", bytes as f64 / 1e6);
    Ok(())
}
