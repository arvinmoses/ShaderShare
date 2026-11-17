//! `sculpt-app --gpu-levels-check [--bench]`: the GPU extractor against the CPU's reference, exactly.
//!
//! A ball is sculpted with plane brushes at every level, undone and redone, and shown at every level cap; after each
//! step the GPU's surface is read back and compared with `LevelVolume::extract` block by block: the word per cell
//! (vertex numbering, pieces, corner signs, transition) and the quads (corner by corner, in order) must be identical,
//! and the vertices must agree to within the 16-bit samples' rounding. A new ball's big rough rocks, set and dragged,
//! must land as the reference, and the cull pass must never leave out a cluster with a quad facing the eye (`--rocks`
//! tries more cases). With `--bench`, a larger ball is timed: a full
//! extraction, and a sync per dab of strokes at each level.

use std::time::Instant;

use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use sculpt_core::generator::level_volume::{BlockSurface, Key, LevelVolume};

use super::{GpuLevels, PAGE_INDICES, SLOT_WORDS, packed_samples};
use crate::ms;

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

pub(super) fn read(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> Vec<u32> {
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
/// A stroke from `a` to `b`, `each` run after every dab and `last` before the stroke ends.
#[allow(clippy::too_many_arguments)]
fn stroke(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool, each: impl FnMut(&mut LevelVolume), last: impl FnOnce(&mut LevelVolume)) {
    stroke_with(v, l, a, b, r, trim, false, each, last)
}

/// [`stroke`], or with `smooth` a smooth stroke.
#[allow(clippy::too_many_arguments)]
fn stroke_with(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool, smooth: bool, mut each: impl FnMut(&mut LevelVolume), last: impl FnOnce(&mut LevelVolume)) {
    v.begin_stroke();
    let steps = (((b - a).length() / (0.15 * r)).ceil() as usize).max(1);
    let mut along = Vec3::ZERO;
    for k in 0..=steps {
        let dir = a.lerp(b, k as f32 / steps as f32).normalize();
        let Some((hit, n)) = v.raycast(dir * 3.0, -dir, l, true) else { continue };
        let h = r * if trim { 0.075 } else { 0.095 };
        let rough = if along == Vec3::ZERO { n } else { along };
        let dabbed = if smooth { v.smooth_dab(l, hit, rough, r, 0.4, 0.5) } else { v.plane_dab(l, hit, rough, r, h, 0.5, 0.4, trim) };
        if let Some((n, _)) = dabbed {
            along = if along == Vec3::ZERO { n } else { (along + n).normalize() };
        }
        each(v);
    }
    last(v);
    v.end_stroke();
}

/// Sync until nothing is left for later (a big change lands over several syncs), checking after each sync on the
/// way that the surface is whole: every pair of neighbouring blocks on the GPU agrees on the samples of their shared
/// face, and no quad was dropped. Returns the last sync's stats.
fn sync_all(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &mut GpuLevels, vol: &mut LevelVolume, when: &str, landed_only: bool) -> Res<super::SyncStats> {
    let mut s = gpu.sync(vol);
    let mut syncs = 1;
    let mut landed: std::collections::HashSet<Key> = std::collections::HashSet::new();
    while gpu.pending() > 0 {
        landed.extend(gpu.last_sent.iter().copied());
        faces_agree(device, queue, gpu, &format!("{when}, after sync {syncs} with {} blocks left to land", gpu.pending()), landed_only.then_some(&landed))?;
        s = gpu.sync(vol);
        syncs += 1;
    }
    if syncs > 1 {
        println!("{when}: landed over {syncs} syncs");
    }
    Ok(s)
}

/// Every pair of neighbouring blocks the GPU holds agrees on the samples of the face, edge or corner they share (a
/// block's first layer past it is the neighbour's own first layer), and no quad was dropped. With `landed`, only
/// pairs with a block among them (the rest may be a join's preview, which the exact field differs a little from).
fn faces_agree(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &GpuLevels, when: &str, landed: Option<&std::collections::HashSet<Key>>) -> Res<()> {
    let samples = read(device, queue, &gpu.buf.samples);
    let status = read(device, queue, &gpu.buf.status);
    let at = |s: usize, i: usize| (samples[s * 365 + i / 2] >> (16 * (i % 2))) & 0xffff;
    let mut pairs = 0;
    for (k, &s) in &gpu.slot_of {
        for q in 1..8usize {
            let o = glam::IVec3::new((q & 1) as i32, ((q >> 1) & 1) as i32, (q >> 2) as i32);
            let d = (k.0, k.1 + o);
            let Some(&t) = gpu.slot_of.get(&d) else { continue };
            if landed.is_some_and(|l| !l.contains(k) && !l.contains(&d)) {
                continue;
            }
            pairs += 1;
            for z in 0..9usize {
                for y in 0..9usize {
                    for x in 0..9usize {
                        if (x == 8) as usize | (((y == 8) as usize) << 1) | (((z == 8) as usize) << 2) != q {
                            continue;
                        }
                        let i = x + 9 * y + 81 * z;
                        let j = (x - 8 * o.x as usize) + 9 * (y - 8 * o.y as usize) + 81 * (z - 8 * o.z as usize);
                        if at(s as usize, i) != at(t as usize, j) {
                            return Err(format!("{when}: block {k:?} and the block {o} past it disagree at its sample ({x},{y},{z}): {} against {}", at(s as usize, i) as u16 as i16, at(t as usize, j) as u16 as i16).into());
                        }
                    }
                }
            }
        }
    }
    if status[3] != 0 {
        return Err(format!("{when}: the GPU dropped {} quads with a corner missing", status[3]).into());
    }
    println!("{when}: {pairs} pairs of neighbouring blocks agree on their shared samples");
    Ok(())
}

/// Compare the GPU's surface with the reference, block by block. Returns (blocks, quads) compared.
fn compare(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &GpuLevels, vol: &LevelVolume, when: &str) -> Res<(usize, usize)> {
    let show = gpu.show();
    let keys: Vec<Key> = vol.keys().into_iter().filter(|k| !show.hides(*k)).collect();
    let (want, _) = vol.extract_in(&keys, &show);
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
            let page = slot_pages[s * SLOT_WORDS as usize + 27 + i / 64] as usize;
            let q = verts[(page * 64 + i % 64) * 2];
            let local = Vec3::new((q & 1023) as f32, ((q >> 10) & 1023) as f32, ((q >> 20) & 1023) as f32) / 1023.0 * 8.0;
            let gp = corner + local * vx;
            let off = (gp - *p).length() / vx;
            if off > 0.05 && std::env::var("SCULPT_CHECK_DIAG").is_ok() {
                let samples = read(device, queue, &gpu.buf.samples);
                let want_s = packed_samples(vol, *k, Some(gpu.view.hidden()));
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

/// The shader's surface noise against `sculpt_core::surface_noise`, at points about the unit ball, for each kind and
/// some octave counts. Returns the largest difference against the amplitude.
fn surface_noise_check(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<f32> {
    use sculpt_core::surface_noise::{SurfaceNoise, WGSL};
    let src = format!(
        "{WGSL}
struct P {{ f: vec4<f32>, i: vec4<u32>, k: vec4<u32> }};
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= arrayLength(&pts)) {{ return; }}
    SN_F = prm.f;
    SN_I = prm.i;
    out[id.x] = sn_height(pts[id.x].xyz, prm.k.x);
}}"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("surface noise check"), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let n = 4096usize;
    let pts: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            let t = i as f32;
            let d = Vec3::new((t * 0.7548).sin(), (t * 0.5698).cos(), (t * 0.3165).sin()).normalize_or(Vec3::Z);
            (d * (1.0 + 0.1 * (t * 0.137).sin())).extend(0.0).to_array()
        })
        .collect();
    use eframe::egui_wgpu::wgpu::util::DeviceExt;
    let pts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pts), usage: wgpu::BufferUsages::STORAGE });
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let mut worst = 0.0f32;
    for kind in [0u32, 1] {
        for octaves in [1u32, 4, 7] {
            let sn = SurfaceNoise { on: true, amplitude: 0.02, scale: 9.0, octaves: 7, kind, seed: 3 };
            let (f, i) = sn.uniform();
            let mut prm = Vec::new();
            prm.extend_from_slice(bytemuck::cast_slice(&f));
            prm.extend_from_slice(bytemuck::cast_slice(&i));
            prm.extend_from_slice(bytemuck::cast_slice(&[octaves, 0u32, 0, 0]));
            let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &prm, usage: wgpu::BufferUsages::UNIFORM });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pts_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                ],
            });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(&pipe);
                cp.set_bind_group(0, &group, &[]);
                cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            queue.submit([enc.finish()]);
            let got: Vec<f32> = read(device, queue, &out).into_iter().map(f32::from_bits).collect();
            let off = pts.iter().zip(&got).map(|(p, g)| (sn.height(Vec3::new(p[0], p[1], p[2]), octaves) - g).abs()).fold(0.0f32, f32::max);
            println!("surface noise, kind {kind}, {octaves} octaves: GPU against CPU within {:.2e} of its height", off / sn.amplitude);
            worst = worst.max(off / sn.amplitude);
        }
    }
    Ok(worst)
}

/// The shader's generator joining (`vg_apply`) against `VolumeGen::apply`, for every primitive, both joins and each
/// roughness, at points about the instances. Returns the largest difference against the instances' size.
fn generator_check(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<f32> {
    use eframe::egui_wgpu::wgpu::util::DeviceExt;
    use sculpt_core::generator::layer::GeneratorSpec;
    use sculpt_core::generator::volume_gen::{Anchor, VolumeGen, WGSL as GEN_WGSL};
    use sculpt_core::generator::{BoolOp, Primitive};
    use sculpt_core::noise::NoiseKind;
    let src = format!(
        "{}
{GEN_WGSL}
@group(0) @binding(0) var<storage, read> vg_inst: array<GpuInstance>;
@group(0) @binding(1) var<storage, read> vg_cand: array<u32>;
@group(0) @binding(2) var<uniform> gen: GpuGen;
@group(0) @binding(3) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= arrayLength(&pts)) {{ return; }}
    let p = pts[id.x];
    out[id.x] = vg_apply(gen, p.w, p.xyz, 0u, arrayLength(&vg_cand));
}}",
        sculpt_core::surface_noise::WGSL
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("generator check"), source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let anchors: Vec<Anchor> = (0..6).map(|i| {
        let t = i as f32;
        let n = Vec3::new((t * 1.7).sin(), 0.4 + 0.3 * (t * 0.9).cos(), (t * 1.7).cos()).normalize();
        Anchor { p: n, n }
    }).collect();
    let n = 4096usize;
    let mut worst = 0.0f32;
    for prim in Primitive::ALL {
        for op in [BoolOp::Add, BoolOp::Subtract] {
            for rough in [None, Some(NoiseKind::Fbm), Some(NoiseKind::Ridged), Some(NoiseKind::Cellular)] {
                let spec = GeneratorSpec { primitive: prim, op, size: 0.2, lift: 0.3, blend: 0.3, stretch_var: 0.3, tilt: 20.0, random_spin: true, rough: if rough.is_some() { 0.25 } else { 0.0 }, rough_kind: rough.unwrap_or_default(), ..GeneratorSpec::default() };
                let g = VolumeGen::from_anchors(&spec, anchors.clone()).fitted(0.005);
                let pts: Vec<[f32; 4]> = (0..n)
                    .map(|i| {
                        let t = i as f32;
                        let a = anchors[i % anchors.len()].p;
                        let q = a * 1.1 + Vec3::new((t * 0.7548).sin(), (t * 0.5698).cos(), (t * 0.3165).sin()) * 0.3;
                        [q.x, q.y, q.z, q.length() - 1.0]
                    })
                    .collect();
                let inst = g.gpu_instances();
                let cand: Vec<u32> = (0..inst.len() as u32).collect();
                let params = g.gpu_params();
                let bufs = [
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&inst), usage: wgpu::BufferUsages::STORAGE }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&cand), usage: wgpu::BufferUsages::STORAGE }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM }),
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pts), usage: wgpu::BufferUsages::STORAGE }),
                    device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
                ];
                let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries });
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(&pipe);
                    cp.set_bind_group(0, &group, &[]);
                    cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
                }
                queue.submit([enc.finish()]);
                let got: Vec<f32> = read(device, queue, &bufs[4]).into_iter().map(f32::from_bits).collect();
                let off = pts.iter().zip(&got).map(|(p, gv)| (g.apply(p[3], Vec3::new(p[0], p[1], p[2]), &cand, 1.0) - gv).abs()).fold(0.0f32, f32::max) / spec.size;
                if off > 1e-4 {
                    println!("generator {prim:?} {op:?} roughness {rough:?}: GPU against CPU {off:.2e} of its size");
                }
                worst = worst.max(off);
            }
        }
    }
    println!("generator joining: GPU against CPU within {worst:.2e} of the instances' size, every primitive, join and roughness");
    Ok(worst)
}

/// An edit of a generator joined on the GPU against the exact field the CPU's job brings after it: the samples near the
/// surface agree but for the distance the CPU makes of steep joins again, and the surface built on the GPU drops
/// nothing.
fn join_check(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    // Rocks at level 2, and at level 1 (where a new generator goes on a new ball).
    for level in [2, 1] {
        println!("rocks at level {level}:");
        join_check_at(device, queue, level)?;
    }
    Ok(())
}

fn join_check_at(device: &wgpu::Device, queue: &wgpu::Queue, level: usize) -> Res<()> {
    use sculpt_core::generator::detail_layer::Layered;
    use sculpt_core::generator::layer::GeneratorSpec;
    use sculpt_core::generator::{BoolOp, Primitive};
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    let mut layer = Layered::new(&mut vol, 2);
    let rocks = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 12, seed: 3, size: 0.12, lift: 0.2, min_spacing: 0.4, blend: 0.3, rough: 0.2, stretch_var: 0.3, tilt: 20.0, ..GeneratorSpec::default() };
    let k = layer.add_generator(&mut vol, "Rocks", level, &rocks);
    // Its mask painted away over the front, half way at its edge: the GPU fades the join as the CPU does.
    layer.set_active(k);
    layer.paint_mask(&mut vol, Vec3::new(0.0, 0.0, 1.0), 0.6, 0.0);
    let mut gpu = GpuLevels::new(device, queue, &vol, 3);
    gpu.sync(&mut vol);
    for (step, spec) in [GeneratorSpec { size: 0.15, ..rocks.clone() }, GeneratorSpec { blend: 0.6, ..rocks.clone() }, GeneratorSpec { op: BoolOp::Subtract, lift: 0.0, ..rocks.clone() }, GeneratorSpec { count: 20, ..rocks.clone() }].into_iter().enumerate() {
        let t = Instant::now();
        let plan = layer.gpu_generator(&mut vol, k, &spec).ok_or("the GPU cannot join this stack")?;
        let plan_ms = ms(t);
        let keys: Vec<Key> = plan.blocks.clone();
        gpu.sync_for_join(&mut vol, &plan);
        let s = gpu.join(plan, &|k| layer.base_samples(k));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let status = read(device, queue, &gpu.buf.status);
        if status[3] != 0 {
            return Err(format!("join step {step}: the GPU dropped {} quads", status[3]).into());
        }
        let samples = read(device, queue, &gpu.buf.samples);
        let got: Vec<(Key, Vec<u32>)> = keys.iter().filter_map(|k| gpu.slot_of.get(k).map(|&sl| (*k, samples[sl as usize * 365..sl as usize * 365 + 365].to_vec()))).collect();
        // The exact field.
        let tj = Instant::now();
        while let Some(job) = layer.generator_take_landing(&vol) {
            layer.generator_apply(&mut vol, job.run());
        }
        let job_ms = ms(tj);
        let (mut flips, mut worst, mut near) = (0usize, 0.0f32, 0usize);
        for (key, words) in &got {
            let want = packed_samples(&vol, *key, Some(gpu.view.hidden()));
            let (l, v) = (key.0 as usize, vol.voxel(key.0 as usize));
            let band = vol.band(l);
            for j in 0..729 {
                let g = ((words[j / 2] >> (16 * (j % 2))) & 0xffff) as u16 as i16 as f32 / 32767.0 * band;
                let w = ((want[j / 2] >> (16 * (j % 2))) & 0xffff) as u16 as i16 as f32 / 32767.0 * band;
                if w.abs() < 2.0 * v {
                    near += 1;
                    worst = worst.max((g - w).abs() / v);
                    if (g < 0.0) != (w < 0.0) && w.abs() > 0.25 * v {
                        flips += 1;
                    }
                }
            }
        }
        println!("join step {step}: {} blocks joined on the GPU (plan {plan_ms:.1} ms, join {:.1} ms CPU) against the CPU's job ({job_ms:.0} ms): near the surface ({near} samples) at most {worst:.2} steps apart, {flips} sides differ", got.len(), s.cpu_ms);
        if flips > near / 1000 {
            return Err(format!("join step {step}: {flips} samples on the other side").into());
        }
        // The exact field landing, region by region.
        gpu.land_limit = Some(4000);
        sync_all(device, queue, &mut gpu, &mut vol, &format!("join step {step}, the exact field"), true)?;
        gpu.land_limit = None;
    }
    Ok(())
}

/// A quad's two triangles' unit normals, as drawn ((0, 1, 2) and (0, 2, 3)), facing out as their winding does.
fn triangle_normals(p: &[(Vec3, Vec3)]) -> Vec<Vec3> {
    [(p[1].0 - p[0].0).cross(p[2].0 - p[0].0), (p[2].0 - p[0].0).cross(p[3].0 - p[0].0)].into_iter().filter(|n| n.length_squared() > 1e-20).map(|n| n.normalize()).collect()
}

/// Quads the cull pass would leave out as facing away though they face the eye, from eyes all round at two distances:
/// (eyes tried, quads wrongly left out, blocks with any).
fn wrongly_culled(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &GpuLevels, vol: &LevelVolume) -> (usize, usize, usize) {
    let show = gpu.show();
    let keys: Vec<Key> = vol.keys().into_iter().filter(|k| !show.hides(*k)).collect();
    let (surf, _) = vol.extract_in(&keys, &show);
    let cl: Vec<f32> = read(device, queue, &gpu.buf.clusters).into_iter().map(f32::from_bits).collect();
    let (mut eyes, mut bad, mut blocks) = (0, 0, std::collections::HashSet::new());
    // How far below each block's cone its quads' own normals reach (the cone's cosine less the least cosine).
    let mut under: Vec<f32> = Vec::new();
    for k in &keys {
        let s = gpu.slot_of[k] as usize;
        let cone = glam::Vec4::new(cl[s * 12 + 8], cl[s * 12 + 9], cl[s * 12 + 10], cl[s * 12 + 11]);
        if cone.w < -1.5 || surf[k].quads.is_empty() {
            continue;
        }
        let mut least = 1.0f32;
        for q in &surf[k].quads {
            let Some(p): Option<Vec<(Vec3, Vec3)>> = q.iter().map(|(b, c, pc)| surf.get(b).and_then(|sb| sb.vertex(*c, *pc).map(|v| (sb.verts[v as usize], sb.normals[v as usize])))).collect() else { continue };
            for n in triangle_normals(&p) {
                least = least.min(n.dot(cone.truncate()));
            }
        }
        under.push(cone.w - least);
    }
    let cullable = keys.iter().filter(|k| cl[gpu.slot_of[*k] as usize * 12 + 11] > 0.1).count();
    under.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |f: f32| if under.is_empty() { 0.0 } else { under[((under.len() - 1) as f32 * f) as usize] };
    println!("cones: {cullable} of {} blocks can be culled as facing away; the quads' normals reach past the cone by at most {:.3} (99.9%: {:.3}) in cosine", keys.len(), pct(1.0), pct(0.999));
    for dist in [1.6f32, 3.0] {
        for i in 0..64 {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / 64.0;
            let a = i as f32 * 2.399963;
            let r = (1.0 - z * z).sqrt();
            let eye = Vec3::new(r * a.cos(), r * a.sin(), z) * dist;
            eyes += 1;
            for k in &keys {
                let s = gpu.slot_of[k] as usize;
                let vx = vol.voxel(k.0 as usize);
                let corner = vol.origin() + (k.1 * 8).as_vec3() * vx;
                let lo = corner + Vec3::new(cl[s * 12], cl[s * 12 + 1], cl[s * 12 + 2]) * vx;
                let hi = corner + Vec3::new(cl[s * 12 + 4], cl[s * 12 + 5], cl[s * 12 + 6]) * vx;
                let cone = glam::Vec4::new(cl[s * 12 + 8], cl[s * 12 + 9], cl[s * 12 + 10], cl[s * 12 + 11]);
                if cone.w <= 0.1 {
                    continue;
                }
                let (centre, radius) = (0.5 * (lo + hi), 0.5 * (hi - lo).length());
                let to = centre - eye;
                let dist = to.length();
                if dist <= radius {
                    continue;
                }
                let sin_cone = (1.0 - cone.w * cone.w).max(0.0).sqrt();
                let sin_view = radius / dist;
                let c = cone.truncate().dot(to / dist);
                let cos_view = (1.0 - sin_view * sin_view).max(0.0).sqrt();
                let cos_sum = cone.w * cos_view - sin_cone * sin_view;
                let sin_sum = sin_cone * cos_view + cone.w * sin_view;
                if !(cos_sum > 0.1 && c > sin_sum + 0.1) {
                    continue;
                }
                // Culled: does any of its quads face the eye?
                for q in &surf[k].quads {
                    let Some(p): Option<Vec<(Vec3, Vec3)>> = q.iter().map(|(b, c, pc)| surf.get(b).and_then(|sb| sb.vertex(*c, *pc).map(|v| (sb.verts[v as usize], sb.normals[v as usize])))).collect() else { continue };
                    let mid = (p[0].0 + p[1].0 + p[2].0 + p[3].0) * 0.25;
                    if triangle_normals(&p).iter().any(|n| n.dot((eye - mid).normalize()) > 0.05) {
                        bad += 1;
                        blocks.insert(*k);
                    }
                }
            }
        }
    }
    (eyes, bad, blocks.len())
}

/// A frame as the viewport has it: a generator edit's plan joined on the GPU, else a plain sync; then the tick. True
/// if the frame changed nothing on the GPU.
fn rocks_frame(gpu: &mut GpuLevels, mode: &mut crate::volume_mode::VolumeMode) -> bool {
    let plan = mode.gpu_plan.take();
    let idle = plan.is_none();
    let s = match &plan {
        Some(plan) => gpu.sync_for_join(&mut mode.vol, plan),
        None => gpu.sync(&mut mode.vol),
    };
    if let (Some(plan), Some(layer)) = (plan, mode.layer.as_ref()) {
        gpu.join(plan, &|k| layer.base_samples(k));
    }
    mode.tick();
    // Nothing joined, sent or rebuilt, and nothing left to land.
    idle && s.uploaded == 0 && s.extracted == 0 && gpu.pending() == 0
}

/// A new ball's rocks made big and rough (a few drag frames, or set at once), the views by distance or not: once
/// landed and idle, the GPU's surface is the CPU's reference exactly.
#[allow(clippy::too_many_arguments)]
fn big_rocks_check(device: &wgpu::Device, queue: &wgpu::Queue, triangles: f32, size: f32, rough: f32, drag: bool, by_distance: bool, level: Option<usize>, seeds: bool) -> Res<()> {
    let when = format!("rocks size {size} rough {rough} drag {drag} by distance {by_distance} level {level:?} seeds {seeds}");
    let mut mode = crate::volume_mode::VolumeMode::ball(triangles);
    mode.add_generator();
    if let Some(l) = level {
        let k = mode.layer.as_ref().unwrap().active();
        let vol = &mut mode.vol;
        mode.layer.as_mut().unwrap().set_level(vol, k, l);
        while let Some(job) = mode.layer.as_mut().unwrap().generator_take() {
            let done = job.run();
            mode.layer.as_mut().unwrap().generator_apply(&mut mode.vol, done);
        }
    }
    let mut gpu = GpuLevels::new(device, queue, &mode.vol, mode.vol.levels() - 1);
    gpu.view.on = by_distance;
    if by_distance {
        gpu.set_eye(Vec3::new(0.3, 0.4, 2.2), 1000.0);
    }
    rocks_frame(&mut gpu, &mut mode);
    let k = mode.layer.as_ref().unwrap().active();
    let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
    spec.rough = rough;
    if drag {
        spec.blend = 0.6;
        for i in 0..=6 {
            spec.size = size * (0.7 + 0.05 * i as f32);
            if seeds {
                spec.seed = 1 + i as u64;
            }
            mode.set_generator(k, &spec, true);
            rocks_frame(&mut gpu, &mut mode);
        }
        mode.end_generator_edit();
        // A second drag while the first lands.
        rocks_frame(&mut gpu, &mut mode);
        for i in 0..=4 {
            spec.rough = rough * (0.8 + 0.05 * i as f32);
            if seeds {
                spec.seed = 20 + i as u64;
            }
            mode.set_generator(k, &spec, true);
            rocks_frame(&mut gpu, &mut mode);
        }
        mode.end_generator_edit();
    } else {
        spec.size = size;
        mode.set_generator(k, &spec, false);
    }
    let start = Instant::now();
    loop {
        let quiet = rocks_frame(&mut gpu, &mut mode);
        if quiet && !mode.busy() && mode.vol.dirty_keys().is_empty() {
            break;
        }
        if start.elapsed().as_secs() > 300 {
            return Err(format!("{when}: never landed").into());
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    rocks_frame(&mut gpu, &mut mode);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    // The samples the GPU holds against those the CPU would send.
    let samples = read(device, queue, &gpu.buf.samples);
    let stale: Vec<Key> = mode.vol.keys().into_iter().filter(|k| gpu.slot_of.get(k).is_some_and(|&sl| samples[sl as usize * 365..sl as usize * 365 + 365] != packed_samples(&mode.vol, *k, Some(gpu.view.hidden()))[..])).collect();
    if !stale.is_empty() {
        let mut per = [0usize; 8];
        for k in &stale {
            per[k.0 as usize] += 1;
        }
        let k = stale[0];
        let sl = gpu.slot_of[&k] as usize;
        let cpu = packed_samples(&mode.vol, k, Some(gpu.view.hidden()));
        let diff = (0..365).filter(|&i| samples[sl * 365 + i] != cpu[i]).count();
        return Err(format!("{when}: once landed, {} blocks on the GPU hold other samples than the CPU's (per level {per:?}; first {k:?}: {diff} words differ, inside0 {})", stale.len(), mode.vol.inside0_has(k.1)).into());
    }
    let (eyes, bad, blocks) = wrongly_culled(device, queue, &gpu, &mode.vol);
    println!("{when}: from {eyes} eyes, {bad} quads facing the eye left out as facing away");
    if bad > 0 {
        return Err(format!("{when}: the cull pass leaves out {bad} quads facing the eye, in {blocks} blocks (holes)").into());
    }
    // Quads dropped while the edit's frames joined its preview (whose blocks' neighbours differ from it a little) are
    // counted on the way; once landed, every quad is compared with the reference below.
    queue.write_buffer(&gpu.buf.status, 12, bytemuck::bytes_of(&0i32));
    compare(device, queue, &gpu, &mode.vol, &when)?;
    Ok(())
}

pub fn run(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}");
    let off = surface_noise_check(&device, &queue)?;
    let gen_off = generator_check(&device, &queue)?;
    join_check(&device, &queue)?;
    // Big rough rocks on a new ball: set at once (over a hundred thousand blocks coming at once on a small slot table),
    // and dragged at level 2 with their seed changing, seen by distance. With `--rocks`, more: by distance or not, on
    // the app's ball, at level 1 and 2.
    let mut rocks = vec![(2e5f32, 0.3f32, 0.6f32, false, false, None, false), (2e5, 0.15, 0.5, true, true, Some(2), true)];
    if args.iter().any(|a| a == "--rocks") {
        rocks.extend([(2e5f32, 0.15f32, 0.5f32, true, false, Some(2), true), (1e6, 0.2, 0.5, true, true, None, true), (1e6, 0.12, 0.5, true, true, Some(2), true)]);
    }
    for (tri, size, rough, drag, dist, level, seeds) in rocks {
        big_rocks_check(&device, &queue, tri, size, rough, drag, dist, level, seeds)?;
    }
    if gen_off > 1e-3 {
        return Err(format!("the shader's generator joining differs from the CPU's by {gen_off} of the instances' size").into());
    }
    if off > 1e-3 {
        return Err(format!("the shader's surface noise differs from the CPU's by {off} of its height").into());
    }
    if args.iter().any(|a| a == "--bench-join") {
        return bench_join(&device, &queue);
    }
    if args.iter().any(|a| a == "--bench") {
        return bench(&device, &queue);
    }
    // `--dumps DIR`: volumes saved from the review's frames (`SCULPT_DEMO_DUMP`), each extracted on the GPU from scratch
    // and compared with the reference.
    if let Some(dir) = args.iter().position(|a| a == "--dumps").and_then(|i| args.get(i + 1)) {
        let mut files: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "svol")).collect();
        files.sort();
        for f in files {
            let mut vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(&f)?), crate::volume_mode::OPEN_LEVELS)?;
            let cap = vol.levels() - 1;
            let mut gpu = GpuLevels::new(&device, &queue, &vol, cap);
            gpu.sync(&mut vol);
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            match compare(&device, &queue, &gpu, &vol, &name) {
                Ok((blocks, verts)) => println!("{name}: matches ({blocks} blocks, {verts} vertices)"),
                Err(e) => println!("{name}: {e}"),
            }
        }
        return Ok(());
    }
    for jobs in [false, true] {
        let pass = if jobs { "jobs: " } else { "" };
        let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
        // Carrying detail a little at a time, so the GPU sees the queued blocks arrive late: first 0.5 ms after each dab,
        // then (with `jobs`) as the app does, in jobs taken after a dab and brought in after the next.
        vol.carry_budget = Some(if jobs { 0.0 } else { 0.5 });
        // The first pass buries finer detail under coarse strokes (the default); the second keeps it, carried in jobs.
        vol.keep_detail = jobs;
        let mut gpu = GpuLevels::new(&device, &queue, &vol, 3);
        // Changes land a few hundred blocks a sync, so a stroke's dabs land over several syncs, as a big change does
        // in the app.
        gpu.land_limit = Some(200);
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("{pass}fresh ball"))?;
        // Clay and trim strokes, then smooth strokes over them (`None`).
        let plan: [(usize, Vec3, Vec3, f32, Option<bool>); 9] = [
            (0, Vec3::new(-0.6, 0.2, 1.0), Vec3::new(0.6, 0.3, 1.0), 0.35, Some(false)),
            (1, Vec3::new(-0.4, -0.1, 1.0), Vec3::new(0.3, -0.05, 1.0), 0.06, Some(false)),
            (2, Vec3::new(0.1, 0.0, 1.0), Vec3::new(0.25, 0.05, 1.0), 0.03, Some(true)),
            (3, Vec3::new(0.0, 0.12, 1.0), Vec3::new(0.05, 0.12, 1.0), 0.012, Some(false)),
            (0, Vec3::new(-0.5, -0.4, 1.0), Vec3::new(0.4, -0.3, 1.0), 0.3, Some(true)),
            (2, Vec3::new(-0.2, 0.1, 1.0), Vec3::new(0.0, 0.2, 1.0), 0.02, Some(false)),
            (0, Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.2, 0.1, 1.0), 0.4, Some(false)),
            (2, Vec3::new(0.05, 0.0, 1.0), Vec3::new(0.3, 0.1, 1.0), 0.04, None),
            (0, Vec3::new(-0.4, -0.3, 1.0), Vec3::new(0.4, 0.2, 1.0), 0.3, None),
        ];
        for (i, (l, a, b, r, kind)) in plan.iter().enumerate() {
            let (trim, smooth) = (&kind.unwrap_or(false), kind.is_none());
            // Synced after every dab, as the app does each frame.
            let every = std::env::var("SCULPT_CHECK_EVERY_DAB").is_ok();
            let mut dab = 0;
            let mut failed: Option<String> = None;
            let held: std::cell::RefCell<Option<sculpt_core::generator::level_volume::CarryJob>> = std::cell::RefCell::new(None);
            let bring_in = |v: &mut LevelVolume| {
                if let Some(job) = held.borrow_mut().take() {
                    v.carry_apply(job.run());
                }
            };
            stroke_with(&mut vol, *l, *a, *b, *r, *trim, smooth, |v| {
                if jobs {
                    bring_in(v);
                    *held.borrow_mut() = v.carry_take(64);
                } else {
                    v.carry_pending(Some(0.5));
                }
                let s = match sync_all(&device, &queue, &mut gpu, v, &format!("{pass}stroke {} dab {}", i + 1, dab + 1), false) {
                    Ok(s) => s,
                    Err(e) => {
                        failed.get_or_insert(e.to_string());
                        return;
                    }
                };
                dab += 1;
                if every && failed.is_none() && let Err(e) = compare(&device, &queue, &gpu, v, &format!("{pass}stroke {} dab {dab} (full {}, released {}, uploaded {})", i + 1, s.full, s.released, s.uploaded)) {
                    failed = Some(e.to_string());
                    if std::env::var("SCULPT_CHECK_DIAG").is_ok() {
                        let samples = read(&device, &queue, &gpu.buf.samples);
                        let stale: Vec<Key> = v.keys().into_iter().filter(|k| gpu.slot_of.get(k).is_some_and(|&sl| samples[sl as usize * 365..sl as usize * 365 + 365] != packed_samples(v, *k, Some(gpu.view.hidden()))[..])).collect();
                        eprintln!("blocks whose GPU samples are stale: {stale:?}");
                        let c = gpu.cap;
                        gpu.set_cap(c.saturating_sub(1));
                        gpu.sync(v);
                        gpu.set_cap(c);
                        gpu.sync(v);
                        eprintln!("after a full rebuild: {:?}", compare(&device, &queue, &gpu, v, "rebuilt").err());
                    let orphans: Vec<Key> = v.keys().into_iter().filter(|k| k.0 > 0 && !v.exists(k.0 as usize - 1, k.1.div_euclid(glam::IVec3::splat(2)))).collect();
                    eprintln!("blocks without a parent: {orphans:?}");
                    }
                }
            }, bring_in);
            if let Some(e) = failed {
                return Err(e.into());
            }
            sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}stroke {}", i + 1), false)?;
            compare(&device, &queue, &gpu, &vol, &format!("{pass}stroke {} (level {l}, {})", i + 1, if smooth { "smooth" } else if *trim { "trim" } else { "clay" }))?;
        }
        for i in 0..4 {
            vol.undo();
            sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}undo {}", i + 1), false)?;
            compare(&device, &queue, &gpu, &vol, &format!("{pass}undo {}", i + 1))?;
        }
        vol.redo();
        sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}redo"), false)?;
        compare(&device, &queue, &gpu, &vol, &format!("{pass}redo"))?;
        for cap in [0, 1, 2, 3] {
            gpu.set_cap(cap);
            gpu.sync(&mut vol);
            compare(&device, &queue, &gpu, &vol, &format!("{pass}levels up to {cap}"))?;
        }
        // The level chosen by distance: the camera moving about, near and far, with a stroke now and then, each
        // view's blocks shown or hidden on the GPU as they are in the reference.
        let eyes = [Vec3::new(0.0, 0.0, 1.3), Vec3::new(0.0, 0.0, 6.0), Vec3::new(2.0, 0.5, -1.5), Vec3::new(0.0, 0.0, 30.0), Vec3::new(0.2, 0.1, 1.15), Vec3::new(0.0, 0.0, 3.0)];
        for (i, eye) in eyes.iter().enumerate() {
            if i % 2 == 1 {
                let a = Vec3::new(-0.2 + 0.1 * i as f32, 0.0, 1.0);
                stroke(&mut vol, 1 + i % 3, a, a + Vec3::new(0.15, 0.05, 0.0), [0.06, 0.03, 0.012][i % 3], false, |v| { v.carry_pending(None); }, |_| {});
            }
            gpu.set_eye(*eye, 1000.0);
            sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}seen from {eye}"), false)?;
            while gpu.view.stats.stale > 0 {
                sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}seen from {eye}, measured"), false)?;
            }
            let st = gpu.view.stats;
            compare(&device, &queue, &gpu, &vol, &format!("{pass}seen from {eye} ({} blocks shown, {} hidden)", st.shown, st.hidden))?;
        }
        // A tight triangle budget, near: only the detail that shows most.
        gpu.view.budget = Some(gpu.view.stats.triangles * 9 / 10);
        gpu.set_eye(Vec3::new(0.0, 0.0, 1.3), 1000.0);
        sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}within a budget"), false)?;
        let st = gpu.view.stats;
        compare(&device, &queue, &gpu, &vol, &format!("{pass}within a budget ({} triangles, detail down to {:.1} px)", st.triangles, st.tau_used))?;
        gpu.view.budget = None;
        gpu.view.on = false;
        sync_all(&device, &queue, &mut gpu, &mut vol, &format!("{pass}every level again"), false)?;
        compare(&device, &queue, &gpu, &vol, &format!("{pass}every level again"))?;
    }
    println!("the GPU surface matches the reference at every step");
    Ok(())
}

pub(super) fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// At the app's scale: a generator's edits joined on the GPU, timed (planning, joining, the GPU's work) against the
/// CPU's job (`--bench-join`).
fn bench_join(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    use sculpt_core::generator::detail_layer::Layered;
    use sculpt_core::generator::layer::GeneratorSpec;
    use sculpt_core::generator::{BoolOp, Primitive};
    // `SCULPT_BENCH_LEVEL=2` for the one level.
    let only: Option<usize> = std::env::var("SCULPT_BENCH_LEVEL").ok().and_then(|v| v.parse().ok());
    for (level, count) in [(2usize, 12u32), (3, 12)].into_iter().filter(|(l, _)| only.is_none_or(|o| o == *l)) {
        let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, 0.007, 6);
        let mut layer = Layered::new(&mut vol, 5);
        let size = 0.028 * 2.5;
        let spec = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count, seed: 1, size, lift: 0.15, random_spin: true, min_spacing: 3.0 * size, size_var: 0.4, stretch_var: 0.35, tilt: 25.0, blend: 0.3, rough: 0.15, ..GeneratorSpec::default() };
        let k = layer.add_generator(&mut vol, "Rocks", level, &spec);
        let mut gpu = GpuLevels::new(device, queue, &vol, 5);
        gpu.sync(&mut vol);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut rows = Vec::new();
        for i in 1..=6 {
            let next = GeneratorSpec { size: size * (1.0 + 0.04 * i as f32), ..spec.clone() };
            let t = Instant::now();
            let plan = layer.gpu_generator(&mut vol, k, &next).ok_or("no GPU join")?;
            let plan_ms = ms(t);
            let n = plan.blocks.len();
            let ts = Instant::now();
            gpu.sync_for_join(&mut vol, &plan);
            let sync_ms = ms(ts);
            let s = gpu.join(plan, &|k| layer.base_samples(k));
            let tg = Instant::now();
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let gpu_ms = ms(tg);
            let tj = Instant::now();
            let mut apply_ms = 0.0;
            while let Some(job) = layer.generator_take_landing(&vol) {
                let done = job.run();
                let ta = Instant::now();
                layer.generator_apply(&mut vol, done);
                apply_ms += ms(ta);
            }
            let job_ms = ms(tj);
            // The exact field landing: the syncs after the job, as the app's next frames do it, region by region,
            // each frame's GPU work done before the next (as presenting a frame has it).
            let tl = Instant::now();
            let (mut syncs, mut worst, mut worst_frame, mut cpu, mut sent, mut extracted) = (0, 0.0f64, 0.0f64, 0.0f64, 0, 0);
            loop {
                let tf = Instant::now();
                let land = gpu.sync(&mut vol);
                let ms = ms(tf);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                worst = worst.max(ms);
                worst_frame = worst_frame.max(crate::ms(tf));
                cpu += ms;
                sent += land.uploaded;
                extracted += land.extracted;
                syncs += 1;
                if gpu.pending() == 0 {
                    break;
                }
            }
            let land_ms = ms(tl);
            rows.push(format!(
                "{n} blocks: plan {plan_ms:.1}, sync {sync_ms:.1}, join {:.1} ms CPU, then the GPU {gpu_ms:.1} ms; the job {job_ms:.0} ms ({apply_ms:.1} taking it in); landing it: {syncs} syncs, the longest {worst:.1} ms CPU ({worst_frame:.1} with its GPU work), {cpu:.1} ms CPU in all ({sent} sent, {extracted} extracted), all done at {land_ms:.1} ms",
                s.cpu_ms
            ));
        }
        println!("rocks at level {level} (blocks per level {:?}):", vol.memory().0);
        for r in rows {
            println!("  {r}");
        }
        gpu.sync(&mut vol);
        let b = &gpu.buf;
        let mb = |x: u64| x as f64 / 1e6;
        println!(
            "  pools: {} blocks in {} slots; quad pages {} used of {} ({:.0} MB); vertex pages {} used of {} ({:.0} MB)",
            gpu.slots(), b.slots, b.pages as i64 - gpu.status[0] as i64, b.pages, mb(b.pages_buf.size()), b.vpages as i64 - gpu.status[4] as i64, b.vpages, mb(b.verts.size())
        );
    }
    Ok(())
}

/// `SCULPT_BENCH_FAR`: wide clay strokes seen from far (a ball of radius 1, eye at 4, focal 1100 px; brushes of radius
/// 0.45 and 0.7), a sync after each dab as the app does a frame, on a new ball (its coarse level 0 under the step of
/// 1M triangles) and on one as balls were made before it (level 0 at that step, six levels): each stroke at the level
/// the app picks for it. Per stroke: the median dab and sync (CPU), the blocks a dab changed, and those sent and
/// extracted.
fn bench_far(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    let eye = Vec3::new(0.0, 0.0, 4.0);
    let focal = 1100.0;
    let old = (4.0 * std::f32::consts::PI / 5e5f32).sqrt();
    for (name, voxel0, levels) in [("new ball (7 levels)", crate::volume_mode::ball_voxel0(1e6), crate::volume_mode::LEVELS), ("ball as before (6 levels)", old, crate::volume_mode::OPEN_LEVELS)] {
        let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, voxel0, levels);
        let mut gpu = GpuLevels::new(device, queue, &vol, levels - 1);
        gpu.set_eye(eye, focal);
        gpu.sync(&mut vol);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let patch = Vec3::new(0.30, 0.34, 0.89).normalize();
        let (t1, t2) = patch.any_orthonormal_pair();
        println!("{name}, level 0 step {voxel0:.4}:");
        println!("| radius | level | dabs | dab ms CPU | sync ms CPU | sync to GPU done ms | blocks changed a dab | sent | extracted | stroke end ms |");
        println!("|---|---|---|---|---|---|---|---|---|---|");
        for (k, r) in [0.45f32, 0.7, 0.45, 0.7].into_iter().enumerate() {
            let (mut dab_ms, mut sync_ms, mut gpu_ms, mut changed, mut sent, mut extracted) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let b = 0.05 * k as f32 - 0.05;
            let steps = (0.4 / (0.15 * r)).ceil() as usize;
            vol.begin_stroke();
            let mut along = Vec3::ZERO;
            let mut level = None;
            for i in 0..=steps {
                let a = -0.2 + 0.4 * i as f32 / steps as f32;
                let d = (patch + t1 * if k % 2 == 0 { a } else { -a } + t2 * (b + 0.05 * a)).normalize();
                let Some((hit, n)) = vol.raycast(d * 3.0, -d, vol.levels() - 1, true) else { continue };
                // The level as the app picks it, on the stroke's first dab.
                let l = *level.get_or_insert_with(|| crate::volume_mode::level_for(&vol, r, (hit - eye).length() / focal));
                let t = Instant::now();
                let rough = if along == Vec3::ZERO { n } else { along };
                if let Some((n, s)) = vol.plane_dab(l, hit, rough, r, r * 0.095, 0.2, 0.4, false) {
                    along = if along == Vec3::ZERO { n } else { (along * 0.5 + n * 0.5).normalize() };
                    changed.push(s.changed + s.carried);
                }
                dab_ms.push(ms(t));
                let t = Instant::now();
                let s = gpu.sync(&mut vol);
                sync_ms.push(s.cpu_ms);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                gpu_ms.push(ms(t));
                sent.push(s.uploaded);
                extracted.push(s.extracted);
            }
            let t = Instant::now();
            vol.end_stroke();
            let end_ms = ms(t);
            while gpu.pending() > 0 {
                gpu.sync(&mut vol);
            }
            gpu.sync(&mut vol);
            let med = |v: Vec<usize>| median(v.into_iter().map(|x| x as f64).collect());
            println!("| {r} | {} | {} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {:.0} | {end_ms:.0} |", level.unwrap_or(0), steps + 1, median(dab_ms), median(sync_ms), median(gpu_ms), med(changed), med(sent), med(extracted));
        }
    }
    Ok(())
}

fn bench(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    if std::env::var_os("SCULPT_BENCH_FAR").is_some() {
        return bench_far(device, queue);
    }
    // Level 0 at about 1M triangles' step; five more levels for detail.
    let voxel0 = 0.0035;
    let t = Instant::now();
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, voxel0, 6);
    vol.carry_budget = Some(6.0);
    println!("ball at level 0 step {voxel0} ({} blocks) in {:.0} ms", vol.keys().len(), ms(t));
    let mut gpu = GpuLevels::new(device, queue, &vol, 5);
    // Seen as the app sees it, from in front, choosing the level by distance (`SCULPT_BENCH_ALL` shows every level).
    gpu.set_eye(Vec3::new(0.0, 0.0, 2.6), 1100.0);
    gpu.view.on = std::env::var("SCULPT_BENCH_ALL").is_err();
    let t = Instant::now();
    let s = gpu.sync(&mut vol);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    println!("first sync: {} blocks uploaded and extracted, {:.1} MB, CPU {:.1} ms, total {:.1} ms", s.uploaded, s.bytes as f64 / 1e6, s.cpu_ms, ms(t));
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
            dab_ms.push(ms(t));
            let t = Instant::now();
            let s = gpu.sync(&mut vol);
            sync_ms.push(s.cpu_ms);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            gpu_ms.push(ms(t));
            most = (most.0.max(s.uploaded), most.1.max(s.extracted));
        }
        let t = Instant::now();
        let end = vol.end_stroke();
        let end_ms = ms(t);
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
    // which runs as jobs on a pool of its own (half the cores) while the frames go on; each frame brings in a job that
    // finished, starts the next, and syncs.
    {
        let threads = std::env::var("SCULPT_CARRY_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or((std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).max(2));
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let mut job: Option<std::sync::mpsc::Receiver<sculpt_core::generator::level_volume::CarryDone>> = None;
        let tick = |vol: &mut LevelVolume, job: &mut Option<std::sync::mpsc::Receiver<sculpt_core::generator::level_volume::CarryDone>>, wait: bool| -> (f64, usize) {
            let t = Instant::now();
            let mut n = 0;
            if let Some(rx) = job.as_ref() {
                let got = if wait { rx.recv().ok() } else { rx.try_recv().ok() };
                if let Some(done) = got {
                    n = done.blocks;
                    vol.carry_apply(done);
                    *job = None;
                }
            }
            if job.is_none()
                && let Some(j) = vol.carry_take(384)
            {
                let (tx, rx) = std::sync::mpsc::channel();
                pool.spawn(move || {
                    let _ = tx.send(j.run());
                });
                *job = Some(rx);
            }
            (ms(t), n)
        };
        let (mut frame_ms, mut dab_ms, mut tick_ms, mut sync_ms, mut most) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0);
        vol.carry_budget = Some(0.0);
        vol.keep_detail = true;
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
            dab_ms.push(ms(t));
            let (ms, n) = tick(&mut vol, &mut job, false);
            tick_ms.push(ms);
            most = most.max(n);
            let s = gpu.sync(&mut vol);
            sync_ms.push(s.cpu_ms);
            frame_ms.push(crate::ms(t));
            // The rest of a frame at 60 fps, when the worker carries on.
            let left = 16.7 - crate::ms(t);
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left / 1e3));
            }
        }
        // Pen up: frames go on (as above, at 60 fps) until the detail is all in, then the stroke ends.
        let t = Instant::now();
        let (mut frames, mut after_ms) = (0, Vec::new());
        while job.is_some() || vol.carry_busy() {
            let tf = Instant::now();
            tick(&mut vol, &mut job, false);
            gpu.sync(&mut vol);
            after_ms.push(ms(tf));
            frames += 1;
            let left = 16.7 - ms(tf);
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left / 1e3));
            }
        }
        let catch_up = ms(t);
        // Then the stroke's end, as a job on the shared threads, frames going on until it is in.
        let t = Instant::now();
        let take = Instant::now();
        let end_job = vol.end_stroke_take().expect("the stroke changed something");
        let take_ms = ms(take);
        let (tx, rx) = std::sync::mpsc::channel();
        rayon::spawn(move || {
            let _ = tx.send(end_job.run());
        });
        let (mut end_frames, mut end_frame_ms) = (0, Vec::new());
        let (end, apply_ms) = loop {
            let tf = Instant::now();
            if let Ok(done) = rx.try_recv() {
                let ta = Instant::now();
                let end = vol.end_stroke_apply(done);
                break (end, ms(ta));
            }
            gpu.sync(&mut vol);
            end_frame_ms.push(ms(tf));
            end_frames += 1;
            let left = 16.7 - ms(tf);
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left / 1e3));
            }
        };
        let end_wall = ms(t);
        let ts = Instant::now();
        gpu.sync(&mut vol);
        let end_sync = ms(ts);
        vol.carry_budget = Some(6.0);
        println!("a level 0 stroke (radius 0.3) over the fine work, two dabs a frame, carried on {threads} threads of its own: frame {:.1} ms on the main thread (median: dabs {:.1}, bringing in and starting jobs {:.1}, sync {:.1}; up to {most} finer blocks a job); after the pen lifts the detail was all in after {frames} frames ({catch_up:.0} ms, {:.1} ms a frame)", median(frame_ms), median(dab_ms), median(tick_ms), median(sync_ms), median(after_ms));
        println!("  then the stroke's end as a job: {:.0} ms of its own, in {end_frames} frames ({end_wall:.0} ms, {:.1} ms a frame); on the main thread {take_ms:.1} ms to start it, {apply_ms:.1} ms to take it in and {end_sync:.1} ms to sync what it changed", end.ms, median(end_frame_ms));
    }
    println!();
    println!("| level | step | brush radius (steps) | dab ms CPU | sync ms CPU | sync to GPU done ms | blocks uploaded / extracted (most) | stroke end ms | extracted after |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("{r}");
    }
    let (counts, bytes) = vol.memory();
    println!("blocks per level {counts:?}, {:.1} MB of samples", bytes as f64 / 1e6);
    while gpu.view.stats.stale > 0 || gpu.pending() > 0 {
        gpu.sync(&mut vol);
    }
    for eye in [Vec3::new(0.0, 0.0, 2.6), Vec3::new(0.0, 0.0, 1.15), Vec3::new(0.0, 0.0, 8.0), Vec3::new(3.0, 0.0, 0.0)] {
        gpu.set_eye(eye, 1100.0);
        let s = gpu.sync(&mut vol);
        let st = gpu.view.stats;
        println!("seen from {eye}: {} blocks shown, {} hidden, choosing {:.2} ms, {} shown or hidden, sync {:.1} ms CPU ({} uploaded, {} extracted)", st.shown, st.hidden, st.ms, st.flips, s.cpu_ms, s.uploaded, s.extracted);
    }
    Ok(())
}

// ---------------------------------------------------------------- mesh layer rows

/// A block's vertex count in its GPU vertex order, from its cells (a cell's word names its first vertex and its
/// pieces); the same as `BlockSurface::verts`, whose order is the GPU's (cells are compared equal in [`compare`]).
fn gpu_vertex_count(s: &BlockSurface) -> usize {
    s.cells.iter().map(|&c| if (c >> 10) & 7 == 0 { 0 } else { (c & 1023) as usize + ((c >> 10) & 7) as usize }).max().unwrap_or(0)
}

type Surfaces = sculpt_core::generator::level_volume::HashMap<Key, BlockSurface>;

/// The shown blocks with surface the GPU holds, each with its slot, and the CPU's surface of every shown block.
fn rows_blocks(vp: &mut crate::viewport::Viewport, vol: &LevelVolume) -> Res<(Vec<(Key, u32)>, Surfaces)> {
    let gpu = vp.levels().ok_or("no volume on the GPU")?;
    let show = gpu.show();
    let keys: Vec<Key> = vol.keys().into_iter().filter(|k| !show.hides(*k)).collect();
    let (surf, _) = vol.extract_in(&keys, &show);
    let mut out = Vec::new();
    for k in keys {
        let s = gpu.slot(k).ok_or_else(|| format!("block {k:?} has no slot"))?;
        let w = &surf[&k];
        if gpu_vertex_count(w) != w.verts.len() {
            return Err(format!("block {k:?}: the cells name {} vertices, the surface has {}", gpu_vertex_count(w), w.verts.len()).into());
        }
        if !w.verts.is_empty() {
            out.push((k, s));
        }
    }
    Ok((out, surf))
}

/// Sync until nothing is left to land.
fn rows_settle(device: &wgpu::Device, queue: &wgpu::Queue, vp: &mut crate::viewport::Viewport, mode: &mut crate::volume_mode::VolumeMode, eye: Vec3) {
    for _ in 0..200 {
        let more = vp.sync_volume_on(device, queue, mode, eye, 1000.0, Some(1000.0));
        if !more && mode.vol.dirty_keys().is_empty() {
            break;
        }
    }
}

fn rows_capture(device: &wgpu::Device, queue: &wgpu::Queue, vp: &mut crate::viewport::Viewport, blocks: &[(Key, u32)], surf: &Surfaces) -> Vec<(Vec3, Vec3)> {
    let slots: Vec<u32> = blocks.iter().map(|b| b.1).collect();
    let counts: Vec<u32> = blocks.iter().map(|b| surf[&b.0].verts.len() as u32).collect();
    vp.capture(device, queue, &slots, &counts)
}

fn same_bits(a: Vec3, b: Vec3) -> bool {
    a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits() && a.z.to_bits() == b.z.to_bits()
}

/// Pixels differing between two images, of those where `keep(x, y)`: (differing, kept).
fn pixels_differ(a: &[u8], b: &[u8], w: u32, keep: impl Fn(u32, u32) -> bool) -> (usize, usize) {
    let (mut n, mut of) = (0, 0);
    for (i, (p, q)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        if keep(x, y) {
            of += 1;
            n += (p != q) as usize;
        }
    }
    (n, of)
}

/// `sculpt-app --gpu-rows-check [--old-shader FILE] [--triangles M]`: mesh layer rows drawn on the GPU's surface.
/// 1. No rows, and rows all zero: the image and every drawn vertex bit-equal to none (and the image to one drawn with
///    the shader in FILE, the viewport's shader before rows, if given).
/// 2. Rows of half a step along the normal, with a change of normal, on the blocks right of x = 0: every drawn vertex
///    is where it was plus its row, exactly; none moves in the other half and no pixel there differs. After a stroke,
///    blocks the GPU extracted again lose their rows and the rest keep them.
/// 3. Timing at about 2.5M triangles (`--triangles`, millions shown): a dab's rows, a swap's, every row; memory.
pub fn rows_run(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}");
    let theme = crate::theme::ThemeLibrary::load(std::path::Path::new("no themes here")).themes[0].clone();
    let old = match args.iter().position(|a| a == "--old-shader").and_then(|i| args.get(i + 1)) {
        Some(f) => Some(std::fs::read_to_string(f)?.trim_start_matches('\u{feff}').to_string()),
        None => None,
    };
    let size = [640u32, 480];
    let cam = crate::camera::Camera { distance: 3.4, yaw: 0.0, pitch: 0.0, ..Default::default() };
    let eye = cam.eye();
    // A ball with strokes at levels 1 and 2, so blocks of three levels are drawn.
    let mut mode = crate::volume_mode::VolumeMode::ball(2e5);
    stroke(&mut mode.vol, 1, Vec3::new(-0.4, 0.1, 1.0), Vec3::new(0.4, 0.15, 1.0), 0.08, false, |v| { v.carry_pending(None); }, |_| {});
    stroke(&mut mode.vol, 2, Vec3::new(-0.2, -0.2, 1.0), Vec3::new(0.3, -0.15, 1.0), 0.03, false, |v| { v.carry_pending(None); }, |_| {});
    let mut vp = crate::viewport::Viewport::with_device(&device, include_str!("../viewport.wgsl"));
    vp.layers_shown = true;
    rows_settle(&device, &queue, &mut vp, &mut mode, eye);
    let (blocks, surf) = rows_blocks(&mut vp, &mode.vol)?;
    let verts: usize = blocks.iter().map(|b| surf[&b.0].verts.len()).sum();
    let levels: std::collections::BTreeSet<u8> = blocks.iter().map(|b| b.0.0).collect();
    println!("{} blocks with surface at levels {levels:?}, {verts} vertices; each block's GPU vertex order from its cells", blocks.len());

    // ---- 1. zero rows
    let none = vp.render_offscreen(&device, &queue, size, &cam, &theme);
    if let Some(src) = &old {
        let mut vp_old = crate::viewport::Viewport::with_device(&device, src);
        vp_old.layers_shown = true;
        rows_settle(&device, &queue, &mut vp_old, &mut mode, eye);
        let img = vp_old.render_offscreen(&device, &queue, size, &cam, &theme);
        let (d, of) = pixels_differ(&none, &img, size[0], |_, _| true);
        println!("gate 1: no rows against the shader before rows: {d} of {of} pixels differ");
        if d > 0 {
            return Err("gate 1: the image without rows differs from the shader before rows".into());
        }
    }
    let p0 = rows_capture(&device, &queue, &mut vp, &blocks, &surf);
    {
        let gpu = vp.levels().unwrap();
        for (k, _) in &blocks {
            gpu.set_block_rows(*k, &vec![super::Row::default(); surf[k].verts.len()]);
        }
    }
    let zero = vp.render_offscreen(&device, &queue, size, &cam, &theme);
    let pz = rows_capture(&device, &queue, &mut vp, &blocks, &surf);
    let (d, of) = pixels_differ(&none, &zero, size[0], |_, _| true);
    let moved = p0.iter().zip(&pz).filter(|(a, b)| !same_bits(a.0, b.0) || !same_bits(a.1, b.1)).count();
    println!("gate 1: zero rows: {d} of {of} pixels differ, {moved} of {} drawn vertices differ in any bit", p0.len());
    if d > 0 || moved > 0 {
        return Err("gate 1: zero rows change the image".into());
    }
    // The drawn vertices are the CPU's, in the same order.
    let mut worst = 0.0f32;
    let mut at = 0;
    for (k, _) in &blocks {
        let vx = mode.vol.voxel(k.0 as usize);
        for p in &surf[k].verts {
            worst = worst.max((p0[at].0 - *p).length() / vx);
            at += 1;
        }
    }
    println!("drawn vertices against the CPU's, in GPU vertex order: within {worst:.4} steps");
    if worst > 0.05 {
        return Err(format!("the drawn vertices are not the CPU's in order ({worst} steps off)").into());
    }

    // ---- 2. a known field on the blocks right of x = 0
    let dn = Vec3::new(0.3, 0.25, -0.2);
    let origin = mode.vol.origin();
    let voxels: Vec<f32> = (0..mode.vol.levels()).map(|l| mode.vol.voxel(l)).collect();
    let voxel = |l: u8| voxels[l as usize];
    let left_x = |k: &Key| origin.x + (k.1.x * 8) as f32 * voxel(k.0);
    let right = |k: &Key| left_x(k) >= 0.0;
    let mut want: Vec<Option<Vec3>> = Vec::with_capacity(p0.len());
    {
        let gpu = vp.levels().unwrap();
        gpu.clear_all_rows();
        for (k, _) in &blocks {
            let s = &surf[k];
            if right(k) {
                let rows: Vec<super::Row> = s.normals.iter().map(|n| super::Row::new(n.normalize_or_zero() * 0.5 * voxel(k.0), dn)).collect();
                want.extend(rows.iter().map(|r| Some(Vec3::from_array(r.offset))));
                gpu.set_block_rows(*k, &rows);
            } else {
                want.extend(std::iter::repeat_n(None, s.verts.len()));
            }
        }
    }
    let with = vp.render_offscreen(&device, &queue, size, &cam, &theme);
    let p1 = rows_capture(&device, &queue, &mut vp, &blocks, &surf);
    let dn_back = super::rows::unpack_normal_change(super::rows::pack_normal_change(dn));
    let (mut wrong, mut bent_wrong, mut moved_left, mut n_right) = (0, 0, 0, 0);
    for i in 0..p0.len() {
        match want[i] {
            Some(off) => {
                n_right += 1;
                wrong += !same_bits(p1[i].0, p0[i].0 + off) as usize;
                bent_wrong += ((p1[i].1 - (p0[i].1 + dn_back).normalize()).length() > 1e-5) as usize;
            }
            None => moved_left += (!same_bits(p1[i].0, p0[i].0) || !same_bits(p1[i].1, p0[i].1)) as usize,
        }
    }
    // Pixels by where the ball (radius about 1) is hit: left of x = -0.05, or right of the rows' edge.
    let edge = blocks.iter().filter(|b| right(&b.0)).map(|b| left_x(&b.0)).fold(f32::MAX, f32::min);
    let hit_x = |x: u32, y: u32| -> Option<f32> {
        let r = cam.ray(glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5), glam::Vec2::new(size[0] as f32, size[1] as f32));
        let b = r.origin.dot(r.dir);
        let c = r.origin.length_squared() - 1.0;
        let h = b * b - c;
        (h > 0.0).then(|| (r.origin + r.dir * (-b - h.sqrt())).x)
    };
    let (d_left, of_left) = pixels_differ(&none, &with, size[0], |x, y| hit_x(x, y).is_some_and(|hx| hx < -0.05));
    let (d_right, of_right) = pixels_differ(&none, &with, size[0], |x, y| hit_x(x, y).is_some_and(|hx| hx > edge + 0.05));
    println!("gate 2: {n_right} vertices with rows: {wrong} not at base + offset exactly, {bent_wrong} normals not bent as packed; {moved_left} of {} without rows moved", p0.len() - n_right);
    println!("gate 2: pixels: {d_left} of {of_left} differ in the half without rows, {d_right} of {of_right} in the half with rows");
    if wrong > 0 || bent_wrong > 0 || moved_left > 0 || d_left > 0 || d_right == 0 {
        return Err("gate 2: the rows are not drawn where they should be".into());
    }
    // A stroke on the right: blocks extracted again lose their rows, the rest keep theirs in place.
    vp.levels().unwrap().take_rows_dropped();
    let mark = vp.levels().unwrap().rows_mark();
    stroke(&mut mode.vol, 1, Vec3::new(0.3, 0.0, 1.0), Vec3::new(0.5, 0.05, 1.0), 0.08, false, |v| { v.carry_pending(None); }, |_| {});
    rows_settle(&device, &queue, &mut vp, &mut mode, eye);
    let dropped: std::collections::HashSet<Key> = vp.levels().unwrap().take_rows_dropped().into_iter().collect();
    let kept_blocks: Vec<(usize, Key, u32)> = {
        let gpu = vp.levels().unwrap();
        let mut at = 0;
        let mut out = Vec::new();
        for (k, _) in &blocks {
            if right(k) && !dropped.contains(k)
                && let Some(s) = gpu.slot(*k)
            {
                out.push((at, *k, s));
            }
            at += surf[k].verts.len();
        }
        out
    };
    let slots: Vec<u32> = kept_blocks.iter().map(|b| b.2).collect();
    let counts: Vec<u32> = kept_blocks.iter().map(|b| surf[&b.1].verts.len() as u32).collect();
    let p2 = vp.capture(&device, &queue, &slots, &counts);
    let (mut kept_wrong, mut j) = (0, 0);
    for (at, k, _) in &kept_blocks {
        for i in 0..surf[k].verts.len() {
            kept_wrong += !same_bits(p2[j].0, p1[at + i].0) as usize;
            j += 1;
        }
    }
    println!("gate 2: after a stroke, {} blocks lost their rows (extracted again); {} kept them, {kept_wrong} of their vertices moved", dropped.len(), kept_blocks.len());
    if dropped.is_empty() || kept_wrong > 0 {
        return Err("gate 2: rows after a stroke are wrong".into());
    }
    // Rows made from the surface before the stroke are refused for a block extracted since; rows made after are taken.
    {
        let gpu = vp.levels().unwrap();
        let k = *dropped.iter().next().unwrap();
        let rows = vec![super::Row::default(); surf.get(&k).map_or(1, |s| s.verts.len())];
        if gpu.set_block_rows_since(k, &rows, mark) || !gpu.set_block_rows_since(k, &rows, gpu.rows_mark()) || !gpu.has_rows() {
            return Err("gate 2: rows for a block extracted since their mark were not refused".into());
        }
        gpu.clear_block_rows(k);
    }

    // ---- 3. timing and memory at about 2.5M triangles
    let tri_m: f32 = args.iter().position(|a| a == "--triangles").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(2.5);
    // A fresh ball shows about 1.5 times what `ball` names over 4 (measured below).
    let mut mode = crate::volume_mode::VolumeMode::ball(tri_m * 4e6 / 1.5);
    let mut vp = crate::viewport::Viewport::with_device(&device, include_str!("../viewport.wgsl"));
    vp.layers_shown = true;
    rows_settle(&device, &queue, &mut vp, &mut mode, eye);
    let (blocks, surf) = rows_blocks(&mut vp, &mode.vol)?;
    let tris: usize = blocks.iter().map(|b| surf[&b.0].quads.len() * 2).sum();
    let verts: usize = blocks.iter().map(|b| surf[&b.0].verts.len()).sum();
    println!("timing: {} blocks, {tris} triangles, {verts} vertices", blocks.len());
    // The offsets and normal changes the rows carry, per block, as a layer would hold them.
    let offsets: Vec<Vec<(Vec3, Vec3)>> = {
        use rayon::prelude::*;
        blocks.par_iter().map(|(k, _)| surf[k].normals.iter().map(|n| (n.normalize_or_zero() * 0.001, Vec3::new(0.01, 0.0, 0.0))).collect()).collect()
    };
    let near = |c: Vec3, want: usize| -> Vec<usize> {
        let mut by: Vec<(f32, usize)> = blocks.iter().enumerate().map(|(i, (k, _))| ((surf[k].verts[0] - c).length(), i)).collect();
        by.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut n = 0;
        by.into_iter()
            .take_while(|(_, i)| {
                let go = n < want;
                n += surf[&blocks[*i].0].verts.len();
                go
            })
            .map(|(_, i)| i)
            .collect()
    };
    let time = |which: &[usize], par: bool, gpu: &mut GpuLevels| -> f64 {
        let t = Instant::now();
        if par {
            let list: Vec<(Key, usize)> = which.iter().map(|&i| (blocks[i].0, offsets[i].len())).collect();
            gpu.set_many_block_rows(&list, u64::MAX, |j, rows| {
                for (r, (o, n)) in rows.iter_mut().zip(&offsets[which[j]]) {
                    *r = super::Row::new(*o, *n);
                }
            });
        } else {
            let mut rows: Vec<super::Row> = Vec::new();
            for &i in which {
                rows.clear();
                rows.extend(offsets[i].iter().map(|(o, n)| super::Row::new(*o, *n)));
                gpu.set_block_rows(blocks[i].0, &rows);
            }
        }
        gpu.flush_rows();
        ms(t)
    };
    let all: Vec<usize> = (0..blocks.len()).collect();
    let dab = near(Vec3::new(0.0, 0.0, 1.0), 3000);
    let swap = near(Vec3::new(0.0, 0.0, 1.0), 60_000);
    let gpu = vp.levels().unwrap();
    // The first time every block is placed and the buffer made.
    let first = time(&all, true, gpu);
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let all_ms = median((0..7).map(|_| { let t = time(&all, true, gpu); let _ = device.poll(wgpu::PollType::wait_indefinitely()); t }).collect());
    let dab_v: usize = dab.iter().map(|&i| surf[&blocks[i].0].verts.len()).sum();
    let dab_ms = median((0..50).map(|j| { let t = time(&dab, false, gpu); if j % 10 == 9 { let _ = device.poll(wgpu::PollType::wait_indefinitely()); } t }).collect());
    let swap_v: usize = swap.iter().map(|&i| surf[&blocks[i].0].verts.len()).sum();
    let swap_ms = median((0..15).map(|_| { let t = time(&swap, false, gpu); let _ = device.poll(wgpu::PollType::wait_indefinitely()); t }).collect());
    let (bytes, used, held) = gpu.rows_memory();
    println!("gate 3: a dab's rows ({} blocks, {dab_v} vertices): {dab_ms:.3} ms (bar 0.5)", dab.len());
    println!("gate 3: a swap's rows ({} blocks, {swap_v} vertices): {swap_ms:.3} ms (bar 2)", swap.len());
    println!("gate 3: every row ({verts} vertices): {all_ms:.2} ms (bar 6); the first time, placing them: {first:.2} ms");
    println!("gate 3: row memory on the GPU {:.1} MB (bar 26): {used} rows used of {held} held", bytes as f64 / 1048576.0);
    // Drawn with every row set: no validation error.
    let _ = vp.render_offscreen(&device, &queue, size, &cam, &theme);
    if dab_ms > 0.5 || swap_ms > 2.0 || all_ms > 6.0 || bytes as f64 > 26.0 * 1048576.0 {
        return Err("gate 3: over a bar".into());
    }
    println!("mesh layer rows: every gate passes");
    Ok(())
}

/// Probe: the Size slider dragged from its default to the top with every other setting at its default, through the
/// GPU as the viewport has it, then let go and landed.
pub fn size_drag_probe(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}, max binding {}", device.limits().max_storage_buffer_binding_size);
    let tris: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(13.2);
    let mut mode = crate::volume_mode::VolumeMode::ball(tris);
    mode.add_generator();
    let mut gpu = GpuLevels::new(&device, &queue, &mode.vol, mode.vol.levels() - 1);
    mode.layer.as_mut().unwrap().block_budget = gpu.block_budget();
    println!("block budget {}", gpu.block_budget());
    rocks_frame(&mut gpu, &mut mode);
    let k = mode.layer.as_ref().unwrap().active();
    let mut spec = mode.layer.as_ref().unwrap().generator(k).unwrap().clone();
    let (a, b) = (spec.size.ln(), 0.5f32.ln());
    let mut worst = 0.0f64;
    let sweeps: usize = std::env::var("PROBE_SWEEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    for sweep in 0..sweeps {
        let (from, to) = if sweep % 2 == 0 { (a, b) } else { (b, a) };
        for i in 1..=40 {
            spec.size = (from + (to - from) * i as f32 / 40.0).exp();
            // Rocks somewhere else each frame: blocks come and go, so slots are freed and given to others.
            if std::env::var_os("PROBE_SEEDS").is_some() {
                spec.seed = 1 + (sweep * 40 + i) as u64;
            }
            let t = Instant::now();
            mode.set_generator(k, &spec, true);
            rocks_frame(&mut gpu, &mut mode);
            if std::mem::take(&mut gpu.join_refused) {
                mode.generator_note = Some("join refused".into());
            }
            let f = t.elapsed().as_secs_f64() * 1e3;
            worst = worst.max(f);
            let (cap, high, free) = gpu.join_pool();
            if i % 10 == 0 {
                println!("sweep {sweep} size {:.3}: {f:.0} ms, slots {} of {}, pool {high} used ({free} free) of {cap}, note {:?}", spec.size, gpu.slot_of.len(), gpu.buf.slots, mode.generator_note);
            }
            if high as usize > gpu.buf.slots as usize {
                return Err(format!("the join pool holds {high} places for {} slots", gpu.buf.slots).into());
            }
        }
        mode.end_generator_edit();
        let start = Instant::now();
        let (mut frames, mut over, mut worst_land) = (0usize, 0usize, 0.0f64);
        loop {
            let tf = Instant::now();
            let quiet = rocks_frame(&mut gpu, &mut mode);
            let f = tf.elapsed().as_secs_f64() * 1e3;
            frames += 1;
            worst_land = worst_land.max(f);
            if f > 50.0 {
                over += 1;
                if std::env::var_os("PROBE_LAND_LOG").is_some() {
                    println!("  landing frame {frames} at {:.2} s: {f:.0} ms, pending {}", start.elapsed().as_secs_f64(), gpu.pending());
                }
            }
            if quiet && !mode.busy() && mode.vol.dirty_keys().is_empty() {
                break;
            }
            if start.elapsed().as_secs() > 300 {
                return Err("never landed".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        println!("sweep {sweep} landed in {:.1} s: {frames} frames, {over} over 50 ms, worst {worst_land:.0} ms", start.elapsed().as_secs_f64());
    }
    println!("worst drag frame {worst:.0} ms");
    println!("end: {} blocks, slots {} of {}", mode.vol.keys().len(), gpu.slot_of.len(), gpu.buf.slots);
    Ok(())
}
