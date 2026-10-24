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
/// A stroke from `a` to `b`, `each` run after every dab and `last` before the stroke ends.
#[allow(clippy::too_many_arguments)]
fn stroke(v: &mut LevelVolume, l: usize, a: Vec3, b: Vec3, r: f32, trim: bool, mut each: impl FnMut(&mut LevelVolume), last: impl FnOnce(&mut LevelVolume)) {
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
    last(v);
    v.end_stroke();
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
            let q = verts[(s * VSTRIDE as usize + i) * 2];
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
    use sculpt_core::generator::detail_layer::Layered;
    use sculpt_core::generator::layer::GeneratorSpec;
    use sculpt_core::generator::{BoolOp, Primitive};
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.3, 0.03, 4);
    let mut layer = Layered::new(&mut vol, 2);
    let rocks = GeneratorSpec { primitive: Primitive::Sphere, op: BoolOp::Add, count: 12, seed: 3, size: 0.12, lift: 0.2, min_spacing: 0.4, blend: 0.3, rough: 0.2, stretch_var: 0.3, tilt: 20.0, ..GeneratorSpec::default() };
    let k = layer.add_generator(&mut vol, "Rocks", 2, &rocks);
    // Its mask painted away over the front, half way at its edge: the GPU fades the join as the CPU does.
    layer.set_active(k);
    layer.paint_mask(&mut vol, Vec3::new(0.0, 0.0, 1.0), 0.6, 0.0);
    let mut gpu = GpuLevels::new(device, queue, &vol, 3);
    gpu.sync(&mut vol);
    for (step, spec) in [GeneratorSpec { size: 0.15, ..rocks.clone() }, GeneratorSpec { blend: 0.6, ..rocks.clone() }, GeneratorSpec { op: BoolOp::Subtract, lift: 0.0, ..rocks.clone() }, GeneratorSpec { count: 20, ..rocks.clone() }].into_iter().enumerate() {
        let t = Instant::now();
        let plan = layer.gpu_generator(&mut vol, k, &spec).ok_or("the GPU cannot join this stack")?;
        let plan_ms = t.elapsed().as_secs_f64() * 1e3;
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
        while let Some(job) = layer.generator_take() {
            layer.generator_apply(&mut vol, job.run());
        }
        let job_ms = tj.elapsed().as_secs_f64() * 1e3;
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
        gpu.sync(&mut vol);
    }
    Ok(())
}

pub fn run(args: &[String]) -> Res<()> {
    let (device, queue, adapter) = device()?;
    println!("adapter: {adapter}");
    let off = surface_noise_check(&device, &queue)?;
    let gen_off = generator_check(&device, &queue)?;
    join_check(&device, &queue)?;
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
            let mut vol = LevelVolume::read_from(&mut std::io::BufReader::new(std::fs::File::open(&f)?), crate::volume_mode::LEVELS)?;
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
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("{pass}fresh ball"))?;
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
            let held: std::cell::RefCell<Option<sculpt_core::generator::level_volume::CarryJob>> = std::cell::RefCell::new(None);
            let bring_in = |v: &mut LevelVolume| {
                if let Some(job) = held.borrow_mut().take() {
                    v.carry_apply(job.run());
                }
            };
            stroke(&mut vol, *l, *a, *b, *r, *trim, |v| {
                if jobs {
                    bring_in(v);
                    *held.borrow_mut() = v.carry_take(64);
                } else {
                    v.carry_pending(Some(0.5));
                }
                let s = gpu.sync(v);
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
            gpu.sync(&mut vol);
            compare(&device, &queue, &gpu, &vol, &format!("{pass}stroke {} (level {l}, {})", i + 1, if *trim { "trim" } else { "clay" }))?;
        }
        for i in 0..4 {
            vol.undo();
            gpu.sync(&mut vol);
            compare(&device, &queue, &gpu, &vol, &format!("{pass}undo {}", i + 1))?;
        }
        vol.redo();
        gpu.sync(&mut vol);
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
            gpu.sync(&mut vol);
            while gpu.view.stats.stale > 0 {
                gpu.sync(&mut vol);
            }
            let st = gpu.view.stats;
            compare(&device, &queue, &gpu, &vol, &format!("{pass}seen from {eye} ({} blocks shown, {} hidden)", st.shown, st.hidden))?;
        }
        // A tight triangle budget, near: only the detail that shows most.
        gpu.view.budget = Some(gpu.view.stats.triangles * 9 / 10);
        gpu.set_eye(Vec3::new(0.0, 0.0, 1.3), 1000.0);
        gpu.sync(&mut vol);
        let st = gpu.view.stats;
        compare(&device, &queue, &gpu, &vol, &format!("{pass}within a budget ({} triangles, detail down to {:.1} px)", st.triangles, st.tau_used))?;
        gpu.view.budget = None;
        gpu.view.on = false;
        gpu.sync(&mut vol);
        compare(&device, &queue, &gpu, &vol, &format!("{pass}every level again"))?;
    }
    println!("the GPU surface matches the reference at every step");
    Ok(())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// At the app's scale: a generator's edits joined on the GPU, timed (planning, joining, the GPU's work) against the
/// CPU's job (`--bench-join`).
fn bench_join(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    use sculpt_core::generator::detail_layer::Layered;
    use sculpt_core::generator::layer::GeneratorSpec;
    use sculpt_core::generator::{BoolOp, Primitive};
    // Fewer rocks at level 3: twelve would take more slots than a 2 GB vertex buffer holds.
    // `SCULPT_BENCH_LEVEL=2` for the one level.
    let only: Option<usize> = std::env::var("SCULPT_BENCH_LEVEL").ok().and_then(|v| v.parse().ok());
    for (level, count) in [(2usize, 12u32), (3, 4)].into_iter().filter(|(l, _)| only.is_none_or(|o| o == *l)) {
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
            let plan_ms = t.elapsed().as_secs_f64() * 1e3;
            let n = plan.blocks.len();
            let ts = Instant::now();
            gpu.sync_for_join(&mut vol, &plan);
            let sync_ms = ts.elapsed().as_secs_f64() * 1e3;
            let s = gpu.join(plan, &|k| layer.base_samples(k));
            let tg = Instant::now();
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let gpu_ms = tg.elapsed().as_secs_f64() * 1e3;
            let tj = Instant::now();
            let mut apply_ms = 0.0;
            while let Some(job) = layer.generator_take() {
                let done = job.run();
                let ta = Instant::now();
                layer.generator_apply(&mut vol, done);
                apply_ms += ta.elapsed().as_secs_f64() * 1e3;
            }
            let job_ms = tj.elapsed().as_secs_f64() * 1e3;
            // The exact field landing: the sync after the job, as the app's next frame does it.
            let tl = Instant::now();
            let land = gpu.sync(&mut vol);
            let land_cpu = tl.elapsed().as_secs_f64() * 1e3;
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let land_ms = tl.elapsed().as_secs_f64() * 1e3;
            rows.push(format!(
                "{n} blocks: plan {plan_ms:.1}, sync {sync_ms:.1}, join {:.1} ms CPU, then the GPU {gpu_ms:.1} ms; the job {job_ms:.0} ms ({apply_ms:.1} taking it in); landing it: sync {land_cpu:.1} ms CPU ({} sent, {} extracted), GPU done at {land_ms:.1} ms",
                s.cpu_ms, land.uploaded, land.extracted
            ));
        }
        println!("rocks at level {level} (blocks per level {:?}):", vol.memory().0);
        for r in rows {
            println!("  {r}");
        }
    }
    Ok(())
}

fn bench(device: &wgpu::Device, queue: &wgpu::Queue) -> Res<()> {
    // Level 0 at about 1M triangles' step; five more levels for detail.
    let voxel0 = 0.0035;
    let t = Instant::now();
    let mut vol = LevelVolume::from_field(&|p: Vec3| p.length() - 1.0, 1.2, voxel0, 6);
    vol.carry_budget = Some(6.0);
    println!("ball at level 0 step {voxel0} ({} blocks) in {:.0} ms", vol.keys().len(), t.elapsed().as_secs_f64() * 1e3);
    let mut gpu = GpuLevels::new(device, queue, &vol, 5);
    // Seen as the app sees it, from in front, choosing the level by distance (`SCULPT_BENCH_ALL` shows every level).
    gpu.set_eye(Vec3::new(0.0, 0.0, 2.6), 1100.0);
    gpu.view.on = std::env::var("SCULPT_BENCH_ALL").is_err();
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
            (t.elapsed().as_secs_f64() * 1e3, n)
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
            dab_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let (ms, n) = tick(&mut vol, &mut job, false);
            tick_ms.push(ms);
            most = most.max(n);
            let s = gpu.sync(&mut vol);
            sync_ms.push(s.cpu_ms);
            frame_ms.push(t.elapsed().as_secs_f64() * 1e3);
            // The rest of a frame at 60 fps, when the worker carries on.
            let left = 16.7 - t.elapsed().as_secs_f64() * 1e3;
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
            after_ms.push(tf.elapsed().as_secs_f64() * 1e3);
            frames += 1;
            let left = 16.7 - tf.elapsed().as_secs_f64() * 1e3;
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left / 1e3));
            }
        }
        let catch_up = t.elapsed().as_secs_f64() * 1e3;
        // Then the stroke's end, as a job on the shared threads, frames going on until it is in.
        let t = Instant::now();
        let take = Instant::now();
        let end_job = vol.end_stroke_take().expect("the stroke changed something");
        let take_ms = take.elapsed().as_secs_f64() * 1e3;
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
                break (end, ta.elapsed().as_secs_f64() * 1e3);
            }
            gpu.sync(&mut vol);
            end_frame_ms.push(tf.elapsed().as_secs_f64() * 1e3);
            end_frames += 1;
            let left = 16.7 - tf.elapsed().as_secs_f64() * 1e3;
            if left > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(left / 1e3));
            }
        };
        let end_wall = t.elapsed().as_secs_f64() * 1e3;
        let ts = Instant::now();
        gpu.sync(&mut vol);
        let end_sync = ts.elapsed().as_secs_f64() * 1e3;
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
    while gpu.view.stats.stale > 0 {
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
