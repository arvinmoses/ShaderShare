//! SPIKE (measurement only): volume sculpting on the GPU.
//!
//! The same field, dabs and surface nets as `sculpt_core::generator::volume`, with the field and the surface kept on
//! the GPU: a dab updates the samples there and re-extracts the touched blocks' vertices and quads straight into GPU
//! buffers, so nothing is uploaded for drawing. The CPU only decides which blocks a dab touches (geometry, cheap) and
//! keeps the table of which block lives in which slot.
//!
//! `sculpt-app --gpu-volume-bench [millions: 1,5,20]` times it against the CPU version on the same strokes.

use std::collections::HashMap;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use glam::{IVec3, Vec3};
use sculpt_core::generator::volume::Volume;

const SHADER: &str = include_str!("volume.wgsl");
const PASSES: [&str; 5] = ["scatter", "fill", "dab", "vertices", "faces"];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin: [f32; 3],
    voxel: f32,
    centre: [f32; 3],
    radius: f32,
    dims: [i32; 3],
    band: f32,
    blend: f32,
    carve: u32,
    count: u32,
    _pad: u32,
}

/// What one dab cost on the GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDab {
    /// Time the GPU spent in the passes, from timestamps.
    pub gpu_ms: f64,
    /// From planning on the CPU to the GPU finishing (what the app would wait for, if it waited).
    pub wall_ms: f64,
    pub plan_ms: f64,
    pub dirty: usize,
    pub quads: u32,
    pub uploaded: usize,
}

struct Pass {
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

pub struct GpuVolume {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter: String,
    passes: Vec<Pass>,
    params: wgpu::Buffer,
    counter: wgpu::Buffer,
    counter_staging: wgpu::Buffer,
    // Work lists: blocks to fill, to dab, to re-extract; slot table changes.
    fill_work: wgpu::Buffer,
    dab_work: wgpu::Buffer,
    dirty_work: wgpu::Buffer,
    changes: wgpu::Buffer,
    ts: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, f32)>,
    // CPU-side bookkeeping.
    slot_of: HashMap<u32, u32>,
    free: Vec<u32>,
    signs: Vec<i8>,
    origin: Vec3,
    dims: IVec3,
    voxel: f32,
    band: f32,
    capacity: usize,
    work_capacity: usize,
}

fn device() -> Result<(wgpu::Device, wgpu::Queue, String), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;
    let info = adapter.get_info();
    let have = adapter.limits();
    let limits = wgpu::Limits {
        max_buffer_size: have.max_buffer_size,
        max_storage_buffer_binding_size: have.max_storage_buffer_binding_size,
        max_storage_buffers_per_shader_stage: have.max_storage_buffers_per_shader_stage.min(16),
        max_compute_workgroups_per_dimension: have.max_compute_workgroups_per_dimension,
        ..Default::default()
    };
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("gpu_volume"), required_features: features, required_limits: limits, ..Default::default() }))
        .map_err(|e| format!("request_device: {e}"))?;
    Ok((device, queue, format!("{} ({:?}), max binding {} MB", info.name, info.backend, have.max_storage_buffer_binding_size >> 20)))
}

impl GpuVolume {
    /// Upload `vol`'s field, with room for `growth` times as many stored blocks and `quads` quads of output.
    pub fn new(vol: &Volume, growth: f32, quads: usize) -> Result<GpuVolume, String> {
        let (device, queue, adapter) = device()?;
        let stored: Vec<(u32, &[f32; 512])> = vol.stored().collect();
        let capacity = ((stored.len() as f32 * growth) as usize).max(1024);
        let work_capacity = capacity * 3;
        let storage = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra, mapped_at_creation: false })
        };
        let samples = storage("samples", capacity as u64 * 512 * 4, wgpu::BufferUsages::empty());
        let (origin, dims) = vol.grid();
        let slots_buf = storage("slots", (dims.x * dims.y * dims.z) as u64 * 4, wgpu::BufferUsages::empty());
        let verts = storage("verts", capacity as u64 * 512 * 4, wgpu::BufferUsages::empty());
        let quads_buf = storage("quads", quads as u64 * 16, wgpu::BufferUsages::empty());
        let counter = storage("counter", 4, wgpu::BufferUsages::COPY_SRC);
        let counter_staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("counter staging"), size: 4, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let fill_work = storage("fill work", work_capacity as u64 * 16, wgpu::BufferUsages::empty());
        let dab_work = storage("dab work", work_capacity as u64 * 16, wgpu::BufferUsages::empty());
        let dirty_work = storage("dirty work", work_capacity as u64 * 16, wgpu::BufferUsages::empty());
        let changes = storage("changes", work_capacity as u64 * 8, wgpu::BufferUsages::empty());
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("params"), size: std::mem::size_of::<Params>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

        // Initial contents: every stored block in its own slot.
        let mut slot_of = HashMap::with_capacity(capacity);
        let mut flat = Vec::with_capacity(stored.len() * 512);
        for (slot, (b, s)) in stored.iter().enumerate() {
            slot_of.insert(*b, slot as u32);
            flat.extend_from_slice(&s[..]);
        }
        queue.write_buffer(&samples, 0, bytemuck::cast_slice(&flat));
        let signs = vol.signs().to_vec();
        let table: Vec<i32> = signs.iter().enumerate().map(|(b, &s)| match s {
            0 => slot_of[&(b as u32)] as i32,
            s if s < 0 => -2,
            _ => -1,
        }).collect();
        queue.write_buffer(&slots_buf, 0, bytemuck::cast_slice(&table));
        let free = (stored.len() as u32..capacity as u32).rev().collect();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("volume.wgsl"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let (ro, rw) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
        let entries = [entry(0, wgpu::BufferBindingType::Uniform), entry(1, rw), entry(2, rw), entry(3, ro), entry(4, rw), entry(5, rw), entry(6, rw), entry(7, ro)];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("volume bgl"), entries: &entries });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let group = |work: &wgpu::Buffer| {
            let bufs = [&params, &samples, &slots_buf, work, &verts, &quads_buf, &counter, &changes];
            let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &entries })
        };
        let passes = PASSES
            .iter()
            .map(|&name| {
                let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: Some(&layout), module: &module, entry_point: Some(name), compilation_options: Default::default(), cache: None });
                let work = match name {
                    "fill" => &fill_work,
                    "dab" => &dab_work,
                    _ => &dirty_work,
                };
                Pass { pipe, group: group(work) }
            })
            .collect();
        let ts = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("volume ts"), ty: wgpu::QueryType::Timestamp, count: 2 * PASSES.len() as u32 });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 * PASSES.len() as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 * PASSES.len() as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            (set, resolve, staging, queue.get_timestamp_period())
        });
        Ok(GpuVolume {
            device,
            queue,
            adapter,
            passes,
            params,
            counter,
            counter_staging,
            fill_work,
            dab_work,
            dirty_work,
            changes,
            ts,
            slot_of,
            free,
            signs,
            origin,
            dims,
            voxel: vol.voxel,
            band: vol.band,
            capacity,
            work_capacity,
        })
    }

    fn block_index(&self, c: IVec3) -> Option<u32> {
        (c.cmpge(IVec3::ZERO).all() && c.cmplt(self.dims).all()).then(|| ((c.z * self.dims.y + c.y) * self.dims.x + c.x) as u32)
    }

    fn coords(&self, b: u32) -> IVec3 {
        let d = self.dims;
        IVec3::new(b as i32 % d.x, (b as i32 / d.x) % d.y, b as i32 / (d.x * d.y))
    }

    fn params(&self, centre: Vec3, radius: f32, blend: f32, carve: bool, count: usize) -> Params {
        Params {
            origin: self.origin.to_array(),
            voxel: self.voxel,
            centre: centre.to_array(),
            radius,
            dims: self.dims.to_array(),
            band: self.band,
            blend,
            carve: carve as u32,
            count: count as u32,
            _pad: 0,
        }
    }

    /// Run the passes for one batch of work, wait, and return (GPU ms per pass summed, quads written).
    fn run(&self, jobs: &[(usize, usize, Params)]) -> (f64, u32) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.counter, 0, None);
        for &(pass, count, params) in jobs {
            if count == 0 {
                continue;
            }
            // Uniforms per pass: write a fresh buffer content through a staging copy so each pass sees its own.
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: std::mem::size_of::<Params>() as u64, usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: true });
            staging.slice(..).get_mapped_range_mut().expect("mapped").copy_from_slice(bytemuck::bytes_of(&params));
            staging.unmap();
            enc.copy_buffer_to_buffer(&staging, 0, &self.params, 0, std::mem::size_of::<Params>() as u64);
            let threads = if PASSES[pass] == "scatter" { count } else { count * 512 };
            let groups = threads.div_ceil(64) as u32;
            let x = groups.min(65535);
            let y = groups.div_ceil(x);
            let writes = self.ts.as_ref().map(|(set, ..)| wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(2 * pass as u32), end_of_pass_write_index: Some(2 * pass as u32 + 1) });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(PASSES[pass]), timestamp_writes: writes });
            cp.set_pipeline(&self.passes[pass].pipe);
            cp.set_bind_group(0, &self.passes[pass].group, &[]);
            cp.dispatch_workgroups(x, y, 1);
        }
        enc.copy_buffer_to_buffer(&self.counter, 0, &self.counter_staging, 0, 4);
        if let Some((set, resolve, staging, _)) = &self.ts {
            enc.resolve_query_set(set, 0..2 * PASSES.len() as u32, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, staging, 0, 16 * PASSES.len() as u64);
        }
        self.queue.submit([enc.finish()]);
        let read = |buf: &wgpu::Buffer, len: u64| -> Vec<u8> {
            let (tx, rx) = std::sync::mpsc::channel();
            buf.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map").expect("map");
            let out = buf.slice(..len).get_mapped_range().expect("mapped").to_vec();
            buf.unmap();
            out
        };
        let quads = u32::from_le_bytes(read(&self.counter_staging, 4)[..4].try_into().unwrap());
        let gpu_ms = match &self.ts {
            Some((_, _, staging, period)) => {
                let raw = read(staging, 16 * PASSES.len() as u64);
                let t: &[u64] = bytemuck::cast_slice(&raw);
                jobs.iter().filter(|j| j.1 > 0).map(|&(pass, ..)| t[2 * pass + 1].saturating_sub(t[2 * pass]) as f64 * *period as f64 * 1e-6).sum()
            }
            None => 0.0,
        };
        (gpu_ms, quads)
    }

    /// Extract every stored block (and the ones just below them).
    pub fn extract_all(&mut self) -> GpuDab {
        let t = Instant::now();
        let mut dirty: Vec<u32> = self.slot_of.keys().flat_map(|&b| self.lower(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        let work: Vec<[i32; 4]> = dirty.iter().map(|&b| self.work_item(b)).collect();
        assert!(work.len() <= self.work_capacity, "work list over capacity");
        self.queue.write_buffer(&self.dirty_work, 0, bytemuck::cast_slice(&work));
        let p = self.params(Vec3::ZERO, 1.0, 1.0, false, work.len());
        let (gpu_ms, quads) = self.run(&[(3, work.len(), p), (4, work.len(), p)]);
        GpuDab { gpu_ms, wall_ms: t.elapsed().as_secs_f64() * 1e3, plan_ms: 0.0, dirty: work.len(), quads, uploaded: work.len() * 16 }
    }

    fn lower(&self, b: u32) -> Vec<u32> {
        let c = self.coords(b);
        (0..8).filter_map(|k| self.block_index(c - IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1))).collect()
    }

    fn work_item(&self, b: u32) -> [i32; 4] {
        let c = self.coords(b);
        [c.x, c.y, c.z, self.slot_of.get(&b).map_or(-1, |&s| s as i32)]
    }

    /// One dab: plan on the CPU (which blocks, which slots), then fill, dab and re-extract on the GPU.
    pub fn dab(&mut self, geometry: &Volume, centre: Vec3, radius: f32, blend: f32, carve: bool) -> GpuDab {
        let t = Instant::now();
        let (touched, swallowed) = geometry.dab_shell(centre, radius, blend);
        let mut changes: Vec<[i32; 2]> = Vec::new();
        let mut fills: Vec<[i32; 4]> = Vec::new();
        let mut dabs: Vec<[i32; 4]> = Vec::new();
        let fill_sign: i8 = if carve { 1 } else { -1 };
        for &b in &swallowed {
            if let Some(slot) = self.slot_of.remove(&b) {
                self.free.push(slot);
            }
            self.signs[b as usize] = fill_sign;
            changes.push([b as i32, if carve { -1 } else { -2 }]);
        }
        for &b in &touched {
            let sign = self.signs[b as usize];
            if sign != 0 && ((sign > 0 && !carve) || (sign < 0 && carve)) {
                let slot = self.free.pop().expect("slot capacity");
                self.slot_of.insert(b, slot);
                self.signs[b as usize] = 0;
                changes.push([b as i32, slot as i32]);
                let c = self.coords(b);
                fills.push([sign as i32, 0, 0, slot as i32]);
                dabs.push([c.x, c.y, c.z, slot as i32]);
            } else if let Some(&slot) = self.slot_of.get(&b) {
                let c = self.coords(b);
                dabs.push([c.x, c.y, c.z, slot as i32]);
            }
        }
        let mut dirty: Vec<u32> = touched.iter().chain(&swallowed).flat_map(|&b| self.lower(b)).collect();
        dirty.sort_unstable();
        dirty.dedup();
        let work: Vec<[i32; 4]> = dirty.iter().map(|&b| self.work_item(b)).collect();
        assert!(work.len() <= self.work_capacity && changes.len() <= self.work_capacity, "work list over capacity");
        self.queue.write_buffer(&self.changes, 0, bytemuck::cast_slice(&changes));
        self.queue.write_buffer(&self.fill_work, 0, bytemuck::cast_slice(&fills));
        self.queue.write_buffer(&self.dab_work, 0, bytemuck::cast_slice(&dabs));
        self.queue.write_buffer(&self.dirty_work, 0, bytemuck::cast_slice(&work));
        let plan_ms = t.elapsed().as_secs_f64() * 1e3;
        let jobs = [
            (0, changes.len(), self.params(centre, radius, blend, carve, changes.len())),
            (1, fills.len(), self.params(centre, radius, blend, carve, fills.len())),
            (2, dabs.len(), self.params(centre, radius, blend, carve, dabs.len())),
            (3, work.len(), self.params(centre, radius, blend, carve, work.len())),
            (4, work.len(), self.params(centre, radius, blend, carve, work.len())),
        ];
        let (gpu_ms, quads) = self.run(&jobs);
        GpuDab { gpu_ms, wall_ms: t.elapsed().as_secs_f64() * 1e3, plan_ms, dirty: work.len(), quads, uploaded: changes.len() * 8 + (fills.len() + dabs.len() + work.len()) * 16 }
    }

    pub fn stored(&self) -> usize {
        self.slot_of.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `--gpu-volume-bench [millions]`
pub fn bench(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let millions: Vec<f32> = args.first().map_or(vec![1.0, 5.0, 20.0], |s| s.split(',').filter_map(|v| v.parse().ok()).collect());
    let mut rows = Vec::new();
    for m in millions {
        let voxel = (4.0 * std::f32::consts::PI / (m * 1e6 / 2.0)).sqrt();
        let mut cpu = Volume::from_field(&|p: Vec3| p.length() - 1.0, 1.4, voxel);
        let (_, _, cpu_quads) = cpu.extract_all();
        // Room to grow, within the 2 GB a single buffer may span (samples and vertex slots are 2 KB a block each).
        let growth = (2.5f32).min(2000.0 * 1024.0 * 1024.0 / (cpu.dense_blocks() as f32 * 2048.0));
        let mut gpu = GpuVolume::new(&cpu, growth, (cpu_quads as f32 * 1.5) as usize + 4_000_000)?;
        if rows.is_empty() {
            println!("adapter: {}", gpu.adapter);
        }
        let full = gpu.extract_all();
        println!("{m}M: {} blocks stored (capacity {}), full extraction GPU {:.1} ms (wall {:.1} ms), {} quads (CPU: {cpu_quads})", gpu.stored(), gpu.capacity(), full.gpu_ms, full.wall_ms, full.quads);
        for (label, radius) in [("small", 0.03f32), ("medium", 0.1), ("large", 0.25)] {
            for carve in [false, true] {
                let (mut g, mut w, mut c, mut plan) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let (mut most, mut upload) = (0, 0);
                for i in 0..24 {
                    let a = (if carve { 1.5 } else { 0.0 }) + i as f32 * radius * 0.25;
                    let dir = Vec3::new(a.sin() * 0.6, 1.0, a.cos() * 0.6).normalize();
                    let centre = dir * (1.0 - 0.5 * radius);
                    let d = gpu.dab(&cpu, centre, radius, 0.2 * radius, carve);
                    let s = cpu.dab(centre, radius, 0.2 * radius, carve);
                    g.push(d.gpu_ms);
                    w.push(d.wall_ms);
                    plan.push(d.plan_ms);
                    c.push(s.field_ms + s.extract_ms);
                    most = most.max(d.dirty);
                    upload = upload.max(d.uploaded);
                }
                let worst = |v: &[f64]| v.iter().cloned().fold(0.0, f64::max);
                rows.push(format!(
                    "| {m}M | {label} ({:.0} voxels) | {} | {:.2} / {:.2} | {:.2} / {:.2} | {:.2} | {:.2} / {:.2} | {most} | {:.0} KB |",
                    radius / voxel,
                    if carve { "carve" } else { "clay" },
                    median(g.clone()),
                    worst(&g),
                    median(w.clone()),
                    worst(&w),
                    median(plan),
                    median(c.clone()),
                    worst(&c),
                    upload as f64 / 1024.0
                ));
            }
        }
        // The two must agree: extract everything again on both and compare quad counts.
        let after_gpu = gpu.extract_all().quads;
        let (_, _, after_cpu) = cpu.extract_all();
        println!("{m}M after the strokes: GPU {after_gpu} quads, CPU {after_cpu}");
    }
    println!();
    println!("| triangles | brush | dab | GPU ms (median / worst) | wall ms incl. CPU planning and wait (median / worst) | planning ms | CPU version ms (median / worst) | blocks re-extracted (most) | uploaded (most) |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        println!("{r}");
    }
    Ok(())
}
