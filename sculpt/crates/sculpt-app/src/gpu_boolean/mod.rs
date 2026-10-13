//! GPU displacement boolean (Generator Boolean layer, spike 1).
//!
//! A wgpu compute port of `sculpt_core::generator` spike A: analytic-SDF displacement boolean
//! (subtract / add / intersect) with one thread per vertex. It only reads the mesh and writes a
//! separate per-vertex `delta` buffer; the CPU path and the sculpt latency path are untouched.
//!
//! Data flow per evaluation (one command buffer, three compute passes):
//!
//! 1. Instances (`Instance` -> 128 B baked struct) are baked on the CPU (rayon), uploaded with
//!    `queue.write_buffer`, and a dense uniform grid is built on the GPU as per-cell linked lists
//!    (`clear_heads` + `build_grid`: one atomic exchange per instance/cell pair).
//! 2. `mark_leaves`: one workgroup per BVH leaf checks whether the leaf bounds touch an occupied
//!    cell and appends the leaf to an active list; its count is the indirect dispatch's x.
//! 3. `eval`: indirect dispatch `(active leaves, chunks per leaf)`, one thread per owned vertex.
//!
//! [`headless`] creates a window-less device; [`bench`] is the benchmark / PNG / CPU comparison.

pub mod bench;

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use rayon::prelude::*;
use sculpt_core::bvh::Bvh;
use sculpt_core::generator::{BoolOp, EvalParams, Instance, Primitive};
use sculpt_core::geom::Aabb;
use wgpu::util::DeviceExt;

/// Upper bound on dense grid cells (4 bytes each). The cell size grows until the grid fits.
const MAX_CELLS: u64 = 1 << 24;
const WG: u32 = 64;
/// Row stride of 2D-linearised dispatches (`4096 * 64`), matches `LIN` in the shader.
const LIN_X: u32 = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInst {
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
    sc: [f32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
    lo: [u32; 4],
    hi: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLeaf {
    a: [u32; 4],
    bmin: [f32; 4],
    bmax: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuParams {
    origin: [f32; 4],
    dims: [u32; 4],
    cfg: [u32; 4],
    misc: [f32; 4],
    pool: [u32; 4],
}

/// CPU-side cost of [`GpuBoolean::set_instances`].
#[derive(Clone, Copy, Debug, Default)]
pub struct UploadStats {
    /// Bake transforms + AABBs + grid sizing + per-instance cell ranges (rayon).
    pub bake_ms: f64,
    /// `write_buffer` plus a submit/wait so the transfer is complete.
    pub upload_ms: f64,
    pub bytes: usize,
    pub cells: u64,
    pub cell_size: f32,
    pub entries: u64,
}

/// Timings of one [`GpuBoolean::evaluate`]. The `gpu_*` fields need TIMESTAMP_QUERY.
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchTimes {
    /// Submit to fence-wait return, wall clock.
    pub wall_ms: f64,
    /// `clear_heads` + `build_grid`.
    pub gpu_grid_ms: Option<f64>,
    /// `mark_leaves`.
    pub gpu_cull_ms: Option<f64>,
    /// `clear_delta` + `eval`.
    pub gpu_eval_ms: Option<f64>,
}

/// Result counters read back from the GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuStats {
    pub active_leaves: u32,
    pub entries_used: u32,
    pub entries_capacity: u32,
}

struct Timestamps {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    period_ns: f32,
}

struct Pipelines {
    clear_heads: wgpu::ComputePipeline,
    build_grid: wgpu::ComputePipeline,
    mark_leaves: wgpu::ComputePipeline,
    clear_delta: wgpu::ComputePipeline,
    eval: wgpu::ComputePipeline,
}

pub struct GpuBoolean {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout_build: wgpu::BindGroupLayout,
    layout_eval: wgpu::BindGroupLayout,
    pipes: Pipelines,
    params_buf: wgpu::Buffer,
    args: wgpu::Buffer,
    args_staging: wgpu::Buffer,
    ts: Option<Timestamps>,

    mesh: Option<wgpu::Buffer>,
    delta: Option<wgpu::Buffer>,
    delta_staging: Option<wgpu::Buffer>,
    leaves: Option<wgpu::Buffer>,
    work: Option<wgpu::Buffer>,
    n_verts: u32,
    n_leaves: u32,
    chunk_y: u32,

    insts: Option<(wgpu::Buffer, usize)>,
    heads: Option<(wgpu::Buffer, u64)>,
    entries: Option<(wgpu::Buffer, u64)>,
    grid: GpuParams,
    n_inst: u32,
    entry_cap: u32,
    margin: f32,

    groups: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl GpuBoolean {
    /// Build the pipelines on an existing device (the app's, or a [`headless`] one).
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> GpuBoolean {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_boolean"),
            source: wgpu::ShaderSource::Wgsl(include_str!("boolean.wgsl").into()),
        });
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        // 1 insts(ro) 2 heads 3 entries 4 mesh(ro) 5 delta 6 leaves(ro) 7 work 8 args
        let mut build = vec![uniform, storage_entry(1, true), storage_entry(2, false), storage_entry(3, false), storage_entry(4, true)];
        build.extend([storage_entry(5, false), storage_entry(6, true), storage_entry(7, false), storage_entry(8, false)]);
        let layout_build = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gb build"), entries: &build });
        // The eval pass dispatches indirectly from `args`, so it must not bind it as storage.
        let layout_eval = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gb eval"), entries: &build[..8] });
        let pl_build = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout_build)], immediate_size: 0 });
        let pl_eval = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout_eval)], immediate_size: 0 });
        let mk = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = Pipelines {
            clear_heads: mk("clear_heads", &pl_build),
            build_grid: mk("build_grid", &pl_build),
            mark_leaves: mk("mark_leaves", &pl_build),
            clear_delta: mk("clear_delta", &pl_eval),
            eval: mk("eval", &pl_eval),
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb args"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let args_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb args staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ts = device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| Timestamps {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("gb ts"), ty: wgpu::QueryType::Timestamp, count: 6 }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gb ts resolve"),
                size: 48,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gb ts staging"),
                size: 48,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: queue.get_timestamp_period(),
        });
        GpuBoolean {
            device,
            queue,
            layout_build,
            layout_eval,
            pipes,
            params_buf,
            args,
            args_staging,
            ts,
            mesh: None,
            delta: None,
            delta_staging: None,
            leaves: None,
            work: None,
            n_verts: 0,
            n_leaves: 0,
            chunk_y: 1,
            insts: None,
            heads: None,
            entries: None,
            grid: GpuParams::default(),
            n_inst: 0,
            entry_cap: 0,
            margin: 0.0,
            groups: None,
        }
    }

    pub fn has_timestamps(&self) -> bool {
        self.ts.is_some()
    }

    /// Upload positions + normals (interleaved) and the per-leaf vertex ranges / bounds.
    /// One-time per topology; not on the slider path.
    pub fn upload_mesh(&mut self, positions: &[Vec3], normals: &[Vec3], bvh: &Bvh) {
        assert_eq!(positions.len(), normals.len());
        // 16 B per vertex: position + octahedral-packed normal. Keeps 6.3M vertices under the
        // 128 MiB default storage-binding limit (a 24 B layout would not fit) and cuts bandwidth.
        let inter: Vec<[u32; 4]> = positions.par_iter().zip(normals.par_iter()).map(|(p, n)| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits(), oct_encode(*n)]).collect();
        let leaves: Vec<GpuLeaf> = bvh
            .leaves
            .iter()
            .map(|l| {
                let r: Range<usize> = l.owned_range();
                GpuLeaf { a: [r.start as u32, r.len() as u32, 0, 0], bmin: l.bounds.min.extend(0.0).to_array(), bmax: l.bounds.max.extend(0.0).to_array() }
            })
            .collect();
        let max_len = leaves.iter().map(|l| l.a[1]).max().unwrap_or(1).max(1);
        self.chunk_y = max_len.div_ceil(WG);
        self.n_verts = positions.len() as u32;
        self.n_leaves = leaves.len() as u32;
        let d = &self.device;
        self.mesh = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gb mesh"), contents: bytemuck::cast_slice(&inter), usage: wgpu::BufferUsages::STORAGE }));
        self.leaves = Some(d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("gb leaves"), contents: bytemuck::cast_slice(&leaves), usage: wgpu::BufferUsages::STORAGE }));
        let dsize = (positions.len() * 12).max(4) as u64;
        self.delta = Some(d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb delta"),
            size: dsize,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.delta_staging = Some(d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb delta staging"),
            size: dsize,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.work = Some(d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb work"),
            size: (leaves.len() * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.groups = None;
        let _ = self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Bake `instances` (AABBs inflated by `margin`, the boolean's reach for Add / Intersect),
    /// size the grid, and upload. This is the cost paid on every slider tick.
    pub fn set_instances(&mut self, instances: &[Instance], margin: f32) -> UploadStats {
        let t0 = std::time::Instant::now();
        let n = instances.len();
        // 1. Bake + inflated AABB.
        let mut baked: Vec<GpuInst> = instances.par_iter().map(|i| bake(i, margin)).collect();
        // 2. Grid parameters (same rule as the CPU hash: cell = mean instance extent).
        let (bounds, ext_sum) = baked
            .par_iter()
            .fold(
                || (Aabb::EMPTY, 0.0f64),
                |(b, s), g| {
                    let a = Aabb { min: Vec3::from_slice(&g.bmin[..3]), max: Vec3::from_slice(&g.bmax[..3]) };
                    (b.union(&a), s + a.extent().max_element() as f64)
                },
            )
            .reduce(|| (Aabb::EMPTY, 0.0), |a, b| (a.0.union(&b.0), a.1 + b.1));
        let mut stats = UploadStats::default();
        if n == 0 {
            self.n_inst = 0;
            self.entry_cap = 0;
            self.grid = GpuParams::default();
            return stats;
        }
        let ext = bounds.extent();
        let mean_ext = (ext_sum / n as f64) as f32;
        let vol = (ext.x.max(1e-9) * ext.y.max(1e-9) * ext.z.max(1e-9)) as f64;
        let mut cell = mean_ext.max((vol / MAX_CELLS as f64).cbrt() as f32).max(1e-9);
        let dims = loop {
            let inv = 1.0 / cell;
            let d = [(ext.x * inv).floor() as u64 + 1, (ext.y * inv).floor() as u64 + 1, (ext.z * inv).floor() as u64 + 1];
            if d[0] * d[1] * d[2] <= MAX_CELLS {
                break d;
            }
            cell *= 1.1;
        };
        let inv = 1.0 / cell;
        let origin = bounds.min;
        // 3. Per-instance cell ranges (shared by the entry count and the GPU build, so they agree).
        let entries: u64 = baked
            .par_iter_mut()
            .map(|g| {
                let f = |v: [f32; 4], sat: [u64; 3]| {
                    let c = (Vec3::from_slice(&v[..3]) - origin) * inv;
                    [(c.x.floor().max(0.0) as u64).min(sat[0] - 1) as u32, (c.y.floor().max(0.0) as u64).min(sat[1] - 1) as u32, (c.z.floor().max(0.0) as u64).min(sat[2] - 1) as u32]
                };
                let (lo, hi) = (f(g.bmin, dims), f(g.bmax, dims));
                g.lo[..3].copy_from_slice(&lo);
                g.hi[..3].copy_from_slice(&hi);
                (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * (hi[2] - lo[2] + 1) as u64
            })
            .sum();
        let bake_ms = t0.elapsed().as_secs_f64() * 1e3;

        // 4. Upload (buffers grow geometrically and are reused between slider ticks).
        let t1 = std::time::Instant::now();
        let bytes: &[u8] = bytemuck::cast_slice(&baked);
        let mut regroup = false;
        if self.insts.as_ref().is_none_or(|(_, cap)| *cap < n) {
            let cap = n.next_power_of_two().max(1024);
            self.insts = Some((
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gb insts"),
                    size: (cap * std::mem::size_of::<GpuInst>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cap,
            ));
            regroup = true;
        }
        let ncells = dims[0] * dims[1] * dims[2];
        if self.heads.as_ref().is_none_or(|(_, cap)| *cap < ncells) {
            self.heads = Some((
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gb heads"),
                    size: ncells * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                ncells,
            ));
            regroup = true;
        }
        if self.entries.as_ref().is_none_or(|(_, cap)| *cap < entries) {
            let cap = entries.next_power_of_two().max(1024);
            self.entries = Some((
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gb entries"),
                    size: cap * 8,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }),
                cap,
            ));
            regroup = true;
        }
        if regroup {
            self.groups = None;
        }
        self.queue.write_buffer(&self.insts.as_ref().unwrap().0, 0, bytes);
        let _ = self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let upload_ms = t1.elapsed().as_secs_f64() * 1e3;

        self.n_inst = n as u32;
        self.entry_cap = entries as u32;
        self.margin = margin;
        self.grid = GpuParams {
            origin: [origin.x, origin.y, origin.z, inv],
            dims: [dims[0] as u32, dims[1] as u32, dims[2] as u32, n as u32],
            pool: [entries as u32, self.chunk_y, ncells as u32, 0],
            ..Default::default()
        };
        stats.bake_ms = bake_ms;
        stats.upload_ms = upload_ms;
        stats.bytes = bytes.len();
        stats.cells = ncells;
        stats.cell_size = cell;
        stats.entries = entries;
        stats
    }

    fn bind_groups(&mut self) {
        if self.groups.is_some() {
            return;
        }
        let (insts, heads, entries) = (&self.insts.as_ref().unwrap().0, &self.heads.as_ref().unwrap().0, &self.entries.as_ref().unwrap().0);
        fn b(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
        }
        let common = [
            b(0, &self.params_buf),
            b(1, insts),
            b(2, heads),
            b(3, entries),
            b(4, self.mesh.as_ref().unwrap()),
            b(5, self.delta.as_ref().unwrap()),
            b(6, self.leaves.as_ref().unwrap()),
            b(7, self.work.as_ref().unwrap()),
        ];
        let eval = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gb eval"), layout: &self.layout_eval, entries: &common });
        let mut full = common.to_vec();
        full.push(b(8, &self.args));
        let build = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gb build"), layout: &self.layout_build, entries: &full });
        self.groups = Some((build, eval));
    }

    /// Run grid build + leaf culling + the per-vertex boolean and wait for completion.
    /// Instances must have been set with `margin >= params.reach` for Add / Intersect.
    pub fn evaluate(&mut self, params: &EvalParams) -> DispatchTimes {
        assert!(self.mesh.is_some(), "upload_mesh first");
        if self.n_inst == 0 {
            return DispatchTimes::default();
        }
        self.bind_groups();
        let op = match params.op {
            BoolOp::Add => 0,
            BoolOp::Subtract => 1,
            BoolOp::Intersect => 2,
        };
        let mut g = self.grid;
        g.cfg = [op, params.iterations, self.n_leaves, self.n_verts];
        g.misc = [params.reach, 0.0, 0.0, 0.0];
        g.pool[1] = self.chunk_y;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&g));
        self.queue.write_buffer(&self.args, 0, bytemuck::cast_slice(&[0u32, self.chunk_y, 1, 0]));

        let (build_bg, eval_bg) = self.groups.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gb") });
        let tsw = |i: u32| {
            self.ts.as_ref().map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(2 * i), end_of_pass_write_index: Some(2 * i + 1) })
        };
        let grid_wgs = g.pool[2].div_ceil(WG);
        let (gx, gy) = (grid_wgs.min(LIN_X), grid_wgs.div_ceil(LIN_X).max(1));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("grid"), timestamp_writes: tsw(0) });
            pass.set_bind_group(0, build_bg, &[]);
            pass.set_pipeline(&self.pipes.clear_heads);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.pipes.build_grid);
            pass.dispatch_workgroups(self.n_inst.div_ceil(WG), 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cull"), timestamp_writes: tsw(1) });
            pass.set_bind_group(0, build_bg, &[]);
            pass.set_pipeline(&self.pipes.mark_leaves);
            pass.dispatch_workgroups(self.n_leaves, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("eval"), timestamp_writes: tsw(2) });
            pass.set_bind_group(0, eval_bg, &[]);
            let dwgs = (self.n_verts * 3).div_ceil(WG);
            pass.set_pipeline(&self.pipes.clear_delta);
            pass.dispatch_workgroups(dwgs.min(LIN_X), dwgs.div_ceil(LIN_X).max(1), 1);
            pass.set_pipeline(&self.pipes.eval);
            pass.dispatch_workgroups_indirect(&self.args, 0);
        }
        if let Some(t) = &self.ts {
            enc.resolve_query_set(&t.set, 0..6, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.staging, 0, 48);
        }
        let t0 = std::time::Instant::now();
        self.queue.submit([enc.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut out = DispatchTimes { wall_ms, ..Default::default() };
        if let Some(t) = &self.ts {
            let raw = read_buffer(&self.device, &t.staging, 48);
            let v: &[u64] = bytemuck::cast_slice(&raw);
            let ms = |a: usize, b: usize| v[b].saturating_sub(v[a]) as f64 * t.period_ns as f64 * 1e-6;
            out.gpu_grid_ms = Some(ms(0, 1));
            out.gpu_cull_ms = Some(ms(2, 3));
            out.gpu_eval_ms = Some(ms(4, 5));
        }
        out
    }

    /// Active leaf count and grid entry usage of the last [`evaluate`](Self::evaluate).
    pub fn stats(&self) -> GpuStats {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.args, 0, &self.args_staging, 0, 16);
        self.queue.submit([enc.finish()]);
        let raw = read_buffer(&self.device, &self.args_staging, 16);
        let v: &[u32] = bytemuck::cast_slice(&raw);
        GpuStats { active_leaves: v[0], entries_used: v[3], entries_capacity: self.entry_cap }
    }

    /// Copy the delta buffer back (testing, PNG, readback of a finished drag).
    pub fn read_delta(&self) -> Vec<Vec3> {
        let (src, dst) = (self.delta.as_ref().unwrap(), self.delta_staging.as_ref().unwrap());
        let size = (self.n_verts as u64 * 12).max(4);
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.queue.submit([enc.finish()]);
        let raw = read_buffer(&self.device, dst, size);
        let f: &[f32] = bytemuck::cast_slice(&raw);
        f.chunks_exact(3).take(self.n_verts as usize).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
    }

    #[allow(dead_code)] // integration point: a vertex shader adds this to the base positions
    /// The per-vertex delta buffer, for a vertex shader to add to the base positions.
    pub fn delta_buffer(&self) -> Option<&wgpu::Buffer> {
        self.delta.as_ref()
    }

    /// Bytes of GPU buffers held (mesh copy, delta, instances, grid).
    pub fn gpu_bytes(&self) -> u64 {
        let sz = |b: &Option<wgpu::Buffer>| b.as_ref().map_or(0, |b| b.size());
        sz(&self.mesh) + sz(&self.delta) + sz(&self.leaves) + sz(&self.work) + self.insts.as_ref().map_or(0, |b| b.0.size()) + self.heads.as_ref().map_or(0, |b| b.0.size()) + self.entries.as_ref().map_or(0, |b| b.0.size())
    }
}

/// Octahedral unit-vector encoding into two snorm16 (matches `oct_decode` in the shader).
fn oct_encode(n: Vec3) -> u32 {
    let sgn = |x: f32| if x >= 0.0 { 1.0 } else { -1.0 };
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut x, mut y) = if l1 > 0.0 { (n.x / l1, n.y / l1) } else { (0.0, 0.0) };
    if n.z < 0.0 {
        (x, y) = ((1.0 - y.abs()) * sgn(x), (1.0 - x.abs()) * sgn(y));
    }
    let q = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i32 as i16) as u16 as u32;
    q(x) | q(y) << 16
}

fn read_buffer(device: &wgpu::Device, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let slice = buf.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    buf.unmap();
    data
}

/// Same transform bake as `sculpt_core::generator::instances::Baked::new`, plus the inflated AABB.
fn bake(i: &Instance, margin: f32) -> GpuInst {
    let rot = glam::Mat3::from_quat(i.rotation);
    let inv = i.scale.recip();
    let min_scale = i.scale.min_element();
    let ball = i.primitive == Primitive::Sphere && (i.scale.max_element() - i.scale.min_element()) <= 1e-6 * i.scale.max_element();
    let prim = Primitive::ALL.iter().position(|p| *p == i.primitive).unwrap() as u32;
    let a = i.aabb();
    let t = i.translation;
    GpuInst {
        c0: [rot.x_axis.x, rot.x_axis.y, rot.x_axis.z, t.x],
        c1: [rot.y_axis.x, rot.y_axis.y, rot.y_axis.z, t.y],
        c2: [rot.z_axis.x, rot.z_axis.y, rot.z_axis.z, t.z],
        sc: [inv.x, inv.y, inv.z, min_scale],
        bmin: (a.min - Vec3::splat(margin)).extend(0.0).to_array(),
        bmax: (a.max + Vec3::splat(margin)).extend(0.0).to_array(),
        lo: [0, 0, 0, prim | (ball as u32) << 8],
        hi: [0; 4],
    }
}

/// A device without a window. Tries the default adapter, then a fallback (software) one.
/// Returns `None` when no adapter exists (headless CI), so callers can skip gracefully.
pub fn headless() -> Option<(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle().with_env());
    let mut adapter = None;
    for fallback in [false, true] {
        let opts = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, force_fallback_adapter: fallback, compatible_surface: None, ..Default::default() };
        if let Ok(a) = pollster::block_on(instance.request_adapter(&opts)) {
            adapter = Some(a);
            break;
        }
    }
    let adapter = adapter?;
    let have = adapter.limits();
    let limits = wgpu::Limits {
        max_buffer_size: have.max_buffer_size.min(4 << 30),
        max_storage_buffer_binding_size: have.max_storage_buffer_binding_size.min(2 << 30),
        ..wgpu::Limits::default()
    };
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let desc = wgpu::DeviceDescriptor { label: Some("gpu_boolean headless"), required_features: features, required_limits: limits, ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;
    Some((device, queue, adapter.get_info()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU vs CPU generator on a small mesh, all three ops, mixed primitives.
    #[test]
    fn gpu_matches_cpu() {
        let Some((device, queue, info)) = headless() else {
            eprintln!("gpu_boolean test skipped: no wgpu adapter");
            return;
        };
        eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
        let mut gb = GpuBoolean::new(device, queue);
        let scene = bench::Scene::new(5, 150, true, 1.0);
        gb.upload_mesh(scene.doc.positions(), scene.doc.normals(), scene.doc.bvh());
        for op in [BoolOp::Subtract, BoolOp::Add, BoolOp::Intersect] {
            let cmp = scene.compare(&mut gb, op);
            eprintln!("{op:?}: {cmp}");
            assert_eq!(cmp.stats.entries_used, cmp.stats.entries_capacity, "grid entry count mismatch");
            assert!(cmp.moved_cpu > 20, "{op:?}: scene should move vertices ({})", cmp.moved_cpu);
            // Float re-association (FMA, sqrt / normalize rounding, 1e-3 tetrahedral gradient differences)
            // moves projections by ~1e-5; the snorm16 octahedral normals add up to ~1e-4 (0.2% of an edge) for Add.
            assert!(cmp.frac_within(3e-3 * scene.edge) > 0.999, "{op:?}: only {:.5} of vertices within tolerance ({cmp})", cmp.frac_within(3e-3 * scene.edge));
            assert!(cmp.max_err < 0.01 * scene.edge, "{op:?}: max error {} exceeds 1% of an edge ({cmp})", cmp.max_err);
        }
    }
}
