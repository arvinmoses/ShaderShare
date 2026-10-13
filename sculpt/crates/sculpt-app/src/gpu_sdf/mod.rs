//! SPIKE 5: GPU sparse-brick SDF boolean for the Generator Boolean layer.
//!
//! Measurement code, not wired into the app. Everything here runs on its own wgpu
//! device (no window), so the benchmark works over SSH and on CI.
//!
//! ```text
//! cargo run --release -p sculpt-app -- --gpu-sdf-bench          # full matrix (see bench.rs)
//! cargo run --release -p sculpt-app -- --gpu-sdf-check          # GPU vs CPU correctness
//! cargo run --release -p sculpt-app -- --gpu-sdf-png out_dir    # images through the headless GPU path
//! ```
//!
//! # Pipeline (all passes are compute, in `sdf.wgsl`)
//!
//! Persistent state (rebuilt only when lower layers change):
//!
//! * `table` + `smeta` + `base`: a GPU hash table `brick key -> slot` and, per slot,
//!   a 10^3 lattice of **base signed distance** (mesh closest point; 8^3 voxels plus a
//!   one-sample pad so extraction never reads a neighbour brick). Bricks are created
//!   lazily where instances need them.
//!
//! Per frame (an instance edit):
//!
//! 1. `enum_insert`, `assign_slots`: make sure every brick an instance overlaps is in the
//!    cache; new ones are queued for `base_classify` / `base_fill` (the expensive part).
//! 2. `reset_heads`, `list_build`: per-brick linked lists of the instances overlapping the
//!    brick (atomic exchange, no sort), and the list of active bricks.
//! 3. In chunks of bricks (bounded scratch): `combine` (cached base + analytic instance
//!    SDFs, `min`/`max` or smooth variants), `verts_pass` (surface-nets vertex per
//!    sign-changing cell, atomic append), `indices_pass` (two triangles per sign-changing
//!    lattice edge, atomic append).
//!
//! The output is an indexed triangle list in a GPU buffer, drawn as an overlay in place of
//! the base mesh leaves whose bricks are active.

pub mod bench;
pub mod check;
pub mod render;

use std::time::Instant;

use eframe::egui_wgpu::wgpu;
use glam::{Mat3, Vec3};
use rayon::prelude::*;
use sculpt_core::generator::voxel::FlatBvh;
use sculpt_core::generator::{Instance, Primitive};

pub const SAMPLES: usize = 1000; // 10^3 lattice per brick (8^3 voxels + 1 sample pad each side)
pub const CELLS: usize = 729; // 9^3 cells (the pad cells included)
const SHADER: &str = include_str!("sdf.wgsl");

// Counter indices, mirrored from the shader.
pub const C_BUILD: usize = 1;
pub const C_ACTIVE: usize = 2;
pub const C_PAIRS: usize = 3;
pub const C_VERTS: usize = 4;
pub const C_IDX: usize = 5;
pub const C_OWNED: usize = 6;
pub const C_OVERFLOW: usize = 7;
pub const C_QUADS_OWNED: usize = 8;

// ------------------------------------------------------------------------------------ device

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

impl Gpu {
    /// Own instance and device; no surface. `Err` when no adapter exists (tests skip then).
    pub fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let request = |fallback: bool| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: fallback,
                apply_limit_buckets: false,
            }))
        };
        let adapter = request(false).or_else(|_| request(true)).map_err(|e| format!("no wgpu adapter: {e}"))?;
        let info = adapter.get_info();
        let have = adapter.limits();
        let want = wgpu::Limits {
            max_storage_buffers_per_shader_stage: have.max_storage_buffers_per_shader_stage.min(16),
            max_storage_buffer_binding_size: have.max_storage_buffer_binding_size.min(4 << 30),
            max_buffer_size: have.max_buffer_size.min(4 << 30),
            max_compute_workgroups_per_dimension: have.max_compute_workgroups_per_dimension,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu_sdf"),
            required_limits: want.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device: {e}"))?;
        if want.max_storage_buffers_per_shader_stage < 14 {
            return Err(format!("adapter supports only {} storage buffers per stage; need 14", want.max_storage_buffers_per_shader_stage));
        }
        Ok(Gpu { device, queue, info, limits: want })
    }

    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}, driver \"{}\"){}",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            if self.is_software() { "  ** SOFTWARE RASTERIZER: timings are not representative of a GPU **" } else { "" }
        )
    }

    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn read_buffer(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: len.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, offset, &staging, 0, len);
        self.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wait();
        rx.recv().expect("map callback").expect("map");
        let data = staging.slice(..len).get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        data
    }

    pub fn read_u32(&self, buf: &wgpu::Buffer, offset_words: u64, n: usize) -> Vec<u32> {
        bytemuck::pod_collect_to_vec::<u8, u32>(&self.read_buffer(buf, offset_words * 4, n as u64 * 4))
    }

    pub fn read_f32(&self, buf: &wgpu::Buffer, offset_words: u64, n: usize) -> Vec<f32> {
        bytemuck::pod_collect_to_vec::<u8, f32>(&self.read_buffer(buf, offset_words * 4, n as u64 * 4))
    }

    fn storage(&self, label: &str, bytes: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(256).next_multiple_of(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | extra,
            mapped_at_creation: false,
        })
    }
}

// ------------------------------------------------------------------------------------ mesh

/// Base mesh closest-point structure on the GPU (flat BVH + triangles + positions + pseudo-normals).
pub struct MeshGpu {
    pub u: wgpu::Buffer,
    pub f: wgpu::Buffer,
    pub tri_off: u32,
    pub pn_off: u32,
    pub root: u32,
    pub bytes: u64,
    pub upload_ms: f64,
}

impl MeshGpu {
    pub fn upload(gpu: &Gpu, flat: &FlatBvh, positions: &[Vec3], pseudo: &[Vec3], root: u32) -> MeshGpu {
        let t = Instant::now();
        let node_words = flat.nodes.len() * 8;
        let u_words = node_words + flat.tris.len();
        let f_words = positions.len() * 3 * 2;
        let u = gpu.storage("mesh_u", u_words as u64 * 4, wgpu::BufferUsages::empty());
        let f = gpu.storage("mesh_f", f_words as u64 * 4, wgpu::BufferUsages::empty());
        let write_in_chunks = |buf: &wgpu::Buffer, offset_words: usize, data: &[u8]| {
            const CH: usize = 64 << 20;
            for (i, c) in data.chunks(CH).enumerate() {
                gpu.queue.write_buffer(buf, (offset_words * 4 + i * CH) as u64, c);
            }
        };
        write_in_chunks(&u, 0, bytemuck::cast_slice(&flat.nodes));
        write_in_chunks(&u, node_words, bytemuck::cast_slice(&flat.tris));
        let flat3 = |v: &[Vec3]| -> Vec<f32> { v.iter().flat_map(|p| p.to_array()).collect() };
        write_in_chunks(&f, 0, bytemuck::cast_slice(&flat3(positions)));
        write_in_chunks(&f, positions.len() * 3, bytemuck::cast_slice(&flat3(pseudo)));
        gpu.wait();
        MeshGpu {
            u,
            f,
            tri_off: node_words as u32,
            pn_off: (positions.len() * 3) as u32,
            root,
            bytes: (u_words + f_words) as u64 * 4,
            upload_ms: t.elapsed().as_secs_f64() * 1e3,
        }
    }
}

/// A sub-tree of the flat BVH holding only what a spatial window can query, with vertices
/// renumbered. Lets one adapter with small binding limits time a window of a huge mesh.
pub struct PrunedMesh {
    pub flat: FlatBvh,
    pub positions: Vec<Vec3>,
    pub pseudo: Vec<Vec3>,
    pub root: u32,
}

pub fn prune_mesh(flat: &FlatBvh, positions: &[Vec3], pseudo: &[Vec3], lo: Vec3, hi: Vec3) -> PrunedMesh {
    struct Ctx<'a> {
        src: &'a FlatBvh,
        pos: &'a [Vec3],
        pn: &'a [Vec3],
        lo: Vec3,
        hi: Vec3,
        out: FlatBvh,
        remap: Vec<u32>,
        p2: Vec<Vec3>,
        n2: Vec<Vec3>,
    }
    fn rec(c: &mut Ctx, o: u32) -> Option<u32> {
        let w = c.src.nodes[o as usize];
        let (bmin, bmax) = (Vec3::new(f32::from_bits(w[0]), f32::from_bits(w[1]), f32::from_bits(w[2])), Vec3::new(f32::from_bits(w[4]), f32::from_bits(w[5]), f32::from_bits(w[6])));
        if bmax.cmplt(c.lo).any() || bmin.cmpgt(c.hi).any() {
            return None;
        }
        if w[7] & FlatBvh::LEAF == 0 {
            match (rec(c, w[3]), rec(c, w[7])) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(l), Some(r)) => {
                    c.out.nodes.push([w[0], w[1], w[2], l, w[4], w[5], w[6], r]);
                    Some(c.out.nodes.len() as u32 - 1)
                }
            }
        } else {
            let start = c.out.tris.len() as u32 / 3;
            let n = w[7] & !FlatBvh::LEAF;
            for t in w[3]..w[3] + n {
                for k in 0..3 {
                    let v = c.src.tris[3 * t as usize + k] as usize;
                    if c.remap[v] == u32::MAX {
                        c.remap[v] = c.p2.len() as u32;
                        c.p2.push(c.pos[v]);
                        c.n2.push(c.pn[v]);
                    }
                    c.out.tris.push(c.remap[v]);
                }
            }
            c.out.nodes.push([w[0], w[1], w[2], start, w[4], w[5], w[6], FlatBvh::LEAF | n]);
            Some(c.out.nodes.len() as u32 - 1)
        }
    }
    let mut c = Ctx { src: flat, pos: positions, pn: pseudo, lo, hi, out: FlatBvh { nodes: Vec::new(), tris: Vec::new() }, remap: vec![u32::MAX; positions.len()], p2: Vec::new(), n2: Vec::new() };
    let root = rec(&mut c, 0).expect("window does not touch the mesh");
    PrunedMesh { flat: c.out, positions: c.p2, pseudo: c.n2, root }
}

// ------------------------------------------------------------------------------------ kernels

const ENTRIES: [&str; 11] = [
    "begin_frame",
    "reset_heads",
    "enum_insert",
    "assign_slots",
    "base_classify",
    "base_fill",
    "list_build",
    "combine",
    "verts_pass",
    "indices_pass",
    "query_stats",
];

/// Compiled compute pipelines, shared by every [`Sdf`].
pub struct Kernels {
    bgl: wgpu::BindGroupLayout,
    pipes: Vec<(&'static str, wgpu::ComputePipeline)>,
    pub compile_ms: f64,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Kernels {
        let t = Instant::now();
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sdf.wgsl"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform), entry(1, ro)];
        entries.extend((2..12).map(|b| entry(b, rw)));
        entries.extend([entry(12, ro), entry(13, ro)]);
        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("sdf bgl"), entries: &entries });
        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipes = ENTRIES
            .iter()
            .map(|&name| {
                let p = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (name, p)
            })
            .collect();
        Kernels { bgl, pipes, compile_ms: t.elapsed().as_secs_f64() * 1e3 }
    }

    fn pipe(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.pipes.iter().find(|(n, _)| *n == name).expect("kernel").1
    }
}

// ------------------------------------------------------------------------------------ instances

/// 6 x vec4 per instance: three scaled inverse-rotation rows (w = -row.t), (t, min_scale),
/// (inflated AABB min, primitive id), (inflated AABB max, uniform-sphere flag).
pub fn gpu_instance(i: &Instance, margin: f32) -> [[f32; 4]; 6] {
    let rot = Mat3::from_quat(i.rotation);
    let inv = rot.transpose();
    let isc = i.scale.recip();
    let row = |k: usize| {
        let r = inv.row(k) * isc[k];
        [r.x, r.y, r.z, -r.dot(i.translation)]
    };
    let a = i.aabb();
    let (lo, hi) = (a.min - Vec3::splat(margin), a.max + Vec3::splat(margin));
    let ms = i.scale.min_element();
    let ball = i.primitive == Primitive::Sphere && (i.scale.max_element() - ms) <= 1e-6 * i.scale.max_element();
    let prim = Primitive::ALL.iter().position(|&p| p == i.primitive).unwrap() as f32;
    [row(0), row(1), row(2), [i.translation.x, i.translation.y, i.translation.z, ms], [lo.x, lo.y, lo.z, prim], [hi.x, hi.y, hi.z, ball as u32 as f32]]
}

// ------------------------------------------------------------------------------------ config

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Subtract = 0,
    Union = 1,
    Intersect = 2,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub voxel: f32,
    pub op: Op,
    /// Smooth min/max blend radius in world units (0 = hard).
    pub smooth_k: f32,
    /// Origin of the brick lattice (brick 0 starts here).
    pub origin: Vec3,
    /// Brick subset to process (benchmarks only).
    pub keep: Keep,
    /// Bricks per extraction chunk (bounds the scratch memory).
    pub chunk: u32,
    /// Bricks the cache can hold, and the number of hash-table entries (power of two).
    pub slot_cap: u32,
    pub table_cap: u32,
    /// Capacity of the per-brick instance-list node pool.
    pub pair_cap: u64,
}

pub const MARGIN_VOXELS: f32 = 2.0;

// ------------------------------------------------------------------------------------ CPU enumeration

fn hash32(k: u32) -> u32 {
    let mut x = k.wrapping_mul(0x9E37_79B1);
    x ^= x >> 15;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^ (x >> 13)
}
/// Which bricks a run processes: every brick, a hash-selected 1-in-`stride`, or those inside a box
/// of brick coordinates. Bounds benchmark cost; the results are extrapolated by the brick ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keep {
    pub stride: u32,
    pub win: Option<([i32; 3], [i32; 3])>,
}

impl Keep {
    pub const ALL: Keep = Keep { stride: 1, win: None };
    pub fn stride(stride: u32) -> Keep {
        Keep { stride, win: None }
    }
    pub fn keeps(&self, key: u32) -> bool {
        if let Some((lo, hi)) = self.win {
            let b = unpack_key(key);
            if (0..3).any(|i| b[i] < lo[i] || b[i] > hi[i]) {
                return false;
            }
        }
        self.stride <= 1 || hash32(key ^ 0x5bd1_e995).is_multiple_of(self.stride)
    }
}
pub fn pack_key(b: [i32; 3]) -> u32 {
    b[0] as u32 | (b[1] as u32) << 10 | (b[2] as u32) << 20
}
pub fn unpack_key(k: u32) -> [i32; 3] {
    [(k & 1023) as i32, ((k >> 10) & 1023) as i32, ((k >> 20) & 1023) as i32]
}

pub struct Enumeration {
    /// Sorted unique brick keys (all bricks, ignoring `stride`).
    pub keys_all: usize,
    /// Sorted unique keys kept by the stride filter.
    pub keys: Vec<u32>,
    pub pairs_all: usize,
    pub pairs_kept: usize,
    pub ms: f64,
}

/// What the GPU enumeration passes will produce, computed on the CPU (sizes the buffers, and
/// is the reference for the correctness check). Same arithmetic as `inst_range` in the shader.
pub fn enumerate_bricks(insts: &[Instance], origin: Vec3, voxel: f32, stride: u32) -> Enumeration {
    enumerate_with(insts, origin, voxel, Keep::stride(stride))
}

pub fn enumerate_with(insts: &[Instance], origin: Vec3, voxel: f32, keep: Keep) -> Enumeration {
    let t = Instant::now();
    let bsz = 8.0 * voxel;
    let pad = voxel; // band - margin
    let margin = MARGIN_VOXELS * voxel;
    let mut keys: Vec<u32> = insts
        .par_iter()
        .flat_map_iter(|inst| {
            let a = inst.aabb();
            let amin = a.min - Vec3::splat(margin);
            let amax = a.max + Vec3::splat(margin);
            let lo = ((amin - Vec3::splat(pad) - origin) / bsz).floor().as_ivec3();
            let hi = ((amax + Vec3::splat(pad) - origin) / bsz).floor().as_ivec3();
            (lo.z..=hi.z).flat_map(move |z| (lo.y..=hi.y).flat_map(move |y| (lo.x..=hi.x).map(move |x| pack_key([x, y, z]))))
        })
        .collect();
    let pairs_all = keys.len();
    let pairs_kept = keys.par_iter().filter(|&&k| keep.keeps(k)).count();
    keys.par_sort_unstable();
    keys.dedup();
    let keys_all = keys.len();
    keys.retain(|&k| keep.keeps(k));
    Enumeration { keys_all, keys, pairs_all, pairs_kept, ms: t.elapsed().as_secs_f64() * 1e3 }
}

// ------------------------------------------------------------------------------------ stats

#[derive(Clone, Debug, Default)]
pub struct ColdStats {
    pub ms_enum: f64,
    pub ms_assign: f64,
    pub ms_classify: f64,
    pub ms_fill: f64,
    pub bricks: u32,
}

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub ms_upload: f64,
    pub ms_enum: f64,
    pub ms_assign: f64,
    pub ms_build_new: f64,
    pub ms_reset: f64,
    pub ms_list: f64,
    pub ms_combine: f64,
    pub ms_verts: f64,
    pub ms_indices: f64,
    pub ms_wall: f64,
    pub active_bricks: u32,
    pub new_bricks: u32,
    pub pairs: u32,
    /// Vertices written including the pad duplicates (what the render buffer holds).
    pub verts: u32,
    /// Vertices in owned cells only (comparable to the CPU spike's `patch verts out`).
    pub verts_owned: u32,
    pub indices: u32,
    pub edges_owned: u32,
    pub overflow: u32,
}

impl FrameStats {
    /// Sum of the stages that run on every instance edit (excludes upload and new-brick builds).
    pub fn ms_eval(&self) -> f64 {
        self.ms_enum + self.ms_assign + self.ms_reset + self.ms_list + self.ms_combine + self.ms_verts + self.ms_indices
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemReport {
    pub mesh: u64,
    pub instances: u64,
    pub table: u64,
    pub slot_meta: u64,
    pub base_pool: u64,
    pub pairs: u64,
    pub lists: u64,
    pub scratch: u64,
    pub out_verts: u64,
    pub out_idx: u64,
}

impl MemReport {
    pub fn total(&self) -> u64 {
        self.mesh + self.instances + self.table + self.slot_meta + self.base_pool + self.pairs + self.lists + self.scratch + self.out_verts + self.out_idx
    }
}

// ------------------------------------------------------------------------------------ pipeline state

pub struct Sdf<'a> {
    pub gpu: &'a Gpu,
    pub cfg: Config,
    kernels: &'a Kernels,
    mesh: &'a MeshGpu,
    params: wgpu::Buffer,
    inst: wgpu::Buffer,
    inst_cap: usize,
    table: wgpu::Buffer,
    ctr: wgpu::Buffer,
    smeta: wgpu::Buffer,
    pairs: wgpu::Buffer,
    lists: wgpu::Buffer,
    base: wgpu::Buffer,
    comb: wgpu::Buffer,
    cellv: wgpu::Buffer,
    pub verts: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    vert_cap: u32,
    idx_cap: u32,
    cap_factor: u32,
    bg: wgpu::BindGroup,
    words: [u32; 24],
    frame_no: u32,
    built: u32,
    n_inst: u32,
    pub last_indices: u32,
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

impl<'a> Sdf<'a> {
    pub fn new(gpu: &'a Gpu, kernels: &'a Kernels, mesh: &'a MeshGpu, cfg: Config, max_instances: usize) -> Sdf<'a> {
        let pool = cfg.slot_cap as u64 * SAMPLES as u64 * 4;
        assert!(
            pool <= gpu.limits.max_storage_buffer_binding_size.min(gpu.limits.max_buffer_size),
            "base brick pool of {} MB exceeds the adapter's storage binding limit of {} MB: use fewer instances, a larger voxel, or --max-bricks",
            pool >> 20,
            gpu.limits.max_storage_buffer_binding_size >> 20
        );
        let rw = wgpu::BufferUsages::empty();
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let inst_cap = max_instances.max(1);
        let inst = gpu.storage("instances", inst_cap as u64 * 96, rw);
        let table = gpu.storage("table", cfg.table_cap as u64 * 8, rw);
        gpu.queue.write_buffer(&table, 0, &vec![0xFFu8; cfg.table_cap as usize * 8]);
        let ctr = gpu.storage("counters", 64, rw);
        let smeta = gpu.storage("slot_meta", cfg.slot_cap as u64 * 16, rw);
        let pairs = gpu.storage("pairs", cfg.pair_cap * 8, rw);
        let lists = gpu.storage("lists", cfg.slot_cap as u64 * 8, rw);
        let base = gpu.storage("base_pool", cfg.slot_cap as u64 * SAMPLES as u64 * 4, rw);
        let comb = gpu.storage("combined_scratch", cfg.chunk as u64 * SAMPLES as u64 * 4, rw);
        let cellv = gpu.storage("cell_vertex_scratch", cfg.chunk as u64 * CELLS as u64 * 4, rw);
        let verts = gpu.storage("out_verts", 4096 * 32, wgpu::BufferUsages::VERTEX);
        let indices = gpu.storage("out_idx", 4096 * 4, wgpu::BufferUsages::INDEX);
        let bg = Self::bind(gpu, kernels, mesh, &params, &inst, &table, &ctr, &smeta, &pairs, &lists, &base, &comb, &cellv, &verts, &indices);
        let mut s = Sdf {
            gpu,
            kernels,
            mesh,
            params,
            inst,
            inst_cap,
            table,
            ctr,
            smeta,
            pairs,
            lists,
            base,
            comb,
            cellv,
            verts,
            indices,
            vert_cap: 4096,
            idx_cap: 4096,
            cap_factor: 1,
            bg,
            words: [0; 24],
            frame_no: 0,
            built: 0,
            n_inst: 0,
            last_indices: 0,
            cfg,
        };
        s.init_words();
        s
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(
        gpu: &Gpu,
        k: &Kernels,
        mesh: &MeshGpu,
        params: &wgpu::Buffer,
        inst: &wgpu::Buffer,
        table: &wgpu::Buffer,
        ctr: &wgpu::Buffer,
        smeta: &wgpu::Buffer,
        pairs: &wgpu::Buffer,
        lists: &wgpu::Buffer,
        base: &wgpu::Buffer,
        comb: &wgpu::Buffer,
        cellv: &wgpu::Buffer,
        verts: &wgpu::Buffer,
        idx: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bufs = [params, inst, table, ctr, smeta, pairs, lists, base, comb, cellv, verts, idx, &mesh.u, &mesh.f];
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("sdf bg"), layout: &k.bgl, entries: &entries })
    }

    fn rebind(&mut self) {
        self.bg = Self::bind(self.gpu, self.kernels, self.mesh, &self.params, &self.inst, &self.table, &self.ctr, &self.smeta, &self.pairs, &self.lists, &self.base, &self.comb, &self.cellv, &self.verts, &self.indices);
    }

    fn init_words(&mut self) {
        let c = &self.cfg;
        let w = &mut self.words;
        w[0] = f(c.origin.x);
        w[1] = f(c.origin.y);
        w[2] = f(c.origin.z);
        w[3] = f(c.voxel);
        w[5] = c.table_cap - 1;
        w[7] = c.op as u32;
        w[8] = f(c.smooth_k);
        w[9] = f(MARGIN_VOXELS * c.voxel);
        w[10] = f((MARGIN_VOXELS + 1.0) * c.voxel);
        w[11] = c.keep.stride;
        let (lo, hi) = c.keep.win.unwrap_or(([0; 3], [1023; 3]));
        w[21] = pack_key(lo.map(|x| x.clamp(0, 1023)));
        w[22] = pack_key(hi.map(|x| x.clamp(0, 1023)));
        w[23] = self.mesh.root;
        w[14] = c.slot_cap;
        w[17] = self.mesh.tri_off;
        w[18] = self.mesh.pn_off;
        w[20] = 1 << 20;
    }

    pub fn memory(&self) -> MemReport {
        let c = &self.cfg;
        MemReport {
            mesh: self.mesh.bytes,
            instances: self.inst_cap as u64 * 96,
            table: c.table_cap as u64 * 8,
            slot_meta: c.slot_cap as u64 * 16,
            base_pool: c.slot_cap as u64 * SAMPLES as u64 * 4,
            pairs: c.pair_cap * 8,
            lists: c.slot_cap as u64 * 8,
            scratch: c.chunk as u64 * (SAMPLES + CELLS) as u64 * 4,
            out_verts: self.vert_cap as u64 * 32,
            out_idx: self.idx_cap as u64 * 4,
        }
    }

    /// Run one kernel over `threads` invocations, wait for it, return milliseconds.
    fn run(&mut self, entry: &str, threads: u64) -> f64 {
        if threads == 0 {
            return 0.0;
        }
        let groups = threads.div_ceil(64);
        let max = self.gpu.limits.max_compute_workgroups_per_dimension as u64;
        let x = groups.min(max);
        let y = groups.div_ceil(x);
        assert!(y <= max, "dispatch too large: {threads} threads");
        self.words[19] = x as u32;
        let t = Instant::now();
        self.gpu.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&self.words));
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(self.kernels.pipe(entry));
            pass.set_bind_group(0, &self.bg, &[]);
            pass.dispatch_workgroups(x as u32, y as u32, 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.wait();
        t.elapsed().as_secs_f64() * 1e3
    }

    pub fn counters(&self) -> [u32; 16] {
        let v = self.gpu.read_u32(&self.ctr, 0, 16);
        let mut a = [0u32; 16];
        a.copy_from_slice(&v);
        a
    }

    /// Upload instances; the caller passes the same list the CPU reference sees.
    pub fn upload_instances(&mut self, insts: &[Instance]) -> f64 {
        assert!(insts.len() <= self.inst_cap);
        let t = Instant::now();
        let margin = MARGIN_VOXELS * self.cfg.voxel;
        let recs: Vec<[[f32; 4]; 6]> = insts.par_iter().map(|i| gpu_instance(i, margin)).collect();
        self.gpu.queue.write_buffer(&self.inst, 0, bytemuck::cast_slice(&recs));
        self.n_inst = insts.len() as u32;
        self.words[4] = self.n_inst;
        self.gpu.wait();
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Cache every brick the current instances need: insert keys, assign slots, build base SDF.
    pub fn cold_build(&mut self) -> ColdStats {
        let mut s = ColdStats { ms_enum: self.run("enum_insert", self.n_inst as u64), ms_assign: self.run("assign_slots", self.cfg.table_cap as u64), ..Default::default() };
        let (c, ms_cl, ms_fill) = self.build_pending();
        s.bricks = c;
        s.ms_classify = ms_cl;
        s.ms_fill = ms_fill;
        s
    }

    /// Build base bricks queued by `assign_slots` and not built yet. Returns (count, classify ms, fill ms).
    fn build_pending(&mut self) -> (u32, f64, f64) {
        let total = self.counters()[C_BUILD].min(self.cfg.slot_cap);
        let (mut ms_cl, mut ms_fill) = (0.0, 0.0);
        const CH: u32 = 8192;
        let mut at = self.built;
        let n = total - self.built.min(total);
        while at < total {
            let cnt = CH.min(total - at);
            self.words[12] = at;
            self.words[13] = cnt;
            ms_cl += self.run("base_classify", cnt as u64);
            ms_fill += self.run("base_fill", cnt as u64 * SAMPLES as u64);
            at += cnt;
        }
        self.built = total;
        (n, ms_cl, ms_fill)
    }

    fn ensure_outputs(&mut self, active: u32) {
        let per = 96u64 * self.cap_factor as u64;
        let max = self.gpu.limits.max_buffer_size / 40;
        let vc = ((active as u64 * per).max(4096)).min(max) as u32;
        if vc > self.vert_cap {
            self.vert_cap = vc;
            self.idx_cap = vc.saturating_mul(7).min((self.gpu.limits.max_buffer_size / 4) as u32);
            self.verts = self.gpu.storage("out_verts", vc as u64 * 32, wgpu::BufferUsages::VERTEX);
            self.indices = self.gpu.storage("out_idx", self.idx_cap as u64 * 4, wgpu::BufferUsages::INDEX);
            self.rebind();
        }
        self.words[15] = self.vert_cap;
        self.words[16] = self.idx_cap;
    }

    /// One edit: re-evaluate the boolean over every active brick and extract the surface.
    /// Instances must have been uploaded (and `cold_build` run once, though new bricks are
    /// also built here if the instances moved).
    pub fn frame(&mut self) -> FrameStats {
        let wall = Instant::now();
        let mut s = FrameStats { ms_enum: self.run("enum_insert", self.n_inst as u64), ms_assign: self.run("assign_slots", self.cfg.table_cap as u64), ..Default::default() };
        let (new, cl, fill) = self.build_pending();
        s.new_bricks = new;
        s.ms_build_new = cl + fill;
        loop {
            self.frame_no += 1;
            self.words[6] = self.frame_no;
            s.ms_reset = self.run("begin_frame", 1) + self.run("reset_heads", self.cfg.slot_cap as u64);
            s.ms_list = self.run("list_build", self.n_inst as u64);
            let c = self.counters();
            let active = c[C_ACTIVE];
            s.active_bricks = active;
            s.pairs = c[C_PAIRS];
            self.ensure_outputs(active);
            (s.ms_combine, s.ms_verts, s.ms_indices) = (0.0, 0.0, 0.0);
            let mut at = 0;
            while at < active {
                let cnt = self.cfg.chunk.min(active - at);
                self.words[12] = at;
                self.words[13] = cnt;
                s.ms_combine += self.run("combine", cnt as u64 * SAMPLES as u64);
                s.ms_verts += self.run("verts_pass", cnt as u64 * CELLS as u64);
                s.ms_indices += self.run("indices_pass", cnt as u64 * 512);
                at += cnt;
            }
            let c = self.counters();
            s.verts = c[C_VERTS];
            s.verts_owned = c[C_OWNED];
            s.indices = c[C_IDX];
            s.edges_owned = c[C_QUADS_OWNED];
            s.overflow = c[C_OVERFLOW];
            // Output capacity is a guess from the brick count; double it and redo if it was short.
            if s.overflow & 12 != 0 && self.cap_factor < 8 {
                self.cap_factor *= 2;
                continue;
            }
            break;
        }
        self.last_indices = s.indices.min(self.idx_cap);
        s.ms_wall = wall.elapsed().as_secs_f64() * 1e3;
        s
    }

    /// Average (node visits, triangle tests) per base query, measured by re-running the queries of the first
    /// bricks with the counting traversal. Returns (visits, tests, queries).
    pub fn query_stats(&mut self) -> (f64, f64, u64) {
        let n = self.built.min(self.cfg.slot_cap).min(256);
        self.gpu.queue.write_buffer(&self.ctr, 9 * 4, &[0u8; 12]);
        self.words[12] = 0;
        self.words[13] = n;
        self.run("query_stats", n as u64 * SAMPLES as u64);
        let c = self.counters();
        let q = c[11].max(1) as f64;
        (c[9] as f64 / q, c[10] as f64 / q, c[11] as u64)
    }

    /// Everything the correctness check needs (valid when all active bricks fit one chunk).
    pub fn read_debug(&self) -> DebugDump {
        let c = self.counters();
        let active = c[C_ACTIVE] as usize;
        let slots = self.gpu.read_u32(&self.lists, self.cfg.slot_cap as u64, active);
        let meta = self.gpu.read_u32(&self.smeta, 0, self.cfg.slot_cap as usize * 4);
        let keys = slots.iter().map(|&s| meta[s as usize * 4]).collect();
        let comb = self.gpu.read_f32(&self.comb, 0, active.min(self.cfg.chunk as usize) * SAMPLES);
        let nv = (c[C_VERTS] as usize).min(self.vert_cap as usize);
        let verts = self.gpu.read_f32(&self.verts, 0, nv * 8);
        let ni = (c[C_IDX] as usize).min(self.idx_cap as usize);
        let idx = self.gpu.read_u32(&self.indices, 0, ni);
        DebugDump { active_keys: keys, comb, verts, idx }
    }

    /// Slot keys of all cached bricks (for tests).
    pub fn cached_bricks(&self) -> u32 {
        self.built
    }

    /// Median distance-to-surface cost proxy: base samples in bricks that needed per-sample queries.
    pub fn near_samples(&self) -> u64 {
        let n = (self.built as usize).min(self.cfg.slot_cap as usize);
        let meta = self.gpu.read_u32(&self.smeta, 0, n * 4);
        let hd = 0.5 * 3f32.sqrt() * 9.0 * self.cfg.voxel;
        let band = (MARGIN_VOXELS + 1.0) * self.cfg.voxel;
        (0..n).filter(|&s| f32::from_bits(meta[s * 4 + 3]).abs() <= hd + band).count() as u64 * SAMPLES as u64
    }
}

pub struct DebugDump {
    pub active_keys: Vec<u32>,
    pub comb: Vec<f32>,
    /// 8 floats per vertex: position, 1, normal, 0.
    pub verts: Vec<f32>,
    pub idx: Vec<u32>,
}

/// `Config` sized for an enumeration (exact brick and pair counts plus slack).
pub fn config_for(en: &Enumeration, origin: Vec3, voxel: f32, op: Op, keep: Keep, chunk: u32) -> Config {
    let slots = (en.keys.len() as f64 * 1.02) as u32 + 64;
    Config {
        voxel,
        op,
        smooth_k: 0.0,
        origin,
        keep,
        chunk,
        slot_cap: slots,
        table_cap: (slots as u64 * 2).next_power_of_two().max(1024) as u32,
        pair_cap: (en.pairs_kept as f64 * 1.02) as u64 + 64,
    }
}

pub fn cli(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        Some("--gpu-sdf-bench") => bench::run(&args[1..]),
        Some("--gpu-sdf-check") => check::run(&args[1..]),
        Some("--gpu-sdf-png") => render::run(&args[1..]),
        _ => Err("usage: sculpt-app --gpu-sdf-bench | --gpu-sdf-check | --gpu-sdf-png <dir>".into()),
    }
}

pub fn is_gpu_sdf_command(arg: &str) -> bool {
    matches!(arg, "--gpu-sdf-bench" | "--gpu-sdf-check" | "--gpu-sdf-png")
}
