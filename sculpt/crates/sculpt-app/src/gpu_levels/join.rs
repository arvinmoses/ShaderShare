//! Generators joined to the field on the GPU, for an edit shown at its own level at once (see
//! `Layered::gpu_generator`): each block's base field (composed without the generators, at 16-bit floats) is kept in a
//! pool while what is under the generators stays the same, and a pass joins the generators to it, writing the block's
//! samples as its own extraction reads them. The CPU's job brings the exact field after (which also makes a distance
//! of it again where the join left it steeper; the GPU leaves that out).
//!
//! The CPU's part is kept small: the plan comes with its lists flat and its new blocks' bases packed, the pool is
//! indexed by slot (no table of blocks), and the pass's buffers are kept from one join to the next.

use eframe::egui_wgpu::wgpu;
use rayon::prelude::*;
use sculpt_core::generator::detail_layer::GpuPlan;

use super::{GpuLevels, SAMPLE_WORDS, SyncStats, grid};

const KERNEL: &str = "
struct JoinP { origin: vec4<f32>, misc: vec4<f32> };
@group(0) @binding(0) var<uniform> JP: JoinP;
@group(0) @binding(1) var<storage, read_write> samples: array<u32>;
@group(0) @binding(2) var<storage, read> blocks: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> base: array<u32>;
@group(0) @binding(4) var<storage, read> vg_inst: array<GpuInstance>;
@group(0) @binding(5) var<storage, read> vg_cand: array<u32>;
@group(0) @binding(6) var<storage, read> gens: array<GpuGen>;
// Per block and generator: its first candidate, how many, its mask's samples (none: whole).
@group(0) @binding(7) var<storage, read> ranges: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> masks: array<u32>;
// Per block: its slot, its first range, how many generators, its base's place in the pool.
@group(0) @binding(8) var<storage, read> work: array<vec4<u32>>;

@compute @workgroup_size(64)
fn join(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let w = wg.x + wg.y * nw.x;
    if (w >= u32(JP.misc.y)) {
        return;
    }
    let item = work[w];
    let slot = item.x;
    let m = blocks[slot];
    let l = u32(max(m.w, 0));
    let vx = JP.misc.x / f32(1u << l);
    let band = select(2.5 * vx, 16.0 * JP.misc.x, l == 0u);
    for (var wd = li; wd < 365u; wd += 64u) {
        var word = 0u;
        for (var h = 0u; h < 2u; h++) {
            let i = wd * 2u + h;
            if (i >= 729u) {
                break;
            }
            let q = vec3<i32>(i32(i % 9u), i32((i / 9u) % 9u), i32(i / 81u));
            let p = JP.origin.xyz + vec3<f32>(m.xyz * 8 + q) * vx;
            let b = unpack2x16float(base[item.w * 365u + i / 2u]);
            var c = select(b.x, b.y, (i & 1u) == 1u);
            for (var g = 0u; g < item.z; g++) {
                let r = ranges[item.y + g];
                let joined = vg_apply(gens[g], c, p, r.x, r.y);
                var m = 1.0;
                if (r.z != 0xffffffffu) {
                    let mv = unpack2x16float(masks[r.z * 365u + i / 2u]);
                    m = clamp(select(mv.x, mv.y, (i & 1u) == 1u), 0.0, 1.0);
                }
                c = c + (joined - c) * m;
            }
            let e = clamp(c / band, -1.0, 1.0);
            var qv = i32(round(e * 32767.0));
            if (c < 0.0 && qv == 0) {
                qv = -1;
            }
            word |= (bitcast<u32>(qv) & 0xffffu) << (16u * h);
        }
        samples[slot * 365u + wd] = word;
    }
}
";

/// A slot whose block's base the pool does not hold.
const NONE: u32 = u32::MAX;

/// The pass's lists, as buffers kept between joins: params, instances, candidates, generators, ranges, work, masks.
const LISTS: usize = 7;

pub(super) struct Join {
    pipe: wgpu::ComputePipeline,
    base: wgpu::Buffer,
    cap: u32,
    /// Per slot, its block's base's place in the pool, for the epoch the bases were read at.
    base_slot: Vec<u32>,
    high: u32,
    epoch: u64,
    lists: [wgpu::Buffer; LISTS],
}

impl Join {
    fn new(device: &wgpu::Device, slots: u32) -> Join {
        let src = format!("{}\n{}\n{KERNEL}", sculpt_core::surface_noise::WGSL, sculpt_core::generator::volume_gen::WGSL);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels join"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("levels join"), layout: None, module: &module, entry_point: Some("join"), compilation_options: Default::default(), cache: None });
        let lists = std::array::from_fn(|i| Self::list(device, i, 1024));
        // The pool holds a base per slot to begin with (it grows past that, as bases outlive their blocks).
        let cap = slots.max(1024);
        Join { pipe, base: Self::pool(device, cap), cap, base_slot: Vec::new(), high: 0, epoch: u64::MAX, lists }
    }

    fn pool(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels join bases"), size: cap as u64 * SAMPLE_WORDS * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }

    fn list(device: &wgpu::Device, i: usize, size: u64) -> wgpu::Buffer {
        let usage = if i == 0 { wgpu::BufferUsages::UNIFORM } else { wgpu::BufferUsages::STORAGE };
        device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels join lists"), size: size.max(16), usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    /// Send list `i`, in a buffer large enough for it (grown to the next power of two when it is not).
    fn send(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, i: usize, data: &[u8]) {
        let need = (data.len() as u64).max(16);
        if self.lists[i].size() < need {
            self.lists[i] = Self::list(device, i, need.next_power_of_two());
        }
        if !data.is_empty() {
            queue.write_buffer(&self.lists[i], 0, data);
        }
    }

    /// Slot `s` was given to another block: its base is not held.
    pub(super) fn slot_taken(&mut self, s: u32) {
        if let Some(b) = self.base_slot.get_mut(s as usize) {
            *b = NONE;
        }
    }
}

impl GpuLevels {
    /// Join the generators to the blocks `plan` lists (those with a slot: shown), `base` giving a block's base field
    /// when neither the plan nor the pool has it, and rebuild their surface.
    pub fn join(&mut self, plan: GpuPlan, base: &(dyn Fn(sculpt_core::generator::level_volume::Key) -> Box<[f32; 729]> + Sync)) -> SyncStats {
        let t = std::time::Instant::now();
        let mut stats = SyncStats::default();
        let device = self.device.clone();
        let slots = self.buf.slots;
        let join = self.join.get_or_insert_with(|| Join::new(&device, slots));
        if plan.epoch != join.epoch {
            join.base_slot.clear();
            join.high = 0;
            join.epoch = plan.epoch;
        }
        if join.base_slot.len() < self.key_of.len() {
            join.base_slot.resize(self.key_of.len(), NONE);
        }
        let slot_of = &self.slot_of;
        // Each block's slot (none: hidden, so not joined).
        let slots: Vec<u32> = plan.blocks.par_iter().map(|k| slot_of.get(k).copied().unwrap_or(NONE)).collect();
        let joined = slots.iter().filter(|s| **s != NONE).count();
        if joined == 0 {
            return stats;
        }
        // Bases the pool lacks: the plan's new blocks come with theirs; the rest are read.
        let fresh_slots: Vec<u32> = plan.fresh.iter().map(|k| slot_of.get(k).copied().unwrap_or(NONE)).collect();
        let mut fresh_take: Vec<usize> = Vec::with_capacity(plan.fresh.len());
        for (i, &s) in fresh_slots.iter().enumerate() {
            if s != NONE && join.base_slot[s as usize] == NONE {
                join.base_slot[s as usize] = join.high + fresh_take.len() as u32;
                fresh_take.push(i);
            }
        }
        let missing: Vec<(usize, u32)> = slots.iter().enumerate().filter(|(_, s)| **s != NONE && join.base_slot[**s as usize] == NONE).map(|(i, s)| (i, *s)).collect();
        let more = fresh_take.len() + missing.len();
        if join.high as usize + more > join.cap as usize {
            // A larger pool, the bases it held copied across.
            let cap = (join.high + more as u32).next_power_of_two();
            let pool = Join::pool(&device, cap);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&join.base, 0, &pool, 0, join.high as u64 * SAMPLE_WORDS * 4);
            self.queue.submit([enc.finish()]);
            join.base = pool;
            join.cap = cap;
        }
        if !fresh_take.is_empty() {
            let all: Vec<[u32; 365]> = fresh_take.par_iter().map(|&i| plan.bases[i]).collect();
            self.queue.write_buffer(&join.base, join.high as u64 * SAMPLE_WORDS * 4, bytemuck::cast_slice(&all));
            stats.bytes += all.len() * SAMPLE_WORDS as usize * 4;
            join.high += all.len() as u32;
        }
        let t_fresh = t.elapsed().as_secs_f64() * 1e3;
        if !missing.is_empty() {
            let blocks = &plan.blocks;
            let all: Vec<[u32; 365]> = missing.par_iter().map(|&(i, _)| sculpt_core::generator::detail_layer::pack_f16(&base(blocks[i]))).collect();
            for (n, &(_, s)) in missing.iter().enumerate() {
                join.base_slot[s as usize] = join.high + n as u32;
            }
            self.queue.write_buffer(&join.base, join.high as u64 * SAMPLE_WORDS * 4, bytemuck::cast_slice(&all));
            stats.bytes += all.len() * SAMPLE_WORDS as usize * 4;
            join.high += all.len() as u32;
        }
        let t_base = t.elapsed().as_secs_f64() * 1e3;
        // The pass's work: per joined block, its slot, its ranges and its base.
        let ngens = plan.gens.len() as u32;
        let mut work: Vec<[u32; 4]> = Vec::with_capacity(joined);
        let mut ex: Vec<u32> = Vec::with_capacity(joined);
        for (i, &s) in slots.iter().enumerate() {
            if s != NONE {
                work.push([s, i as u32 * ngens, ngens, join.base_slot[s as usize]]);
                ex.push(s);
            }
        }
        let params: [f32; 8] = [self.origin.x, self.origin.y, self.origin.z, 0.0, self.voxel0, work.len() as f32, 0.0, 0.0];
        let queue = self.queue.clone();
        join.send(&device, &queue, 0, bytemuck::cast_slice(&params));
        join.send(&device, &queue, 1, bytemuck::cast_slice(&plan.instances));
        join.send(&device, &queue, 2, bytemuck::cast_slice(&plan.cands));
        join.send(&device, &queue, 3, bytemuck::cast_slice(&plan.gens));
        join.send(&device, &queue, 4, bytemuck::cast_slice(&plan.ranges));
        join.send(&device, &queue, 5, bytemuck::cast_slice(&work));
        join.send(&device, &queue, 6, bytemuck::cast_slice(&plan.masks));
        stats.bytes += (plan.instances.len() * 80 + plan.cands.len() * 4 + plan.ranges.len() * 16 + work.len() * 16 + plan.masks.len() * SAMPLE_WORDS as usize * 4) as usize;
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: join.lists[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.buf.samples.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.buf.meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: join.base.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: join.lists[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: join.lists[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: join.lists[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: join.lists[4].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: join.lists[5].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: join.lists[6].as_entire_binding() },
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("levels join"), layout: &join.pipe.get_bind_group_layout(0), entries: &entries });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("levels join") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("levels join"), timestamp_writes: None });
            cp.set_pipeline(&join.pipe);
            cp.set_bind_group(0, &group, &[]);
            let (x, y) = grid(work.len() as u32);
            cp.dispatch_workgroups(x, y, 1);
        }
        self.queue.submit([enc.finish()]);
        // What the slots hold is no longer what was sent: the next upload of them goes through.
        for &s in &ex {
            self.sent[s as usize] = 0;
        }
        let t_pass = t.elapsed().as_secs_f64() * 1e3;
        // Rebuilt: the blocks joined, and what the sync before left to be extracted with them (the blocks around
        // them, which read their vertices; the generators change nothing there).
        let (around, released) = self.deferred.take().unwrap_or_default();
        ex.extend(around);
        self.extract_slots(ex, &released, &mut stats);
        stats.uploaded = joined;
        stats.cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("SCULPT_JOIN_LOG").is_some() {
            eprintln!("join: {joined} blocks ({} bases from the plan, {} read): slots and plan bases {t_fresh:.1}, bases read {:.1}, lists and pass {:.1}, extraction {:.1} ms", fresh_take.len(), missing.len(), t_base - t_fresh, t_pass - t_base, stats.cpu_ms - t_pass);
        }
        stats
    }
}
