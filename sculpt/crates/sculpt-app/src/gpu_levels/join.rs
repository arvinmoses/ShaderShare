//! Generators joined to the field on the GPU, for an edit shown at its own level at once (see
//! `Layered::gpu_generator`): each block's base field (composed without the generators, at 16-bit floats) is kept in a
//! pool while what is under the generators stays the same, and a pass joins the generators to it, writing the block's
//! samples as its own extraction reads them. The CPU's job brings the exact field after (which also makes a distance
//! of it again where the join left it steeper; the GPU leaves that out).

use std::collections::HashMap;

use eframe::egui_wgpu::wgpu;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use rayon::prelude::*;
use sculpt_core::generator::detail_layer::GpuPlan;
use sculpt_core::generator::level_volume::Key;

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

/// An f32 as an IEEE half (round to nearest; tiny values to zero, huge ones to the largest half).
fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7bff;
    }
    let m = b & 0x7f_ffff;
    let h = ((e as u32) << 10) | (m >> 13);
    // Round to nearest (half up: a carry into the exponent is right).
    let h = h + ((m >> 12) & 1);
    sign | (h.min(0x7bff) as u16)
}

pub(super) struct Join {
    pipe: wgpu::ComputePipeline,
    base: wgpu::Buffer,
    cap: u32,
    /// Each block's base's place in the pool, for the epoch the bases were read at.
    base_of: HashMap<Key, u32>,
    high: u32,
    epoch: u64,
}

impl Join {
    fn new(device: &wgpu::Device) -> Join {
        let src = format!("{}\n{}\n{KERNEL}", sculpt_core::surface_noise::WGSL, sculpt_core::generator::volume_gen::WGSL);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels join"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("levels join"), layout: None, module: &module, entry_point: Some("join"), compilation_options: Default::default(), cache: None });
        Join { pipe, base: Self::pool(device, 1024), cap: 1024, base_of: HashMap::new(), high: 0, epoch: u64::MAX }
    }

    fn pool(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels join bases"), size: cap as u64 * SAMPLE_WORDS * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }
}

impl GpuLevels {
    /// Join the generators to the blocks `plan` lists (those with a slot: shown), `base` giving a block's base field
    /// when the pool has none for it, and rebuild their surface and the blocks around them.
    pub fn join(&mut self, plan: GpuPlan, base: &(dyn Fn(Key) -> Box<[f32; 729]> + Sync)) -> SyncStats {
        let t = std::time::Instant::now();
        let mut stats = SyncStats::default();
        let device = self.device.clone();
        let join = self.join.get_or_insert_with(|| Join::new(&device));
        if plan.epoch != join.epoch {
            join.base_of.clear();
            join.high = 0;
            join.epoch = plan.epoch;
        }
        let blocks: Vec<(Key, u32, Vec<[u32; 4]>)> = plan.blocks.into_iter().filter_map(|(k, r)| self.slot_of.get(&k).map(|&s| (k, s, r))).collect();
        if blocks.is_empty() {
            return stats;
        }
        // Bases the pool lacks: room for them (a larger pool starts afresh), then read and sent.
        let missing: Vec<Key> = blocks.iter().map(|b| b.0).filter(|k| !join.base_of.contains_key(k)).collect();
        if join.high as usize + missing.len() > join.cap as usize {
            // A larger pool, the bases it held copied across.
            let cap = (join.high + missing.len() as u32).next_power_of_two();
            let pool = Join::pool(&device, cap);
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&join.base, 0, &pool, 0, join.high as u64 * SAMPLE_WORDS * 4);
            self.queue.submit([enc.finish()]);
            join.base = pool;
            join.cap = cap;
        }
        let missing: Vec<Key> = blocks.iter().map(|b| b.0).filter(|k| !join.base_of.contains_key(k)).collect();
        let packed: Vec<(Key, Vec<u32>)> = missing
            .par_iter()
            .map(|&k| {
                let b = base(k);
                let mut w = vec![0u32; SAMPLE_WORDS as usize];
                for (i, v) in b.iter().enumerate() {
                    w[i / 2] |= (f16(*v) as u32) << (16 * (i % 2));
                }
                (k, w)
            })
            .collect();
        let t_base = t.elapsed().as_secs_f64() * 1e3;
        let first = join.high;
        let mut all = Vec::with_capacity(packed.len() * SAMPLE_WORDS as usize);
        for (k, w) in packed {
            join.base_of.insert(k, join.high);
            join.high += 1;
            all.extend(w);
        }
        if !all.is_empty() {
            self.queue.write_buffer(&join.base, first as u64 * SAMPLE_WORDS * 4, bytemuck::cast_slice(&all));
            stats.bytes += all.len() * 4;
        }
        // The pass's lists.
        let ngens = plan.gens.len() as u32;
        let mut ranges: Vec<[u32; 4]> = Vec::with_capacity(blocks.len() * ngens as usize);
        let mut work: Vec<[u32; 4]> = Vec::with_capacity(blocks.len());
        for (k, slot, r) in &blocks {
            work.push([*slot, ranges.len() as u32, ngens, join.base_of[k]]);
            ranges.extend(r.iter().copied());
        }
        let pad = |mut v: Vec<u8>| {
            if v.is_empty() {
                v.resize(16, 0);
            }
            v
        };
        let buf = |data: Vec<u8>, usage: wgpu::BufferUsages| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("levels join"), contents: &pad(data), usage });
        let st = wgpu::BufferUsages::STORAGE;
        let params: [f32; 8] = [self.origin.x, self.origin.y, self.origin.z, 0.0, self.voxel0, work.len() as f32, 0.0, 0.0];
        let bufs = [
            buf(bytemuck::cast_slice(&params).to_vec(), wgpu::BufferUsages::UNIFORM),
            buf(bytemuck::cast_slice(&plan.instances).to_vec(), st),
            buf(bytemuck::cast_slice(&plan.cands).to_vec(), st),
            buf(bytemuck::cast_slice(&plan.gens).to_vec(), st),
            buf(bytemuck::cast_slice(&ranges).to_vec(), st),
            buf(bytemuck::cast_slice(&work).to_vec(), st),
            buf(plan.masks.iter().flat_map(|m| {
                let mut w = vec![0u32; SAMPLE_WORDS as usize];
                for (i, v) in m.iter().enumerate() {
                    w[i / 2] |= (f16(*v) as u32) << (16 * (i % 2));
                }
                w
            }).flat_map(|w| w.to_le_bytes()).collect(), st),
        ];
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: bufs[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.buf.samples.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.buf.meta.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: join.base.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs[4].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs[5].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: bufs[6].as_entire_binding() },
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
        for (_, slot, _) in &blocks {
            self.sent[*slot as usize] = 0;
        }
        let t_pass = t.elapsed().as_secs_f64() * 1e3;
        // Rebuilt: the blocks joined (which blocks exist is the sync's, which went before). Those around them read
        // them only where the generators change nothing.
        let keys: Vec<Key> = blocks.iter().map(|b| b.0).collect();
        let extract = keys.clone();
        let t_set = t.elapsed().as_secs_f64() * 1e3;
        self.extract(&extract, &[], &mut stats);
        stats.uploaded = keys.len();
        stats.cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        if std::env::var_os("SCULPT_JOIN_LOG").is_some() {
            eprintln!("join: {} blocks ({} bases read): bases {t_base:.1}, lists and pass {:.1}, extract set {:.1} ({} blocks), extraction {:.1} ms", keys.len(), missing.len(), t_pass - t_base, t_set - t_pass, extract.len(), stats.cpu_ms - t_set);
        }
        stats
    }
}
