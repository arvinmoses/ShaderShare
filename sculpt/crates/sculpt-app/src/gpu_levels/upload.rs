//! Samples sent to the GPU in bulk. A sync that changes many blocks (a generator's exact field landing after its
//! preview) would otherwise send a write per run of slots, tens of thousands of them: instead the blocks go up in one
//! write, packed end to end with the slot of each, and a pass scatters them to their slots.
//!
//! A change that lands over several syncs leaves some blocks as the GPU holds them for a while. The first layer of
//! samples past a block that is sent reads such a neighbour's samples: the pass takes that layer from the neighbour's
//! slot, so both sides of their face agree and nothing opens between what has landed and what has not.
//!
//! Also here: the pass that puts new pages on the free stacks when the pools grow in place.

use std::num::NonZeroU64;

use eframe::egui_wgpu::wgpu;
use eframe::egui_wgpu::wgpu::WriteOnly;
use rayon::prelude::*;

use super::{SAMPLE_WORDS, grid};

const W: usize = SAMPLE_WORDS as usize;

const KERNEL: &str = "
@group(0) @binding(0) var<storage, read_write> samples: array<u32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
// The count, then per block its slot and the slots of the seven neighbours past it (by octant) whose samples the
// block's first layer past it must agree with (0xffffffff: as sent).
@group(0) @binding(2) var<storage, read> slots: array<u32>;

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let b = wg.x + wg.y * nw.x;
    if (b >= slots[0]) {
        return;
    }
    let s = slots[1u + b * 8u];
    for (var w = li; w < 365u; w += 64u) {
        var word = data[b * 365u + w];
        for (var h = 0u; h < 2u; h++) {
            let i = w * 2u + h;
            if (i >= 729u) {
                break;
            }
            let x = i % 9u;
            let y = (i / 9u) % 9u;
            let z = i / 81u;
            let q = u32(x == 8u) | (u32(y == 8u) << 1u) | (u32(z == 8u) << 2u);
            if (q == 0u) {
                continue;
            }
            let nb = slots[1u + b * 8u + q];
            if (nb == 0xffffffffu) {
                continue;
            }
            let j = (x - 8u * (q & 1u)) + 9u * (y - 8u * ((q >> 1u) & 1u)) + 81u * (z - 8u * ((q >> 2u) & 1u));
            let v = (samples[nb * 365u + j / 2u] >> (16u * (j & 1u))) & 0xffffu;
            word = (word & ~(0xffffu << (16u * h))) | (v << (16u * h));
        }
        samples[s * 365u + w] = word;
    }
}
";

const GROW_KERNEL: &str = "
@group(0) @binding(0) var<storage, read_write> free_pages: array<u32>;
@group(0) @binding(1) var<storage, read_write> status: array<i32>;
// Quad pages before and after, vertex pages before and after.
@group(0) @binding(2) var<uniform> counts: vec4<u32>;

@compute @workgroup_size(1)
fn grow() {
    let n = u32(status[0]);
    for (var i = 0u; i < counts.y - counts.x; i++) {
        free_pages[n + i] = counts.x + i;
    }
    status[0] = i32(n + counts.y - counts.x);
    let m = u32(status[4]);
    for (var i = 0u; i < counts.w - counts.z; i++) {
        free_pages[counts.y + m + i] = counts.z + i;
    }
    status[4] = i32(m + counts.w - counts.z);
}
";

/// Fewer blocks than this are written slot by slot: a pass is not worth setting up for them.
pub(super) const IN_BULK: usize = 64;

pub(super) struct Scatter {
    pipe: wgpu::ComputePipeline,
    /// The blocks' samples end to end, and the count then the slots of each: kept between syncs, grown as needed.
    data: wgpu::Buffer,
    slots: wgpu::Buffer,
}

impl Scatter {
    pub(super) fn new(device: &wgpu::Device) -> Scatter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels upload"), source: wgpu::ShaderSource::Wgsl(KERNEL.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("levels upload"), layout: None, module: &module, entry_point: Some("scatter"), compilation_options: Default::default(), cache: None });
        Scatter { pipe, data: buffer(device, "levels upload data", 1 << 20), slots: buffer(device, "levels upload slots", 1 << 15) }
    }

    /// Write `blocks` (slot, neighbours' slots, samples) to their slots of `samples`. Returns the time (ms) after the
    /// staging write, the slot list and the pass.
    pub(super) fn send(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, samples: &wgpu::Buffer, blocks: &[(u32, [u32; 7], &[u32; W])]) -> [f64; 3] {
        let t = std::time::Instant::now();
        let n = blocks.len();
        let bytes = (n * W * 4) as u64;
        if self.data.size() < bytes {
            self.data = buffer(device, "levels upload data", bytes.next_power_of_two());
        }
        let list_bytes = (n as u64 * 8 + 1) * 4;
        if self.slots.size() < list_bytes {
            self.slots = buffer(device, "levels upload slots", list_bytes.next_power_of_two());
        }
        {
            // Straight into the staging memory, over the threads.
            let mut view = queue.write_buffer_with(&self.data, 0, NonZeroU64::new(bytes).expect("blocks to send")).expect("a staging write");
            // A handle per block: those are sized, so they can go to other threads where the whole view cannot.
            let (chunks, _) = view.slice(..).into_chunks::<{ W * 4 }>();
            let handles: Vec<WriteOnly<'_, [u8; W * 4]>> = chunks.into_iter().collect();
            handles.into_par_iter().zip(blocks.par_iter()).for_each(|(w, b)| WriteOnly::<[u8]>::from(w).copy_from_slice(bytemuck::cast_slice(b.2)));
        }
        let t_staging = t.elapsed().as_secs_f64() * 1e3;
        let mut list: Vec<u32> = Vec::with_capacity(n * 8 + 1);
        list.push(n as u32);
        for b in blocks {
            list.push(b.0);
            list.extend_from_slice(&b.1);
        }
        queue.write_buffer(&self.slots, 0, bytemuck::cast_slice(&list));
        let t_slots = t.elapsed().as_secs_f64() * 1e3;
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: samples.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.slots.as_entire_binding() },
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("levels upload"), layout: &self.pipe.get_bind_group_layout(0), entries: &entries });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("levels upload") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("levels upload"), timestamp_writes: None });
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &group, &[]);
            let (x, y) = grid(n as u32);
            cp.dispatch_workgroups(x, y, 1);
        }
        queue.submit([enc.finish()]);
        [t_staging, t_slots, t.elapsed().as_secs_f64() * 1e3]
    }
}

fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

/// Puts the pages of grown pools on the free stacks.
pub(super) struct Grow {
    pipe: wgpu::ComputePipeline,
    counts: wgpu::Buffer,
}

impl Grow {
    pub(super) fn new(device: &wgpu::Device) -> Grow {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels grow"), source: wgpu::ShaderSource::Wgsl(GROW_KERNEL.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("levels grow"), layout: None, module: &module, entry_point: Some("grow"), compilation_options: Default::default(), cache: None });
        let counts = device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels grow counts"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Grow { pipe, counts }
    }

    /// The free stacks in `free_pages` (quad pages, then from the new quad page count on the vertex pages) get the
    /// pages from the old counts to the new: `[quad pages before, after, vertex pages before, after]`.
    pub(super) fn extend(&self, device: &wgpu::Device, queue: &wgpu::Queue, free_pages: &wgpu::Buffer, status: &wgpu::Buffer, counts: [u32; 4]) {
        queue.write_buffer(&self.counts, 0, bytemuck::cast_slice(&counts));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: free_pages.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: status.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.counts.as_entire_binding() },
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("levels grow"), layout: &self.pipe.get_bind_group_layout(0), entries: &entries });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("levels grow") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("levels grow"), timestamp_writes: None });
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &group, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([enc.finish()]);
    }
}
