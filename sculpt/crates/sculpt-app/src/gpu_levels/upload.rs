//! Samples sent to the GPU in bulk. A sync that changes many blocks (a generator's exact field landing after its
//! preview) would otherwise send a write per run of slots, tens of thousands of them: instead the blocks go up in one
//! write, packed end to end with the slot of each, and a pass scatters them to their slots.

use std::num::NonZeroU64;

use eframe::egui_wgpu::wgpu;
use eframe::egui_wgpu::wgpu::WriteOnly;
use rayon::prelude::*;

use super::{SAMPLE_WORDS, grid};

const W: usize = SAMPLE_WORDS as usize;

const KERNEL: &str = "
@group(0) @binding(0) var<storage, read_write> samples: array<u32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
// The count, then the slot of each block.
@group(0) @binding(2) var<storage, read> slots: array<u32>;

@compute @workgroup_size(64)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let b = wg.x + wg.y * nw.x;
    if (b >= slots[0]) {
        return;
    }
    let s = slots[b + 1u];
    for (var w = li; w < 365u; w += 64u) {
        samples[s * 365u + w] = data[b * 365u + w];
    }
}
";

/// Fewer blocks than this are written slot by slot: a pass is not worth setting up for them.
pub(super) const IN_BULK: usize = 64;

pub(super) struct Scatter {
    pipe: wgpu::ComputePipeline,
    /// The blocks' samples end to end, and the count then the slot of each: kept between syncs, grown as needed.
    data: wgpu::Buffer,
    slots: wgpu::Buffer,
}

impl Scatter {
    pub(super) fn new(device: &wgpu::Device) -> Scatter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels upload"), source: wgpu::ShaderSource::Wgsl(KERNEL.into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("levels upload"), layout: None, module: &module, entry_point: Some("scatter"), compilation_options: Default::default(), cache: None });
        Scatter { pipe, data: Self::buffer(device, "levels upload data", 1 << 20), slots: Self::buffer(device, "levels upload slots", 4096) }
    }

    fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
    }

    /// Write `blocks` (slot, samples) to their slots of `samples`. Returns the time (ms) after the staging write,
    /// the slot list and the pass.
    pub(super) fn send(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, samples: &wgpu::Buffer, blocks: &[(u32, &[u32; W])]) -> [f64; 3] {
        let t = std::time::Instant::now();
        let n = blocks.len();
        let bytes = (n * W * 4) as u64;
        if self.data.size() < bytes {
            self.data = Self::buffer(device, "levels upload data", bytes.next_power_of_two());
        }
        let list_bytes = (n as u64 + 1) * 4;
        if self.slots.size() < list_bytes {
            self.slots = Self::buffer(device, "levels upload slots", list_bytes.next_power_of_two());
        }
        {
            // Straight into the staging memory, over the threads.
            let mut view = queue.write_buffer_with(&self.data, 0, NonZeroU64::new(bytes).expect("blocks to send")).expect("a staging write");
            // A handle per block: those are sized, so they can go to other threads where the whole view cannot.
            let (chunks, _) = view.slice(..).into_chunks::<{ W * 4 }>();
            let handles: Vec<WriteOnly<'_, [u8; W * 4]>> = chunks.into_iter().collect();
            handles.into_par_iter().zip(blocks.par_iter()).for_each(|(w, b)| WriteOnly::<[u8]>::from(w).copy_from_slice(bytemuck::cast_slice(b.1)));
        }
        let t_staging = t.elapsed().as_secs_f64() * 1e3;
        let mut list: Vec<u32> = Vec::with_capacity(n + 1);
        list.push(n as u32);
        list.extend(blocks.iter().map(|b| b.0));
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
