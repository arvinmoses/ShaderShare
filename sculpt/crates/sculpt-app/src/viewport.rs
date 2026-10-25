//! GPU viewport.
//!
//! Vertex data lives in one GPU buffer of 8-byte vertices (position as three 16-bit values inside the model's box, plus
//! the overlay scalar), plus a normals buffer only in smooth shading, laid out in the engine's internal vertex order.
//! Flat shading takes each triangle's normal from the screen derivatives of its position, so it needs no normals at all. Because every
//! spatial leaf owns a contiguous vertex range, a brush dab becomes a handful
//! of `write_buffer` calls for exactly the leaves it touched — the GPU never
//! sees a full re-upload during sculpting.
//!
//! The scene renders (4x MSAA) into an offscreen texture that egui shows as
//! an image, so panels and viewport share one device and one queue.

use std::ops::Range;
use std::time::Instant;

use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3, Vec4};
use rayon::prelude::*;
use sculpt_core::bvh::Bvh;
use sculpt_core::geom::Aabb;
use sculpt_core::lod::{LodTree, View};
use sculpt_core::{DirtySet, Document};

use crate::camera::Camera;
use crate::theme::Theme;

const MSAA: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    clay: [f32; 4],
    overlay: [f32; 4],
    bg_top: [f32; 4],
    bg_bottom: [f32; 4],
    cursor: [f32; 4],
    cursor_color: [f32; 4],
    /// Corner and size of the box the 16-bit vertex positions are measured in.
    box_min: [f32; 4],
    box_size: [f32; 4],
}

/// What the overlay buffer currently shows.
#[derive(Clone, Debug, PartialEq)]
pub enum OverlayKind {
    None,
    Freeze,
    LayerMask,
    Channel(String),
    /// App-provided values (pose weights).
    Custom,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UploadStats {
    pub bytes: usize,
    pub ranges: usize,
    pub ms: f32,
}

struct Targets {
    size: [u32; 2],
    msaa: wgpu::TextureView,
    resolve: wgpu::TextureView,
    depth: wgpu::TextureView,
}

/// How the surface is lit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shading {
    /// Each triangle has one hard normal, derived on the GPU. Nothing is stored or uploaded, and the CPU skips normals.
    #[default]
    Flat,
    /// Smooth per-vertex normals, computed on the CPU and uploaded.
    Smooth,
}

/// The box that 16-bit vertex positions are measured in: a margin around the model, so the model can move a little
/// before everything has to be re-sent. At 16 bits the step is about 1/65000 of the box, far finer than a triangle.
#[derive(Clone, Copy, Debug)]
struct Quant {
    min: Vec3,
    size: Vec3,
}

impl Quant {
    fn fit(b: Aabb) -> Quant {
        let centre = (b.min + b.max) * 0.5;
        let half = ((b.max - b.min) * 0.5 * 1.15).max(Vec3::splat(1e-3));
        Quant { min: centre - half, size: half * 2.0 }
    }

    fn contains(&self, b: &Aabb) -> bool {
        b.min.cmpge(self.min).all() && b.max.cmple(self.min + self.size).all()
    }
}

/// Pack positions (and the overlay scalar, 0 to 1) as `Unorm16x4`. `positions` and `ov` cover the same vertices.
fn pack(q: &Quant, positions: &[Vec3], ov: Option<&[f32]>, out: &mut [[u16; 4]]) {
    let inv = Vec3::ONE / q.size;
    out.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
        let base = ci * 4096;
        for (k, o) in chunk.iter_mut().enumerate() {
            let p = ((positions[base + k] - q.min) * inv).clamp(Vec3::ZERO, Vec3::ONE) * 65535.0 + Vec3::splat(0.5);
            let w = ov.map_or(0.0, |s| s[base + k].clamp(0.0, 1.0)) * 65535.0 + 0.5;
            *o = [p.x as u16, p.y as u16, p.z as u16, w as u16];
        }
    });
}

struct MeshBuffers {
    topology_id: u64,
    /// 8 bytes per vertex: see [`pack`].
    vertices: wgpu::Buffer,
    /// Smooth shading only.
    normals: Option<wgpu::Buffer>,
    quant: Quant,
    /// Plain full-detail indices. Dropped once the LOD pool (whose leaf prefix is the same mesh) is resident.
    indices: Option<wgpu::Buffer>,
    index_count: u32,
}

/// The LOD pool on the GPU plus the buffer that holds this frame's draw list.
struct LodGpu {
    topology_id: u64,
    /// Identity of the uploaded tree, so a rebuilt one is noticed.
    tree: usize,
    pool: wgpu::Buffer,
    indirect: wgpu::Buffer,
    capacity: u32,
}

/// What the renderer needs to draw a dense mesh through its LOD tree.
pub struct LodInput<'a> {
    pub tree: &'a std::sync::Arc<LodTree>,
    pub bvh: &'a Bvh,
    pub topology_id: u64,
    /// Screen-space error tolerance, in pixels.
    pub tau_px: f32,
    /// Hard cap on triangles per frame.
    pub budget: u64,
}

#[derive(Clone, Copy, Default)]
pub struct LodStats {
    pub active: bool,
    pub triangles: u64,
    pub nodes: u32,
    pub tau_px: f32,
    pub select_ms: f32,
}

pub struct Viewport {
    flat_pipeline: wgpu::RenderPipeline,
    smooth_pipeline: wgpu::RenderPipeline,
    /// Set by the app; a change rebuilds the vertex buffers on the next sync.
    pub shading: Shading,
    scratch: Vec<[u16; 4]>,
    bg_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    targets: Option<Targets>,
    mesh: Option<MeshBuffers>,
    pub texture_id: Option<egui::TextureId>,
    overlay_kind: OverlayKind,
    pub last_upload: UploadStats,
    lod: Option<LodGpu>,
    pub lod_stats: LodStats,
    /// Block until the GPU finishes each frame (benchmarking).
    pub wait_gpu: bool,
    /// The volume playground's surface, drawn instead of the mesh while it is open.
    volume: Option<VolumeGpu>,
    levels_pipeline: wgpu::RenderPipeline,
    levels_bgl: wgpu::BindGroupLayout,
    /// Hide clusters outside the view or facing away (the volume playground).
    pub cull: bool,
    /// What the last frame drew of the volume: clusters' pages drawn, of those holding surface.
    pub volume_draws: u32,
}

/// The volume playground's surface on the GPU: built there by `gpu_levels`, and drawn from its buffers, a page of
/// quads per indirect draw for the clusters the cull pass keeps.
struct VolumeGpu {
    gpu: crate::gpu_levels::GpuLevels,
    uniforms: wgpu::Buffer,
    group: wgpu::BindGroup,
    /// The surface noise's mask: a hash of its blocks (at level 0's lattice: x, y, z, place or -1) and their 9x9x9
    /// samples; the noise epoch they are of, and the hash's size less one (0: no mask).
    mask_hash: wgpu::Buffer,
    mask_samples: wgpu::Buffer,
    mask_of: Option<u64>,
    mask_bits: u32,
    /// The GPU buffers `group` binds, and the volume `gpu` copies.
    buffers_made: u64,
    generation: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelsUniforms {
    origin: [f32; 4],
    misc: [f32; 4],
    noise_f: [f32; 4],
    noise_i: [u32; 4],
    mask_i: [u32; 4],
}

/// The surface noise mask's blocks as the shader looks them up: an open-addressed hash of (x, y, z, place), and each
/// block's 9x9x9 samples (its own and the first layer past it; whole where none is painted).
fn noise_mask(m: &sculpt_core::generator::level_volume::Shared<glam::IVec3, sculpt_core::generator::level_volume::Stored>) -> (Vec<[i32; 4]>, Vec<f32>, u32) {
    let keys: Vec<glam::IVec3> = m.keys().copied().collect();
    let len = (keys.len() * 2).next_power_of_two().max(16);
    let mut hash = vec![[0i32, 0, 0, -1]; len];
    let mut samples = Vec::with_capacity(keys.len() * 729);
    let at = |j: glam::IVec3| -> f32 {
        let b = j.div_euclid(glam::IVec3::splat(8));
        let l = j.rem_euclid(glam::IVec3::splat(8));
        m.get(&b).map_or(1.0, |s| s[(l.x + 8 * l.y + 64 * l.z) as usize])
    };
    for (n, bc) in keys.iter().enumerate() {
        let mut h = ((bc.x as u32).wrapping_mul(73856093) ^ (bc.y as u32).wrapping_mul(19349663) ^ (bc.z as u32).wrapping_mul(83492791)) as usize & (len - 1);
        while hash[h][3] >= 0 {
            h = (h + 1) & (len - 1);
        }
        hash[h] = [bc.x, bc.y, bc.z, n as i32];
        for k in 0..729 {
            samples.push(at(*bc * 8 + glam::IVec3::new(k % 9, (k / 9) % 9, k / 81)));
        }
    }
    if samples.is_empty() {
        samples.push(1.0);
    }
    (hash, samples, (len - 1) as u32)
}

fn linear(c: egui::Color32) -> [f32; 4] {
    let [r, g, b, a] = egui::Rgba::from(c).to_array();
    [r, g, b, a]
}

impl Viewport {
    pub fn new(rs: &egui_wgpu::RenderState) -> Viewport {
        let device = &rs.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", include_str!("viewport.wgsl"), sculpt_core::surface_noise::WGSL).into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let attr = |loc: u32, format: wgpu::VertexFormat, stride: u64| wgpu::VertexBufferLayout {
            array_stride: stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Box::leak(Box::new([wgpu::VertexAttribute { format, offset: 0, shader_location: loc }])),
        };
        let flat_buffers = [Some(attr(0, wgpu::VertexFormat::Unorm16x4, 8))];
        let smooth_buffers = [Some(attr(0, wgpu::VertexFormat::Unorm16x4, 8)), Some(attr(1, wgpu::VertexFormat::Float32x3, 12))];
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let ms = wgpu::MultisampleState { count: MSAA, ..Default::default() };
        let mesh_pipeline = |label: &'static str, vs: &'static str, fs: &'static str, buffers: &[Option<wgpu::VertexBufferLayout<'_>>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &target }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: ms,
                multiview_mask: None,
                cache: None,
            })
        };
        let flat_pipeline = mesh_pipeline("mesh flat", "vs_flat", "fs_flat", &flat_buffers);
        // The volume's surface: vertices pulled from the GPU extractor's buffers.
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let levels_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport levels"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ro(1),
                ro(2),
                ro(3),
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ro(6),
                ro(7),
                ro(8),
                ro(9),
                ro(10),
            ],
        });
        let levels_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("viewport levels"), bind_group_layouts: &[Some(&bgl), Some(&levels_bgl)], immediate_size: 0 });
        let levels_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume levels"),
            layout: Some(&levels_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_levels"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_levels"), compilation_options: Default::default(), targets: &target }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            multiview_mask: None,
            cache: None,
        });
        let smooth_pipeline = mesh_pipeline("mesh smooth", "vs_smooth", "fs_smooth", &smooth_buffers);
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_bg"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_bg"), compilation_options: Default::default(), targets: &target }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            multiview_mask: None,
            cache: None,
        });
        Viewport {
            flat_pipeline,
            smooth_pipeline,
            shading: Shading::default(),
            scratch: Vec::new(),
            bg_pipeline,
            uniforms,
            bind_group,
            targets: None,
            mesh: None,
            volume: None,
            levels_pipeline,
            levels_bgl,
            cull: true,
            volume_draws: 0,
            texture_id: None,
            overlay_kind: OverlayKind::None,
            last_upload: UploadStats::default(),
            lod: None,
            lod_stats: LodStats::default(),
            wait_gpu: false,
        }
    }

    fn ensure_targets(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2]) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let tex = |label, format, samples, usage| {
            rs.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let msaa = tex("viewport msaa", COLOR_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolve = tex("viewport color", COLOR_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let depth = tex("viewport depth", DEPTH_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let mut renderer = rs.renderer.write();
        match self.texture_id {
            Some(id) => renderer.update_egui_texture_from_wgpu_texture(&rs.device, &resolve, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(renderer.register_native_texture(&rs.device, &resolve, wgpu::FilterMode::Linear)),
        }
        self.targets = Some(Targets { size, msaa, resolve, depth });
    }

    /// Bring the volume playground's surface up to date: send the blocks that changed and rebuild the surface around them
    /// on the GPU (everything, for a new volume or a new level cap). `clock` is the time in seconds (the wall clock if
    /// none). True while levels are still fading as they hand over, or a big change is still landing region by region
    /// (more frames are wanted).
    pub fn sync_volume(&mut self, rs: &egui_wgpu::RenderState, mode: &mut crate::volume_mode::VolumeMode, eye: glam::Vec3, focal_px: f32, clock: Option<f64>) -> bool {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let now = clock.unwrap_or_else(|| START.get_or_init(Instant::now).elapsed().as_secs_f64());
        let t = Instant::now();
        if self.volume.as_ref().is_none_or(|v| v.generation != mode.generation) {
            let gpu = crate::gpu_levels::GpuLevels::new(&rs.device, &rs.queue, &mode.vol, mode.cap);
            let uniforms = rs.device.create_buffer(&wgpu::BufferDescriptor { label: Some("volume levels uniforms"), size: std::mem::size_of::<LevelsUniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let empty = |label| rs.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 256, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let (mask_hash, mask_samples) = (empty("noise mask hash"), empty("noise mask samples"));
            let group = Self::levels_group(&rs.device, &self.levels_bgl, &uniforms, &gpu, &mask_hash, &mask_samples);
            self.volume = Some(VolumeGpu { buffers_made: gpu.buffers_made, gpu, uniforms, group, mask_hash, mask_samples, mask_of: None, mask_bits: 0, generation: mode.generation });
        }
        let v = self.volume.as_mut().unwrap();
        v.gpu.set_cap(mode.cap);
        v.gpu.view.on = mode.by_distance;
        v.gpu.view.tau_px = mode.level_px;
        v.gpu.view.budget = mode.budget_on.then(|| (mode.budget_m as f64 * 1e6) as u64);
        // Levels hand over by fading (seconds; `SCULPT_FADE=0` switches it to at once).
        static FADE: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        v.gpu.view.fade_s = *FADE.get_or_init(|| std::env::var("SCULPT_FADE").ok().and_then(|s| s.parse().ok()).unwrap_or(0.25));
        v.gpu.set_clock(now);
        v.gpu.set_eye(eye, focal_px);
        // A generator edit the GPU joins at its own level (the CPU's job brings the exact field after): the sync
        // leaves the plan's blocks to the join.
        let plan = mode.gpu_plan.take();
        let s = match &plan {
            Some(plan) => v.gpu.sync_for_join(&mut mode.vol, plan),
            None => v.gpu.sync(&mut mode.vol),
        };
        mode.view = v.gpu.view.stats;
        if let (Some(plan), Some(layer)) = (plan, mode.layer.as_ref()) {
            let j = v.gpu.join(plan, &|k| layer.base_samples(k));
            mode.join_ms = j.cpu_ms;
        }
        // The surface noise shown, and its mask (sent again when the noise changed).
        let noise = mode.shown_noise();
        let mut regroup = v.gpu.buffers_made != v.buffers_made;
        if let Some((_, m, epoch)) = &noise
            && v.mask_of != Some(*epoch)
        {
            let (hash, samples, bits) = noise_mask(m);
            use eframe::egui_wgpu::wgpu::util::DeviceExt;
            let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            v.mask_hash = rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("noise mask hash"), contents: bytemuck::cast_slice(&hash), usage: st });
            v.mask_samples = rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("noise mask samples"), contents: bytemuck::cast_slice(&samples), usage: st });
            v.mask_bits = if m.is_empty() { 0 } else { bits };
            v.mask_of = Some(*epoch);
            regroup = true;
        }
        if regroup {
            v.group = Self::levels_group(&rs.device, &self.levels_bgl, &v.uniforms, &v.gpu, &v.mask_hash, &v.mask_samples);
            v.buffers_made = v.gpu.buffers_made;
        }
        let (noise_f, noise_i) = noise.as_ref().map_or(sculpt_core::surface_noise::SurfaceNoise { on: false, ..Default::default() }, |n| n.0).uniform();
        let fading = v.gpu.view.fading();
        let mask_i = [v.mask_bits, (v.mask_bits > 0 && noise.is_some()) as u32, v.gpu.fade_buffers().4, v.gpu.cap as u32];
        let misc = [v.gpu.voxel0(), if mode.tint { 1.0 } else { 0.0 }, if fading { 1.0 } else { 0.0 }, now as f32];
        let u = LevelsUniforms { origin: v.gpu.origin().extend(0.0).to_array(), misc, noise_f, noise_i, mask_i };
        rs.queue.write_buffer(&v.uniforms, 0, bytemuck::bytes_of(&u));
        mode.blocks = v.gpu.slots();
        if s.uploaded > 0 || s.full {
            mode.sync = s;
        } else if mode.sync.pending > 0 {
            mode.sync.pending = s.pending;
        }
        let fading = fading || s.pending > 0;
        self.volume_draws = v.gpu.status[2].max(0) as u32;
        mode.pages_drawn = self.volume_draws;
        mode.gpu_ms = v.gpu.gpu_ms;
        mode.pages_used = v.gpu.page_capacity().saturating_sub(v.gpu.status[0].max(0) as u32);
        self.last_upload = UploadStats { bytes: s.bytes, ranges: s.uploaded, ms: t.elapsed().as_secs_f32() * 1e3 };
        fading
    }

    fn levels_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, uniforms: &wgpu::Buffer, gpu: &crate::gpu_levels::GpuLevels, mask_hash: &wgpu::Buffer, mask_samples: &wgpu::Buffer) -> wgpu::BindGroup {
        let (_, _, _, verts, cells, slot_pages) = gpu.buffers_for_draw();
        let (morph, hash, samples, table, _) = gpu.fade_buffers();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume levels"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: verts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.meta_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mask_hash.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: mask_samples.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: slot_pages.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: morph.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: hash.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: samples.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: table.as_entire_binding() },
            ],
        })
    }

    /// Leave the volume playground: draw the document again.
    pub fn clear_volume(&mut self) {
        self.volume = None;
        self.mesh = None;
    }

    /// Bring GPU buffers up to date with the document. Cheap when nothing changed.
    pub fn sync(&mut self, rs: &egui_wgpu::RenderState, doc: &mut Document, overlay: &OverlayKind, custom: Option<&[f32]>, custom_changed: bool) {
        let t = Instant::now();
        let mut stats = UploadStats::default();
        let queue = &rs.queue;
        let smooth = self.shading == Shading::Smooth;
        if smooth {
            doc.ensure_normals();
        }
        let rebuild = self.mesh.as_ref().is_none_or(|m| m.topology_id != doc.topology_id() || m.normals.is_some() != smooth);
        if rebuild {
            let old = self.mesh.take();
            let mesh = create_mesh_buffers(&rs.device, doc, smooth, overlay_source(doc, overlay, custom), old);
            stats.bytes = doc.vertex_count() * (8 + if smooth { 12 } else { 0 }) + doc.face_count() * 24;
            stats.ranges = 1;
            self.mesh = Some(mesh);
            self.overlay_kind = overlay.clone();
            doc.take_geometry_dirty();
            doc.take_scalar_dirty();
            stats.ms = t.elapsed().as_secs_f32() * 1e3;
            self.last_upload = stats;
            return;
        }
        // The 16-bit box follows the surface: if the model outgrew it, refit and send every vertex again.
        let bounds = doc.bounds();
        let geometry = doc.take_geometry_dirty();
        let scalar = doc.take_scalar_dirty();
        let lod_updates = if self.lod.as_ref().is_some_and(|l| l.topology_id == doc.topology_id()) { doc.take_lod_updates() } else { Vec::new() };
        let mut resend = false;
        let mesh = self.mesh.as_mut().unwrap();
        if !mesh.quant.contains(&bounds) {
            mesh.quant = Quant::fit(bounds);
            resend = true;
        }
        let mesh = self.mesh.as_ref().unwrap();
        let src = overlay_source(doc, overlay, custom);
        let all = doc.vertex_count();
        match if resend { DirtySet::All } else { geometry } {
            DirtySet::None => {}
            DirtySet::All => {
                write_vertices(queue, mesh, doc, 0..all, src, &mut self.scratch, &mut stats);
                if smooth {
                    write_normals(queue, mesh, doc, 0..all, &mut stats);
                }
            }
            DirtySet::Leaves(leaves) => {
                for r in runs(doc, &leaves) {
                    write_vertices(queue, mesh, doc, r.clone(), src, &mut self.scratch, &mut stats);
                    if smooth {
                        write_normals(queue, mesh, doc, r, &mut stats);
                    }
                }
            }
        }
        for (first, count) in lod_updates {
            if let (Some(l), Some(tree)) = (&self.lod, doc.lod()) {
                let r = first as usize..(first + count) as usize;
                queue.write_buffer(&l.pool, first as u64 * 4, bytemuck::cast_slice(&tree.indices[r]));
                stats.bytes += count as usize * 4;
                stats.ranges += 1;
            }
        }
        // The overlay lives in each vertex's fourth component, so changing it re-sends those vertices.
        let kind_changed = *overlay != self.overlay_kind;
        let partial = match (&scalar, overlay) {
            (_, OverlayKind::None) => None,
            (_, OverlayKind::Custom) => custom_changed.then_some(DirtySet::All),
            (DirtySet::None, _) => None,
            (s, _) => Some(s.clone()),
        };
        if kind_changed {
            if !resend {
                write_vertices(queue, mesh, doc, 0..all, src, &mut self.scratch, &mut stats);
            }
            self.overlay_kind = overlay.clone();
        } else if let (Some(set), false) = (partial, resend) {
            match set {
                DirtySet::Leaves(leaves) => {
                    for r in runs(doc, &leaves) {
                        write_vertices(queue, mesh, doc, r, src, &mut self.scratch, &mut stats);
                    }
                }
                DirtySet::None => {}
                DirtySet::All => write_vertices(queue, mesh, doc, 0..all, src, &mut self.scratch, &mut stats),
            }
        }
        stats.ms = t.elapsed().as_secs_f32() * 1e3;
        self.last_upload = stats;
    }

    /// Upload the LOD pool when needed, choose this frame's cut and write the draw list.
    /// Returns the number of draws, or `None` when the plain mesh should be drawn.
    fn prepare_lod(&mut self, rs: &egui_wgpu::RenderState, input: Option<LodInput<'_>>, cam: &Camera, size: [u32; 2], view_proj: Mat4) -> Option<u32> {
        self.lod_stats = LodStats::default();
        let input = input?;
        let mesh = self.mesh.as_mut()?;
        if mesh.topology_id != input.topology_id {
            return None;
        }
        let key = std::sync::Arc::as_ptr(input.tree) as usize;
        if self.lod.as_ref().is_none_or(|l| l.tree != key || l.topology_id != input.topology_id) {
            let bytes = (input.tree.indices.len() * 4) as u64;
            if bytes > rs.device.limits().max_buffer_size {
                return None;
            }
            use wgpu::util::DeviceExt;
            let pool = rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lod pool"),
                contents: bytemuck::cast_slice(&input.tree.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
            let capacity = input.tree.nodes.len() as u32;
            let indirect = rs.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("lod draws"),
                size: capacity as u64 * 20,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            mesh.indices = None;
            self.lod = Some(LodGpu { topology_id: input.topology_id, tree: key, pool, indirect, capacity });
        }
        let lod = self.lod.as_ref()?;
        let t = Instant::now();
        let focal_px = size[1] as f32 / (2.0 * (cam.fov_y * 0.5).tan());
        let view = View::from_view_proj(view_proj, cam.eye(), focal_px);
        let cut = input.tree.select(input.bvh, &view, input.tau_px, input.budget);
        // DrawIndexedIndirect: index_count, instance_count, first_index, base_vertex, first_instance.
        let args: Vec<[u32; 5]> = cut.ranges.iter().take(lod.capacity as usize).map(|&(first, count)| [count, 1, first, 0, 0]).collect();
        if !args.is_empty() {
            rs.queue.write_buffer(&lod.indirect, 0, bytemuck::cast_slice(&args));
        }
        self.lod_stats = LodStats { active: true, triangles: cut.triangles, nodes: cut.nodes, tau_px: cut.tau_px, select_ms: t.elapsed().as_secs_f32() * 1e3 };
        Some(args.len() as u32)
    }

    /// Render into the offscreen texture. Returns the egui texture to display.
    #[allow(clippy::too_many_arguments)]
    pub fn render(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2], cam: &Camera, theme: &Theme, overlay_strength: f32, cursor: Option<(Vec3, f32, f32)>, lod_in: Option<LodInput<'_>>) -> Option<egui::TextureId> {
        let size = [size[0].max(1), size[1].max(1)];
        self.ensure_targets(rs, size);
        let vp = &theme.viewport;
        let overlay_color = if self.overlay_kind == OverlayKind::Freeze { vp.freeze.0 } else { vp.overlay.0 };
        let mut overlay = linear(overlay_color);
        overlay[3] = if self.overlay_kind == OverlayKind::None { 0.0 } else { overlay_strength };
        // Viewing a layer's mask shows it as greyscale (white = applies, dark = hidden), still lit.
        let mut clay = linear(vp.clay.0);
        if self.overlay_kind == OverlayKind::LayerMask {
            clay = linear(egui::Color32::from_gray(18));
            overlay = linear(egui::Color32::from_gray(235));
            overlay[3] = 1.0;
        }
        let view_proj: Mat4 = cam.proj(size[0] as f32 / size[1] as f32) * cam.view();
        let u = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            eye: cam.eye().extend(1.0).to_array(),
            right: cam.right().extend(0.0).to_array(),
            up: cam.up().extend(0.0).to_array(),
            clay,
            overlay,
            bg_top: linear(vp.background_top.0),
            bg_bottom: linear(vp.background_bottom.0),
            cursor: cursor.map_or(Vec4::ZERO, |(c, r, _)| c.extend(r)).to_array(),
            // Alpha carries the falloff hardness (inner cursor ring).
            cursor_color: { let mut cc = linear(vp.cursor.0); cc[3] = cursor.map_or(0.0, |(_, _, h)| h); cc },
            box_min: self.mesh.as_ref().map(|m| &m.quant).map_or(Vec4::ZERO, |q| q.min.extend(0.0)).to_array(),
            box_size: self.mesh.as_ref().map(|m| &m.quant).map_or(Vec4::ONE, |q| q.size.extend(1.0)).to_array(),
        };
        rs.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let draws = self.prepare_lod(rs, lod_in, cam, size, view_proj);
        let targets = self.targets.as_ref().unwrap();
        let mut enc = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewport") });
        if let Some(v) = &self.volume {
            v.gpu.cull(&mut enc, view_proj, cam.eye(), self.cull);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.msaa,
                    depth_slice: None,
                    resolve_target: Some(&targets.resolve),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.bg_pipeline);
            pass.draw(0..3, 0..1);
            if let Some(v) = &self.volume {
                let (pages, draws, status, _, _, _) = v.gpu.buffers_for_draw();
                pass.set_pipeline(&self.levels_pipeline);
                pass.set_bind_group(1, &v.group, &[]);
                pass.set_index_buffer(pages.slice(..), wgpu::IndexFormat::Uint32);
                if v.gpu.indirect_count {
                    pass.multi_draw_indexed_indirect_count(draws, 0, status, 8, v.gpu.page_capacity());
                } else {
                    pass.multi_draw_indexed_indirect(draws, 0, v.gpu.page_capacity());
                }
            } else if let Some(m) = &self.mesh {
                pass.set_pipeline(if m.normals.is_some() { &self.smooth_pipeline } else { &self.flat_pipeline });
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                if let Some(n) = &m.normals {
                    pass.set_vertex_buffer(1, n.slice(..));
                }
                match (&self.lod, draws) {
                    (Some(l), Some(n)) => {
                        pass.set_index_buffer(l.pool.slice(..), wgpu::IndexFormat::Uint32);
                        pass.multi_draw_indexed_indirect(&l.indirect, 0, n);
                    }
                    (lod, _) => {
                        // Without a cut the whole mesh is the pool's leaf prefix (or the plain buffer).
                        let buf = m.indices.as_ref().or(lod.as_ref().filter(|l| l.topology_id == m.topology_id).map(|l| &l.pool));
                        if let Some(buf) = buf {
                            pass.set_index_buffer(buf.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..m.index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
        rs.queue.submit([enc.finish()]);
        if let Some(v) = &self.volume {
            v.gpu.after_submit();
        }
        if self.wait_gpu {
            let _ = rs.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.texture_id
    }
}

/// The values the overlay shows right now, or `None` (all zero) when it shows nothing.
fn overlay_source<'a>(doc: &'a Document, kind: &OverlayKind, custom: Option<&'a [f32]>) -> Option<&'a [f32]> {
    let active_mask = doc.active_layer().and_then(|id| doc.layer(id)).and_then(|l| l.mask_values());
    let src: Option<&[f32]> = match kind {
        OverlayKind::None => None,
        OverlayKind::Freeze => Some(doc.freeze()),
        OverlayKind::LayerMask => active_mask,
        OverlayKind::Channel(n) => doc.channel(n),
        OverlayKind::Custom => custom,
    };
    src.filter(|s| s.len() == doc.vertex_count())
}

/// Quantize and upload the vertices in `r`.
fn write_vertices(queue: &wgpu::Queue, mesh: &MeshBuffers, doc: &Document, r: Range<usize>, ov: Option<&[f32]>, scratch: &mut Vec<[u16; 4]>, stats: &mut UploadStats) {
    scratch.resize(r.len(), [0; 4]);
    pack(&mesh.quant, &doc.positions()[r.clone()], ov.map(|s| &s[r.clone()]), scratch);
    queue.write_buffer(&mesh.vertices, r.start as u64 * 8, bytemuck::cast_slice(scratch));
    stats.bytes += r.len() * 8;
    stats.ranges += 1;
}

/// Upload the normals in `r` (smooth shading only).
fn write_normals(queue: &wgpu::Queue, mesh: &MeshBuffers, doc: &Document, r: Range<usize>, stats: &mut UploadStats) {
    if let Some(buf) = &mesh.normals {
        queue.write_buffer(buf, r.start as u64 * 12, bytemuck::cast_slice(&doc.normals()[r.clone()]));
        stats.bytes += r.len() * 12;
    }
}

/// Build the vertex buffers (and, unless `old` is the same topology, the index buffer).
fn create_mesh_buffers(device: &wgpu::Device, doc: &Document, smooth: bool, ov: Option<&[f32]>, old: Option<MeshBuffers>) -> MeshBuffers {
    use wgpu::util::DeviceExt;
    let vb = |label, data: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    };
    let quant = Quant::fit(doc.bounds());
    let mut packed = vec![[0u16; 4]; doc.vertex_count()];
    pack(&quant, doc.positions(), ov, &mut packed);
    let vertices = vb("vertices", bytemuck::cast_slice(&packed));
    let normals = smooth.then(|| vb("normals", bytemuck::cast_slice(doc.normals())));
    let (indices, index_count) = match old.filter(|o| o.topology_id == doc.topology_id()) {
        Some(o) => (o.indices, o.index_count),
        None => {
            let mut idx: Vec<u32> = Vec::with_capacity(doc.face_count() * 6);
            for f in doc.faces() {
                for t in sculpt_core::mesh::face_triangles(f) {
                    idx.extend_from_slice(&t);
                }
            }
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("indices"),
                contents: bytemuck::cast_slice(&idx),
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buffer), idx.len() as u32)
        }
    };
    MeshBuffers { topology_id: doc.topology_id(), vertices, normals, quant, indices, index_count }
}

/// Merge the owned vertex ranges of sorted leaves into contiguous runs.
fn runs(doc: &Document, leaves: &[u32]) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    for &l in leaves {
        let r = doc.bvh().leaves[l as usize].owned_range();
        match out.last_mut() {
            Some(last) if last.end == r.start => last.end = r.end,
            _ => out.push(r),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(q: &Quant, v: [u16; 4]) -> Vec3 {
        q.min + Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32) / 65535.0 * q.size
    }

    #[test]
    fn packed_positions_round_trip_within_one_step() {
        let bounds = Aabb { min: Vec3::new(-1.0, -0.5, -2.0), max: Vec3::new(1.0, 0.5, 2.0) };
        let q = Quant::fit(bounds);
        let positions: Vec<Vec3> = (0..10_000).map(|i| Vec3::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos() * 0.5, (i as f32 * 0.07).sin() * 2.0)).collect();
        let overlay: Vec<f32> = (0..positions.len()).map(|i| (i % 100) as f32 / 99.0).collect();
        let mut out = vec![[0u16; 4]; positions.len()];
        pack(&q, &positions, Some(&overlay), &mut out);
        let step = q.size / 65535.0;
        for ((p, o), w) in positions.iter().zip(&out).zip(&overlay) {
            let err = (decode(&q, *o) - *p).abs();
            assert!(err.cmple(step * 0.51).all(), "error {err} exceeds half a step {step}");
            assert!((o[3] as f32 / 65535.0 - w).abs() < 1e-4, "overlay value survives");
        }
        // Without an overlay the fourth component is zero.
        pack(&q, &positions, None, &mut out);
        assert!(out.iter().all(|v| v[3] == 0));
    }

    #[test]
    fn the_box_contains_its_surface_with_a_margin_and_notices_growth() {
        let bounds = Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) };
        let q = Quant::fit(bounds);
        assert!(q.contains(&bounds));
        let nudged = Aabb { min: Vec3::splat(-1.1), max: Vec3::splat(1.1) };
        assert!(q.contains(&nudged), "a small move stays inside the margin");
        let grown = Aabb { min: Vec3::splat(-1.5), max: Vec3::splat(1.5) };
        assert!(!q.contains(&grown), "growing past the margin forces a refit");
        // At 6.3M faces a triangle edge is about 1.4e-3 of a unit sphere; one step must be far finer.
        assert!(q.size.max_element() / 65535.0 < 1.4e-3 / 20.0);
    }

    #[test]
    fn positions_outside_the_box_clamp_instead_of_wrapping() {
        let q = Quant::fit(Aabb { min: Vec3::splat(-1.0), max: Vec3::splat(1.0) });
        let mut out = [[0u16; 4]; 2];
        pack(&q, &[Vec3::splat(50.0), Vec3::splat(-50.0)], None, &mut out);
        assert_eq!(out[0][..3], [65535; 3]);
        assert_eq!(out[1][..3], [0; 3]);
    }
}
