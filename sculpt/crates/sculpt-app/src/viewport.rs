//! GPU viewport.
//!
//! The volume's surface is built on the GPU (`gpu_levels`) and drawn straight from its buffers: a page of quads per
//! indirect draw, for the clusters the cull pass keeps. The scene renders (4x MSAA) into an offscreen texture that egui
//! shows as an image, so panels and viewport share one device and one queue.

use std::time::Instant;

use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Vec3, Vec4};

use crate::camera::Camera;
use crate::theme::Theme;

const MSAA: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

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
    /// Unused by the volume (the shader's mesh entry points read them).
    box_min: [f32; 4],
    box_size: [f32; 4],
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
    resolve_tex: wgpu::Texture,
    depth: wgpu::TextureView,
}

pub struct Viewport {
    bg_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    targets: Option<Targets>,
    pub texture_id: Option<egui::TextureId>,
    pub last_upload: UploadStats,
    /// The volume's surface.
    volume: Option<VolumeGpu>,
    levels_pipeline: wgpu::RenderPipeline,
    levels_bgl: wgpu::BindGroupLayout,
    /// Hide clusters outside the view or facing away (the volume playground).
    pub cull: bool,
    /// What the last frame drew of the volume: clusters' pages drawn, of those holding surface.
    pub volume_draws: u32,
    /// Mesh layers are shown: every block at its finest level (no level by distance, no budget) and no cluster
    /// culling (the clusters' bounds and cones are of the surface without the layers).
    pub layers_shown: bool,
    /// Mesh layer rows kept up to date (see [`Self::sync_layer_rows`]); its stats say what the last frame did.
    pub layer_rows: crate::layer_rows::LayerRowsSync,
    /// How many volume surfaces were made on the GPU (each new one holds no rows).
    volumes_made: u64,
    /// The volume's vertices as drawn, for checks (made when first needed).
    capture: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    shader: wgpu::ShaderModule,
    pub(crate) bgl: wgpu::BindGroupLayout,
    /// The decimation preview, drawn in place of the volume while it shows (`decimate_ui`).
    pub decimate: Option<crate::decimate_draw::DecimateDraw>,
    /// A generator edit held as meshes, drawn over the volume (made when first needed).
    proxy: Option<crate::proxy_draw::ProxyDraw>,
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

/// The last GPU error not yet shown (see [`Viewport::new`]).
pub static GPU_ERROR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

impl Viewport {
    pub fn new(rs: &egui_wgpu::RenderState) -> Viewport {
        // A GPU error (a buffer past the device's limits, say) is noted and the app goes on: by default it ends the app.
        rs.device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            eprintln!("GPU error: {e}");
            if let Ok(mut last) = GPU_ERROR.lock() {
                *last = Some(e.to_string());
            }
        }));
        Self::with_device(&rs.device, include_str!("viewport.wgsl"))
    }

    /// A viewport on `device` drawing with the shader `wgsl` (the surface noise's appended), with no window.
    pub fn with_device(device: &wgpu::Device, wgsl: &str) -> Viewport {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", wgsl, sculpt_core::surface_noise::WGSL).into()),
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
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
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
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let ms = wgpu::MultisampleState { count: MSAA, ..Default::default() };
        // The volume's surface: vertices pulled from the GPU extractor's buffers.
        let vc = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE;
        let vfc = wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry { binding, visibility: vc, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let levels_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport levels"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: vfc, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ro(1),
                ro(2),
                ro(3),
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: vfc, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: vfc, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                ro(6),
                ro(7),
                ro(8),
                ro(9),
                ro(10),
                ro(11),
                ro(12),
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
            bg_pipeline,
            uniforms,
            bind_group,
            targets: None,
            volume: None,
            levels_pipeline,
            levels_bgl,
            cull: true,
            volume_draws: 0,
            layers_shown: false,
            layer_rows: Default::default(),
            volumes_made: 0,
            capture: None,
            shader,
            bgl,
            decimate: None,
            proxy: None,
            texture_id: None,
            last_upload: UploadStats::default(),
        }
    }

    fn ensure_targets(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2]) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let targets = Self::make_targets(&rs.device, size);
        let mut renderer = rs.renderer.write();
        match self.texture_id {
            Some(id) => renderer.update_egui_texture_from_wgpu_texture(&rs.device, &targets.resolve, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(renderer.register_native_texture(&rs.device, &targets.resolve, wgpu::FilterMode::Linear)),
        }
        self.targets = Some(targets);
    }

    fn make_targets(device: &wgpu::Device, size: [u32; 2]) -> Targets {
        let tex = |label, format, samples, usage| {
            device
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
        let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport color"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolve = resolve_tex.create_view(&Default::default());
        let depth = tex("viewport depth", DEPTH_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        Targets { size, msaa, resolve, resolve_tex, depth }
    }

    /// Bring the volume playground's surface up to date: send the blocks that changed and rebuild the surface around them
    /// on the GPU (everything, for a new volume or a new level cap). `clock` is the time in seconds (the wall clock if
    /// none). True while levels are still fading as they hand over, or a big change is still landing region by region
    /// (more frames are wanted).
    pub fn sync_volume(&mut self, rs: &egui_wgpu::RenderState, mode: &mut crate::volume_mode::VolumeMode, eye: glam::Vec3, focal_px: f32, clock: Option<f64>) -> bool {
        self.sync_volume_on(&rs.device, &rs.queue, mode, eye, focal_px, clock)
    }

    /// [`Self::sync_volume`] on a device and queue.
    pub fn sync_volume_on(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mode: &mut crate::volume_mode::VolumeMode, eye: glam::Vec3, focal_px: f32, clock: Option<f64>) -> bool {
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let now = clock.unwrap_or_else(|| START.get_or_init(Instant::now).elapsed().as_secs_f64());
        let t = Instant::now();
        if self.volume.as_ref().is_none_or(|v| v.generation != mode.generation) {
            let gpu = crate::gpu_levels::GpuLevels::new(&device, &queue, &mode.vol, mode.cap);
            let uniforms = device.create_buffer(&wgpu::BufferDescriptor { label: Some("volume levels uniforms"), size: std::mem::size_of::<LevelsUniforms>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let empty = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 256, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let (mask_hash, mask_samples) = (empty("noise mask hash"), empty("noise mask samples"));
            let group = Self::levels_group(&device, &self.levels_bgl, &uniforms, &gpu, &mask_hash, &mask_samples);
            self.volume = Some(VolumeGpu { buffers_made: gpu.buffers_made, gpu, uniforms, group, mask_hash, mask_samples, mask_of: None, mask_bits: 0, generation: mode.generation });
            self.volumes_made += 1;
        }
        let v = self.volume.as_mut().unwrap();
        v.gpu.set_cap(mode.cap);
        // A generator edit leaves no more blocks shown than the GPU can hold (it is refused with a note past that).
        if let Some(layer) = mode.layer.as_mut() {
            layer.block_budget = sculpt_core::generator::detail_layer::BLOCK_BUDGET.min(v.gpu.block_budget());
        }
        // With mesh layers shown every block is at its finest level: layers are rows on those blocks' vertices.
        v.gpu.view.on = mode.by_distance && !self.layers_shown;
        v.gpu.view.tau_px = mode.level_px;
        v.gpu.view.budget = (mode.budget_on && !self.layers_shown).then(|| (mode.budget_m as f64 * 1e6) as u64);
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
        if std::mem::take(&mut v.gpu.join_refused) {
            mode.generator_note = Some("Too big to preview on the GPU while dragging: it is shown once let go. Lower the size or the count.".into());
        }
        if let Some(e) = GPU_ERROR.lock().ok().and_then(|mut e| e.take()) {
            mode.generator_note = Some(format!("The GPU refused part of this edit, so some of it may not show: {}", e.lines().next().unwrap_or("")));
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
            v.mask_hash = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("noise mask hash"), contents: bytemuck::cast_slice(&hash), usage: st });
            v.mask_samples = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("noise mask samples"), contents: bytemuck::cast_slice(&samples), usage: st });
            v.mask_bits = if m.is_empty() { 0 } else { bits };
            v.mask_of = Some(*epoch);
            regroup = true;
        }
        if regroup {
            v.group = Self::levels_group(&device, &self.levels_bgl, &v.uniforms, &v.gpu, &v.mask_hash, &v.mask_samples);
            v.buffers_made = v.gpu.buffers_made;
        }
        let (noise_f, noise_i) = noise.as_ref().map_or(sculpt_core::surface_noise::SurfaceNoise { on: false, ..Default::default() }, |n| n.0).uniform();
        let fading = v.gpu.view.fading();
        let mask_i = [v.mask_bits, (v.mask_bits > 0 && noise.is_some()) as u32, v.gpu.fade_buffers().4, v.gpu.cap as u32];
        let misc = [v.gpu.voxel0(), if mode.tint { 1.0 } else { 0.0 }, if fading { 1.0 } else { 0.0 }, now as f32];
        let u = LevelsUniforms { origin: v.gpu.origin().extend(0.0).to_array(), misc, noise_f, noise_i, mask_i };
        queue.write_buffer(&v.uniforms, 0, bytemuck::bytes_of(&u));
        mode.blocks = v.gpu.slots();
        if s.uploaded > 0 || s.full {
            mode.sync = s;
        } else if mode.sync.pending > 0 {
            mode.sync.pending = s.pending;
        }
        let fading = fading || s.pending > 0;
        v.gpu.flush_rows();
        if v.gpu.buffers_made != v.buffers_made {
            v.group = Self::levels_group(device, &self.levels_bgl, &v.uniforms, &v.gpu, &v.mask_hash, &v.mask_samples);
            v.buffers_made = v.gpu.buffers_made;
        }
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
                wgpu::BindGroupEntry { binding: 11, resource: gpu.row_buffers().0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: gpu.row_buffers().1.as_entire_binding() },
            ],
        })
    }

    /// Drop the GPU's copy of the volume: the next sync builds it afresh (a new or opened volume).
    /// Show a generator edit held as meshes (see `VolumeMode::proxy`), or none.
    pub fn set_proxy(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, shown: Option<(&[sculpt_core::generator::Instance], bool, u64)>) {
        if shown.is_none() && self.proxy.is_none() {
            return;
        }
        let bgl = &self.bgl;
        self.proxy.get_or_insert_with(|| crate::proxy_draw::ProxyDraw::new(device, bgl)).set(device, queue, shown);
    }

    pub fn clear_volume(&mut self) {
        self.volume = None;
    }

    /// Mesh layers on the volume's surface, once a frame right after [`Self::sync_volume`]: `ml` the live mesh layers
    /// (never a run-ahead copy) and `dirty` what changed since the last call. Sets the rows of the blocks listed (every
    /// block's with `dirty.all`, on the first call that shows layers, or on a new GPU surface), drops those of blocks
    /// removed, and turns [`Self::layers_shown`] on with the first rows and off with `dirty.none` or no `ml`. A block
    /// the GPU extracted again for a volume edit after `ml`'s mesh was taken shows the plain volume until a later
    /// `dirty.blocks` lists it; one extracted for the view only shows its rows again by itself.
    /// The rows go up with the next draw or capture.
    pub fn sync_layer_rows(&mut self, ml: Option<&sculpt_core::mesh_layers::MeshLayers>, dirty: crate::layer_rows::RowsDirty) {
        match self.volume.as_mut() {
            Some(v) => self.layers_shown = self.layer_rows.sync(&mut v.gpu, self.volumes_made, ml, dirty, self.layers_shown),
            None => {
                if dirty.none || ml.is_none() {
                    self.layers_shown = false;
                } else {
                    self.layer_rows.want_all(&dirty);
                }
            }
        }
    }

    /// The volume's GPU surface (where mesh layer rows are set), once synced.
    pub fn levels(&mut self) -> Option<&mut crate::gpu_levels::GpuLevels> {
        self.volume.as_mut().map(|v| &mut v.gpu)
    }

    /// Rows set since the last sync sent, and the volume's bind group made again if a buffer was made afresh.
    fn flush_rows(&mut self, device: &wgpu::Device) {
        if let Some(v) = self.volume.as_mut() {
            v.gpu.flush_rows();
            if v.gpu.buffers_made != v.buffers_made {
                v.group = Self::levels_group(device, &self.levels_bgl, &v.uniforms, &v.gpu, &v.mask_hash, &v.mask_samples);
                v.buffers_made = v.gpu.buffers_made;
            }
        }
    }

    /// Render offscreen (no window) and read the image back: `size` pixels, RGBA8, rows top to bottom.
    pub fn render_offscreen(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: [u32; 2], cam: &Camera, theme: &Theme) -> Vec<u8> {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(Self::make_targets(device, size));
        }
        self.draw(device, queue, cam, theme, None);
        let t = self.targets.as_ref().unwrap();
        let row = (size[0] * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * size[1]) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.resolve_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size[1]) } },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = staging.slice(..).get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((size[0] * size[1] * 4) as usize);
        for y in 0..size[1] as usize {
            out.extend_from_slice(&data[y * row as usize..y * row as usize + size[0] as usize * 4]);
        }
        out
    }

    /// The vertices of `slots` as drawn (after rows and surface noise): per slot its vertices in its GPU vertex order,
    /// `counts[i]` of them, as (position, normal).
    pub fn capture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, slots: &[u32], counts: &[u32]) -> Vec<(Vec3, Vec3)> {
        self.flush_rows(device);
        let Some(v) = self.volume.as_ref() else { return Vec::new() };
        let (pipe, layout) = self.capture.get_or_insert_with(|| {
            let ro = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
            let rw = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("capture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: ro, count: None },
                    wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: rw, count: None },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("capture"), bind_group_layouts: &[Some(&self.bgl), Some(&self.levels_bgl), Some(&layout)], immediate_size: 0 });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("capture"), layout: Some(&pl), module: &self.shader, entry_point: Some("cs_capture"), compilation_options: Default::default(), cache: None });
            (pipe, layout)
        });
        let mut work: Vec<[u32; 2]> = Vec::with_capacity(slots.len());
        let mut total = 0u32;
        for (s, n) in slots.iter().zip(counts) {
            work.push([*s, total]);
            total += n;
        }
        use eframe::egui_wgpu::wgpu::util::DeviceExt;
        let work_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("capture work"), contents: bytemuck::cast_slice(&work), usage: wgpu::BufferUsages::STORAGE });
        let size = (total.max(1) as u64) * 32;
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: Some("capture out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("capture"), layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: work_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() }] });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &self.bind_group, &[]);
            cp.set_bind_group(1, &v.group, &[]);
            cp.set_bind_group(2, &group, &[]);
            let n = (work.len() as u32 * 2048).div_ceil(64);
            let x = n.clamp(1, 65535);
            cp.dispatch_workgroups(x, n.div_ceil(x).max(1), 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &staging, 0, size);
        queue.submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = staging.slice(..).get_mapped_range().expect("mapped");
        let f: &[[f32; 4]] = bytemuck::cast_slice(&data);
        (0..total as usize).map(|i| (Vec3::from_slice(&f[2 * i][..3]), Vec3::from_slice(&f[2 * i + 1][..3]))).collect()
    }

    /// Render into the offscreen texture. Returns the egui texture to display.
    pub fn render(&mut self, rs: &egui_wgpu::RenderState, size: [u32; 2], cam: &Camera, theme: &Theme, cursor: Option<(Vec3, f32, f32)>) -> Option<egui::TextureId> {
        let size = [size[0].max(1), size[1].max(1)];
        self.ensure_targets(rs, size);
        self.draw(&rs.device, &rs.queue, cam, theme, cursor);
        self.texture_id
    }

    /// Draw the scene into the targets.
    fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cam: &Camera, theme: &Theme, cursor: Option<(Vec3, f32, f32)>) {
        self.flush_rows(device);
        let size = self.targets.as_ref().unwrap().size;
        let vp = &theme.viewport;
        let clay = linear(vp.clay.0);
        let overlay = [0.0; 4];
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
            box_min: [0.0; 4],
            box_size: [1.0; 4],
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let targets = self.targets.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewport") });
        if let Some(v) = &self.volume {
            v.gpu.cull(&mut enc, view_proj, cam.eye(), self.cull && !self.layers_shown);
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
                    // (Marks where the surface lies inside held rocks that cut: see `proxy_draw`.)
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0), store: wgpu::StoreOp::Discard }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.bg_pipeline);
            pass.draw(0..3, 0..1);
            if let Some(d) = self.decimate.as_ref().filter(|d| d.showing()) {
                d.draw(&mut pass);
            } else if let Some(v) = &self.volume {
                let (pages, draws, status, _, _, _) = v.gpu.buffers_for_draw();
                pass.set_pipeline(&self.levels_pipeline);
                pass.set_bind_group(1, &v.group, &[]);
                pass.set_index_buffer(pages.slice(..), wgpu::IndexFormat::Uint32);
                if v.gpu.indirect_count {
                    pass.multi_draw_indexed_indirect_count(draws, 0, status, 8, v.gpu.page_capacity());
                } else {
                    pass.multi_draw_indexed_indirect(draws, 0, v.gpu.page_capacity());
                }
                if let Some(p) = self.proxy.as_ref().filter(|p| p.showing()) {
                    p.draw(&mut pass);
                }
            }
        }
        queue.submit([enc.finish()]);
        if let Some(v) = &self.volume {
            v.gpu.after_submit();
        }
    }
}
