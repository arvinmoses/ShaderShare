//! GPU viewport.
//!
//! Vertex data lives in three GPU buffers (positions, normals, overlay
//! scalar) laid out in the engine's internal vertex order. Because every
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
use sculpt_core::bvh::Bvh;
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

struct MeshBuffers {
    topology_id: u64,
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    overlay: wgpu::Buffer,
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
    mesh_pipeline: wgpu::RenderPipeline,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("viewport.wgsl").into()),
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
        let buffers = [Some(attr(0, wgpu::VertexFormat::Float32x3, 12)), Some(attr(1, wgpu::VertexFormat::Float32x3, 12)), Some(attr(2, wgpu::VertexFormat::Float32, 4))];
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let ms = wgpu::MultisampleState { count: MSAA, ..Default::default() };
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_mesh"), compilation_options: Default::default(), buffers: &buffers },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_mesh"), compilation_options: Default::default(), targets: &target }),
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
            mesh_pipeline,
            bg_pipeline,
            uniforms,
            bind_group,
            targets: None,
            mesh: None,
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

    /// Bring GPU buffers up to date with the document. Cheap when nothing changed.
    pub fn sync(&mut self, rs: &egui_wgpu::RenderState, doc: &mut Document, overlay: &OverlayKind, custom: Option<&[f32]>, custom_changed: bool) {
        let t = Instant::now();
        let mut stats = UploadStats::default();
        let queue = &rs.queue;
        if self.mesh.as_ref().is_none_or(|m| m.topology_id != doc.topology_id()) {
            self.mesh = Some(create_mesh_buffers(&rs.device, doc));
            self.overlay_kind = OverlayKind::None;
            doc.take_geometry_dirty();
            doc.take_scalar_dirty();
            stats.bytes = doc.vertex_count() * 24 + doc.face_count() * 24;
            stats.ranges = 1;
            self.write_overlay(queue, doc, overlay, custom, None, &mut stats);
            self.overlay_kind = overlay.clone();
            stats.ms = t.elapsed().as_secs_f32() * 1e3;
            self.last_upload = stats;
            return;
        }
        let mesh = self.mesh.as_ref().unwrap();
        match doc.take_geometry_dirty() {
            DirtySet::None => {}
            DirtySet::All => {
                queue.write_buffer(&mesh.positions, 0, bytemuck::cast_slice(doc.positions()));
                queue.write_buffer(&mesh.normals, 0, bytemuck::cast_slice(doc.normals()));
                stats.bytes += doc.vertex_count() * 24;
                stats.ranges += 1;
            }
            DirtySet::Leaves(leaves) => {
                for r in runs(doc, &leaves) {
                    queue.write_buffer(&mesh.positions, r.start as u64 * 12, bytemuck::cast_slice(&doc.positions()[r.clone()]));
                    queue.write_buffer(&mesh.normals, r.start as u64 * 12, bytemuck::cast_slice(&doc.normals()[r.clone()]));
                    stats.bytes += r.len() * 24;
                    stats.ranges += 1;
                }
            }
        }
        if self.lod.as_ref().is_some_and(|l| l.topology_id == doc.topology_id()) {
            for (first, count) in doc.take_lod_updates() {
                if let (Some(l), Some(tree)) = (&self.lod, doc.lod()) {
                    let r = first as usize..(first + count) as usize;
                    queue.write_buffer(&l.pool, first as u64 * 4, bytemuck::cast_slice(&tree.indices[r]));
                    stats.bytes += count as usize * 4;
                    stats.ranges += 1;
                }
            }
        }
        let scalar = doc.take_scalar_dirty();
        let kind_changed = *overlay != self.overlay_kind;
        let partial = match (&scalar, overlay) {
            (_, OverlayKind::None) => None,
            (_, OverlayKind::Custom) => custom_changed.then_some(DirtySet::All),
            (DirtySet::None, _) => None,
            (s, _) => Some(s.clone()),
        };
        if kind_changed {
            self.write_overlay(queue, doc, overlay, custom, None, &mut stats);
            self.overlay_kind = overlay.clone();
        } else if let Some(set) = partial {
            self.write_overlay(queue, doc, overlay, custom, Some(set), &mut stats);
        }
        stats.ms = t.elapsed().as_secs_f32() * 1e3;
        self.last_upload = stats;
    }

    fn write_overlay(&self, queue: &wgpu::Queue, doc: &Document, kind: &OverlayKind, custom: Option<&[f32]>, set: Option<DirtySet>, stats: &mut UploadStats) {
        let Some(mesh) = &self.mesh else { return };
        let active_mask = doc.active_layer().and_then(|id| doc.layer(id)).and_then(|l| l.mask_values());
        let src: Option<&[f32]> = match kind {
            OverlayKind::None => None,
            OverlayKind::Freeze => Some(doc.freeze()),
            OverlayKind::LayerMask => active_mask,
            OverlayKind::Channel(n) => doc.channel(n),
            OverlayKind::Custom => custom,
        };
        let Some(src) = src.filter(|s| s.len() == doc.vertex_count()) else {
            // Nothing to show: clear once.
            let zeros = vec![0f32; doc.vertex_count()];
            queue.write_buffer(&mesh.overlay, 0, bytemuck::cast_slice(&zeros));
            stats.bytes += zeros.len() * 4;
            return;
        };
        match set {
            Some(DirtySet::Leaves(leaves)) => {
                for r in runs(doc, &leaves) {
                    queue.write_buffer(&mesh.overlay, r.start as u64 * 4, bytemuck::cast_slice(&src[r.clone()]));
                    stats.bytes += r.len() * 4;
                    stats.ranges += 1;
                }
            }
            Some(DirtySet::None) => {}
            _ => {
                queue.write_buffer(&mesh.overlay, 0, bytemuck::cast_slice(src));
                stats.bytes += src.len() * 4;
                stats.ranges += 1;
            }
        }
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
        };
        rs.queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        let draws = self.prepare_lod(rs, lod_in, cam, size, view_proj);
        let targets = self.targets.as_ref().unwrap();
        let mut enc = rs.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("viewport") });
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
            if let Some(m) = &self.mesh {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_vertex_buffer(0, m.positions.slice(..));
                pass.set_vertex_buffer(1, m.normals.slice(..));
                pass.set_vertex_buffer(2, m.overlay.slice(..));
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
        if self.wait_gpu {
            let _ = rs.device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.texture_id
    }
}

fn create_mesh_buffers(device: &wgpu::Device, doc: &Document) -> MeshBuffers {
    use wgpu::util::DeviceExt;
    let vb = |label, data: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    };
    let positions = vb("positions", bytemuck::cast_slice(doc.positions()));
    let normals = vb("normals", bytemuck::cast_slice(doc.normals()));
    let overlay = vb("overlay", bytemuck::cast_slice(&vec![0f32; doc.vertex_count()]));
    let mut idx: Vec<u32> = Vec::with_capacity(doc.face_count() * 6);
    for f in doc.faces() {
        for t in sculpt_core::mesh::face_triangles(f) {
            idx.extend_from_slice(&t);
        }
    }
    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indices"),
        contents: bytemuck::cast_slice(&idx),
        usage: wgpu::BufferUsages::INDEX,
    });
    MeshBuffers { topology_id: doc.topology_id(), positions, normals, overlay, indices: Some(indices), index_count: idx.len() as u32 }
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
