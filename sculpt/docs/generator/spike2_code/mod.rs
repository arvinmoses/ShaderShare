//! SPIKE 2: render-time CSG preview for the generator Boolean layer.
//!
//! The base mesh is never modified. Per pixel *sample* the shader combines
//! the mesh's first front surface and exit surface (two depth+normal
//! prepasses) with exact ray/primitive intervals found through a uniform
//! world-space grid of instances (storage buffers). Edges are therefore as
//! crisp as the analytic primitives, independent of mesh density.
//!
//! Enabled only by `--csg-preview`; the default render path is untouched.

use std::time::Instant;

use eframe::egui_wgpu::{self, wgpu};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

const MSAA: u32 = 4;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Subtract,
    Add,
    Intersect,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "subtract" | "sub" => Some(Mode::Subtract),
            "add" | "union" => Some(Mode::Add),
            "intersect" => Some(Mode::Intersect),
            _ => None,
        }
    }
}

/// Command-line configuration of the preview.
#[derive(Clone, Debug)]
pub struct CsgOptions {
    pub mode: Mode,
    /// Number of instances to place.
    pub count: u32,
    /// Instance size as a fraction of the mean spacing between neighbours
    /// (0.5 = neighbours just touch).
    pub size: f32,
    pub seed: u32,
    /// Boxes instead of spheres.
    pub boxes: bool,
    /// Show instance tests per sample as a heat map instead of shading.
    pub heat: bool,
    /// Wait for the GPU every frame and print timing on exit.
    pub bench: bool,
}

impl Default for CsgOptions {
    fn default() -> Self {
        CsgOptions { mode: Mode::Subtract, count: 60, size: 0.7, seed: 1, boxes: false, heat: false, bench: false }
    }
}

/// One analytic primitive instance (std430, 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Inst {
    /// xyz centre, w = kind (0 sphere, 1 box).
    pub c: [f32; 4],
    /// Sphere: x = radius. Box: xyz half extents.
    pub h: [f32; 4],
    /// Rotation quaternion (xyzw).
    pub q: [f32; 4],
}

impl Inst {
    fn bounding_radius(&self) -> f32 {
        if self.c[3] < 0.5 { self.h[0] } else { Vec3::new(self.h[0], self.h[1], self.h[2]).length() }
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}

fn rnd(seed: u32, i: u32, k: u32) -> f32 {
    hash(seed.wrapping_mul(0x9e3779b9) ^ hash(i.wrapping_mul(0x85ebca6b) ^ k)) as f32 / u32::MAX as f32
}

/// Deterministic test placement: a jittered Fibonacci lattice over the sphere
/// of radius `radius` about `centre`.
///
/// NOTE: the generator module described in the task brief (`sculpt-core`
/// `generator/`, SDF primitives, instance placement) does not exist in this
/// repository yet, so this stands in for "N instances from the core
/// placement". The instance record (`Inst`) is what a real placement would
/// upload.
pub fn place(opts: &CsgOptions, centre: Vec3, radius: f32) -> Vec<Inst> {
    let n = opts.count.max(1);
    let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt() * radius;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
        let rr = (1.0 - y * y).max(0.0).sqrt();
        let phi = i as f32 * golden;
        let mut dir = Vec3::new(phi.cos() * rr, y, phi.sin() * rr);
        let jitter = Vec3::new(rnd(opts.seed, i, 1), rnd(opts.seed, i, 2), rnd(opts.seed, i, 3)) - Vec3::splat(0.5);
        dir = (dir + jitter * (spacing / radius) * 0.9).normalize();
        let scale = 0.65 + 0.7 * rnd(opts.seed, i, 4);
        let r = opts.size * spacing * scale;
        // Where the primitive sits relative to the surface.
        let lift = match opts.mode {
            Mode::Subtract => 0.35 * r,
            Mode::Add => -0.2 * r,
            Mode::Intersect => 0.0,
        };
        let c = centre + dir * (radius + lift);
        if opts.boxes || opts.mode == Mode::Add {
            let q = Quat::from_axis_angle(
                Vec3::new(rnd(opts.seed, i, 5) - 0.5, rnd(opts.seed, i, 6) - 0.5, rnd(opts.seed, i, 7) - 0.5).normalize_or(Vec3::Y),
                rnd(opts.seed, i, 8) * std::f32::consts::TAU,
            );
            let hx = r * (0.7 + 0.5 * rnd(opts.seed, i, 9));
            let hy = r * (0.7 + 0.5 * rnd(opts.seed, i, 10));
            let hz = r * (0.7 + 0.5 * rnd(opts.seed, i, 11));
            out.push(Inst { c: [c.x, c.y, c.z, 1.0], h: [hx, hy, hz, 0.0], q: q.to_array() });
        } else {
            out.push(Inst { c: [c.x, c.y, c.z, 0.0], h: [r, 0.0, 0.0, 0.0], q: [0.0, 0.0, 0.0, 1.0] });
        }
    }
    out
}

/// Uniform world-space grid. `cells[c] = (offset, count)` into `indices`;
/// an instance is listed in every cell its bounding sphere overlaps.
pub struct Grid {
    pub min: Vec3,
    pub cell: f32,
    pub dim: [u32; 3],
    pub cells: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
}

impl Grid {
    pub fn build(insts: &[Inst]) -> Grid {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        let mut rsum = 0.0;
        for i in insts {
            let c = Vec3::new(i.c[0], i.c[1], i.c[2]);
            let r = i.bounding_radius();
            lo = lo.min(c - Vec3::splat(r));
            hi = hi.max(c + Vec3::splat(r));
            rsum += r;
        }
        if insts.is_empty() {
            lo = Vec3::splat(-1.0);
            hi = Vec3::splat(1.0);
        }
        let rmean = rsum / insts.len().max(1) as f32;
        let ext = (hi - lo).max_element().max(1e-4);
        // Cells about as large as an instance, but never more than ~128 per axis.
        let cell = (ext / 128.0).max(rmean * 2.0).max(1e-5);
        let min = lo - Vec3::splat(cell * 0.01);
        let dim = ((hi - min) / cell).ceil().max(Vec3::ONE).as_uvec3();
        let dim = [dim.x.max(1), dim.y.max(1), dim.z.max(1)];
        let ncell = (dim[0] * dim[1] * dim[2]) as usize;
        let range = |i: &Inst| {
            let c = Vec3::new(i.c[0], i.c[1], i.c[2]);
            let r = Vec3::splat(i.bounding_radius());
            let a = ((c - r - min) / cell).floor().max(Vec3::ZERO).as_uvec3();
            let b = ((c + r - min) / cell).floor().max(Vec3::ZERO).as_uvec3();
            (a, b.min(glam::UVec3::new(dim[0] - 1, dim[1] - 1, dim[2] - 1)))
        };
        let cid = |x: u32, y: u32, z: u32| ((z * dim[1] + y) * dim[0] + x) as usize;
        let mut counts = vec![0u32; ncell];
        for i in insts {
            let (a, b) = range(i);
            for z in a.z..=b.z {
                for y in a.y..=b.y {
                    for x in a.x..=b.x {
                        counts[cid(x, y, z)] += 1;
                    }
                }
            }
        }
        let mut cells = vec![[0u32; 2]; ncell];
        let mut total = 0u32;
        for (c, &n) in cells.iter_mut().zip(&counts) {
            *c = [total, n];
            total += n;
        }
        let mut fill = vec![0u32; ncell];
        let mut indices = vec![0u32; total.max(1) as usize];
        for (k, i) in insts.iter().enumerate() {
            let (a, b) = range(i);
            for z in a.z..=b.z {
                for y in a.y..=b.y {
                    for x in a.x..=b.x {
                        let c = cid(x, y, z);
                        indices[(cells[c][0] + fill[c]) as usize] = k as u32;
                        fill[c] += 1;
                    }
                }
            }
        }
        Grid { min, cell, dim, cells, indices }
    }

    pub fn stats(&self) -> String {
        let used = self.cells.iter().filter(|c| c[1] > 0).count();
        let max = self.cells.iter().map(|c| c[1]).max().unwrap_or(0);
        format!(
            "grid {}x{}x{} cell {:.4}, {} non-empty cells, {:.1} inst/cell avg (max {}), {} index entries",
            self.dim[0],
            self.dim[1],
            self.dim[2],
            self.cell,
            used,
            self.indices.len() as f32 / used.max(1) as f32,
            max,
            self.indices.len()
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CsgUniforms {
    inv_view_proj: [[f32; 4]; 4],
    size: [f32; 4],
    grid_min: [f32; 4],
    grid_dim: [u32; 4],
    tint: [f32; 4],
}

struct Targets {
    size: [u32; 2],
    depth_f: wgpu::TextureView,
    depth_b: wgpu::TextureView,
    nrm_f: wgpu::TextureView,
    nrm_b: wgpu::TextureView,
    bind_main: Option<wgpu::BindGroup>,
}

struct SceneBuffers {
    insts: wgpu::Buffer,
    cells: wgpu::Buffer,
    indices: wgpu::Buffer,
    grid_min: [f32; 4],
    grid_dim: [u32; 4],
}

/// Borrowed GPU mesh buffers owned by the regular viewport.
pub struct MeshRefs<'a> {
    pub positions: &'a wgpu::Buffer,
    pub normals: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub index_count: u32,
}

pub struct CsgPreview {
    pub opts: CsgOptions,
    pre_front: wgpu::RenderPipeline,
    pre_back: wgpu::RenderPipeline,
    full: wgpu::RenderPipeline,
    bgl_main: wgpu::BindGroupLayout,
    bind_pre: wgpu::BindGroup,
    uniforms_buf: wgpu::Buffer,
    viewport_uniforms: wgpu::Buffer,
    targets: Option<Targets>,
    scene: Option<SceneBuffers>,
    frame_ms: Vec<f32>,
    pub grid_info: String,
}

impl CsgPreview {
    pub fn new(rs: &egui_wgpu::RenderState, opts: CsgOptions, viewport_uniforms: &wgpu::Buffer, color_format: wgpu::TextureFormat) -> CsgPreview {
        let device = &rs.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport csg"),
            source: wgpu::ShaderSource::Wgsl(include_str!("csg.wgsl").into()),
        });
        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("csg uniforms"),
            size: std::mem::size_of::<CsgUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ubo = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let tex = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: true },
            count: None,
        };
        let float = wgpu::TextureSampleType::Float { filterable: false };
        let bgl_pre = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("csg pre"), entries: &[ubo(0), ubo(1)] });
        let bgl_main = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("csg main"),
            entries: &[
                ubo(0),
                ubo(1),
                storage(2),
                storage(3),
                storage(4),
                tex(5, wgpu::TextureSampleType::Depth),
                tex(6, wgpu::TextureSampleType::Depth),
                tex(7, float),
                tex(8, float),
            ],
        });
        let bind_pre = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csg pre"),
            layout: &bgl_pre,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: viewport_uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniforms_buf.as_entire_binding() },
            ],
        });
        let layout_pre = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg pre"), bind_group_layouts: &[Some(&bgl_pre)], immediate_size: 0 });
        let layout_main = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("csg main"), bind_group_layouts: &[Some(&bgl_main)], immediate_size: 0 });
        let attr = |loc: u32| wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: Box::leak(Box::new([wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: loc }])),
        };
        let buffers = [Some(attr(0)), Some(attr(1))];
        let ms = wgpu::MultisampleState { count: MSAA, ..Default::default() };
        let pre = |label, entry| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout_pre),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_pre"), compilation_options: Default::default(), buffers: &buffers },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format: NORMAL_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
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
        let pre_front = pre("csg front", "fs_front");
        let pre_back = pre("csg back", "fs_back");
        let full = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("csg resolve"),
            layout: Some(&layout_main),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_csg"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            // Must match the main pass attachment; the CSG pass neither tests nor writes depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ms,
            multiview_mask: None,
            cache: None,
        });
        CsgPreview {
            opts,
            pre_front,
            pre_back,
            full,
            bgl_main,
            bind_pre,
            uniforms_buf,
            viewport_uniforms: viewport_uniforms.clone(),
            targets: None,
            scene: None,
            frame_ms: Vec::new(),
            grid_info: String::new(),
        }
    }

    /// (Re)build the instance set and grid for a mesh occupying a sphere-ish
    /// region. Called when the document's topology changes.
    pub fn set_scene(&mut self, device: &wgpu::Device, centre: Vec3, radius: f32) {
        let t = Instant::now();
        let insts = place(&self.opts, centre, radius);
        let tp = t.elapsed().as_secs_f32() * 1e3;
        let grid = Grid::build(&insts);
        let tg = t.elapsed().as_secs_f32() * 1e3 - tp;
        self.grid_info = format!("{} instances placed in {tp:.1} ms; {} built in {tg:.1} ms", insts.len(), grid.stats());
        println!("csg: {}", self.grid_info);
        let mk = |label, data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE })
        };
        self.scene = Some(SceneBuffers {
            insts: mk("csg instances", bytemuck::cast_slice(&insts)),
            cells: mk("csg cells", bytemuck::cast_slice(&grid.cells)),
            indices: mk("csg cell indices", bytemuck::cast_slice(&grid.indices)),
            grid_min: [grid.min.x, grid.min.y, grid.min.z, grid.cell],
            grid_dim: [grid.dim[0], grid.dim[1], grid.dim[2], insts.len() as u32],
        });
        if let Some(t) = &mut self.targets {
            t.bind_main = None;
        }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let tex = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: MSAA,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        self.targets = Some(Targets {
            size,
            depth_f: tex("csg depth front", DEPTH_FORMAT),
            depth_b: tex("csg depth back", DEPTH_FORMAT),
            nrm_f: tex("csg normal front", NORMAL_FORMAT),
            nrm_b: tex("csg normal back", NORMAL_FORMAT),
            bind_main: None,
        });
    }

    /// Rasterise the front/back prepasses into `enc` and prepare the resolve
    /// bind group. Returns false if there is nothing to draw yet.
    pub fn prepare(&mut self, rs: &egui_wgpu::RenderState, enc: &mut wgpu::CommandEncoder, size: [u32; 2], view_proj: Mat4, mesh: &MeshRefs) -> bool {
        let device = &rs.device;
        self.ensure_targets(device, size);
        let Some(scene) = &self.scene else { return false };
        let tint = match self.opts.mode {
            Mode::Subtract => [0.95, 0.55, 0.30],
            Mode::Add => [0.35, 0.60, 0.95],
            Mode::Intersect => [0.45, 0.85, 0.50],
        };
        let u = CsgUniforms {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            size: [size[0] as f32, size[1] as f32, self.opts.mode as u32 as f32, if self.opts.heat { 1.0 } else { 0.0 }],
            grid_min: scene.grid_min,
            grid_dim: scene.grid_dim,
            tint: [tint[0], tint[1], tint[2], 1.0],
        };
        // The srgb-ish conversion of the tint happens in the viewport's gamma step; keep linear.
        rs.queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));
        let t = self.targets.as_mut().unwrap();
        if t.bind_main.is_none() {
            let e = |binding, resource| wgpu::BindGroupEntry { binding, resource };
            t.bind_main = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("csg main"),
                layout: &self.bgl_main,
                entries: &[
                    e(0, self.viewport_uniforms.as_entire_binding()),
                    e(1, self.uniforms_buf.as_entire_binding()),
                    e(2, scene.insts.as_entire_binding()),
                    e(3, scene.cells.as_entire_binding()),
                    e(4, scene.indices.as_entire_binding()),
                    e(5, wgpu::BindingResource::TextureView(&t.depth_f)),
                    e(6, wgpu::BindingResource::TextureView(&t.depth_b)),
                    e(7, wgpu::BindingResource::TextureView(&t.nrm_f)),
                    e(8, wgpu::BindingResource::TextureView(&t.nrm_b)),
                ],
            }));
        }
        for (pipeline, depth, nrm, label) in [(&self.pre_front, &t.depth_f, &t.nrm_f, "csg front"), (&self.pre_back, &t.depth_b, &t.nrm_b, "csg back")] {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: nrm,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_pre, &[]);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, mesh.positions.slice(..));
            pass.set_vertex_buffer(1, mesh.normals.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        true
    }

    /// Draw the per-sample CSG resolve inside the viewport's main pass.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if let Some(bg) = self.targets.as_ref().and_then(|t| t.bind_main.as_ref()) {
            pass.set_pipeline(&self.full);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Record one frame's wall time (only meaningful after a GPU wait).
    pub fn record_frame(&mut self, ms: f32) {
        self.frame_ms.push(ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_is_deterministic_and_counted() {
        let o = CsgOptions { count: 500, ..Default::default() };
        let a = place(&o, Vec3::ZERO, 1.0);
        let b = place(&o, Vec3::ZERO, 1.0);
        assert_eq!(a.len(), 500);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a), bytemuck::cast_slice::<_, u8>(&b));
    }

    #[test]
    fn every_instance_is_listed_in_the_cell_of_its_centre() {
        for mode in [Mode::Subtract, Mode::Add, Mode::Intersect] {
            let o = CsgOptions { count: 2000, mode, ..Default::default() };
            let insts = place(&o, Vec3::new(0.3, -0.2, 0.1), 1.5);
            let g = Grid::build(&insts);
            for (k, i) in insts.iter().enumerate() {
                let c = Vec3::new(i.c[0], i.c[1], i.c[2]);
                let cell = ((c - g.min) / g.cell).floor().as_uvec3();
                let id = ((cell.z * g.dim[1] + cell.y) * g.dim[0] + cell.x) as usize;
                let [off, n] = g.cells[id];
                assert!(g.indices[off as usize..(off + n) as usize].contains(&(k as u32)), "instance {k} missing from its centre cell");
            }
        }
    }
}

impl Drop for CsgPreview {
    fn drop(&mut self) {
        if !self.opts.bench || self.frame_ms.len() < 4 {
            return;
        }
        // Skip the first frames (shader/pipeline warm-up, mesh upload).
        let mut v: Vec<f32> = self.frame_ms.iter().skip(self.frame_ms.len() / 4).copied().collect();
        v.sort_by(f32::total_cmp);
        let med = v[v.len() / 2];
        println!(
            "CSG_BENCH mode={:?} count={} boxes={} frames={} gpu_wall_ms median={:.1} min={:.1} max={:.1} | {}",
            self.opts.mode,
            self.opts.count,
            self.opts.boxes,
            v.len(),
            med,
            v[0],
            v[v.len() - 1],
            self.grid_info
        );
    }
}
