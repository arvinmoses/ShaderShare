//! The decimation preview on the GPU: a triangle mesh drawn in place of the volume's surface, lit by the viewport's
//! own clay shading (`lit` in `viewport.wgsl`), with an optional wireframe.
//!
//! The mesh lives in three storage buffers (positions, vertex normals, the triangles' corners) and is drawn without
//! an index buffer: vertex `i` is corner `i % 3` of triangle `i / 3`, pulled from the corner list, so each corner knows
//! its barycentric coordinate and the fragment shader draws the edges itself (no line pass, any size of mesh). The
//! pipelines share group 0 (the viewport's uniforms: camera, clay colour) with the volume's and bind the mesh at
//! group 2.
//!
//! A corner's top bit marks its triangle as bad (thin or knotted); those are tinted red.
//!
//! With corner UVs and a normal map (an object-space map baked by `decimate::Baker`), the shading normal comes from
//! the map instead of the mesh.

use eframe::egui_wgpu::wgpu;
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use sculpt_core::decimate::{Maps, TriMesh};

// As the viewport's render targets (`viewport.rs`).
const MSAA: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

const WGSL: &str = r#"
// ------------------------------------------------------------------ the decimation preview (decimate_draw.rs)

@group(2) @binding(0) var<storage, read> dm_pos: array<f32>;
@group(2) @binding(1) var<storage, read> dm_nrm: array<f32>;
@group(2) @binding(2) var<storage, read> dm_corner: array<u32>;
@group(2) @binding(3) var<storage, read> dm_uv: array<f32>;
@group(2) @binding(4) var dm_nmap: texture_2d<f32>;
@group(2) @binding(5) var dm_samp: sampler;

struct DecimOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) bary: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) bad: u32,
};

fn decim_vert(idx: u32) -> DecimOut {
    let raw = dm_corner[idx];
    let v = raw & 0x7fffffffu;
    let p = vec3<f32>(dm_pos[3u * v], dm_pos[3u * v + 1u], dm_pos[3u * v + 2u]);
    let n = vec3<f32>(dm_nrm[3u * v], dm_nrm[3u * v + 1u], dm_nrm[3u * v + 2u]);
    var o: DecimOut;
    o.clip = u.view_proj * vec4<f32>(p, 1.0);
    o.world = p;
    o.normal = n;
    let c = idx % 3u;
    o.bary = vec3<f32>(f32(c == 0u), f32(c == 1u), f32(c == 2u));
    o.uv = vec2<f32>(0.0);
    o.bad = raw >> 31u;
    return o;
}

@vertex
fn vs_decim(@builtin(vertex_index) idx: u32) -> DecimOut {
    return decim_vert(idx);
}

@vertex
fn vs_decim_nmap(@builtin(vertex_index) idx: u32) -> DecimOut {
    var o = decim_vert(idx);
    o.uv = vec2<f32>(dm_uv[2u * idx], dm_uv[2u * idx + 1u]);
    return o;
}

fn with_map(i: DecimOut) -> DecimOut {
    var o = i;
    o.normal = textureSample(dm_nmap, dm_samp, i.uv).xyz * 2.0 - 1.0;
    return o;
}

fn decim_lit(i: DecimOut) -> vec4<f32> {
    var m: MeshOut;
    m.clip = i.clip;
    m.world = i.world;
    m.normal = i.normal;
    m.overlay = 0.0;
    m.base = i.world;
    let l2 = dot(i.normal, i.normal);
    let n = select(vec3<f32>(0.0, 1.0, 0.0), i.normal * inverseSqrt(max(l2, 1e-30)), l2 > 1e-30);
    let c = lit(m, n, u.clay.rgb);
    return select(c, vec4<f32>(mix(c.rgb, vec3<f32>(0.95, 0.12, 0.08), 0.7), c.a), i.bad != 0u);
}

@fragment
fn fs_decim(i: DecimOut) -> @location(0) vec4<f32> {
    return decim_lit(i);
}

@fragment
fn fs_decim_nmap(i: DecimOut) -> @location(0) vec4<f32> {
    return decim_lit(with_map(i));
}

// Edges about a pixel wide, darker than the clay; faded out where triangles shrink to a few pixels, where the lines
// would only grey the whole surface.
fn wire_over(i: DecimOut, c: vec4<f32>) -> vec4<f32> {
    let w = max(fwidth(i.bary), vec3<f32>(1e-6));
    let a = smoothstep(w * 0.2, w * 1.1, i.bary);
    let edge = 1.0 - min(min(a.x, a.y), a.z);
    let dense = 1.0 - smoothstep(0.12, 0.45, max(w.x, max(w.y, w.z)));
    return vec4<f32>(mix(c.rgb, c.rgb * 0.28, edge * dense * 0.85), 1.0);
}

@fragment
fn fs_decim_wire(i: DecimOut) -> @location(0) vec4<f32> {
    return wire_over(i, decim_lit(i));
}

@fragment
fn fs_decim_nmap_wire(i: DecimOut) -> @location(0) vec4<f32> {
    return wire_over(i, decim_lit(with_map(i)));
}
"#;

/// The mesh's buffers on the GPU and how many triangles they hold.
struct MeshGpu {
    pos: wgpu::Buffer,
    nrm: wgpu::Buffer,
    corner: wgpu::Buffer,
    uv: wgpu::Buffer,
    group: wgpu::BindGroup,
    triangles: u32,
    /// The corner UVs hold this mesh's (else the normal map is not used).
    has_uv: bool,
}

pub struct DecimateDraw {
    solid: wgpu::RenderPipeline,
    wire_pipe: wgpu::RenderPipeline,
    nmap_solid: wgpu::RenderPipeline,
    nmap_wire: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    mesh: Option<MeshGpu>,
    /// The normal map (1x1 until one is sent) and its view and sampler.
    nmap: wgpu::Texture,
    nmap_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Shade from the normal map (when the mesh has UVs).
    pub use_nmap: bool,
    /// Draw the edges over the shading.
    pub wire: bool,
    /// Draw the mesh in place of the volume.
    pub shown: bool,
}

impl DecimateDraw {
    /// Pipelines on `device`; `view_bgl` is the viewport's group 0 (its uniforms).
    pub fn new(device: &wgpu::Device, view_bgl: &wgpu::BindGroupLayout) -> DecimateDraw {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decimate"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}\n{}", include_str!("viewport.wgsl"), sculpt_core::surface_noise::WGSL, WGSL).into()),
        });
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let frag = |binding: u32, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decimate mesh"),
            entries: &[
                ro(0),
                ro(1),
                ro(2),
                ro(3),
                frag(4, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }),
                frag(5, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("decimate"), bind_group_layouts: &[Some(view_bgl), None, Some(&bgl)], immediate_size: 0 });
        let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let pipe = |vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("decimate"),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &target }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let (nmap, nmap_view) = map_texture(device, 1);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("decimate normal map"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        DecimateDraw {
            solid: pipe("vs_decim", "fs_decim"),
            wire_pipe: pipe("vs_decim", "fs_decim_wire"),
            nmap_solid: pipe("vs_decim_nmap", "fs_decim_nmap"),
            nmap_wire: pipe("vs_decim_nmap", "fs_decim_nmap_wire"),
            bgl,
            mesh: None,
            nmap,
            nmap_view,
            sampler,
            use_nmap: false,
            wire: true,
            shown: false,
        }
    }

    /// Send a normal map up (made again when its size changes).
    pub fn upload_map(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, maps: &Maps) {
        let r = maps.res as u32;
        if self.nmap.width() != r {
            let (t, v) = map_texture(device, r);
            self.nmap = t;
            self.nmap_view = v;
            if let Some(m) = self.mesh.as_mut() {
                m.group = bind(device, &self.bgl, &m.pos, &m.nrm, &m.corner, &m.uv, &self.nmap_view, &self.sampler);
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.nmap, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&maps.normal),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * r), rows_per_image: Some(r) },
            wgpu::Extent3d { width: r, height: r, depth_or_array_layers: 1 },
        );
    }

    /// Send `mesh` (with its vertex normals, each triangle's corner UVs if any, and the triangles to tint as bad if
    /// any) up, in place of what was there. Buffers are kept when they are big enough.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &TriMesh, corner_uvs: Option<&[[Vec2; 3]]>, bad: Option<&[bool]>) {
        let normals = vertex_normals(mesh);
        let pos: &[f32] = bytemuck::cast_slice(&mesh.positions);
        let nrm: &[f32] = bytemuck::cast_slice(&normals);
        let flagged: Vec<[u32; 3]>;
        let corner: &[u32] = match bad {
            Some(b) if b.len() == mesh.triangles.len() && b.iter().any(|x| *x) => {
                flagged = mesh.triangles.par_iter().zip(b).map(|(t, &x)| if x { t.map(|v| v | 1 << 31) } else { *t }).collect();
                bytemuck::cast_slice(&flagged)
            }
            _ => bytemuck::cast_slice(&mesh.triangles),
        };
        let bytes = |n: usize| ((n.max(1) * 4) as u64).next_multiple_of(16);
        let uv: &[f32] = corner_uvs.map_or(&[], |u| bytemuck::cast_slice(u));
        let fits = self.mesh.as_ref().is_some_and(|m| m.pos.size() >= bytes(pos.len()) && m.corner.size() >= bytes(corner.len()) && m.uv.size() >= bytes(uv.len()));
        if !fits {
            // Room for a little more, so a count slider moving up does not make them again every frame.
            let make = |label, n: usize| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes(n + n / 4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let (p, n, c) = (make("decimate positions", pos.len()), make("decimate normals", nrm.len()), make("decimate corners", corner.len()));
            let uvb = make("decimate corner uvs", uv.len());
            let group = bind(device, &self.bgl, &p, &n, &c, &uvb, &self.nmap_view, &self.sampler);
            self.mesh = Some(MeshGpu { pos: p, nrm: n, corner: c, uv: uvb, group, triangles: 0, has_uv: false });
        }
        let m = self.mesh.as_mut().unwrap();
        // (Writes are padded to 4 bytes already: every element is.)
        if !pos.is_empty() {
            queue.write_buffer(&m.pos, 0, bytemuck::cast_slice(pos));
            queue.write_buffer(&m.nrm, 0, bytemuck::cast_slice(nrm));
        }
        if !corner.is_empty() {
            queue.write_buffer(&m.corner, 0, bytemuck::cast_slice(corner));
        }
        if !uv.is_empty() {
            queue.write_buffer(&m.uv, 0, bytemuck::cast_slice(uv));
        }
        m.has_uv = !uv.is_empty();
        m.triangles = mesh.triangles.len() as u32;
    }

    /// Free the mesh's buffers (the pipelines stay).
    pub fn clear(&mut self) {
        self.mesh = None;
        self.shown = false;
    }

    /// Whether there is a mesh to draw in place of the volume.
    pub fn showing(&self) -> bool {
        self.shown && self.mesh.is_some()
    }

    /// Draw into the viewport's pass (group 0 already set to the viewport's uniforms).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(m) = self.mesh.as_ref() else { return };
        if m.triangles == 0 {
            return;
        }
        let nmap = self.use_nmap && m.has_uv;
        pass.set_pipeline(match (nmap, self.wire) {
            (false, false) => &self.solid,
            (false, true) => &self.wire_pipe,
            (true, false) => &self.nmap_solid,
            (true, true) => &self.nmap_wire,
        });
        pass.set_bind_group(2, &m.group, &[]);
        pass.draw(0..m.triangles * 3, 0..1);
    }
}

fn map_texture(device: &wgpu::Device, r: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("decimate normal map"),
        size: wgpu::Extent3d { width: r, height: r, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}

#[allow(clippy::too_many_arguments)]
fn bind(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    p: &wgpu::Buffer,
    n: &wgpu::Buffer,
    c: &wgpu::Buffer,
    uv: &wgpu::Buffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("decimate mesh"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: n.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uv.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(sampler) },
        ],
    })
}

/// Area-weighted vertex normals.
pub fn vertex_normals(mesh: &TriMesh) -> Vec<Vec3> {
    let p = &mesh.positions;
    let face: Vec<Vec3> = mesh.triangles.par_iter().map(|t| (p[t[1] as usize] - p[t[0] as usize]).cross(p[t[2] as usize] - p[t[0] as usize])).collect();
    let mut n = vec![Vec3::ZERO; p.len()];
    for (t, f) in mesh.triangles.iter().zip(&face) {
        for &v in t {
            n[v as usize] += *f;
        }
    }
    n.par_iter_mut().for_each(|v| *v = v.normalize_or_zero());
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a headless device: the preview's pipelines build against the viewport's shader, and drawing a decimated
    /// ball in place of the (empty) volume puts it on screen, with the wireframe changing the image. Needs a GPU:
    /// `cargo test --release -p sculpt-app decimate_draw -- --ignored --nocapture` (through the GPU queue).
    #[test]
    #[ignore]
    fn preview_draws_on_a_headless_device() {
        use sculpt_core::decimate::{Analysis, AnalyzeOptions, Target};
        let (device, queue, adapter) = crate::gpu_levels::check::device().expect("a GPU");
        println!("adapter: {adapter}");
        let theme = crate::theme::ThemeLibrary::load(std::path::Path::new("no themes here")).themes[0].clone();
        let cam = crate::camera::Camera { distance: 3.4, yaw: 0.3, pitch: 0.2, ..Default::default() };
        let size = [320u32, 240];
        let mut mode = crate::volume_mode::VolumeMode::ball(2e5);
        let (input, _) = crate::decimate_ui::take_input(&mut mode).unwrap();
        let mesh = Analysis::build(&input, &AnalyzeOptions::default(), None).unwrap().extract(Target::KeepPercent(5.0)).mesh;

        let mut vp = crate::viewport::Viewport::with_device(&device, include_str!("viewport.wgsl"));
        let empty = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        let mut draw = DecimateDraw::new(&device, &vp.bgl);
        draw.upload(&device, &queue, &mesh, None, None);
        draw.shown = true;
        draw.wire = false;
        vp.decimate = Some(draw);
        let solid = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        vp.decimate.as_mut().unwrap().wire = true;
        let wire = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        // `DECIMATE_SHOT=out.png` saves the wireframe image to look at.
        if let Ok(p) = std::env::var("DECIMATE_SHOT") {
            let image = egui::ColorImage::from_rgba_unmultiplied([size[0] as usize, size[1] as usize], &wire);
            crate::app::write_png(std::path::Path::new(&p), &image, false).unwrap();
        }
        let differ = |a: &[u8], b: &[u8]| a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p != q).count();
        let px = (size[0] * size[1]) as usize;
        let (shown, edges) = (differ(&empty, &solid), differ(&solid, &wire));
        println!("{} triangles: {shown} of {px} pixels show the mesh, {edges} change with the wireframe", mesh.triangles.len());
        assert!(shown > px / 10, "the mesh barely shows: {shown} of {px} pixels");
        assert!(edges > px / 100 && edges < shown, "wireframe changed {edges} pixels of {shown}");
        // The output split along quick UV seams, shaded from a normal map baked off the input: on screen, and shaded
        // differently from the mesh's own normals.
        let surface = std::sync::Arc::new(sculpt_core::decimate::Surface::new(std::sync::Arc::new(input.clone())));
        let uv = sculpt_core::decimate::quick_uv(&mesh, 4.0 / 1024.0);
        let maps = sculpt_core::decimate::Baker::new(surface).bake(&uv, 256, 0.05, None).unwrap();
        let low = TriMesh { positions: uv.positions.clone(), triangles: uv.triangles.clone() };
        let corners: Vec<[Vec2; 3]> = uv.triangles.iter().map(|t| t.map(|v| uv.uvs[v as usize])).collect();
        {
            let d = vp.decimate.as_mut().unwrap();
            d.wire = false;
            d.upload(&device, &queue, &low, Some(&corners), None);
            d.upload_map(&device, &queue, &maps);
        }
        let plain = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        vp.decimate.as_mut().unwrap().use_nmap = true;
        let mapped = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        let (on, shading) = (differ(&empty, &mapped), differ(&plain, &mapped));
        println!("normal map: {on} pixels show the mesh, {shading} shaded differently from its own normals");
        assert!(on > px / 10 && shading > px / 50, "{on} / {shading}");
        if let Ok(p) = std::env::var("DECIMATE_SHOT") {
            let image = egui::ColorImage::from_rgba_unmultiplied([size[0] as usize, size[1] as usize], &mapped);
            crate::app::write_png(std::path::Path::new(&p.replace(".png", "_nmap.png")), &image, false).unwrap();
        }

        // Hidden again: the volume's place is empty as before.
        vp.decimate.as_mut().unwrap().clear();
        assert_eq!(differ(&empty, &vp.render_offscreen(&device, &queue, size, &cam, &theme)), 0);
    }
}
