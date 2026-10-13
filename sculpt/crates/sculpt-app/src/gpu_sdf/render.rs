//! Headless render of the GPU overlay: own device, offscreen colour + depth targets, PNG out.
//! The overlay buffers come straight from the extraction pass (no CPU copy); the untouched base mesh is
//! drawn from the document with every triangle inside an active brick removed (the "replace the
//! touched leaves" step of the proposed viewport integration).

use std::path::{Path, PathBuf};

use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use sculpt_core::Document;
use sculpt_core::generator::place::SurfaceSampler;
use sculpt_core::generator::voxel::MeshSdf;
use sculpt_core::mesh::face_triangles;

use super::bench::{make_doc, make_instances, mean_edge, ms};
use super::{Gpu, Kernels, MeshGpu, Op, Sdf, config_for, enumerate_bricks, pack_key};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const W: u32 = 2048; // rendered at 2x, box-filtered to 1024 x 768
const H: u32 = 1536;
const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Camera {
    pub center: Vec3,
    pub dir: Vec3,
    pub dist: f32,
    pub fov: f32,
}

pub struct Draw<'a> {
    pub verts: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub count: u32,
    pub color: [f32; 3],
}

/// Render the draws and write `path` (PNG). Returns the render time in ms.
pub fn render_png(gpu: &Gpu, draws: &[Draw], cam: &Camera, path: &Path) -> Res<f64> {
    let t = std::time::Instant::now();
    let dev = &gpu.device;
    let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("render.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()) });
    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let attrs = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 1 },
    ];
    let vb = wgpu::VertexBufferLayout { array_stride: 32, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attrs };
    let target = [Some(wgpu::ColorTargetState { format: COLOR, blend: None, write_mask: wgpu::ColorWrites::ALL })];
    let pipe = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("overlay"),
        layout: Some(&layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[Some(vb)] },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), compilation_options: Default::default(), targets: &target }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let eye = cam.center + cam.dir.normalize() * cam.dist;
    let view = glam::camera::rh::view::look_at_mat4(eye, cam.center, Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(cam.fov, W as f32 / H as f32, cam.dist * 0.02, cam.dist * 8.0);
    let vp = (proj * view).to_cols_array();
    let light = (view.inverse().transform_vector3(Vec3::new(-0.4, 0.6, 0.7))).normalize();
    let groups: Vec<(wgpu::Buffer, wgpu::BindGroup)> = draws
        .iter()
        .map(|d| {
            let mut words = [0f32; 24];
            words[..16].copy_from_slice(&vp);
            words[16..19].copy_from_slice(&d.color);
            words[20..23].copy_from_slice(&light.to_array());
            let b = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: 96, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            gpu.queue.write_buffer(&b, 0, bytemuck::cast_slice(&words));
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: b.as_entire_binding() }] });
            (b, bg)
        })
        .collect();

    let tex = |format, usage| {
        dev.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let color = tex(COLOR, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let depth = tex(DEPTH, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let (cv, dv) = (color.create_view(&Default::default()), depth.create_view(&Default::default()));
    let out = dev.create_buffer(&wgpu::BufferDescriptor { label: None, size: (W * H * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut enc = dev.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &cv,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.075, g: 0.08, b: 0.095, a: 1.0 }), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &dv,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipe);
        for (d, (_, bg)) in draws.iter().zip(&groups) {
            if d.count == 0 {
                continue;
            }
            pass.set_bind_group(0, bg, &[]);
            pass.set_vertex_buffer(0, d.verts.slice(..));
            pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..d.count, 0, 0..1);
        }
    }
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &out, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) } },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    out.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.wait();
    rx.recv()??;
    let data = out.slice(..).get_mapped_range()?.to_vec();
    out.unmap();

    // 2x2 box filter to RGB.
    let (w, h) = ((W / 2) as usize, (H / 2) as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut s = 0u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    s += data[((2 * y + dy) * W as usize + 2 * x + dx) * 4 + c] as u32;
                }
                rgb[(y * w + x) * 3 + c] = (s / 4) as u8;
            }
        }
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    Ok(ms(t))
}

/// The base mesh as render buffers, minus triangles whose centroid lies in an active brick.
fn base_buffers(gpu: &Gpu, doc: &Document, active: &[u32], origin: Vec3, voxel: f32) -> (wgpu::Buffer, wgpu::Buffer, u32, usize) {
    let (pos, nrm) = (doc.positions(), doc.normals());
    let mut v = Vec::with_capacity(pos.len() * 8);
    for (p, n) in pos.iter().zip(nrm) {
        v.extend_from_slice(&[p.x, p.y, p.z, 1.0, n.x, n.y, n.z, 0.0]);
    }
    let bsz = 8.0 * voxel;
    let (mut idx, mut skipped) = (Vec::new(), 0usize);
    for f in doc.faces() {
        for t in face_triangles(f) {
            let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
            let b = ((c - origin) / bsz).floor().as_ivec3();
            if active.binary_search(&pack_key([b.x, b.y, b.z])).is_ok() {
                skipped += 1;
            } else {
                idx.extend_from_slice(&t);
            }
        }
    }
    let vb = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (v.len() * 4) as u64, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    gpu.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&v));
    let ib = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (idx.len() * 4).max(4) as u64, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    gpu.queue.write_buffer(&ib, 0, bytemuck::cast_slice(&idx));
    (vb, ib, idx.len() as u32, skipped)
}

pub fn run(args: &[String]) -> Res<()> {
    let dir = PathBuf::from(args.first().map_or("gpu_sdf_png", |s| s.as_str()));
    std::fs::create_dir_all(&dir)?;
    let level: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(7);
    let gpu = Gpu::new()?;
    println!("adapter: {}", gpu.describe());
    let kernels = Kernels::new(&gpu);
    let doc = make_doc(level)?;
    let (pos, nrm, faces) = (doc.positions(), doc.normals(), doc.faces());
    let edge = mean_edge(&doc);
    let msdf = MeshSdf::build(doc.bvh(), pos, faces, doc.topology());
    let mesh = MeshGpu::upload(&gpu, &msdf.flatten(), pos, msdf.pseudo_normals(), 0);
    let sampler = SurfaceSampler::new(pos, faces, None);
    println!("{} faces, mean edge {:.5}", doc.face_count(), edge);

    struct Scene {
        name: &'static str,
        op: Op,
        vx: f32,
        n: usize,
        size: (f32, f32),
        seed: u64,
        close: bool,
        smooth: f32,
    }
    let scenes = [
        Scene { name: "overview_subtract_0.5e", op: Op::Subtract, vx: 0.5, n: 220, size: (6.0, 14.0), seed: 5, close: false, smooth: 0.0 },
        Scene { name: "close_subtract_0.5e", op: Op::Subtract, vx: 0.5, n: 220, size: (6.0, 14.0), seed: 5, close: true, smooth: 0.0 },
        Scene { name: "close_subtract_1e", op: Op::Subtract, vx: 1.0, n: 220, size: (6.0, 14.0), seed: 5, close: true, smooth: 0.0 },
        Scene { name: "close_union_0.5e", op: Op::Union, vx: 0.5, n: 90, size: (6.0, 14.0), seed: 9, close: true, smooth: 0.0 },
        Scene { name: "close_smooth_union_0.5e", op: Op::Union, vx: 0.5, n: 90, size: (6.0, 14.0), seed: 9, close: true, smooth: 3.0 },
        Scene { name: "close_smooth_subtract_0.5e", op: Op::Subtract, vx: 0.5, n: 220, size: (6.0, 14.0), seed: 5, close: true, smooth: 3.0 },
    ];
    for sc in &scenes {
        let samples = sampler.sample(pos, nrm, faces, None, sc.n, sc.seed);
        let insts = make_instances(&samples, edge, sc.size.0, sc.size.1, true);
        let voxel = sc.vx * edge;
        let origin = doc.bounds().min - Vec3::splat(4.0 * 8.0 * voxel);
        let en = enumerate_bricks(&insts, origin, voxel, 1);
        let mut cfg = config_for(&en, origin, voxel, sc.op, super::Keep::ALL, 16384);
        cfg.smooth_k = sc.smooth * voxel;
        let mut sdf = Sdf::new(&gpu, &kernels, &mesh, cfg, insts.len());
        sdf.upload_instances(&insts);
        let cold = sdf.cold_build();
        let fr = sdf.frame();
        let (bv, bi, bcount, skipped) = base_buffers(&gpu, &doc, &en.keys, origin, voxel);
        let cam = if sc.close {
            let c = insts[0].translation;
            Camera { center: c, dir: c.normalize() + Vec3::new(0.35, 0.3, 0.0), dist: 0.55, fov: 0.7 }
        } else {
            Camera { center: Vec3::ZERO, dir: Vec3::new(0.45, 0.35, 1.0), dist: 3.4, fov: 0.7 }
        };
        let draws = [
            Draw { verts: &bv, indices: &bi, count: bcount, color: [0.62, 0.64, 0.68] },
            Draw { verts: &sdf.verts, indices: &sdf.indices, count: sdf.last_indices, color: [0.95, 0.55, 0.28] },
        ];
        let path = dir.join(format!("{}.png", sc.name));
        let t = render_png(&gpu, &draws, &cam, &path)?;
        println!(
            "{}: {} instances, {} bricks, cold {:.0} ms, frame eval {:.1} ms, overlay {} verts / {} tris, {} base triangles replaced; render {:.0} ms -> {}",
            sc.name,
            insts.len(),
            fr.active_bricks,
            cold.ms_enum + cold.ms_assign + cold.ms_classify + cold.ms_fill,
            fr.ms_eval(),
            fr.verts,
            fr.indices / 3,
            skipped,
            t,
            path.display()
        );
    }
    Ok(())
}
