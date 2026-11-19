//! A generator edit held as plain meshes (see `VolumeMode::hold_generators`): each instance drawn as its primitive at a
//! fixed tessellation, so an edit of any size or count costs a frame about the same. No blend or roughness: those come
//! with the volume, once the user leaves the layer.
//!
//! The primitives' unit meshes live end to end in one storage buffer (position and normal per vertex, no index buffer);
//! the instances, sorted by primitive, in another. One instanced draw per primitive present. The pipeline shares
//! group 0 (the viewport's uniforms) with the volume's, and is lit by the same `lit`, tinted so it reads as a preview.
//!
//! Rocks that cut are drawn as the holes they make, in two draws after the volume's: their faces in front of the
//! surface count in the stencil (up where a ray goes into one, down where it comes out), so a pixel whose surface lies
//! inside a rock is left above zero; there, the rock's far side behind the surface is drawn, as the hole's wall. Exact
//! for rocks that are convex; the parts of rocks out in the air show nothing, as they cut nothing. Where rocks overlap,
//! the part of one's wall inside another is cut away too (each rock tests the few it overlaps), so the wall shown is
//! where the cut ends.

use eframe::egui_wgpu::wgpu;
use glam::{Mat3, Vec2, Vec3};
use sculpt_core::generator::{Instance, Primitive};
use sculpt_core::generator::sdf::TORUS_MINOR;

// As the viewport's render targets (`viewport.rs`).
const MSAA: u32 = 4;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24PlusStencil8;

const WGSL: &str = r#"
// ------------------------------------------------------------------ a generator held as meshes (proxy_draw.rs)

struct ProxyInst {
    // The columns of rotation times scale, with the translation in c0.w, c1.w, c2.w.
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    // The columns of rotation over scale (normals), and in n0.w whether the instances cut.
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    // Its primitive (in `Primitive::ALL`'s order), and where the rocks it overlaps start and how many in `px_near`.
    x: vec4<u32>,
};

@group(2) @binding(0) var<storage, read> px_vert: array<f32>;
@group(2) @binding(1) var<storage, read> px_inst: array<ProxyInst>;
@group(2) @binding(2) var<storage, read> px_near: array<u32>;

// Whether world point `w` lies inside rock `j` (as `sdf.rs` has its primitives, at unit size).
fn px_inside(j: u32, w: vec3<f32>) -> bool {
    let m = px_inst[j];
    let d = w - vec3<f32>(m.c0.w, m.c1.w, m.c2.w);
    let p = vec3<f32>(dot(d, m.n0.xyz), dot(d, m.n1.xyz), dot(d, m.n2.xyz));
    let a = abs(p);
    let r = length(p.xz);
    switch m.x.x {
        case 0u: { return all(a < vec3<f32>(1.0)); }
        case 1u: { return dot(p, p) < 1.0; }
        case 2u: { return r < 1.0 && a.y < 1.0; }
        case 3u: { return a.y < 1.0 && r < 0.5 * (1.0 - p.y); }
        case 4u: { return a.z < 1.0 && p.x > -1.0 && p.y > -1.0 && p.x + p.y < 0.0; }
        case 5u: { return a.z < 1.0 && a.x + a.y < 1.0; }
        default: { return (r - 1.0) * (r - 1.0) + p.y * p.y < PX_TORUS_MINOR * PX_TORUS_MINOR; }
    }
}

struct ProxyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) cut: f32,
    @location(3) @interpolate(flat) inst: u32,
};

@vertex
fn vs_proxy(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> ProxyOut {
    let p = vec3<f32>(px_vert[6u * v], px_vert[6u * v + 1u], px_vert[6u * v + 2u]);
    let n = vec3<f32>(px_vert[6u * v + 3u], px_vert[6u * v + 4u], px_vert[6u * v + 5u]);
    let m = px_inst[i];
    let w = m.c0.xyz * p.x + m.c1.xyz * p.y + m.c2.xyz * p.z + vec3<f32>(m.c0.w, m.c1.w, m.c2.w);
    var o: ProxyOut;
    o.clip = u.view_proj * vec4<f32>(w, 1.0);
    o.world = w;
    o.normal = m.n0.xyz * n.x + m.n1.xyz * n.y + m.n2.xyz * n.z;
    o.cut = m.n0.w;
    o.inst = i;
    return o;
}

@fragment
fn fs_proxy(i: ProxyOut) -> @location(0) vec4<f32> {
    var m: MeshOut;
    m.clip = i.clip;
    m.world = i.world;
    m.normal = i.normal;
    m.overlay = 0.0;
    m.base = i.world;
    let l2 = dot(i.normal, i.normal);
    let n = select(vec3<f32>(0.0, 1.0, 0.0), i.normal * inverseSqrt(max(l2, 1e-30)), l2 > 1e-30);
    // Tinted: cool where they add, warm where they cut.
    let tint = select(vec3<f32>(0.55, 0.75, 1.0), vec3<f32>(1.0, 0.55, 0.45), i.cut > 0.5);
    let c = lit(m, n, u.clay.rgb * tint);
    return vec4<f32>(c.rgb, 1.0);
}

// A cutting rock's far side, seen through the surface it cuts: the wall of its hole, facing out of the hole.
@fragment
fn fs_proxy_wall(i: ProxyOut) -> @location(0) vec4<f32> {
    // Inside a rock it overlaps: cut away by that one too, not where the cut ends.
    let x = px_inst[i.inst].x;
    for (var q = x.y; q < x.y + x.z; q++) {
        if px_inside(px_near[q], i.world) {
            discard;
        }
    }
    var m: MeshOut;
    m.clip = i.clip;
    m.world = i.world;
    m.overlay = 0.0;
    m.base = i.world;
    let l2 = dot(i.normal, i.normal);
    let n = -select(vec3<f32>(0.0, 1.0, 0.0), i.normal * inverseSqrt(max(l2, 1e-30)), l2 > 1e-30);
    m.normal = n;
    let c = lit(m, n, u.clay.rgb);
    return vec4<f32>(c.rgb, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProxyInst {
    c: [[f32; 4]; 3],
    n: [[f32; 4]; 3],
    x: [u32; 4],
}

/// Most rocks one rock's wall is tested against (the nearest it overlaps).
const NEAR_MOST: usize = 16;

pub struct ProxyDraw {
    pipe: wgpu::RenderPipeline,
    /// Rocks that cut: their faces in front of the surface counted in the stencil, then their far sides drawn behind it
    /// where the count is above zero.
    mark: wgpu::RenderPipeline,
    wall: wgpu::RenderPipeline,
    cut: bool,
    bgl: wgpu::BindGroupLayout,
    verts: wgpu::Buffer,
    /// Each primitive's vertices in `verts` (in `Primitive::ALL`'s order).
    ranges: Vec<std::ops::Range<u32>>,
    inst: wgpu::Buffer,
    /// Each rock's overlapping rocks, end to end (see `ProxyInst::x`).
    near: wgpu::Buffer,
    group: wgpu::BindGroup,
    /// One draw per primitive present: its vertices and its instances.
    draws: Vec<(std::ops::Range<u32>, std::ops::Range<u32>)>,
    /// The edit shown (`None`: none).
    pub epoch: Option<u64>,
    /// The unit meshes, until the first [`Self::set`] sends them.
    pending_verts: Option<Vec<[f32; 6]>>,
}

impl ProxyDraw {
    /// The pipeline on `device`; `view_bgl` is the viewport's group 0 (its uniforms).
    pub fn new(device: &wgpu::Device, view_bgl: &wgpu::BindGroupLayout) -> ProxyDraw {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("proxy"),
            source: wgpu::ShaderSource::Wgsl(format!("{}\n{}\nconst PX_TORUS_MINOR: f32 = {TORUS_MINOR:?};\n{}", include_str!("viewport.wgsl"), sculpt_core::surface_noise::WGSL, WGSL).into()),
        });
        let ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("proxy"), entries: &[ro(0), ro(1), ro(2)] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("proxy"), bind_group_layouts: &[Some(view_bgl), None, Some(&bgl)], immediate_size: 0 });
        let pipeline = |label: &str, fs: &str, writes: wgpu::ColorWrites, cull: Option<wgpu::Face>, write: bool, compare: wgpu::CompareFunction, stencil: wgpu::StencilState| {
            let target = [Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: writes })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_proxy"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(fs), compilation_options: Default::default(), targets: &target }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: cull, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: Some(write), depth_compare: Some(compare), stencil, bias: Default::default() }),
                multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };
        let face = |compare: wgpu::CompareFunction, pass_op: wgpu::StencilOperation| wgpu::StencilFaceState { compare, fail_op: wgpu::StencilOperation::Keep, depth_fail_op: wgpu::StencilOperation::Keep, pass_op };
        let pipe = pipeline("proxy", "fs_proxy", wgpu::ColorWrites::ALL, None, true, wgpu::CompareFunction::Less, Default::default());
        // (The unit meshes wind counter-clockwise seen from outside: see `unit_mesh`.)
        let mark = pipeline(
            "proxy cut mark",
            "fs_proxy",
            wgpu::ColorWrites::empty(),
            None,
            false,
            wgpu::CompareFunction::Less,
            wgpu::StencilState { front: face(wgpu::CompareFunction::Always, wgpu::StencilOperation::IncrementWrap), back: face(wgpu::CompareFunction::Always, wgpu::StencilOperation::DecrementWrap), read_mask: 0xff, write_mask: 0xff },
        );
        let inside = face(wgpu::CompareFunction::NotEqual, wgpu::StencilOperation::Keep);
        let wall = pipeline("proxy cut wall", "fs_proxy_wall", wgpu::ColorWrites::ALL, Some(wgpu::Face::Front), true, wgpu::CompareFunction::Greater, wgpu::StencilState { front: inside, back: inside, read_mask: 0xff, write_mask: 0 });
        let mut all: Vec<[f32; 6]> = Vec::new();
        let mut ranges = Vec::new();
        for p in Primitive::ALL {
            let a = all.len() as u32;
            all.extend(unit_mesh(p));
            ranges.push(a..all.len() as u32);
        }
        let verts = buffer(device, "proxy vertices", (all.len() * 24) as u64);
        let inst = buffer(device, "proxy instances", 64 * std::mem::size_of::<ProxyInst>() as u64);
        let near = buffer(device, "proxy near", 1024);
        let group = bind(device, &bgl, &verts, &inst, &near);
        ProxyDraw { pipe, mark, wall, cut: false, bgl, verts, ranges, inst, near, group, draws: Vec::new(), epoch: None, pending_verts: Some(all) }
    }

    /// Show `instances` (cutting ones if `cut`) as the edit numbered `epoch`; `None` shows none.
    pub fn set(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, shown: Option<(&[Instance], bool, u64)>) {
        if let Some(v) = self.pending_verts.take() {
            queue.write_buffer(&self.verts, 0, bytemuck::cast_slice(&v));
        }
        let Some((instances, cut, epoch)) = shown else {
            self.epoch = None;
            self.draws.clear();
            return;
        };
        if self.epoch == Some(epoch) {
            return;
        }
        self.epoch = Some(epoch);
        self.cut = cut;
        // Sorted by primitive, one run each.
        let mut order: Vec<(usize, &Instance)> = instances.iter().map(|i| (Primitive::ALL.iter().position(|p| *p == i.primitive).unwrap_or(1), i)).collect();
        order.sort_by_key(|x| x.0);
        // Each rock's overlapping ones (by bounding spheres), nearest first: only rocks that cut need them.
        let ball = |i: &Instance| (i.translation, (i.primitive.local_half_extent() * i.scale).length());
        let mut near: Vec<u32> = Vec::new();
        let mut spans = vec![(0u32, 0u32); order.len()];
        if cut {
            let balls: Vec<(Vec3, f32)> = order.iter().map(|(_, i)| ball(i)).collect();
            for (a, &(ca, ra)) in balls.iter().enumerate() {
                let mut by: Vec<(f32, u32)> = balls
                    .iter()
                    .enumerate()
                    .filter(|&(b, &(cb, rb))| b != a && ca.distance_squared(cb) < (ra + rb) * (ra + rb))
                    .map(|(b, &(cb, _))| (ca.distance_squared(cb), b as u32))
                    .collect();
                by.sort_by(|x, y| x.0.total_cmp(&y.0));
                by.truncate(NEAR_MOST);
                spans[a] = (near.len() as u32, by.len() as u32);
                near.extend(by.iter().map(|x| x.1));
            }
        }
        let cut = if cut { 1.0 } else { 0.0 };
        let data: Vec<ProxyInst> = order
            .iter()
            .zip(&spans)
            .map(|((p, i), &(from, n))| {
                let r = Mat3::from_quat(i.rotation);
                let s = i.scale;
                let si = s.recip();
                let t = i.translation;
                let col = |v: Vec3, w: f32| [v.x, v.y, v.z, w];
                ProxyInst {
                    c: [col(r.x_axis * s.x, t.x), col(r.y_axis * s.y, t.y), col(r.z_axis * s.z, t.z)],
                    n: [col(r.x_axis * si.x, cut), col(r.y_axis * si.y, 0.0), col(r.z_axis * si.z, 0.0)],
                    x: [*p as u32, from, n, 0],
                }
            })
            .collect();
        let bytes = (data.len().max(1) * std::mem::size_of::<ProxyInst>()) as u64;
        let near_bytes = (near.len().max(1) * 4) as u64;
        if self.inst.size() < bytes || self.near.size() < near_bytes {
            if self.inst.size() < bytes {
                self.inst = buffer(device, "proxy instances", bytes.next_power_of_two());
            }
            if self.near.size() < near_bytes {
                self.near = buffer(device, "proxy near", near_bytes.next_power_of_two());
            }
            self.group = bind(device, &self.bgl, &self.verts, &self.inst, &self.near);
        }
        if !data.is_empty() {
            queue.write_buffer(&self.inst, 0, bytemuck::cast_slice(&data));
        }
        if !near.is_empty() {
            queue.write_buffer(&self.near, 0, bytemuck::cast_slice(&near));
        }
        self.draws.clear();
        let mut a = 0;
        while a < order.len() {
            let p = order[a].0;
            let mut z = a;
            while z < order.len() && order[z].0 == p {
                z += 1;
            }
            self.draws.push((self.ranges[p].clone(), a as u32..z as u32));
            a = z;
        }
    }

    pub fn showing(&self) -> bool {
        !self.draws.is_empty()
    }

    /// Draw into `pass` (its group 0 bound already).
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_bind_group(2, &self.group, &[]);
        let pipes: &[&wgpu::RenderPipeline] = if self.cut { &[&self.mark, &self.wall] } else { &[&self.pipe] };
        for pipe in pipes {
            pass.set_pipeline(pipe);
            pass.set_stencil_reference(0);
            for (v, i) in &self.draws {
                pass.draw(v.clone(), i.clone());
            }
        }
    }
}

fn buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, verts: &wgpu::Buffer, inst: &wgpu::Buffer, near: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("proxy"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: verts.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: inst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: near.as_entire_binding() },
        ],
    })
}

/// Segments around a round primitive.
const AROUND: usize = 24;

/// Primitive `p` at unit size (as `sdf.rs` has it), as triangles: position and normal per corner.
fn unit_mesh(p: Primitive) -> Vec<[f32; 6]> {
    let mut out = Vec::new();
    // (Wound counter-clockwise seen from outside, the side its normals face: the stencil counts on it.)
    let mut tri = |a: (Vec3, Vec3), b: (Vec3, Vec3), c: (Vec3, Vec3)| {
        let (b, c) = if (b.0 - a.0).cross(c.0 - a.0).dot(a.1 + b.1 + c.1) < 0.0 { (c, b) } else { (b, c) };
        for (q, n) in [a, b, c] {
            out.push([q.x, q.y, q.z, n.x, n.y, n.z]);
        }
    };
    let tau = std::f32::consts::TAU;
    let ring = |k: usize| {
        let a = tau * k as f32 / AROUND as f32;
        Vec2::new(a.cos(), a.sin())
    };
    match p {
        Primitive::Sphere => {
            let rows = AROUND / 2;
            let at = |k: usize, j: usize| {
                let t = std::f32::consts::PI * j as f32 / rows as f32;
                let r = ring(k);
                let v = Vec3::new(t.sin() * r.x, t.cos(), t.sin() * r.y);
                (v, v)
            };
            for j in 0..rows {
                for k in 0..AROUND {
                    let (a, b, c, d) = (at(k, j), at(k + 1, j), at(k + 1, j + 1), at(k, j + 1));
                    tri(a, b, c);
                    tri(a, c, d);
                }
            }
        }
        Primitive::Torus => {
            let small = AROUND / 2;
            let at = |k: usize, j: usize| {
                let r = ring(k);
                let a = tau * j as f32 / small as f32;
                let n = Vec3::new(a.cos() * r.x, a.sin(), a.cos() * r.y);
                (Vec3::new(r.x, 0.0, r.y) + n * TORUS_MINOR, n)
            };
            for j in 0..small {
                for k in 0..AROUND {
                    let (a, b, c, d) = (at(k, j), at(k + 1, j), at(k + 1, j + 1), at(k, j + 1));
                    tri(a, b, c);
                    tri(a, c, d);
                }
            }
        }
        Primitive::Cylinder | Primitive::Cone => {
            // Radius 1 at y = -1; 1 (cylinder) or 0 (cone) at y = +1.
            let top = if p == Primitive::Cylinder { 1.0 } else { 0.0 };
            for k in 0..AROUND {
                let (r0, r1) = (ring(k), ring(k + 1));
                let side = |r: Vec2| Vec3::new(r.x * 2.0, 1.0 - top, r.y * 2.0).normalize();
                let (b0, b1) = (Vec3::new(r0.x, -1.0, r0.y), Vec3::new(r1.x, -1.0, r1.y));
                let (t0, t1) = (Vec3::new(r0.x * top, 1.0, r0.y * top), Vec3::new(r1.x * top, 1.0, r1.y * top));
                let (n0, n1) = (side(r0), side(r1));
                tri((b0, n0), (b1, n1), (t1, n1));
                tri((b0, n0), (t1, n1), (t0, n0));
                tri((Vec3::NEG_Y, Vec3::NEG_Y), (b1, Vec3::NEG_Y), (b0, Vec3::NEG_Y));
                if top > 0.0 {
                    tri((Vec3::Y, Vec3::Y), (t0, Vec3::Y), (t1, Vec3::Y));
                }
            }
        }
        Primitive::Box | Primitive::Wedge | Primitive::ChamferCutter => {
            // A prism along z (-1..1) over a convex outline in xy.
            let outline: Vec<Vec2> = match p {
                Primitive::Box => vec![Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)],
                Primitive::Wedge => vec![Vec2::new(-1.0, -1.0), Vec2::new(1.0, -1.0), Vec2::new(-1.0, 1.0)],
                _ => vec![Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(-1.0, 0.0), Vec2::new(0.0, -1.0)],
            };
            let n = outline.len();
            for e in 0..n {
                let (a, b) = (outline[e], outline[(e + 1) % n]);
                let d = b - a;
                let side = Vec3::new(d.y, -d.x, 0.0).normalize();
                let q = |v: Vec2, z: f32| (Vec3::new(v.x, v.y, z), side);
                tri(q(a, -1.0), q(b, -1.0), q(b, 1.0));
                tri(q(a, -1.0), q(b, 1.0), q(a, 1.0));
            }
            for e in 1..n - 1 {
                let (a, b, c) = (outline[0], outline[e], outline[e + 1]);
                let f = |v: Vec2, z: f32| (Vec3::new(v.x, v.y, z), Vec3::new(0.0, 0.0, z));
                tri(f(a, 1.0), f(b, 1.0), f(c, 1.0));
                tri(f(a, -1.0), f(c, -1.0), f(b, -1.0));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use sculpt_core::generator::Primitive;

    /// Every primitive's unit mesh is whole triangles with unit normals, within its unit box.
    #[test]
    fn unit_meshes_are_whole() {
        for p in Primitive::ALL {
            let m = super::unit_mesh(p);
            assert!(!m.is_empty() && m.len() % 3 == 0, "{p:?}");
            let h = p.local_half_extent() + glam::Vec3::splat(1e-4);
            for v in &m {
                let (q, n) = (glam::Vec3::new(v[0], v[1], v[2]), glam::Vec3::new(v[3], v[4], v[5]));
                assert!((n.length() - 1.0).abs() < 1e-3, "{p:?} normal {n}");
                assert!(q.abs().cmple(h).all(), "{p:?} corner {q} past {h}");
                // On the primitive's surface, about (the sphere and torus cut flat between rows).
                assert!(p.sdf(q).abs() < 0.02, "{p:?} corner {q} off its surface by {}", p.sdf(q));
            }
        }
    }

    /// A generator edit held as meshes draws over the volume on a headless device, and is gone when let go of.
    #[test]
    #[ignore]
    fn held_rocks_draw_on_a_headless_device() {
        let (device, queue, adapter) = crate::gpu_levels::check::device().expect("a GPU");
        println!("adapter: {adapter}");
        let theme = crate::theme::ThemeLibrary::load(std::path::Path::new("no themes here")).themes[0].clone();
        let cam = crate::camera::Camera { distance: 3.4, yaw: 0.3, pitch: 0.2, ..Default::default() };
        let size = [320u32, 240];
        let mut mode = crate::volume_mode::VolumeMode::ball(2e5);
        mode.hold_generators = true;
        mode.add_generator();
        mode.settle();
        let mut vp = crate::viewport::Viewport::with_device(&device, include_str!("viewport.wgsl"));
        let settle = |vp: &mut crate::viewport::Viewport, mode: &mut crate::volume_mode::VolumeMode| {
            for _ in 0..200 {
                let more = vp.sync_volume_on(&device, &queue, mode, cam.eye(), 1000.0, Some(1000.0));
                if !more && mode.vol.dirty_keys().is_empty() {
                    break;
                }
            }
        };
        settle(&mut vp, &mut mode);
        let before = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        let k = mode.layer.as_ref().unwrap().active();
        let mut spec = mode.generator_spec(k).unwrap();
        spec.size *= 2.0;
        mode.set_generator(k, &spec, false);
        vp.set_proxy(&device, &queue, mode.proxy());
        let held = vp.render_offscreen(&device, &queue, size, &cam, &theme);
        if let Ok(p) = std::env::var("PROXY_SHOT") {
            let image = egui::ColorImage::from_rgba_unmultiplied([size[0] as usize, size[1] as usize], &held);
            crate::app::write_png(std::path::Path::new(&p), &image, false).unwrap();
        }
        let differ = |a: &[u8], b: &[u8]| a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| p != q).count();
        let px = (size[0] * size[1]) as usize;
        let shown = differ(&before, &held);
        println!("{} rocks held: {shown} of {px} pixels change", mode.proxy().unwrap().0.len());
        assert!(shown > px / 100, "the meshes barely show: {shown} of {px} pixels");
        vp.set_proxy(&device, &queue, None);
        assert_eq!(differ(&before, &vp.render_offscreen(&device, &queue, size, &cam, &theme)), 0, "the meshes stayed");
        // Rocks that cut, held: drawn as the holes they make, so closer to the cut surface once built than the surface
        // before them is. Then many, so they overlap (where one's wall is inside another, it is cut away too): shown and
        // measured, not gated (the built one's soft blend merges them).
        spec.op = sculpt_core::generator::BoolOp::Subtract;
        spec.lift = -0.2;
        for (count, spacing, gate) in [(spec.count, spec.min_spacing, true), (60, 0.0, false)] {
            spec.count = count;
            spec.min_spacing = spacing;
            vp.set_proxy(&device, &queue, None);
            let before = vp.render_offscreen(&device, &queue, size, &cam, &theme);
            mode.set_generator(k, &spec, false);
            vp.set_proxy(&device, &queue, mode.proxy());
            let held = vp.render_offscreen(&device, &queue, size, &cam, &theme);
            mode.build_generator();
            mode.settle();
            vp.set_proxy(&device, &queue, mode.proxy());
            settle(&mut vp, &mut mode);
            let built = vp.render_offscreen(&device, &queue, size, &cam, &theme);
            if let Ok(p) = std::env::var("PROXY_SHOT") {
                for (what, shot) in [("cut", &held), ("built", &built)] {
                    let image = egui::ColorImage::from_rgba_unmultiplied([size[0] as usize, size[1] as usize], shot);
                    crate::app::write_png(std::path::Path::new(&p.replace(".png", &format!("_{what}{count}.png"))), &image, false).unwrap();
                }
            }
            // Pixels far apart in colour, as the eye would see a hole that is there or not.
            let far = |a: &[u8], b: &[u8]| a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| (0..3).map(|c| (p[c] as i32 - q[c] as i32).abs()).sum::<i32>() > 40).count();
            let (was, now) = (far(&before, &built), far(&held, &built));
            println!("{count} cutting rocks: {was} pixels far from the built surface before them, {now} with them held");
            // (Not alike: the built one's blend is soft, and rocks at the outline show their far side held.)
            assert!(!gate || (was > px / 100 && (now as f64) < 0.7 * was as f64), "the held cut does not look like the built one: {now} pixels off, {was} before");
        }
    }
}
