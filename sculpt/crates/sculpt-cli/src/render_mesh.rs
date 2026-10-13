//! Triangle-soup variant of the software rasterizer (for refined patches and close-ups),
//! with optional wireframe and marker dots. Orthographic cameras, several side by side.

use std::path::Path;

use glam::{Mat3, Vec3};

/// One camera: looks at `center`, `half_extent` world units from the centre to the image edge.
#[derive(Clone, Copy)]
pub struct Cam {
    pub center: Vec3,
    pub half_extent: f32,
    pub yaw: f32,
    pub pitch: f32,
}

pub struct Scene<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub tris: &'a [[u32; 3]],
    /// Draw triangle edges.
    pub wire: bool,
    /// Small red markers (e.g. rim vertices).
    pub dots: &'a [Vec3],
}

/// One `size x size` panel per `(scene, camera)`, side by side.
pub fn render(panels: &[(&Scene, Cam)], size: usize, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (size * panels.len(), size);
    let mut rgb = vec![0u8; w * h * 3];
    let lim = size as f32;
    let light = Vec3::new(-0.4, 0.6, 0.7).normalize();
    for (vi, (scene, cam)) in panels.iter().enumerate() {
        let rot = Mat3::from_rotation_x(cam.pitch) * Mat3::from_rotation_y(cam.yaw);
        let scale = 0.5 * size as f32 / cam.half_extent;
        let half = size as f32 * 0.5;
        let project = |p: Vec3| {
            let q = rot * (p - cam.center);
            Vec3::new(q.x * scale + half, half - q.y * scale, q.z * scale)
        };
        let mut depth = vec![f32::NEG_INFINITY; size * size];
        let mut shade = vec![[0.16f32, 0.17, 0.19]; size * size];
        let proj: Vec<Vec3> = scene.positions.iter().map(|&p| project(p)).collect();
        for &[a, bb, c] in scene.tris {
            let (pa, pb, pc) = (proj[a as usize], proj[bb as usize], proj[c as usize]);
            let (minx, maxx) = (pa.x.min(pb.x).min(pc.x), pa.x.max(pb.x).max(pc.x));
            let (miny, maxy) = (pa.y.min(pb.y).min(pc.y), pa.y.max(pb.y).max(pc.y));
            if maxx < 0.0 || maxy < 0.0 || minx >= lim || miny >= lim {
                continue;
            }
            let area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = minx.floor().max(0.0) as usize;
            let x1 = (maxx.ceil() as usize).min(size - 1);
            let y0 = miny.floor().max(0.0) as usize;
            let y1 = (maxy.ceil() as usize).min(size - 1);
            let (na, nb, nc) = (rot * scene.normals[a as usize], rot * scene.normals[bb as usize], rot * scene.normals[c as usize]);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((pb.x - px) * (pc.y - py) - (pb.y - py) * (pc.x - px)) / area;
                    let w1 = ((pc.x - px) * (pa.y - py) - (pc.y - py) * (pa.x - px)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let z = pa.z * w0 + pb.z * w1 + pc.z * w2;
                    let i = y * size + x;
                    if z <= depth[i] {
                        continue;
                    }
                    depth[i] = z;
                    let n = (na * w0 + nb * w1 + nc * w2).normalize_or(Vec3::Z);
                    let diff = n.dot(light).max(0.0);
                    let rim = (1.0 - n.z.abs()).powi(3) * 0.2;
                    let spec = n.dot((light + Vec3::Z).normalize()).max(0.0).powi(24) * 0.2;
                    let col = Vec3::new(0.78, 0.62, 0.52) * (0.18 + 0.82 * diff) + Vec3::splat(rim + spec);
                    shade[i] = [col.x, col.y, col.z];
                }
            }
        }
        if scene.wire {
            for &[a, bb, c] in scene.tris {
                let pts = [proj[a as usize], proj[bb as usize], proj[c as usize]];
                for k in 0..3 {
                    let (p, q) = (pts[k], pts[(k + 1) % 3]);
                    if p.x.max(q.x) < 0.0 || p.y.max(q.y) < 0.0 || p.x.min(q.x) >= lim || p.y.min(q.y) >= lim {
                        continue;
                    }
                    let steps = ((q.x - p.x).abs().max((q.y - p.y).abs()).ceil() as usize).clamp(1, 4 * size);
                    for s in 0..=steps {
                        let t = s as f32 / steps as f32;
                        let (x, y, z) = (p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t, p.z + (q.z - p.z) * t);
                        if x < 0.0 || y < 0.0 || x >= lim || y >= lim {
                            continue;
                        }
                        let i = y as usize * size + x as usize;
                        if z >= depth[i] - 1.5 {
                            shade[i] = [0.06, 0.07, 0.10];
                        }
                    }
                }
            }
        }
        for &d in scene.dots {
            let p = project(d);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (x, y) = (p.x as i32 + dx, p.y as i32 + dy);
                    if x >= 0 && y >= 0 && (x as usize) < size && (y as usize) < size && p.z >= depth[y as usize * size + x as usize] - 3.0 {
                        shade[y as usize * size + x as usize] = [0.95, 0.15, 0.1];
                    }
                }
            }
        }
        for y in 0..size {
            for x in 0..size {
                let c = shade[y * size + x];
                let o = (y * w + vi * size + x) * 3;
                for k in 0..3 {
                    rgb[o + k] = (c[k].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
                }
            }
        }
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgb)?;
    Ok(())
}
