//! Minimal CPU rasterizer for thumbnails and visual checks (no GPU needed).

use std::path::Path;

use glam::{Mat3, Vec3};
use sculpt_core::Document;
use sculpt_core::mesh::face_triangles;

pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    /// Per-vertex overlay in `[0, 1]` (e.g. a mask), tinted onto the clay.
    pub overlay: Option<Vec<f32>>,
}

/// Render `views` side by side, each `size x size`, clay shaded.
pub fn render(doc: &Document, views: &[View], size: usize, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (size * views.len(), size);
    let mut rgb = vec![0u8; w * h * 3];
    let b = doc.bounds();
    let (center, scale) = (b.center(), 0.9 * size as f32 / b.diagonal());
    for (vi, view) in views.iter().enumerate() {
        let rot = Mat3::from_rotation_x(view.pitch) * Mat3::from_rotation_y(view.yaw);
        let pts: Vec<Vec3> = doc
            .positions()
            .iter()
            .map(|p| {
                let q = rot * (*p - center);
                Vec3::new(q.x * scale + size as f32 * 0.5, size as f32 * 0.5 - q.y * scale, q.z)
            })
            .collect();
        let nrm: Vec<Vec3> = doc.normals().iter().map(|n| rot * *n).collect();
        let mut depth = vec![f32::NEG_INFINITY; size * size];
        let mut shade = vec![[0.16f32, 0.17, 0.19]; size * size];
        for f in doc.faces() {
            for [a, bb, c] in face_triangles(f) {
                let (pa, pb, pc) = (pts[a as usize], pts[bb as usize], pts[c as usize]);
                let area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
                if area.abs() < 1e-9 {
                    continue;
                }
                let x0 = pa.x.min(pb.x).min(pc.x).floor().max(0.0) as usize;
                let x1 = (pa.x.max(pb.x).max(pc.x).ceil() as usize).min(size - 1);
                let y0 = pa.y.min(pb.y).min(pc.y).floor().max(0.0) as usize;
                let y1 = (pa.y.max(pb.y).max(pc.y).ceil() as usize).min(size - 1);
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
                        let n = (nrm[a as usize] * w0 + nrm[bb as usize] * w1 + nrm[c as usize] * w2).normalize_or(Vec3::Z);
                        let light = Vec3::new(-0.4, 0.6, 0.7).normalize();
                        let diff = n.dot(light).max(0.0);
                        let rim = (1.0 - n.z.abs()).powi(3) * 0.35;
                        let spec = n.dot((light + Vec3::Z).normalize()).max(0.0).powi(24) * 0.25;
                        let mut base = Vec3::new(0.78, 0.62, 0.52);
                        if let Some(o) = &view.overlay {
                            let m = o[a as usize] * w0 + o[bb as usize] * w1 + o[c as usize] * w2;
                            base = base.lerp(Vec3::new(0.25, 0.55, 0.95), m.clamp(0.0, 1.0) * 0.85);
                        }
                        let col = base * (0.18 + 0.82 * diff) + Vec3::splat(rim + spec);
                        shade[i] = [col.x, col.y, col.z];
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
