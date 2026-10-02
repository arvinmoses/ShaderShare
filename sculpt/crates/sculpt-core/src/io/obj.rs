//! Wavefront OBJ (positions + polygons). Vertex order is preserved exactly,
//! which is what makes per-vertex data from other programs line up.

use std::fmt::Write as _;
use std::path::Path;

use glam::Vec3;

use crate::mesh::{NO_VERT, PolyMesh, face_verts};
use crate::{Error, Result};

pub fn parse(text: &str) -> Result<PolyMesh> {
    let mut m = PolyMesh::default();
    for (ln, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        let bad = |what: &str| Error::Format(format!("obj line {}: {what}", ln + 1));
        match it.next() {
            Some("v") => {
                let c: Vec<f32> = it.take(3).map(|s| s.parse().map_err(|_| bad("bad vertex"))).collect::<Result<_>>()?;
                if c.len() != 3 {
                    return Err(bad("vertex needs 3 coordinates"));
                }
                m.positions.push(Vec3::new(c[0], c[1], c[2]));
            }
            Some("f") => {
                let n = m.positions.len() as i64;
                let idx: Vec<u32> = it
                    .map(|tok| {
                        let i: i64 = tok.split('/').next().unwrap().parse().map_err(|_| bad("bad face index"))?;
                        let i = if i < 0 { n + i } else { i - 1 };
                        if i < 0 || i >= n { Err(bad("face index out of range")) } else { Ok(i as u32) }
                    })
                    .collect::<Result<_>>()?;
                match idx.len() {
                    0..=2 => return Err(bad("face needs at least 3 vertices")),
                    3 => m.faces.push([idx[0], idx[1], idx[2], NO_VERT]),
                    4 => m.faces.push([idx[0], idx[1], idx[2], idx[3]]),
                    _ => {
                        for k in 1..idx.len() - 1 {
                            m.faces.push([idx[0], idx[k], idx[k + 1], NO_VERT]);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(m)
}

pub fn read(path: &Path) -> Result<PolyMesh> {
    parse(&std::fs::read_to_string(path)?)
}

pub fn to_string(m: &PolyMesh) -> String {
    let mut s = String::with_capacity(m.positions.len() * 40 + m.faces.len() * 32);
    for p in &m.positions {
        let _ = writeln!(s, "v {} {} {}", p.x, p.y, p.z);
    }
    for f in &m.faces {
        s.push('f');
        for v in face_verts(f) {
            let _ = write!(s, " {}", v + 1);
        }
        s.push('\n');
    }
    s
}

pub fn write(path: &Path, m: &PolyMesh) -> Result<()> {
    Ok(std::fs::write(path, to_string(m))?)
}

