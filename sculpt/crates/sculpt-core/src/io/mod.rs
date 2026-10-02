//! File I/O: OBJ interchange, per-vertex channel import, and the project format.

pub mod obj;
pub mod project;

use std::path::Path;

use crate::{Error, Result};

/// Read per-vertex scalar data written by another program, in canonical
/// (import) vertex order.
///
/// * `.f32` / `.bin`: raw little-endian `f32` array.
/// * anything else: text, one value per line, or `index,value` pairs
///   (`#` comments allowed). Missing indices default to 0.
pub fn read_channel_file(path: &Path, vertex_count: usize) -> Result<Vec<f32>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "f32" || ext == "bin" {
        let bytes = std::fs::read(path)?;
        if bytes.len() != vertex_count * 4 {
            return Err(Error::Format(format!("{}: expected {} bytes, got {}", path.display(), vertex_count * 4, bytes.len())));
        }
        return Ok(bytemuck::pod_collect_to_vec(&bytes));
    }
    let text = std::fs::read_to_string(path)?;
    let mut out = vec![0.0f32; vertex_count];
    let mut next = 0usize;
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let bad = || Error::Format(format!("{}:{}: cannot parse '{line}'", path.display(), ln + 1));
        let (idx, val) = match line.split_once([',', ' ', '\t']) {
            Some((i, v)) => (i.trim().parse::<usize>().map_err(|_| bad())?, v.trim().parse::<f32>().map_err(|_| bad())?),
            None => (next, line.parse::<f32>().map_err(|_| bad())?),
        };
        if idx >= vertex_count {
            return Err(Error::Format(format!("{}:{}: vertex {idx} out of range", path.display(), ln + 1)));
        }
        out[idx] = val;
        next = idx + 1;
    }
    Ok(out)
}
