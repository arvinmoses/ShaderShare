//! Data-driven project format.
//!
//! A project is a directory:
//!
//! ```text
//! my_head.sculpt/
//!   project.json        manifest: mesh, layers, mask stacks, channels, metadata
//!   blobs/*.bin         raw little-endian arrays described by the manifest
//! ```
//!
//! Every blob is referenced with its dtype, component count and element
//! count, and all per-vertex data is in *canonical* vertex order (the import
//! order, then Catmull-Clark order per subdivision level). Layer deltas are
//! stored sparsely as `(indices, deltas)` pairs. Mask stacks are stored inline
//! as JSON. Anything that can write JSON and a float array can therefore
//! produce or consume project data.

use std::collections::BTreeMap;
use std::path::Path;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::document::Document;
use crate::layers::LayerId;
use crate::mask::MaskStack;
use crate::mesh::Face;
use crate::{Error, Result};

pub const FORMAT: &str = "sculpt-project";
pub const VERSION: u32 = 1;
pub const MANIFEST: &str = "project.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    F32,
    U32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlobRef {
    pub path: String,
    pub dtype: DType,
    pub components: u32,
    pub count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    #[serde(default)]
    pub generator: String,
    pub mesh: MeshEntry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freeze: Option<BlobRef>,
    #[serde(default)]
    pub channels: Vec<ChannelEntry>,
    #[serde(default)]
    pub layers: Vec<LayerEntry>,
    #[serde(default)]
    pub active_layer: Option<u32>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MeshEntry {
    pub vertex_count: u64,
    pub face_count: u64,
    /// Subdivision levels applied since import.
    #[serde(default)]
    pub level: u32,
    pub base_positions: BlobRef,
    /// `u32 x 4` per face; `0xFFFFFFFF` in the last slot marks a triangle.
    pub faces: BlobRef,
    /// Reference space for procedural masks; defaults to `base_positions`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rest_positions: Option<BlobRef>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChannelEntry {
    pub name: String,
    pub data: BlobRef,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LayerEntry {
    pub id: u32,
    pub name: String,
    pub opacity: f32,
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default)]
    pub locked: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<MaskStack>,
    /// Sparse: vertex indices (u32) and matching deltas (f32 x 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indices: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deltas: Option<BlobRef>,
}

fn yes() -> bool {
    true
}

pub(crate) struct PartLayer {
    pub id: LayerId,
    pub name: String,
    pub opacity: f32,
    pub visible: bool,
    pub locked: bool,
    pub mask: Option<MaskStack>,
    pub indices: Vec<u32>,
    pub deltas: Vec<Vec3>,
}

/// Canonical-order document contents, between disk and [`Document`].
pub(crate) struct Parts {
    pub base: Vec<Vec3>,
    pub rest: Option<Vec<Vec3>>,
    pub faces: Vec<Face>,
    pub freeze: Vec<f32>,
    pub channels: BTreeMap<String, Vec<f32>>,
    pub layers: Vec<PartLayer>,
    pub active: Option<LayerId>,
    pub level: u32,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

fn write_blob<T: bytemuck::Pod>(dir: &Path, name: &str, dtype: DType, components: u32, data: &[T]) -> Result<BlobRef> {
    let rel = format!("blobs/{name}.bin");
    std::fs::write(dir.join(&rel), bytemuck::cast_slice(data))?;
    let scalars = std::mem::size_of_val(data) / 4;
    Ok(BlobRef { path: rel, dtype, components, count: (scalars / components as usize) as u64 })
}

fn read_blob<T: bytemuck::Pod>(dir: &Path, b: &BlobRef, dtype: DType, components: u32) -> Result<Vec<T>> {
    if b.dtype != dtype || b.components != components {
        return Err(Error::Format(format!("{}: expected {dtype:?}x{components}, found {:?}x{}", b.path, b.dtype, b.components)));
    }
    if b.path.contains("..") || Path::new(&b.path).is_absolute() {
        return Err(Error::Format(format!("blob path escapes project: {}", b.path)));
    }
    let bytes = std::fs::read(dir.join(&b.path))?;
    let expected = b.count as usize * components as usize * 4;
    if bytes.len() != expected {
        return Err(Error::Format(format!("{}: expected {expected} bytes, found {}", b.path, bytes.len())));
    }
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}

fn safe_name(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect()
}

pub fn save(doc: &Document, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir.join("blobs"))?;
    let base = doc.to_canonical(doc.base_positions());
    let faces = doc.canonical_faces();
    let rest = doc.to_canonical(doc.rest_positions());
    let mesh = MeshEntry {
        vertex_count: base.len() as u64,
        face_count: faces.len() as u64,
        level: doc.level(),
        base_positions: write_blob(dir, "base_positions", DType::F32, 3, &base)?,
        faces: write_blob(dir, "faces", DType::U32, 4, &faces)?,
        rest_positions: if rest != base { Some(write_blob(dir, "rest_positions", DType::F32, 3, &rest)?) } else { None },
    };
    let freeze = if doc.freeze().iter().any(|&f| f != 0.0) {
        Some(write_blob(dir, "freeze", DType::F32, 1, &doc.to_canonical(doc.freeze()))?)
    } else {
        None
    };
    let mut channels = Vec::new();
    for (i, (name, values)) in doc.channels().iter().enumerate() {
        let blob = format!("channel_{i}_{}", safe_name(name));
        channels.push(ChannelEntry { name: name.clone(), data: write_blob(dir, &blob, DType::F32, 1, &doc.to_canonical(values))? });
    }

    let mut internal_to_canonical = vec![0u32; doc.vertex_count()];
    for c in 0..doc.vertex_count() {
        internal_to_canonical[doc.canonical_index(c)] = c as u32;
    }
    let mut layers = Vec::new();
    for layer in doc.layers() {
        let mut sparse: Vec<(u32, Vec3)> = Vec::new();
        for l in layer.allocated_leaves() {
            let start = doc.bvh().leaves[l as usize].owned.start as usize;
            for (k, d) in layer.chunks[l as usize].as_ref().unwrap().iter().enumerate() {
                if *d != Vec3::ZERO {
                    sparse.push((internal_to_canonical[start + k], *d));
                }
            }
        }
        sparse.sort_unstable_by_key(|s| s.0);
        let (idx, deltas): (Vec<u32>, Vec<Vec3>) = sparse.into_iter().unzip();
        let tag = format!("layer_{}", layer.id.0);
        layers.push(LayerEntry {
            id: layer.id.0,
            name: layer.name.clone(),
            opacity: layer.opacity,
            visible: layer.visible,
            locked: layer.locked,
            mask: layer.mask.clone(),
            indices: Some(write_blob(dir, &format!("{tag}_indices"), DType::U32, 1, &idx)?),
            deltas: Some(write_blob(dir, &format!("{tag}_deltas"), DType::F32, 3, &deltas)?),
        });
    }

    let manifest = Manifest {
        format: FORMAT.into(),
        version: VERSION,
        generator: concat!("sculpt-core ", env!("CARGO_PKG_VERSION")).into(),
        mesh,
        freeze,
        channels,
        layers,
        active_layer: doc.active_layer().map(|l| l.0),
        metadata: doc.metadata.clone(),
    };
    std::fs::write(dir.join(MANIFEST), serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

pub fn load(dir: &Path) -> Result<Document> {
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST))?)?;
    if manifest.format != FORMAT {
        return Err(Error::Format(format!("not a {FORMAT} manifest")));
    }
    if manifest.version > VERSION {
        return Err(Error::Format(format!("project version {} is newer than supported {VERSION}", manifest.version)));
    }
    let n = manifest.mesh.vertex_count as usize;
    let base: Vec<Vec3> = read_blob(dir, &manifest.mesh.base_positions, DType::F32, 3)?;
    let faces: Vec<Face> = read_blob(dir, &manifest.mesh.faces, DType::U32, 4)?;
    if base.len() != n || faces.len() as u64 != manifest.mesh.face_count {
        return Err(Error::Format("mesh counts do not match blobs".into()));
    }
    let rest: Option<Vec<Vec3>> = match &manifest.mesh.rest_positions {
        Some(b) => {
            let r: Vec<Vec3> = read_blob(dir, b, DType::F32, 3)?;
            if r.len() != n {
                return Err(Error::Format("rest positions do not match vertex count".into()));
            }
            Some(r)
        }
        None => None,
    };
    let per_vertex = |b: &BlobRef| -> Result<Vec<f32>> {
        let v: Vec<f32> = read_blob(dir, b, DType::F32, 1)?;
        if v.len() != n {
            return Err(Error::Format(format!("{}: {} values for {n} vertices", b.path, v.len())));
        }
        Ok(v)
    };
    let freeze = match &manifest.freeze {
        Some(b) => per_vertex(b)?,
        None => vec![0.0; n],
    };
    let mut channels = BTreeMap::new();
    for c in &manifest.channels {
        channels.insert(c.name.clone(), per_vertex(&c.data)?);
    }
    let mut layers = Vec::new();
    for l in manifest.layers {
        let indices: Vec<u32> = match &l.indices {
            Some(b) => read_blob(dir, b, DType::U32, 1)?,
            None => Vec::new(),
        };
        let deltas: Vec<Vec3> = match &l.deltas {
            Some(b) => read_blob(dir, b, DType::F32, 3)?,
            None => Vec::new(),
        };
        if indices.len() != deltas.len() || indices.iter().any(|&i| i as usize >= n) {
            return Err(Error::Format(format!("layer '{}' has inconsistent sparse data", l.name)));
        }
        layers.push(PartLayer {
            id: LayerId(l.id),
            name: l.name,
            opacity: l.opacity,
            visible: l.visible,
            locked: l.locked,
            mask: l.mask,
            indices,
            deltas,
        });
    }
    Document::from_parts(Parts {
        base,
        rest,
        faces,
        freeze,
        channels,
        layers,
        active: manifest.active_layer.map(LayerId),
        level: manifest.mesh.level,
        metadata: manifest.metadata,
    })
}
