//! Saving layers against vertex names, and reading them back onto a fresh extraction.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::Path;

use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{MeshLayerSettings, MeshLayers};
use crate::generator::level_volume::LevelVolume;
use crate::io::project::LayerEntry;
use crate::layers::{LayerId, SculptLayer};
use crate::mesh::NO_VERT;
use crate::{Error, Result};

impl MeshLayers {
    /// Write the layers, channels and freeze, keyed by vertex name, to `dir` (a directory). The mesh is not written.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let (m, blobs) = self.to_parts()?;
        std::fs::create_dir_all(dir.join("blobs"))?;
        for (path, bytes) in &blobs {
            std::fs::write(dir.join(path), bytes)?;
        }
        std::fs::write(dir.join(MANIFEST), serde_json::to_string_pretty(&m)?)?;
        Ok(())
    }

    /// [`Self::save`] as one stream: a header, the manifest and the blobs, each with its length, so
    /// [`Self::read_from`] reads exactly what this wrote.
    pub fn write_to(&self, w: &mut impl Write) -> Result<()> {
        let (m, blobs) = self.to_parts()?;
        let manifest = serde_json::to_vec(&m)?;
        w.write_all(STREAM_MAGIC)?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(manifest.len() as u64).to_le_bytes())?;
        w.write_all(&manifest)?;
        w.write_all(&(blobs.len() as u32).to_le_bytes())?;
        for (path, bytes) in &blobs {
            w.write_all(&(path.len() as u32).to_le_bytes())?;
            w.write_all(path.as_bytes())?;
            w.write_all(&(bytes.len() as u64).to_le_bytes())?;
            w.write_all(bytes)?;
        }
        Ok(())
    }

    /// Read what [`Self::write_to`] wrote (and nothing past it), extracting the mesh again from `vol`, as
    /// [`Self::load`].
    pub fn read_from(r: &mut impl Read, vol: &LevelVolume) -> Result<(MeshLayers, usize)> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != STREAM_MAGIC {
            return Err(Error::Format("not a mesh layers stream".into()));
        }
        let version = read_u32(r)?;
        if version != 1 {
            return Err(Error::Format(format!("mesh layers stream version {version} is not known")));
        }
        let len = read_u64(r)?;
        let manifest = read_bytes(r, len)?;
        let m: Manifest = serde_json::from_slice(&manifest)?;
        let count = read_u32(r)?;
        let mut blobs: HashMap<String, Vec<u8>> = HashMap::new();
        for _ in 0..count {
            let len = read_u32(r)? as u64;
            let path = String::from_utf8(read_bytes(r, len)?).map_err(|_| Error::Format("a blob name is not text".into()))?;
            let len = read_u64(r)?;
            blobs.insert(path, read_bytes(r, len)?);
        }
        Self::from_parts(m, vol, |b| blobs.get(&b.path).cloned().ok_or_else(|| Error::Format(format!("{}: not in the stream", b.path))))
    }

    /// The manifest and the blobs (path, bytes).
    fn to_parts(&self) -> Result<(Manifest, Vec<(String, Vec<u8>)>)> {
        let mut blobs: Vec<(String, Vec<u8>)> = Vec::new();
        let dir = &mut blobs;
        let doc = &self.doc;
        let to_canon = self.internal_to_canonical();
        let keys_blob = write_blob(dir, "keys", "u64", &self.keys)?;
        let freeze = if doc.freeze.iter().any(|&f| f != 0.0) { Some(write_blob(dir, "freeze", "f32", &doc.to_canonical(&doc.freeze))?) } else { None };
        let mut channels = Vec::new();
        for (i, (name, values)) in doc.channels.iter().enumerate() {
            channels.push((name.clone(), write_blob(dir, &format!("channel_{i}"), "f32", &doc.to_canonical(values))?));
        }
        let mut layers = Vec::new();
        for layer in &doc.layers {
            let mut sparse: Vec<(u32, Vec3)> = Vec::new();
            for l in layer.allocated_leaves() {
                let start = doc.bvh.leaves[l as usize].owned.start as usize;
                for (k, d) in layer.chunks[l as usize].as_ref().unwrap().iter().enumerate() {
                    if *d != Vec3::ZERO {
                        sparse.push((to_canon[start + k], *d));
                    }
                }
            }
            sparse.sort_unstable_by_key(|s| s.0);
            let (idx, deltas): (Vec<u32>, Vec<Vec3>) = sparse.into_iter().unzip();
            let tag = format!("layer_{}", layer.id.0);
            layers.push(SavedLayer {
                entry: LayerEntry {
                    id: layer.id.0,
                    name: layer.name.clone(),
                    opacity: layer.opacity,
                    visible: layer.visible,
                    locked: layer.locked,
                    kind: layer.kind,
                    parent: layer.parent.map(|p| p.0),
                    collapsed: layer.collapsed,
                    blend: layer.blend,
                    mask: layer.mask.clone(),
                    generator: layer.generator.clone(),
                    indices: None,
                    deltas: None,
                },
                indices: write_blob(dir, &format!("{tag}_indices"), "u32", &idx)?,
                deltas: write_blob(dir, &format!("{tag}_deltas"), "f32x3", &deltas)?,
            });
        }
        let m = Manifest {
            format: FORMAT.into(),
            version: 1,
            cap: self.settings.cap,
            margin_steps: MARGIN_STEPS,
            step: self.settings.step,
            keys: keys_blob,
            freeze,
            channels,
            layers,
            active: doc.active.map(|l| l.0),
            solo: doc.solo.map(|l| l.0),
            next_layer_id: doc.next_layer_id,
            metadata: doc.metadata.clone(),
        };
        Ok((m, blobs))
    }

    /// Read what [`Self::save`] wrote, extracting the mesh again from `vol`. Data whose vertex name the surface no
    /// longer has is dropped (the second value counts those vertices).
    pub fn load(dir: &Path, vol: &LevelVolume) -> Result<(MeshLayers, usize)> {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST))?)?;
        Self::from_parts(m, vol, |b| {
            if b.path.contains("..") || Path::new(&b.path).is_absolute() {
                return Err(Error::Format(format!("blob path escapes the directory: {}", b.path)));
            }
            Ok(std::fs::read(dir.join(&b.path))?)
        })
    }

    /// The layers of manifest `m` on `vol`'s surface, each blob's bytes from `bytes`.
    fn from_parts(m: Manifest, vol: &LevelVolume, bytes: impl Fn(&Blob) -> Result<Vec<u8>>) -> Result<(MeshLayers, usize)> {
        let dir = &bytes;
        if m.format != FORMAT {
            return Err(Error::Format(format!("not a {FORMAT} manifest")));
        }
        let settings = MeshLayerSettings { cap: m.cap, step: m.step };
        let mut out = MeshLayers::from_volume(vol, settings)?;
        let saved_keys: Vec<u64> = read_blob(dir, &m.keys, "u64")?;
        // Saved canonical index -> internal index of the new document.
        let to_new: Vec<u32> = saved_keys.par_iter().map(|k| out.keys.binary_search(k).map_or(NO_VERT, |c| out.doc.canonical_to_internal[c])).collect();
        let lost = to_new.iter().filter(|&&v| v == NO_VERT).count();
        let n = out.doc.vertex_count();
        let scalar = |b: &Blob| -> Result<Vec<f32>> {
            let vals: Vec<f32> = read_blob(dir, b, "f32")?;
            if vals.len() != saved_keys.len() {
                return Err(Error::Format(format!("{}: {} values for {} vertices", b.path, vals.len(), saved_keys.len())));
            }
            let mut dense = vec![0.0f32; n];
            for (v, x) in to_new.iter().zip(vals) {
                if *v != NO_VERT {
                    dense[*v as usize] = x;
                }
            }
            Ok(dense)
        };
        let doc = &mut out.doc;
        if let Some(b) = &m.freeze {
            doc.freeze = scalar(b)?;
        }
        for (name, b) in &m.channels {
            let vals = scalar(b)?;
            doc.channels.insert(name.clone(), vals);
        }
        let leaves = doc.bvh.leaves.len();
        for sl in m.layers {
            let e = sl.entry;
            let idx: Vec<u32> = read_blob(dir, &sl.indices, "u32")?;
            let deltas: Vec<Vec3> = read_blob(dir, &sl.deltas, "f32x3")?;
            if idx.len() != deltas.len() || idx.iter().any(|&i| i as usize >= saved_keys.len()) {
                return Err(Error::Format(format!("layer '{}' has inconsistent sparse data", e.name)));
            }
            let mut layer = SculptLayer::new(LayerId(e.id), &e.name, leaves);
            layer.opacity = e.opacity;
            layer.visible = e.visible;
            layer.locked = e.locked;
            layer.kind = e.kind;
            layer.parent = e.parent.map(LayerId);
            layer.collapsed = e.collapsed;
            layer.blend = e.blend;
            layer.mask = e.mask;
            layer.generator = e.generator;
            for (ci, d) in idx.iter().zip(deltas) {
                let v = to_new[*ci as usize];
                if v == NO_VERT {
                    continue;
                }
                let v = v as usize;
                let l = doc.bvh.vert_leaf[v] as usize;
                let leaf = &doc.bvh.leaves[l];
                let len = leaf.owned_len();
                let chunk = layer.chunks[l].get_or_insert_with(|| vec![Vec3::ZERO; len].into_boxed_slice());
                chunk[v - leaf.owned.start as usize] = d;
            }
            doc.layers.push(layer);
        }
        doc.next_layer_id = m.next_layer_id.max(doc.layers.iter().map(|l| l.id.0 + 1).max().unwrap_or(1));
        doc.active = m.active.map(LayerId).filter(|id| doc.layers.iter().any(|l| l.id == *id && l.is_sculpt_target()));
        doc.solo = m.solo.map(LayerId).filter(|id| doc.layers.iter().any(|l| l.id == *id));
        doc.metadata = m.metadata;
        doc.refresh_scales();
        for i in 0..doc.layers.len() {
            if let Some(stack) = doc.layers[i].mask.clone() {
                doc.layers[i].mask_values = doc.mask_values_for(&stack)?;
            }
        }
        doc.recomposite_all();
        doc.dirty_channels.clear();
        doc.painted_leaves.clear();
        doc.undo.clear();
        Ok((out, lost))
    }
}

// ---------------------------------------------------------------------- file format

const FORMAT: &str = "sculpt-mesh-layers";
const MANIFEST: &str = "mesh_layers.json";
const STREAM_MAGIC: &[u8; 4] = b"SMLS";

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// `len` bytes, read in pieces (a bad length fails at the end of the stream rather than allocating it all).
fn read_bytes(r: &mut impl Read, len: u64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    r.take(len).read_to_end(&mut out)?;
    if out.len() as u64 != len {
        return Err(Error::Format("mesh layers stream ends early".into()));
    }
    Ok(out)
}

#[derive(Serialize, Deserialize)]
struct Blob {
    path: String,
    dtype: String,
    count: u64,
}

#[derive(Serialize, Deserialize)]
struct SavedLayer {
    #[serde(flatten)]
    entry: LayerEntry,
    #[serde(rename = "key_indices")]
    indices: Blob,
    #[serde(rename = "key_deltas")]
    deltas: Blob,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    version: u32,
    cap: usize,
    /// Not read: a setting the carry no longer has. Still written (as it always was) so older versions read the file,
    /// and optional so files without it read too.
    #[serde(default = "margin_steps")]
    margin_steps: f32,
    step: Option<f32>,
    /// Vertex names in the order of every per-vertex blob.
    keys: Blob,
    freeze: Option<Blob>,
    channels: Vec<(String, Blob)>,
    layers: Vec<SavedLayer>,
    active: Option<u32>,
    solo: Option<u32>,
    next_layer_id: u32,
    metadata: BTreeMap<String, serde_json::Value>,
}

/// What [`Manifest::margin_steps`] always held.
const MARGIN_STEPS: f32 = 2.0;

fn margin_steps() -> f32 {
    MARGIN_STEPS
}

fn write_blob<T: bytemuck::Pod>(blobs: &mut Vec<(String, Vec<u8>)>, name: &str, dtype: &str, data: &[T]) -> Result<Blob> {
    let rel = format!("blobs/{name}.bin");
    blobs.push((rel.clone(), bytemuck::cast_slice(data).to_vec()));
    Ok(Blob { path: rel, dtype: dtype.into(), count: data.len() as u64 })
}

fn read_blob<T: bytemuck::Pod>(bytes: &impl Fn(&Blob) -> Result<Vec<u8>>, b: &Blob, dtype: &str) -> Result<Vec<T>> {
    if b.dtype != dtype {
        return Err(Error::Format(format!("{}: expected {dtype}, found {}", b.path, b.dtype)));
    }
    let bytes = bytes(b)?;
    if bytes.len() != b.count as usize * std::mem::size_of::<T>() {
        return Err(Error::Format(format!("{}: expected {} values, found {} bytes", b.path, b.count, bytes.len())));
    }
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}
