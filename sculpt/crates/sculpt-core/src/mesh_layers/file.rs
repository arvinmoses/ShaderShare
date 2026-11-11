//! Saving layers against vertex names, and reading them back onto a fresh extraction.

use std::collections::BTreeMap;
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
        std::fs::create_dir_all(dir.join("blobs"))?;
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
            margin_steps: self.settings.margin_steps,
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
        std::fs::write(dir.join(MANIFEST), serde_json::to_string_pretty(&m)?)?;
        Ok(())
    }

    /// Read what [`Self::save`] wrote, extracting the mesh again from `vol`. Data whose vertex name the surface no
    /// longer has is dropped (the second value counts those vertices).
    pub fn load(dir: &Path, vol: &LevelVolume) -> Result<(MeshLayers, usize)> {
        let m: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST))?)?;
        if m.format != FORMAT {
            return Err(Error::Format(format!("not a {FORMAT} manifest")));
        }
        let settings = MeshLayerSettings { cap: m.cap, margin_steps: m.margin_steps, step: m.step };
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
        doc.undo.clear();
        Ok((out, lost))
    }
}

// ---------------------------------------------------------------------- file format

const FORMAT: &str = "sculpt-mesh-layers";
const MANIFEST: &str = "mesh_layers.json";

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

fn write_blob<T: bytemuck::Pod>(dir: &Path, name: &str, dtype: &str, data: &[T]) -> Result<Blob> {
    let rel = format!("blobs/{name}.bin");
    std::fs::write(dir.join(&rel), bytemuck::cast_slice(data))?;
    Ok(Blob { path: rel, dtype: dtype.into(), count: data.len() as u64 })
}

fn read_blob<T: bytemuck::Pod>(dir: &Path, b: &Blob, dtype: &str) -> Result<Vec<T>> {
    if b.dtype != dtype {
        return Err(Error::Format(format!("{}: expected {dtype}, found {}", b.path, b.dtype)));
    }
    if b.path.contains("..") || Path::new(&b.path).is_absolute() {
        return Err(Error::Format(format!("blob path escapes the directory: {}", b.path)));
    }
    let bytes = std::fs::read(dir.join(&b.path))?;
    if bytes.len() != b.count as usize * std::mem::size_of::<T>() {
        return Err(Error::Format(format!("{}: expected {} values, found {} bytes", b.path, b.count, bytes.len())));
    }
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}
