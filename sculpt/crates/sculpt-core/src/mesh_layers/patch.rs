//! Layer and mask strokes as name-keyed patches, so their undo survives a refit swap (which clears the document's own
//! undo).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use glam::Vec3;

use super::{MeshLayers, Patch, carry};
use crate::layers::LayerId;
use crate::undo::{Data, Target};
use crate::{Error, Result};

impl MeshLayers {
    /// The last finished stroke of the document (its newest undo group, as `Document::end_stroke` left it) as a
    /// [`Patch`] holding what the stroke overwrote: every vertex of the leaves it touched, by name, with the values
    /// before the stroke of each layer, channel and the freeze it changed. [`Self::restore_swap`] of it undoes the
    /// stroke and gives the patch that redoes it. Call it right after `end_stroke`, before anything else changes the
    /// document. A channel the stroke created comes back as zeros (it is not removed). Errors when the group changed
    /// the base or the layer list (not a layer or mask stroke), or a stroke is still open.
    pub fn stroke_patch(&self) -> Result<Patch> {
        let doc = &self.doc;
        if doc.undo.open.as_ref().is_some_and(|g| !g.records.is_empty()) {
            return Err(Error::InvalidData("a stroke is still open".into()));
        }
        let Some(group) = doc.undo.undo.last() else { return Ok(Patch::default()) };
        // What the group holds, per (target, leaf).
        let mut before: HashMap<(Target, u32), &Data> = HashMap::new();
        let mut leaves: BTreeSet<u32> = BTreeSet::new();
        let mut layers: BTreeSet<LayerId> = BTreeSet::new();
        let mut channels: BTreeSet<String> = BTreeSet::new();
        let mut freeze = false;
        for r in &group.records {
            match (&r.target, &r.data) {
                (Target::Layer(id), Data::Deltas(_)) => {
                    layers.insert(*id);
                }
                (Target::Channel(n), Data::Scalars(_) | Data::Absent) => {
                    channels.insert(n.clone());
                }
                (Target::Freeze, Data::Scalars(_)) => freeze = true,
                _ => return Err(Error::InvalidData(format!("'{}' is not a layer or mask stroke", group.label))),
            }
            if r.leaf != u32::MAX {
                leaves.insert(r.leaf);
            }
            before.insert((r.target.clone(), r.leaf), &r.data);
        }
        // Every vertex of the touched leaves, by name.
        let i2c = self.internal_to_canonical();
        let mut verts: Vec<(u64, u32, u32)> = Vec::new();
        for &l in &leaves {
            for v in doc.bvh.leaves[l as usize].owned_range() {
                verts.push((self.keys[i2c[v] as usize], v as u32, l));
            }
        }
        verts.sort_unstable_by_key(|x| x.0);
        let start = |l: u32| doc.bvh.leaves[l as usize].owned.start as usize;
        let layer_vals = |id: LayerId| -> Option<Vec<Vec3>> {
            let layer = doc.layers.iter().find(|x| x.id == id)?;
            let vals: Vec<Vec3> = verts
                .iter()
                .map(|&(_, v, l)| {
                    let chunk = match before.get(&(Target::Layer(id), l)) {
                        Some(Data::Deltas(c)) => c.as_ref(),
                        _ => layer.chunks[l as usize].as_ref(),
                    };
                    chunk.map_or(Vec3::ZERO, |c| c[v as usize - start(l)])
                })
                .collect();
            Some(vals)
        };
        let scalar_vals = |target: Target, now: Option<&[f32]>| -> Vec<f32> {
            let absent = matches!(before.get(&(target.clone(), u32::MAX)), Some(Data::Absent));
            verts
                .iter()
                .map(|&(_, v, l)| match before.get(&(target.clone(), l)) {
                    Some(Data::Scalars(s)) => s[v as usize - start(l)],
                    _ if absent => 0.0,
                    _ => now.map_or(0.0, |n| n[v as usize]),
                })
                .collect()
        };
        let mut out_layers = Vec::new();
        for id in layers {
            if let Some(vals) = layer_vals(id) {
                out_layers.push((id, vals.iter().any(|d| *d != Vec3::ZERO).then_some(vals)));
            }
        }
        let out_channels: BTreeMap<String, Vec<f32>> =
            channels.into_iter().map(|n| (n.clone(), scalar_vals(Target::Channel(n.clone()), doc.channels.get(&n).map(|c| c.as_slice())))).collect();
        Ok(Patch {
            names: verts.iter().map(|x| x.0).collect(),
            base: verts.iter().map(|x| doc.base[x.1 as usize]).collect(),
            layers: out_layers,
            freeze: freeze.then(|| scalar_vals(Target::Freeze, Some(&doc.freeze))),
            channels: out_channels.into_iter().collect(),
        })
    }

    /// Put `patch` back (as [`Self::restore`]) and return its inverse: what it overwrote at the vertices it went back
    /// to, which restored in turn undoes it. Entries with no vertex of the same name and base are left out of both.
    pub fn restore_swap(&mut self, patch: &Patch) -> Result<Patch> {
        carry::restore_swap(&mut self.doc, &self.keys, patch)
    }
}
