//! Mesh layers shown on the volume's GPU surface: per drawn vertex a row (the composite's offset from the base mesh and
//! its change of normal), packed from [`MeshLayers`] block by block and kept up to date a frame at a time.
//!
//! A mesh layer vertex is named by the same (block, cell, piece) the GPU draws (see [`MeshLayers::blocks`]), so block
//! `k`'s rows go in `k`'s GPU vertex order: canonical vertex `c` at `vertex_vi()[c]`. What changed comes as a
//! [`RowsDirty`] each frame; [`LayerRowsSync`] packs those blocks and sets their rows on the GPU.
//!
//! Rows are made from the mesh as the refit job took it from the volume; the GPU may have extracted a block again
//! since (a volume stroke under way). Each mesh is given a mark (the GPU rows' clock when its job began, seen as the
//! layers turning busy) and a block extracted for an edit after it shows the plain volume until a later swap lists it.

use std::time::Instant;

use sculpt_core::DirtySet;
use sculpt_core::generator::level_volume::{Key, key_order};
use sculpt_core::mesh_layers::MeshLayers;

use crate::gpu_levels::{GpuLevels, Row};

pub use crate::mesh_stack::RowsDirty;

/// Block `block` of [`MeshLayers::blocks`]'s rows: one per GPU vertex up to its last used one (0 for a block no face
/// uses). Its vertices' `vi` ascend, so that is the last one's plus one.
pub fn block_row_count(ml: &MeshLayers, block: usize) -> usize {
    let r = ml.block_vertices(block);
    if r.is_empty() { 0 } else { ml.vertex_vi()[r.end - 1] as usize + 1 }
}

/// Block `block`'s rows, in its GPU vertex order (`BlockSurface::verts`): per vertex the composite position less the
/// base position, and the composite normal less the base mesh normal (the shader adds that to the volume's gradient
/// normal, which is the base mesh normal near enough). GPU vertices no face uses get zero rows. `out` holds
/// [`block_row_count`] rows or more. Composite normals must be current (`Document::ensure_normals` with lazy normals).
pub fn layer_rows(ml: &MeshLayers, block: usize, out: &mut [Row]) {
    out.fill(Row::default());
    let doc = &ml.doc;
    let (pos, base, nrm, base_n, vi) = (doc.positions(), doc.base_positions(), doc.normals(), ml.base_normals(), ml.vertex_vi());
    for c in ml.block_vertices(block) {
        let v = doc.canonical_index(c);
        let off = pos[v] - base[v];
        let dn = nrm[v] - base_n[v];
        out[vi[c] as usize] = Row::new(off, dn);
    }
}

/// Finds the block of canonical vertices met one after another, mostly in a few blocks: the blocks met so far, the
/// last met first.
struct Near(Vec<(std::ops::Range<usize>, usize)>);

impl Near {
    /// Block (index into [`MeshLayers::blocks`]) of canonical vertex `c`, and whether it was not among those met.
    fn find(&mut self, ml: &MeshLayers, c: usize) -> (usize, bool) {
        if let Some((_, at)) = self.0.first().filter(|(r, _)| r.contains(&c)) {
            return (*at, false);
        }
        if let Some(j) = self.0.iter().position(|(r, _)| r.contains(&c)) {
            self.0.swap(0, j);
            return (self.0[0].1, false);
        }
        let at = ml.block_of(c);
        self.0.push((ml.block_vertices(at), at));
        let last = self.0.len() - 1;
        self.0.swap(0, last);
        (at, true)
    }
}

/// What the rows of some leaves' vertices change (see [`LeafIndex`]).
#[derive(Default)]
pub struct LeafChanges {
    /// (row index, row) for every row that differs from what is held.
    pub writes: Vec<(u32, Row)>,
    /// Blocks holding no rows or too few: to be set whole.
    pub whole: Vec<Key>,
    /// Blocks with a row in `writes`.
    pub changed: Vec<Key>,
    /// Rows made.
    pub made: usize,
}

/// Per leaf of the mesh, each owned vertex's row index in the rows held (or none), kept while the mesh and the places
/// stay: a dab's leaves are then read in order and only their rows compared.
#[derive(Default)]
pub struct LeafIndex {
    of: Option<((usize, usize), u64)>,
    leaves: Vec<Option<Box<[u32]>>>,
}

const NO_ROW: u32 = u32::MAX;

impl LeafIndex {
    /// The rows of the vertices of BVH leaves `leaves` of `ml.doc` against the rows held (`held`), on many threads:
    /// only those that differ are listed. Leaves the document does not have are skipped. The indices are kept for the
    /// mesh named `lay` while no place moves.
    fn changes_for(&mut self, lay: (usize, usize), ml: &MeshLayers, table: &BlockTable, leaves: &[u32], held: &crate::gpu_levels::rows::RowsRead) -> LeafChanges {
        if self.of != Some((lay, held.moved())) {
            self.of = Some((lay, held.moved()));
            self.leaves.clear();
        }
        self.changes(ml, table, leaves, held, leaves.len() >= 4)
    }

    fn changes(&mut self, ml: &MeshLayers, table: &BlockTable, leaves: &[u32], held: &crate::gpu_levels::rows::RowsRead, par: bool) -> LeafChanges {
        use rayon::prelude::*;
        let doc = &ml.doc;
        let (pos, base, nrm, base_n, vi, to_c) = (doc.positions(), doc.base_positions(), doc.normals(), ml.base_normals(), ml.vertex_vi(), ml.internal_to_canonical());
        let bvh = doc.bvh();
        let blocks = ml.blocks();
        self.leaves.resize_with(bvh.leaves.len(), || None);
        let mut work: Vec<(u32, Option<Box<[u32]>>)> = leaves.iter().filter(|&&l| (l as usize) < bvh.leaves.len()).map(|&l| (l, self.leaves[l as usize].take())).collect();
        work.sort_unstable_by_key(|w| w.0);
        work.dedup_by_key(|w| w.0);
        // Each vertex's row index, found once while the places stay.
        let find = |(l, idx): &mut (u32, Option<Box<[u32]>>)| {
            let r = bvh.leaves[*l as usize].owned_range();
            idx.get_or_insert_with(|| {
                const NO: u32 = u32::MAX;
                // The places of the blocks met, by block index (one way, 64 sets): (block, first row, rows).
                let mut seen = [(NO, NO, 0u32); 64];
                r.map(|v| {
                    let c = to_c[v] as usize;
                    let at = table.find(blocks, c);
                    let e = &mut seen[at & 63];
                    if e.0 != at as u32 {
                        *e = match held.place(blocks[at].0) {
                            Some((start, len)) => (at as u32, start, len),
                            None => (at as u32, NO, 0),
                        };
                    }
                    let i = vi[c] as u32;
                    if e.1 == NO || i >= e.2 { NO_ROW } else { e.1 + i }
                })
                .collect()
            });
        };
        if par {
            work.par_iter_mut().filter(|w| w.1.is_none()).for_each(find);
        } else {
            work.iter_mut().for_each(find);
        }
        // Then the rows, in pieces of a few hundred vertices (many reads in flight at once).
        const PIECE: usize = 256;
        let pieces: Vec<(usize, &[u32])> = work
            .iter()
            .flat_map(|(l, idx)| {
                let first = bvh.leaves[*l as usize].owned_range().start;
                idx.as_deref().unwrap_or(&[]).chunks(PIECE).enumerate().map(move |(j, c)| (first + j * PIECE, c))
            })
            .collect();
        let one = |&(first, idx): &(usize, &[u32])| -> LeafChanges {
            let mut out = LeafChanges::default();
            for (j, &at) in idx.iter().enumerate() {
                let v = first + j;
                if at == NO_ROW {
                    // A block with no rows held, or too few: made whole.
                    let k = blocks[table.find(blocks, to_c[v] as usize)].0;
                    if out.whole.last() != Some(&k) {
                        out.whole.push(k);
                    }
                    continue;
                }
                out.made += 1;
                let row = Row::new(pos[v] - base[v], nrm[v] - base_n[v]);
                if held.row(at) != row {
                    out.writes.push((at, row));
                    let k = blocks[table.find(blocks, to_c[v] as usize)].0;
                    if out.changed.last() != Some(&k) {
                        out.changed.push(k);
                    }
                }
            }
            out
        };
        let parts: Vec<LeafChanges> = if par { pieces.par_iter().map(one).collect() } else { pieces.iter().map(one).collect() };
        drop(pieces);
        for (l, idx) in work {
            self.leaves[l as usize] = idx;
        }
        let mut all = LeafChanges::default();
        for mut p in parts {
            all.writes.append(&mut p.writes);
            all.whole.append(&mut p.whole);
            all.changed.append(&mut p.changed);
            all.made += p.made;
        }
        all.whole.sort_unstable_by_key(key_order);
        all.whole.dedup();
        all
    }
}

/// Which block holds a canonical vertex, at a lookup or two: per 64 canonical vertices, the block holding the first.
pub struct BlockTable {
    first: Vec<u32>,
}

impl BlockTable {
    pub fn new(ml: &MeshLayers) -> BlockTable {
        let blocks = ml.blocks();
        let n = ml.keys().len();
        let mut first = Vec::with_capacity((n >> 6) + 1);
        let mut b = 0usize;
        for j in 0..=(n >> 6) {
            let c = j << 6;
            while b + 1 < blocks.len() && blocks[b + 1].1 as usize <= c {
                b += 1;
            }
            first.push(b as u32);
        }
        BlockTable { first }
    }

    /// The block (index into [`MeshLayers::blocks`]) holding canonical vertex `c`, as [`MeshLayers::block_of`].
    #[inline]
    pub fn find(&self, blocks: &[(Key, u32)], c: usize) -> usize {
        let mut b = self.first[c >> 6] as usize;
        while b + 1 < blocks.len() && blocks[b + 1].1 as usize <= c {
            b += 1;
        }
        b
    }
}

/// The blocks holding the vertices of leaves `dirty` (from `Document::take_geometry_dirty` after a layer dab or a
/// restore): what a [`RowsDirty::blocks`] lists for them. `DirtySet::All` gives every block.
pub fn dab_blocks(ml: &MeshLayers, dirty: &DirtySet) -> Vec<Key> {
    let blocks = ml.blocks();
    match dirty {
        DirtySet::None => Vec::new(),
        DirtySet::All => blocks.iter().map(|b| b.0).collect(),
        DirtySet::Leaves(leaves) => {
            let to_c = ml.internal_to_canonical();
            let bvh = ml.doc.bvh();
            let mut seen = vec![false; blocks.len()];
            let mut out = Vec::new();
            let mut near = Near(Vec::with_capacity(64));
            for &l in leaves {
                near.0.clear();
                for v in bvh.leaves[l as usize].owned_range() {
                    let (at, new) = near.find(ml, to_c[v] as usize);
                    if new && !seen[at] {
                        seen[at] = true;
                        out.push(blocks[at].0);
                    }
                }
            }
            out
        }
    }
}

/// Block `k`'s index in [`MeshLayers::blocks`].
pub fn block_index(ml: &MeshLayers, k: Key) -> Option<usize> {
    ml.blocks().binary_search_by(|b| key_order(&b.0).cmp(&key_order(&k))).ok()
}

/// What one [`LayerRowsSync::sync`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerRowsStats {
    /// Every block's rows were made.
    pub all: bool,
    /// A new mesh was swapped in.
    pub swapped: bool,
    /// Blocks whose rows were made whole, and their rows.
    pub blocks: usize,
    pub rows: usize,
    /// Of `blocks`, those shown now.
    pub shown: usize,
    /// Rows of the leaves given: made, and changed.
    pub leaf_rows: usize,
    pub leaf_changed: usize,
    /// Of `ms`: making the leaves' rows and comparing them with those held.
    pub leaf_ms: f64,
    /// Blocks shown again with their rows as they were (came into view, or extracted for the view only).
    pub repointed: usize,
    /// Blocks with rows not shown (no slot, or extracted for an edit after their mesh was taken).
    pub unshown: usize,
    /// The mark of the mesh the rows are made from.
    pub mark: u64,
    /// The whole call (packing, setting; sending happens with the draw).
    pub ms: f64,
}

/// Keeps a frame's mesh layer rows on the GPU (see the module).
#[derive(Default)]
pub struct LayerRowsSync {
    /// The mesh the rows are of (its block list), and the GPU surface they were set on.
    layout: Option<(usize, usize)>,
    gpu: Option<u64>,
    /// The mark of that mesh, of the job under way (taken when the layers turned busy), and whether they were busy.
    mark: u64,
    job: Option<u64>,
    busy: bool,
    /// Blocks whose rows layer dabs changed since the last swap: a swap composites the layers afresh, which can round
    /// their offsets a little differently, so they are made again then.
    dabbed: Vec<Key>,
    /// `dabbed` is made unique when it grows past this.
    dabbed_room: usize,
    /// For the leaves' rows, of the mesh `layout` names.
    table: Option<((usize, usize), BlockTable)>,
    leaf_index: LeafIndex,
    /// Every row is wanted (asked for while there was no GPU surface).
    want_all: bool,
    pub stats: LayerRowsStats,
}

impl LayerRowsSync {
    /// Once a frame after the volume's sync: bring the rows on `gpu` (a GPU surface named `gpu_id`, new when the
    /// viewport made it afresh) up to date with `ml` and `dirty`. `shown`: rows were shown before. Returns whether
    /// they are shown now.
    pub fn sync(&mut self, gpu: &mut GpuLevels, gpu_id: u64, ml: Option<&MeshLayers>, dirty: RowsDirty, shown: bool) -> bool {
        let t = Instant::now();
        let mut stats = LayerRowsStats::default();
        let Some(ml) = ml.filter(|_| !dirty.none) else {
            if shown || gpu.has_rows() {
                gpu.clear_all_rows();
            }
            // Blocks dabbed are still to be made again at the next swap once rows show again.
            let dabbed = if ml.is_some() { std::mem::take(&mut self.dabbed) } else { Vec::new() };
            *self = LayerRowsSync { stats, dabbed, ..Default::default() };
            return false;
        };
        let now = gpu.rows_mark();
        // A new mesh: swapped in (its mark is its job's) or first seen (now).
        let lay = (ml.blocks().as_ptr() as usize, ml.blocks().len());
        let first = self.layout.is_none();
        let swapped = self.layout != Some(lay);
        if swapped {
            self.mark = if first { now } else { self.job.take().unwrap_or(now) };
            self.layout = Some(lay);
        }
        let busy = ml.busy();
        if busy && (!self.busy || swapped) {
            self.job = Some(now);
        }
        self.busy = busy;
        stats.mark = self.mark;
        stats.swapped = swapped && !first;
        let fresh_gpu = self.gpu != Some(gpu_id);
        let asked = dirty.all || !dirty.blocks.is_empty() || !dirty.leaves.is_empty() || self.want_all;
        if !shown && !asked {
            self.stats = stats;
            return false;
        }
        // Turning on: the view changes with it (every block at its finest level), so rows are only kept this frame
        // and shown from the next sync on (`repoint_rows`).
        let turning_on = !shown;
        // Leaves of the mesh before a swap name other vertices: everything is made again.
        let all = dirty.all || self.want_all || fresh_gpu || turning_on || (swapped && !dirty.leaves.is_empty());
        self.want_all = false;
        self.gpu = Some(gpu_id);
        for k in &dirty.removed {
            gpu.clear_block_rows(*k);
        }
        if swapped && all {
            self.dabbed.clear();
        }
        let mut list: Vec<usize> = if all {
            if fresh_gpu || turning_on {
                gpu.clear_all_rows();
            } else {
                // Places kept (rows that did not change are not sent again); blocks the mesh no longer has go.
                for k in gpu.rows_keys() {
                    if block_index(ml, k).is_none() {
                        gpu.clear_block_rows(k);
                    }
                }
            }
            (0..ml.blocks().len()).filter(|&i| !ml.block_vertices(i).is_empty()).collect()
        } else {
            let mut keys = dirty.blocks;
            if swapped {
                // (`dabbed` is empty after this.) A dab given by blocks in the frame of a swap is not told apart from
                // the swap's blocks: dabs go by `leaves` (see `RowsDirty::leaves`).
                keys.append(&mut self.dabbed);
            } else {
                self.dabbed.extend_from_slice(&keys);
            }
            let mut list: Vec<usize> = Vec::with_capacity(keys.len());
            for k in &keys {
                match block_index(ml, *k) {
                    Some(i) if !ml.block_vertices(i).is_empty() => list.push(i),
                    _ => gpu.clear_block_rows(*k),
                }
            }
            list
        };
        // Layer dabs by leaves: only the rows that changed; blocks with no rows yet are made whole.
        if !all && !dirty.leaves.is_empty() {
            let tl = Instant::now();
            if self.table.as_ref().is_none_or(|t| t.0 != lay) {
                self.table = Some((lay, BlockTable::new(ml)));
            }
            let mut ch = self.leaf_index.changes_for(lay, ml, &self.table.as_ref().unwrap().1, &dirty.leaves, &gpu.rows_read());
            stats.leaf_ms = tl.elapsed().as_secs_f64() * 1e3;
            stats.leaf_changed = gpu.write_rows(&ch.writes, &ch.changed);
            stats.leaf_rows = ch.made;
            self.dabbed.append(&mut ch.changed);
            list.extend(ch.whole.iter().filter_map(|k| block_index(ml, *k)));
        } else if all && !dirty.leaves.is_empty() {
            // Made whole, but still to be made again at the next swap (see `dabbed`).
            self.dabbed.extend(dab_blocks(ml, &DirtySet::Leaves(dirty.leaves.iter().copied().filter(|&l| (l as usize) < ml.doc.bvh().leaves.len()).collect())));
        }
        list.sort_unstable();
        list.dedup();
        if !list.is_empty() {
            let blocks: Vec<(Key, usize)> = list.iter().map(|&i| (ml.blocks()[i].0, block_row_count(ml, i))).collect();
            stats.rows = blocks.iter().map(|b| b.1).sum();
            stats.blocks = blocks.len();
            stats.shown = gpu.set_many_block_rows_held(&blocks, self.mark, !turning_on, |j, rows| layer_rows(ml, list[j], rows));
        }
        if swapped && !all {
            // Blocks waiting for a newer mesh that this swap did not change: the rows they hold are the new mesh's too.
            let listed = |k: Key| list.binary_search_by(|&i| key_order(&ml.blocks()[i].0).cmp(&key_order(&k))).is_ok();
            gpu.remark_rows(self.mark, |k| !listed(k));
        }
        if self.dabbed.len() > self.dabbed_room {
            self.dabbed.sort_unstable_by_key(key_order);
            self.dabbed.dedup();
            self.dabbed_room = (self.dabbed.len() * 2).max(4096);
        }
        stats.all = all;
        // Blocks extracted again: those for the view only show their rows again; the rest wait for a later swap.
        let _ = gpu.take_rows_dropped();
        stats.repointed = if turning_on { 0 } else { gpu.repoint_rows() };
        stats.unshown = gpu.rows_unshown();
        stats.ms = t.elapsed().as_secs_f64() * 1e3;
        self.stats = stats;
        true
    }

    /// Rows were asked for while there was no GPU surface to set them on: make every one when there is.
    pub fn want_all(&mut self, dirty: &RowsDirty) {
        if dirty.all || !dirty.blocks.is_empty() || !dirty.leaves.is_empty() {
            self.want_all = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sculpt_core::brush::{self, BrushSettings, ClayBuildup};
    use sculpt_core::geom::Ray;
    use sculpt_core::mesh_layers::MeshLayerSettings;
    use std::time::Instant;

    fn ms(t: Instant) -> f64 {
        t.elapsed().as_secs_f64() * 1e3
    }

    fn med(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn dab(ml: &mut MeshLayers, d: glam::Vec3, r: f32) {
        let d = d.normalize();
        let h = ml.doc.raycast(&Ray::new(d * 3.0, -d)).unwrap();
        let s = BrushSettings { radius: r, strength: 0.5, ..Default::default() };
        let mut clay = ClayBuildup { height: 0.1, ..Default::default() };
        brush::stroke(&mut ml.doc, &mut clay, &s, &[(h.point, 1.0)]).unwrap();
    }

    /// Packing costs at about 2.5M triangles, without the GPU (`cargo test --release -p sculpt-app bench_layer_rows
    /// -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_layer_rows() {
        use rayon::prelude::*;
        let mode = crate::volume_mode::VolumeMode::ball(2.5 * 4e6 / 1.5);
        let mut ml = MeshLayers::from_volume(&mode.vol, MeshLayerSettings { cap: mode.cap, ..Default::default() }).unwrap();
        ml.doc.add_layer("a");
        for k in 0..4 {
            dab(&mut ml, glam::Vec3::new(-0.2 + 0.1 * k as f32, 0.0, 1.0), 0.12);
        }
        let _ = ml.doc.take_geometry_dirty();
        let pack_seq = |ml: &MeshLayers, list: &[usize]| {
            let mut buf = Vec::new();
            for &i in list {
                buf.resize(block_row_count(ml, i), Row::default());
                layer_rows(ml, i, &mut buf);
            }
            buf.len()
        };
        let pack_par = |ml: &MeshLayers, list: &[usize]| -> usize {
            list.par_iter().map(|&i| {
                let mut buf = vec![Row::default(); block_row_count(ml, i)];
                layer_rows(ml, i, &mut buf);
                buf.len()
            }).sum()
        };
        let (mut tb, mut ts, mut tp, mut nb, mut nr) = (vec![], vec![], vec![], vec![], vec![]);
        for k in 0..30 {
            dab(&mut ml, glam::Vec3::new(-0.2 + 0.015 * k as f32, 0.05, 1.0), 0.03);
            let g = ml.doc.take_geometry_dirty();
            let t = Instant::now();
            let keys = dab_blocks(&ml, &g);
            tb.push(ms(t));
            let list: Vec<usize> = keys.iter().filter_map(|k| block_index(&ml, *k)).collect();
            nb.push(list.len() as f64);
            nr.push(list.iter().map(|&i| block_row_count(&ml, i)).sum::<usize>() as f64);
            let t = Instant::now();
            pack_seq(&ml, &list);
            ts.push(ms(t));
            let t = Instant::now();
            pack_par(&ml, &list);
            tp.push(ms(t));
        }
        eprintln!("dab: {:.0} blocks, {:.0} rows; blocks from leaves {:.3} ms, packing {:.3} ms one thread, {:.3} ms many", med(nb), med(nr), med(tb), med(ts), med(tp));
        // Leaves: a 3k-vertex dab's rows against every row held.
        let blocks: Vec<(Key, u32)> = (0..ml.blocks().len()).map(|i| (ml.blocks()[i].0, block_row_count(&ml, i) as u32)).collect();
        let mut held = crate::gpu_levels::rows::HeldRows::new(&blocks);
        let t = Instant::now();
        let _ = BlockTable::new(&ml);
        eprintln!("block table: {:.3} ms", ms(t));
        {
            let r = held.read();
            let all: Vec<u32> = (0..ml.doc.bvh().leaves.len() as u32).collect();
            let ch = LeafIndex::default().changes(&ml, &BlockTable::new(&ml), &all, &r, true);
            drop(r);
            for (i, row) in ch.writes {
                held.mirror[i as usize] = row;
            }
        }
        let (mut tl, mut ts, mut nl, mut nc) = (vec![], vec![], vec![], vec![]);
        let mut index = LeafIndex::default();
        let table = BlockTable::new(&ml);
        {
            // Every leaf's rows found once (as a stroke's first dabs would).
            let all: Vec<u32> = (0..ml.doc.bvh().leaves.len() as u32).collect();
            let _ = index.changes(&ml, &table, &all, &held.read(), true);
        }
        let mut t1 = vec![];
        for k in 0..30 {
            dab(&mut ml, glam::Vec3::new(-0.3 + 0.01 * k as f32, 0.05, 1.0), 0.1);
            let DirtySet::Leaves(l) = ml.doc.take_geometry_dirty() else { panic!() };
            let t = Instant::now();
            let ch = if k % 2 == 0 { index.changes(&ml, &table, &l, &held.read(), true) } else { index.changes(&ml, &table, &l, &held.read(), false) };
            if k % 2 == 0 { tl.push(ms(t)) } else { t1.push(ms(t)) };
            nl.push(ch.made as f64);
            nc.push(ch.writes.len() as f64);
            for (i, row) in ch.writes {
                held.mirror[i as usize] = row;
            }
            let t = Instant::now();
            let _ = LeafIndex::default().changes(&ml, &table, &l, &held.read(), false);
            ts.push(ms(t));
        }
        eprintln!("leaves, found already, right after the dab: {:.3} ms one thread", med(t1));
        eprintln!("leaves: {:.0} rows made, {:.0} changed: {:.3} ms one thread finding the rows, {:.3} ms many with them found", med(nl), med(nc), med(ts), med(tl));
        let mut tr = vec![];
        for _ in 0..30 {
            let t = Instant::now();
            let v: Vec<u64> = (0..40u64).into_par_iter().map(|i| i * 3).collect();
            std::hint::black_box(v);
            tr.push(ms(t));
        }
        eprintln!("rayon: 40 empty tasks {:.3} ms", med(tr));
        let all: Vec<usize> = (0..ml.blocks().len()).filter(|&i| !ml.block_vertices(i).is_empty()).collect();
        let t = Instant::now();
        let n = pack_par(&ml, &all);
        eprintln!("all: {} blocks, {n} rows packed in {:.2} ms many threads", all.len(), ms(t));
    }
}
