//! Mesh layer rows: per drawn vertex, an offset and a change of normal, which the viewport adds to the surface.
//!
//! A block's rows are in its GPU vertex order (the order of `BlockSurface::verts`, the vertex a cell's word names), one
//! per vertex. They live in one storage buffer at a place kept per block (`Key`); per slot a pair (first row, rows)
//! says where its block's rows are, or `NONE`. A block the GPU extracts again (its vertices may be renumbered) or whose
//! slot goes loses its pair until its rows are set again; its rows stay where they are, and the block is listed in
//! [`Rows::take_dropped`]. A CPU copy of every row is kept, so growing or packing the buffer needs no GPU copy.

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use glam::Vec3;
use sculpt_core::generator::level_volume::{HashMap as KeyMap, Key};

/// No rows for this slot.
pub const NONE: u32 = u32::MAX;
/// A block's place has a sixteenth more room than its rows, in whole fours (so a block that gains a few vertices
/// mostly stays put).
const ROUND: u32 = 4;

/// One vertex's row: the offset added to its position, and its change of normal packed (see [`pack_normal_change`]).
/// All zero changes nothing, bit for bit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Row {
    pub offset: [f32; 3],
    pub normal: u32,
}

impl Row {
    pub fn new(offset: Vec3, normal_change: Vec3) -> Row {
        Row { offset: offset.to_array(), normal: pack_normal_change(normal_change) }
    }
}

/// A change of normal (each part within -2..2) in 32 bits: x and y in 11 bits, z in 10, signed, a 512th (z a 256th)
/// a step. Zero packs to 0.
pub fn pack_normal_change(d: Vec3) -> u32 {
    let q = |v: f32, scale: f32, bits: u32| -> u32 {
        let lim = (1i32 << (bits - 1)) - 1;
        let i = (v * scale).round().clamp(-(lim as f32) - 1.0, lim as f32) as i32;
        (i as u32) & ((1u32 << bits) - 1)
    };
    q(d.x, 512.0, 11) | (q(d.y, 512.0, 11) << 11) | (q(d.z, 256.0, 10) << 22)
}

/// [`pack_normal_change`] undone, as the shader does it.
pub fn unpack_normal_change(w: u32) -> Vec3 {
    let x = ((w << 21) as i32 >> 21) as f32 / 512.0;
    let y = ((w << 10) as i32 >> 21) as f32 / 512.0;
    let z = (w as i32 >> 22) as f32 / 256.0;
    Vec3::new(x, y, z)
}

/// Where a block's rows are: first row, rows set, rows of room.
#[derive(Clone, Copy, Debug)]
struct Place {
    start: u32,
    len: u32,
    room: u32,
}

pub(super) struct Rows {
    /// Per slot (first row, rows), `NONE` for none; and the GPU's copy.
    slot_rows: Vec<[u32; 2]>,
    slot_key: Vec<Option<Key>>,
    pub start_buf: wgpu::Buffer,
    slot_dirty: Vec<u32>,
    /// Per slot, the clock when the GPU last extracted it (or its slot was let go).
    extracted_at: Vec<u64>,
    pub clock: u64,
    /// Every row, and the GPU's copy with room for `room` rows.
    mirror: Vec<Row>,
    pub rows_buf: wgpu::Buffer,
    room: u32,
    at: KeyMap<Key, Place>,
    /// Free places by size.
    free: BTreeMap<u32, Vec<u32>>,
    free_rows: u64,
    /// Row ranges changed since the last flush.
    dirty: Vec<(u32, u32)>,
    dropped: Vec<Key>,
    /// Bumped when a buffer is made afresh (whoever binds them binds again).
    pub made: u64,
}

fn round_up(n: u32) -> u32 {
    (n + n / 16).div_ceil(ROUND).max(1) * ROUND
}

impl Rows {
    pub fn new(device: &wgpu::Device, slots: u32) -> Rows {
        let start_buf = Self::make(device, "levels row starts", slots as u64 * 8);
        let rows_buf = Self::make(device, "levels rows", 16 * 16);
        Rows {
            slot_rows: Vec::new(),
            slot_key: Vec::new(),
            start_buf,
            slot_dirty: Vec::new(),
            extracted_at: Vec::new(),
            clock: 1,
            mirror: Vec::new(),
            rows_buf,
            room: 16,
            at: KeyMap::default(),
            free: BTreeMap::new(),
            free_rows: 0,
            dirty: Vec::new(),
            dropped: Vec::new(),
            made: 0,
        }
        .with_slots(slots)
    }

    fn with_slots(mut self, slots: u32) -> Rows {
        self.slot_rows = vec![[NONE, 0]; slots as usize];
        self.slot_key = vec![None; slots as usize];
        self.extracted_at = vec![0; slots as usize];
        self.slot_dirty = (0..slots).collect();
        self
    }

    fn make(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }

    /// Room for `slots` slots (each keeping its pair).
    pub fn fit_slots(&mut self, device: &wgpu::Device, slots: u32) {
        if slots as usize <= self.slot_rows.len() {
            return;
        }
        self.slot_rows.resize(slots as usize, [NONE, 0]);
        self.slot_key.resize(slots as usize, None);
        self.extracted_at.resize(slots as usize, 0);
        self.start_buf = Self::make(device, "levels row starts", slots as u64 * 8);
        self.made += 1;
        // All of it goes up again.
        self.slot_dirty.clear();
        self.slot_dirty.extend(0..slots);
    }

    /// The GPU extracts `slots` again, or lets them go: their pairs go (the blocks they had are listed as dropped).
    pub fn extracted(&mut self, slots: impl IntoIterator<Item = u32>) {
        self.clock += 1;
        for s in slots {
            let i = s as usize;
            if i >= self.slot_rows.len() {
                continue;
            }
            self.extracted_at[i] = self.clock;
            if self.slot_rows[i][0] != NONE {
                self.slot_rows[i] = [NONE, 0];
                self.slot_dirty.push(s);
                if let Some(k) = self.slot_key[i].take() {
                    self.dropped.push(k);
                }
            }
        }
    }

    /// Every slot handed out afresh (a rebuild): no pairs.
    pub fn reset_slots(&mut self) {
        self.clock += 1;
        for i in 0..self.slot_rows.len() {
            self.extracted_at[i] = self.clock;
            if self.slot_rows[i][0] != NONE {
                self.slot_rows[i] = [NONE, 0];
                self.slot_dirty.push(i as u32);
                if let Some(k) = self.slot_key[i].take() {
                    self.dropped.push(k);
                }
            }
        }
    }

    pub fn take_dropped(&mut self) -> Vec<Key> {
        std::mem::take(&mut self.dropped)
    }

    /// A place for `n` rows of `k` (its old one if there is room).
    fn place(&mut self, k: Key, n: u32) -> Place {
        if let Some(p) = self.at.get_mut(&k) {
            if n <= p.room {
                p.len = n;
                return *p;
            }
            let old = *p;
            self.at.remove(&k);
            self.give_back(old);
        }
        let room = round_up(n);
        // The smallest free place that fits, if not much bigger.
        let found = self.free.range_mut(room..room + room / 8 + ROUND).find(|(_, v)| !v.is_empty()).map(|(r, v)| (*r, v.pop().unwrap()));
        let p = match found {
            Some((r, start)) => {
                self.free_rows -= r as u64;
                Place { start, len: n, room: r }
            }
            None => {
                let start = self.mirror.len() as u32;
                self.mirror.resize(self.mirror.len() + room as usize, Row::default());
                Place { start, len: n, room }
            }
        };
        self.at.insert(k, p);
        p
    }

    fn give_back(&mut self, p: Place) {
        self.free.entry(p.room).or_default().push(p.start);
        self.free_rows += p.room as u64;
    }

    /// Set `k`'s rows; pointed at from its slot `slot` unless the GPU extracted that slot after `since`. True if
    /// they show.
    pub fn set(&mut self, k: Key, slot: Option<u32>, rows: &[Row], since: u64) -> bool {
        let p = self.place(k, rows.len() as u32);
        self.mirror[p.start as usize..p.start as usize + rows.len()].copy_from_slice(rows);
        self.dirty.push((p.start, p.start + rows.len() as u32));
        let Some(s) = slot else { return false };
        let i = s as usize;
        if self.extracted_at[i] > since {
            return false;
        }
        self.slot_rows[i] = [p.start, p.len];
        self.slot_key[i] = Some(k);
        self.slot_dirty.push(s);
        true
    }

    /// Set the rows of many blocks at once: `blocks[i]` is (block, its slot, rows), and `fill(i, rows)` writes block
    /// i's rows in place, on many threads. Returns how many show (see [`Rows::set`]).
    pub fn set_many(&mut self, blocks: &[(Key, Option<u32>, u32)], since: u64, fill: &(dyn Fn(usize, &mut [Row]) + Sync)) -> usize {
        use rayon::prelude::*;
        let places: Vec<Place> = blocks.iter().map(|(k, _, n)| self.place(*k, *n)).collect();
        let mut order: Vec<usize> = (0..places.len()).collect();
        order.sort_unstable_by_key(|&i| places[i].start);
        {
            // Each block's rows as a slice of its own, in the order they lie.
            let mut parts: Vec<(usize, &mut [Row])> = Vec::with_capacity(order.len());
            let mut rest: &mut [Row] = &mut self.mirror;
            let mut at = 0u32;
            for &i in &order {
                let p = places[i];
                let tail = std::mem::take(&mut rest).split_at_mut((p.start - at) as usize).1;
                let (mine, tail) = tail.split_at_mut(p.len as usize);
                parts.push((i, mine));
                rest = tail;
                at = p.start + p.len;
            }
            parts.into_par_iter().for_each(|(i, rows)| fill(i, rows));
        }
        let mut shown = 0;
        for (i, (k, slot, _)) in blocks.iter().enumerate() {
            let p = places[i];
            self.dirty.push((p.start, p.start + p.len));
            if let Some(s) = *slot
                && self.extracted_at[s as usize] <= since
            {
                self.slot_rows[s as usize] = [p.start, p.len];
                self.slot_key[s as usize] = Some(*k);
                self.slot_dirty.push(s);
                shown += 1;
            }
        }
        shown
    }

    /// Drop `k`'s rows (from slot `slot` too).
    pub fn clear(&mut self, k: Key, slot: Option<u32>) {
        if let Some(p) = self.at.remove(&k) {
            self.give_back(p);
        }
        if let Some(s) = slot
            && self.slot_key.get(s as usize).copied().flatten() == Some(k)
        {
            self.slot_rows[s as usize] = [NONE, 0];
            self.slot_key[s as usize] = None;
            self.slot_dirty.push(s);
        }
    }

    pub fn clear_all(&mut self) {
        for i in 0..self.slot_rows.len() {
            if self.slot_rows[i][0] != NONE {
                self.slot_rows[i] = [NONE, 0];
                self.slot_key[i] = None;
                self.slot_dirty.push(i as u32);
            }
        }
        self.at.clear();
        self.free.clear();
        self.free_rows = 0;
        self.mirror.clear();
        self.dirty.clear();
    }

    pub fn has_rows(&self) -> bool {
        !self.at.is_empty()
    }

    /// Bytes on the GPU: the rows' buffer and the slots' pairs.
    pub fn gpu_bytes(&self) -> u64 {
        self.rows_buf.size() + self.start_buf.size()
    }

    /// Rows in use and rows held (in use, rounding and free places).
    pub fn counts(&self) -> (u64, u64) {
        (self.at.values().map(|p| p.len as u64).sum(), self.mirror.len() as u64)
    }

    /// Send what changed. Packs the places afresh when over a quarter is free, and grows the buffer (a sixteenth to
    /// spare) when they outgrow it.
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.free_rows * 4 > self.mirror.len() as u64 && self.mirror.len() > 65536 {
            self.repack();
        }
        let mut whole = false;
        if self.mirror.len() as u32 > self.room {
            let n = self.mirror.len() as u32;
            self.room = (n + n / 16).div_ceil(4096) * 4096;
            self.rows_buf = Self::make(device, "levels rows", self.room as u64 * 16);
            self.made += 1;
            whole = true;
        }
        if whole {
            queue.write_buffer(&self.rows_buf, 0, bytemuck::cast_slice(&self.mirror));
            self.dirty.clear();
        } else if !self.dirty.is_empty() {
            // Ranges near one another in one write each.
            let mut d = std::mem::take(&mut self.dirty);
            d.sort_unstable();
            let mut i = 0;
            while i < d.len() {
                let (a, mut b) = d[i];
                let mut j = i + 1;
                while j < d.len() && d[j].0 <= b + 256 {
                    b = b.max(d[j].1);
                    j += 1;
                }
                queue.write_buffer(&self.rows_buf, a as u64 * 16, bytemuck::cast_slice(&self.mirror[a as usize..b as usize]));
                i = j;
            }
            self.dirty = d;
            self.dirty.clear();
        }
        if !self.slot_dirty.is_empty() {
            let mut m = std::mem::take(&mut self.slot_dirty);
            m.sort_unstable();
            m.dedup();
            let mut i = 0;
            while i < m.len() {
                let mut j = i + 1;
                while j < m.len() && m[j] - m[j - 1] <= 32 {
                    j += 1;
                }
                let (a, b) = (m[i] as usize, m[j - 1] as usize);
                queue.write_buffer(&self.start_buf, a as u64 * 8, bytemuck::cast_slice(&self.slot_rows[a..=b]));
                i = j;
            }
            self.slot_dirty = m;
            self.slot_dirty.clear();
        }
    }

    /// Every place packed end to end (in the order they are now), the slots' pairs moved with them.
    fn repack(&mut self) {
        let mut places: Vec<(Key, Place)> = self.at.iter().map(|(k, p)| (*k, *p)).collect();
        places.sort_unstable_by_key(|(_, p)| p.start);
        let mut mirror = Vec::with_capacity(places.iter().map(|(_, p)| p.room as usize).sum());
        let mut moved: KeyMap<u32, u32> = KeyMap::default();
        for (k, p) in &mut places {
            let start = mirror.len() as u32;
            mirror.extend_from_slice(&self.mirror[p.start as usize..(p.start + p.room) as usize]);
            moved.insert(p.start, start);
            p.start = start;
            self.at.insert(*k, *p);
        }
        self.mirror = mirror;
        self.free.clear();
        self.free_rows = 0;
        for (i, r) in self.slot_rows.iter_mut().enumerate() {
            if r[0] != NONE {
                r[0] = moved[&r[0]];
                self.slot_dirty.push(i as u32);
            }
        }
        // Everything goes up again (the buffer is as big as before or bigger).
        self.dirty.clear();
        self.dirty.push((0, self.mirror.len() as u32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_change_packs_and_unpacks() {
        assert_eq!(pack_normal_change(Vec3::ZERO), 0);
        for d in [Vec3::new(0.3, -0.7, 1.2), Vec3::new(-1.99, 1.99, -1.99), Vec3::new(0.001, 0.0, -0.002)] {
            let back = unpack_normal_change(pack_normal_change(d));
            assert!((back - d).abs().max_element() <= 1.0 / 256.0, "{d} -> {back}");
        }
    }
}
