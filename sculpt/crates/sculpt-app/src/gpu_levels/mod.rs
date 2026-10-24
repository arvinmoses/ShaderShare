//! The multi-resolution volume's surface, built and drawn on the GPU.
//!
//! The CPU holds the field (`sculpt_core::generator::level_volume`) and edits it; this keeps a copy of every stored
//! block on the GPU (its 9x9x9 samples at 16 bits) and extracts the surface there, block by block, straight into the
//! buffers the viewport draws from: nothing about the surface crosses the bus. Each block of any level has a slot; a
//! hash table on the GPU finds the slot of a (level, block). A stroke uploads the samples of the blocks it changed and
//! re-extracts the blocks around them ([`LevelVolume::upload_set`], [`LevelVolume::extract_set`]), by the same rules as
//! the CPU's reference extractor, which `--gpu-levels-check` compares it against exactly.
//!
//! Each block's surface is a cluster: its quads in pages of 64 (as triangle indices naming cell and piece, resolved to
//! a vertex when drawn), its bounds and its normal cone. Every frame a cull pass writes the pages of the clusters in
//! view and facing the eye as indirect draws, and the viewport draws them in one call.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;
use sculpt_core::generator::level_view::LevelView;
use sculpt_core::generator::level_volume::{HashSet as KeySet, Key, LevelVolume, Show, VERTS_PER_BLOCK, piece_table, quantize};

pub mod check;
mod join;

const SHADER: &str = include_str!("levels.wgsl");
const PAGE_INDICES: u64 = 384;
const SLOT_WORDS: u64 = 32;
const SAMPLE_WORDS: u64 = 365;
const VSTRIDE: u64 = VERTS_PER_BLOCK as u64;
/// Blocks whose detail is measured again per sync, after they changed (for choosing the level by distance).
const MEASURE_PER_SYNC: usize = 768;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    origin: [f32; 3],
    voxel0: f32,
    cap: u32,
    levels: u32,
    work_count: u32,
    hash_mask: u32,
    release_count: u32,
    page_capacity: u32,
    slot_count: u32,
    cull: u32,
    eye: [f32; 4],
    planes: [[f32; 4]; 6],
}

/// What one sync did.
#[derive(Clone, Copy, Debug, Default)]
pub struct SyncStats {
    pub uploaded: usize,
    pub extracted: usize,
    pub released: usize,
    /// CPU time: working out what to send, gathering and sending it, recording the passes.
    pub cpu_ms: f64,
    pub bytes: usize,
    /// Everything was rebuilt (first sync, a cap change, or buffers that had to grow).
    pub full: bool,
}

struct Buffers {
    slots: u32,
    pages: u32,
    hash_len: u32,
    params: wgpu::Buffer,
    cull_params: wgpu::Buffer,
    samples: wgpu::Buffer,
    hash: wgpu::Buffer,
    meta: wgpu::Buffer,
    work: wgpu::Buffer,
    work_len: u32,
    pub verts: wgpu::Buffer,
    pub cells: wgpu::Buffer,
    pub pages_buf: wgpu::Buffer,
    slot_pages: wgpu::Buffer,
    free_pages: wgpu::Buffer,
    pub status: wgpu::Buffer,
    clusters: wgpu::Buffer,
    table: wgpu::Buffer,
    pub draws: wgpu::Buffer,
    group: wgpu::BindGroup,
    cull_group: wgpu::BindGroup,
}

/// The GPU side of a [`LevelVolume`].
pub struct GpuLevels {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    release: wgpu::ComputePipeline,
    vertices: wgpu::ComputePipeline,
    faces: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
    buf: Buffers,
    /// Slot per stored block, and block per slot.
    slot_of: HashMap<Key, u32>,
    key_of: Vec<Option<Key>>,
    free: Vec<u32>,
    /// One past the highest slot in use.
    high: u32,
    /// The hash table as on the GPU: (block coordinates, level << 24 | slot), -1 empty, -2 removed.
    hash: Vec<[i32; 4]>,
    hash_live: usize,
    hash_dead: usize,
    hash_dirty: Vec<u32>,
    pub cap: usize,
    levels: usize,
    origin: Vec3,
    voxel0: f32,
    /// Status read back from the GPU (free pages, ran out, draws, quads dropped).
    status_staging: wgpu::Buffer,
    status_state: Arc<AtomicU8>,
    pub status: [i32; 4],
    /// Set when a cull pass recorded a status copy that the frame's submit has not mapped yet.
    status_copy: std::sync::atomic::AtomicBool,
    /// GPU timestamps around the extraction passes, when the device has them: the queries, where they resolve, a
    /// buffer to read them from and its state, and nanoseconds per tick.
    timer: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer, Arc<AtomicU8>, f32)>,
    /// Per slot, a hash of the samples last sent (0: none yet): a block whose 16-bit samples come out the same is
    /// neither sent nor rebuilt, since the GPU would build exactly what it has.
    sent: Vec<u64>,
    /// GPU time of the last extraction read back, in milliseconds.
    pub gpu_ms: f64,
    pub indirect_count: bool,
    /// Set when the pages ran out: the next sync grows them and rebuilds.
    grow_pages: bool,
    cap_changed: bool,
    /// Which blocks are shown, chosen by distance from the eye (a hidden block is not on the GPU at all, as if it
    /// were not stored), and the eye with its pixels per unit at distance one.
    pub view: LevelView,
    eye: Option<(Vec3, f32)>,
    /// Generators joined on the GPU (made when first needed).
    join: Option<join::Join>,
    /// An extraction a sync left for the join after it (slots to extract, slots to release): the blocks around a
    /// join's read its vertices, so they are extracted with it.
    deferred: Option<(Vec<u32>, Vec<u32>)>,
    /// Bumped whenever the buffers are made afresh, so whoever draws from them binds them again.
    pub buffers_made: u64,
}

fn hash_of(c: IVec3, l: u32) -> u32 {
    (c.x as u32).wrapping_mul(73856093) ^ (c.y as u32).wrapping_mul(19349663) ^ (c.z as u32).wrapping_mul(83492791) ^ l.wrapping_mul(2654435761)
}

/// A block's 9x9x9 samples as the GPU holds them: 16 bits each, two to a word.
/// A hash of a block's packed samples (never 0, which means none sent).
fn sample_hash(data: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in data {
        h = (h ^ w as u64).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h | 1
}

pub(super) fn packed_samples(vol: &LevelVolume, k: Key, hidden: Option<&KeySet<Key>>) -> Vec<u32> {
    let (l, bc) = (k.0 as usize, k.1);
    let band = vol.band(l);
    let mut g = [0.0f32; 729];
    vol.gather9_hiding(l, bc, &mut g, hidden);
    let mut out = vec![0u32; SAMPLE_WORDS as usize];
    for (i, v) in g.iter().enumerate() {
        let q = quantize(*v, band) as u16 as u32;
        out[i / 2] |= q << (16 * (i % 2));
    }
    out
}

impl GpuLevels {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, vol: &LevelVolume, cap: usize) -> GpuLevels {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("levels.wgsl"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let (ro, rw) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
        let entries = [
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, ro),
            entry(2, ro),
            entry(3, ro),
            entry(4, ro),
            entry(5, rw),
            entry(6, rw),
            entry(7, rw),
            entry(8, rw),
            entry(9, rw),
            entry(10, rw),
            entry(11, rw),
            entry(12, ro),
            entry(13, rw),
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("levels"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("levels"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipe = |name: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: Some(&pl), module: &module, entry_point: Some(name), compilation_options: Default::default(), cache: None });
        let indirect_count = device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let blocks = vol.keys().len() as u32;
        let slots = (blocks * 2).max(4096);
        let buf = Self::buffers(device, &layout, slots, slots * 4, 1024);
        let status_staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels status staging"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let g = GpuLevels {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            release: pipe("release"),
            vertices: pipe("vertices"),
            faces: pipe("faces"),
            cull: pipe("cull"),
            buf,
            slot_of: HashMap::new(),
            key_of: Vec::new(),
            free: Vec::new(),
            high: 0,
            hash: Vec::new(),
            hash_live: 0,
            hash_dead: 0,
            hash_dirty: Vec::new(),
            cap: cap.min(vol.levels() - 1),
            levels: vol.levels(),
            origin: vol.origin(),
            voxel0: vol.voxel0,
            status_staging,
            status_state: Arc::new(AtomicU8::new(0)),
            status: [0; 4],
            status_copy: std::sync::atomic::AtomicBool::new(false),
            timer: device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
                let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("levels timer"), ty: wgpu::QueryType::Timestamp, count: 2 });
                let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels timer resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels timer staging"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                (set, resolve, staging, Arc::new(AtomicU8::new(0)), queue.get_timestamp_period())
            }),
            gpu_ms: 0.0,
            sent: Vec::new(),
            indirect_count,
            grow_pages: false,
            buffers_made: 1,
            cap_changed: false,
            view: LevelView::default(),
            eye: None,
            join: None,
            deferred: None,
        };
        g.queue.write_buffer(&g.buf.table, 0, bytemuck::cast_slice(&piece_table()));
        g
    }

    fn buffers(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, slots: u32, pages: u32, hash_len: u32) -> Buffers {
        let make = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let st = wgpu::BufferUsages::STORAGE;
        let s = slots as u64;
        let params = make("levels params", std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM);
        let cull_params = make("levels cull params", std::mem::size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM);
        let samples = make("levels samples", s * SAMPLE_WORDS * 4, st);
        let hash = make("levels hash", hash_len as u64 * 16, st);
        let meta = make("levels meta", s * 16, st);
        let work_len = slots * 2;
        let work = make("levels work", work_len as u64 * 4, st);
        let verts = make("levels verts", s * VSTRIDE * 8, st);
        let cells = make("levels cells", s * 512 * 4, st);
        let pages_buf = make("levels pages", pages as u64 * PAGE_INDICES * 4, st | wgpu::BufferUsages::INDEX);
        let slot_pages = make("levels slot pages", s * SLOT_WORDS * 4, st);
        let free_pages = make("levels free pages", pages as u64 * 4, st);
        let status = make("levels status", 16, st | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC);
        let clusters = make("levels clusters", s * 48, st);
        let table = make("levels table", 256 * 4, st);
        let draws = make("levels draws", pages as u64 * 20, st | wgpu::BufferUsages::INDIRECT);
        let group_with = |p: &wgpu::Buffer| {
            let bufs = [p, &samples, &hash, &meta, &work, &verts, &cells, &pages_buf, &slot_pages, &free_pages, &status, &clusters, &table, &draws];
            let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("levels"), layout, entries: &entries })
        };
        let group = group_with(&params);
        let cull_group = group_with(&cull_params);
        Buffers { slots, pages, hash_len, params, cull_params, samples, hash, meta, work, work_len, verts, cells, pages_buf, slot_pages, free_pages, status, clusters, table, draws, group, cull_group }
    }

    pub fn page_capacity(&self) -> u32 {
        self.buf.pages
    }

    pub fn buffers_for_draw(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        (&self.buf.pages_buf, &self.buf.draws, &self.buf.status, &self.buf.verts, &self.buf.cells)
    }

    pub fn meta_buffer(&self) -> &wgpu::Buffer {
        &self.buf.meta
    }

    pub fn slots(&self) -> usize {
        self.slot_of.len()
    }

    // ---------------------------------------------------------------- the slot table

    fn hash_insert(&mut self, k: Key, slot: u32) {
        let mask = self.hash.len() as u32 - 1;
        let mut h = hash_of(k.1, k.0 as u32) & mask;
        loop {
            let e = &mut self.hash[h as usize];
            if e[3] < 0 {
                if e[3] == -2 {
                    self.hash_dead -= 1;
                }
                *e = [k.1.x, k.1.y, k.1.z, ((k.0 as i32) << 24) | slot as i32];
                self.hash_live += 1;
                self.hash_dirty.push(h);
                return;
            }
            h = (h + 1) & mask;
        }
    }

    fn hash_remove(&mut self, k: Key) {
        let mask = self.hash.len() as u32 - 1;
        let mut h = hash_of(k.1, k.0 as u32) & mask;
        loop {
            let e = &mut self.hash[h as usize];
            if e[3] == -1 {
                return;
            }
            if e[3] >= 0 && e[0] == k.1.x && e[1] == k.1.y && e[2] == k.1.z && (e[3] >> 24) == k.0 as i32 {
                e[3] = -2;
                self.hash_live -= 1;
                self.hash_dead += 1;
                self.hash_dirty.push(h);
                return;
            }
            h = (h + 1) & mask;
        }
    }

    /// Size the table for the blocks there are (at most half full, counting removed entries) and fill it afresh.
    /// True if it was rebuilt (and must be sent whole).
    fn hash_fit(&mut self) -> bool {
        let want = ((self.slot_of.len() + 64) * 2).next_power_of_two().max(1024);
        if self.hash.len() >= want && (self.hash_live + self.hash_dead) * 2 <= self.hash.len() {
            return false;
        }
        self.hash = vec![[0, 0, 0, -1]; want.max(self.hash.len())];
        self.hash_live = 0;
        self.hash_dead = 0;
        let all: Vec<(Key, u32)> = self.slot_of.iter().map(|(k, s)| (*k, *s)).collect();
        for (k, s) in all {
            self.hash_insert(k, s);
        }
        self.hash_dirty.clear();
        true
    }

    fn take_slot(&mut self, k: Key) -> Option<u32> {
        let s = match self.free.pop() {
            Some(s) => s,
            None if (self.high as usize) < self.buf.slots as usize => {
                self.high += 1;
                self.high - 1
            }
            None => return None,
        };
        if self.key_of.len() <= s as usize {
            self.key_of.resize(s as usize + 1, None);
        }
        self.key_of[s as usize] = Some(k);
        if self.sent.len() <= s as usize {
            self.sent.resize(s as usize + 1, 0);
        }
        self.sent[s as usize] = 0;
        if let Some(j) = self.join.as_mut() {
            j.slot_taken(s);
        }
        self.slot_of.insert(k, s);
        Some(s)
    }

    // ---------------------------------------------------------------- syncing

    fn params(&self, work: u32, release: u32) -> Params {
        Params {
            origin: self.origin.to_array(),
            voxel0: self.voxel0,
            cap: self.cap as u32,
            levels: self.levels as u32,
            work_count: work,
            hash_mask: self.hash.len() as u32 - 1,
            release_count: release,
            page_capacity: self.buf.pages,
            slot_count: self.high,
            cull: 0,
            eye: [0.0; 4],
            planes: [[0.0; 4]; 6],
        }
    }

    /// Show levels up to `cap` (rebuilding the whole surface on the next sync if that changes it).
    pub fn set_cap(&mut self, cap: usize) {
        let cap = cap.min(self.levels - 1);
        if cap != self.cap {
            self.cap = cap;
            self.cap_changed = true;
        }
    }

    /// Choose the levels shown from `eye` (with `focal_px` pixels per unit at distance one) on the next syncs.
    pub fn set_eye(&mut self, eye: Vec3, focal_px: f32) {
        self.eye = Some((eye, focal_px));
    }

    /// The view the GPU's surface is of: levels up to the cap, less the blocks hidden.
    pub fn show(&self) -> Show<'_> {
        self.view.show(self.cap)
    }

    /// Everything from scratch: buffers sized for the volume, every block shown uploaded and extracted.
    fn rebuild(&mut self, vol: &LevelVolume, stats: &mut SyncStats) {
        let blocks: Vec<Key> = vol.keys().into_iter().filter(|k| !self.view.hides(*k)).collect();
        let slots = ((blocks.len() as u32) * 2).max(self.buf.slots).max(4096);
        let mut pages = self.buf.pages.max(slots * 4);
        if self.grow_pages {
            pages *= 2;
            self.grow_pages = false;
        }
        if slots != self.buf.slots || pages != self.buf.pages {
            self.buf = Self::buffers(&self.device, &self.layout, slots, pages, 1024);
            self.buffers_made += 1;
            self.queue.write_buffer(&self.buf.table, 0, bytemuck::cast_slice(&piece_table()));
        }
        self.slot_of.clear();
        self.key_of.clear();
        self.free.clear();
        self.high = 0;
        for k in &blocks {
            self.take_slot(*k);
        }
        self.hash.clear();
        self.hash_fit();
        self.send_hash_whole();
        // All pages free, every slot without pages.
        let free: Vec<u32> = (0..self.buf.pages).rev().collect();
        self.queue.write_buffer(&self.buf.free_pages, 0, bytemuck::cast_slice(&free));
        self.queue.write_buffer(&self.buf.status, 0, bytemuck::cast_slice(&[self.buf.pages as i32, 0, 0, 0]));
        self.queue.write_buffer(&self.buf.slot_pages, 0, &vec![0u8; (self.buf.slots as u64 * SLOT_WORDS * 4) as usize]);
        self.upload(vol, &blocks, stats, None);
        self.send_meta(&blocks);
        self.extract(&blocks, &[], stats);
        stats.full = true;
    }

    fn send_hash_whole(&mut self) {
        if self.hash.len() as u32 > self.buf.hash_len {
            // A bigger table needs a bigger buffer, and so a new bind group.
            let len = self.hash.len() as u32;
            let pages = self.buf.pages;
            let slots = self.buf.slots;
            let old = std::mem::replace(&mut self.buf, Self::buffers(&self.device, &self.layout, slots, pages, len));
            self.buffers_made += 1;
            self.copy_contents(&old);
        }
        self.queue.write_buffer(&self.buf.hash, 0, bytemuck::cast_slice(&self.hash));
        self.hash_dirty.clear();
    }

    /// Copy what the GPU built into freshly made buffers of the same slot and page counts.
    fn copy_contents(&self, old: &Buffers) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (a, b) in [(&old.samples, &self.buf.samples), (&old.meta, &self.buf.meta), (&old.verts, &self.buf.verts), (&old.cells, &self.buf.cells), (&old.pages_buf, &self.buf.pages_buf), (&old.slot_pages, &self.buf.slot_pages), (&old.free_pages, &self.buf.free_pages), (&old.status, &self.buf.status), (&old.clusters, &self.buf.clusters), (&old.table, &self.buf.table)] {
            enc.copy_buffer_to_buffer(a, 0, b, 0, a.size().min(b.size()));
        }
        self.queue.submit([enc.finish()]);
    }

    fn send_meta(&self, keys: &[Key]) {
        // Slots near one another in one write each (the slots between them as they are): a write per slot costs far
        // more than its 16 bytes.
        let mut m: Vec<u32> = keys.iter().filter_map(|k| self.slot_of.get(k).copied()).collect();
        m.sort_unstable();
        let mut i = 0;
        while i < m.len() {
            let mut j = i + 1;
            while j < m.len() && m[j] - m[j - 1] <= 32 {
                j += 1;
            }
            let (a, b) = (m[i], m[j - 1]);
            let data: Vec<[i32; 4]> = (a..=b).map(|s| self.key_of[s as usize].map_or([0, 0, 0, -1], |k| [k.1.x, k.1.y, k.1.z, k.0 as i32])).collect();
            self.queue.write_buffer(&self.buf.meta, a as u64 * 16, bytemuck::cast_slice(&data));
            i = j;
        }
    }

    /// Send the samples of `keys` that differ from what their slots hold; returns those sent. Those in `skip` are
    /// neither read nor sent (a join writes their samples after), only marked as not held.
    fn upload(&mut self, vol: &LevelVolume, keys: &[Key], stats: &mut SyncStats, skip: Option<&KeySet<Key>>) -> Vec<Key> {
        let sent = &self.sent;
        let hidden = Some(self.view.hidden());
        let packed: Vec<(u32, Vec<u32>, u64, Key)> = keys
            .par_iter()
            .filter_map(|k| {
                let s = *self.slot_of.get(k)?;
                if skip.is_some_and(|x| x.contains(k)) {
                    return Some((s, Vec::new(), 0, *k));
                }
                let data = packed_samples(vol, *k, hidden);
                let h = sample_hash(&data);
                (sent[s as usize] != h).then_some((s, data, h, *k))
            })
            .collect();
        for p in &packed {
            self.sent[p.0 as usize] = p.2;
        }
        let out: Vec<Key> = packed.iter().filter(|p| p.2 != 0).map(|p| p.3).collect();
        // Runs of consecutive slots go up in one write.
        let mut sorted: Vec<(u32, Vec<u32>)> = packed.into_iter().filter(|p| p.2 != 0).map(|p| (p.0, p.1)).collect();
        sorted.sort_unstable_by_key(|p| p.0);
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i + 1;
            while j < sorted.len() && sorted[j].0 == sorted[j - 1].0 + 1 {
                j += 1;
            }
            let data: Vec<u32> = sorted[i..j].iter().flat_map(|p| p.1.iter().copied()).collect();
            self.queue.write_buffer(&self.buf.samples, sorted[i].0 as u64 * SAMPLE_WORDS * 4, bytemuck::cast_slice(&data));
            stats.bytes += data.len() * 4;
            i = j;
        }
        stats.uploaded += sorted.len();
        out
    }

    /// Record and submit the passes for `extract` (blocks re-extracted) and `released` (slots that only give pages back).
    fn extract(&mut self, extract: &[Key], released: &[u32], stats: &mut SyncStats) {
        let ex: Vec<u32> = extract.iter().filter_map(|k| self.slot_of.get(k).copied()).collect();
        self.extract_slots(ex, released, stats);
    }

    /// [`Self::extract`] by slot.
    fn extract_slots(&mut self, ex: Vec<u32>, released: &[u32], stats: &mut SyncStats) {
        let mut work: Vec<u32> = released.to_vec();
        work.extend_from_slice(&ex);
        if work.is_empty() {
            return;
        }
        if work.len() as u32 > self.buf.work_len {
            // The work list grows with the slots; a longer list than that is a full rebuild's.
            let len = (work.len() as u32).next_power_of_two();
            self.buf.work = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("levels work"), size: len as u64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            self.buf.work_len = len;
            let rebuilt = Self::rebind(&self.device, &self.layout, &self.buf);
            self.buf.group = rebuilt.0;
            self.buf.cull_group = rebuilt.1;
        }
        self.queue.write_buffer(&self.buf.work, 0, bytemuck::cast_slice(&work));
        let p = self.params(ex.len() as u32, released.len() as u32);
        self.queue.write_buffer(&self.buf.params, 0, bytemuck::bytes_of(&p));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("levels extract") });
        {
            let timing = self.timer.as_ref().filter(|t| t.3.load(Ordering::Acquire) == 0);
            let writes = timing.map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.0, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("levels extract"), timestamp_writes: writes });
            cp.set_bind_group(0, &self.buf.group, &[]);
            let threads = work.len() as u32;
            let (x, y) = grid(threads.div_ceil(64));
            cp.set_pipeline(&self.release);
            cp.dispatch_workgroups(x, y, 1);
            if !ex.is_empty() {
                let (x, y) = grid(ex.len() as u32);
                cp.set_pipeline(&self.vertices);
                cp.dispatch_workgroups(x, y, 1);
                cp.set_pipeline(&self.faces);
                cp.dispatch_workgroups(x, y, 1);
            }
        }
        let timed = if let Some((set, resolve, staging, state, _)) = &self.timer
            && state.load(Ordering::Acquire) == 0
        {
            enc.resolve_query_set(set, 0..2, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, staging, 0, 16);
            true
        } else {
            false
        };
        // Read back the status (free pages, whether they ran out) when the GPU gets to it.
        if self.status_state.load(Ordering::Acquire) == 0 {
            enc.copy_buffer_to_buffer(&self.buf.status, 0, &self.status_staging, 0, 16);
        }
        self.queue.submit([enc.finish()]);
        if self.status_state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let state = self.status_state.clone();
            self.status_staging.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release));
        }
        if timed && let Some((_, _, staging, state, _)) = &self.timer {
            state.store(1, Ordering::Release);
            let state = state.clone();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release));
        }
        stats.extracted += ex.len();
        stats.released += released.len();
    }

    fn rebind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, b: &Buffers) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let group_with = |p: &wgpu::Buffer| {
            let bufs = [p, &b.samples, &b.hash, &b.meta, &b.work, &b.verts, &b.cells, &b.pages_buf, &b.slot_pages, &b.free_pages, &b.status, &b.clusters, &b.table, &b.draws];
            let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, x)| wgpu::BindGroupEntry { binding: i as u32, resource: x.as_entire_binding() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("levels"), layout, entries: &entries })
        };
        (group_with(&b.params), group_with(&b.cull_params))
    }

    /// Pick up the status the GPU last reported.
    fn read_status(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some((_, _, staging, state, period)) = &self.timer
            && state.load(Ordering::Acquire) == 2
        {
            {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                let t: &[u64] = bytemuck::cast_slice(&data[..16]);
                self.gpu_ms = t[1].saturating_sub(t[0]) as f64 * *period as f64 * 1e-6;
            }
            staging.unmap();
            state.store(0, Ordering::Release);
        }
        if self.status_state.load(Ordering::Acquire) == 2 {
            {
                let data = self.status_staging.slice(..).get_mapped_range().expect("mapped");
                self.status.copy_from_slice(bytemuck::cast_slice(&data[..16]));
            }
            self.status_staging.unmap();
            self.status_state.store(0, Ordering::Release);
            if self.status[1] != 0 {
                self.grow_pages = true;
            }
        }
    }

    /// Bring the GPU's copy up to date with `vol`: upload what changed and re-extract around it.
    pub fn sync(&mut self, vol: &mut LevelVolume) -> SyncStats {
        self.sync_in(vol, None)
    }

    /// [`Self::sync`] just before [`Self::join`] of `plan`: the plan's blocks are neither sent nor extracted here
    /// (the join writes their samples and extracts them), only given slots.
    pub fn sync_for_join(&mut self, vol: &mut LevelVolume, plan: &sculpt_core::generator::detail_layer::GpuPlan) -> SyncStats {
        let skip: KeySet<Key> = plan.blocks.iter().copied().collect();
        self.sync_in(vol, Some(&skip))
    }

    fn sync_in(&mut self, vol: &mut LevelVolume, skip: Option<&KeySet<Key>>) -> SyncStats {
        let t = Instant::now();
        let mut stats = SyncStats::default();
        let log = std::env::var_os("SCULPT_JOIN_LOG").is_some();
        let mut laps: Vec<(&str, f64)> = Vec::new();
        let mut lap = |name: &'static str| {
            if log {
                laps.push((name, t.elapsed().as_secs_f64() * 1e3));
            }
        };
        self.read_status();
        // An extraction left for a join that never came: the slots are as they were, so it still applies.
        if let Some((ex, rel)) = self.deferred.take() {
            self.extract_slots(ex, &rel, &mut stats);
        }
        lap("status");
        let mut dirty = vol.drain_dirty();
        let n_dirty = dirty.len();
        lap("dirty");
        let first = self.slot_of.is_empty() && self.high == 0;
        // The levels shown: blocks that changed are measured again (showing until they are), and those shown or
        // hidden since the last sync go up or come down as blocks that came or went do.
        if first || vol.origin() != self.origin {
            self.view.clear();
            self.view.note(vol, &vol.keys());
        } else {
            self.view.note(vol, &dirty);
        }
        lap("note");
        self.view.refresh(vol, MEASURE_PER_SYNC);
        lap("refresh");
        if let Some((eye, focal)) = self.eye {
            dirty.extend(self.view.choose(vol, eye, focal));
            dirty.sort_unstable_by_key(|k| (k.0, k.1.z, k.1.y, k.1.x));
            dirty.dedup();
        }
        lap("choose");
        if first || self.grow_pages || self.cap_changed || vol.origin() != self.origin {
            self.cap_changed = false;
            self.origin = vol.origin();
            self.rebuild(vol, &mut stats);
            stats.cpu_ms = t.elapsed().as_secs_f64() * 1e3;
            return stats;
        }
        if dirty.is_empty() {
            return stats;
        }
        // Blocks that went, and blocks that came.
        let mut released = Vec::new();
        let mut gone = Vec::new();
        for k in &dirty {
            if (vol.block(*k).is_none() || self.view.hides(*k))
                && let Some(s) = self.slot_of.remove(k)
            {
                self.key_of[s as usize] = None;
                self.free.push(s);
                released.push(s);
                gone.push(*k);
            }
        }
        for k in &gone {
            self.hash_remove(*k);
        }
        lap("gone");
        let uploads = vol.upload_set_in(&dirty, &self.view.show(self.levels - 1));
        lap("upload_set");
        let mut out_of_slots = false;
        // Blocks given a slot now: only their slots' block coordinates need sending.
        let mut placed = Vec::new();
        for k in &uploads {
            if !self.slot_of.contains_key(k) {
                if let Some(s) = self.take_slot(*k) {
                    self.hash_insert(*k, s);
                    placed.push(*k);
                } else {
                    out_of_slots = true;
                }
            }
        }
        if out_of_slots {
            self.rebuild(vol, &mut stats);
            stats.cpu_ms = t.elapsed().as_secs_f64() * 1e3;
            return stats;
        }
        // A slot freed here and handed straight to a new block is released with that block's extraction: listing it
        // twice would have two threads give its pages back at once, and two blocks would then share them.
        released.retain(|&s| self.key_of[s as usize].is_none());
        lap("place");
        if self.hash_fit() {
            self.send_hash_whole();
            lap("hash whole");
        } else {
            // Changed entries near one another go up as one span (a write costs far more than a few entries).
            let mut dirty_h = std::mem::take(&mut self.hash_dirty);
            dirty_h.sort_unstable();
            dirty_h.dedup();
            let mut i = 0;
            while i < dirty_h.len() {
                let mut j = i + 1;
                while j < dirty_h.len() && dirty_h[j] - dirty_h[j - 1] <= 32 {
                    j += 1;
                }
                let (a, b) = (dirty_h[i] as usize, dirty_h[j - 1] as usize);
                self.queue.write_buffer(&self.buf.hash, a as u64 * 16, bytemuck::cast_slice(&self.hash[a..=b]));
                i = j;
            }
            lap("hash");
        }
        // Freed slots read as free.
        for &s in &released {
            self.queue.write_buffer(&self.buf.meta, s as u64 * 16, bytemuck::cast_slice(&[0i32, 0, 0, -1]));
        }
        self.send_meta(&placed);
        lap("slots");
        let n_up = uploads.len();
        let uploads = self.upload(vol, &uploads, &mut stats, skip);
        lap("upload");
        let mut changed: Vec<Key> = uploads.clone();
        changed.extend(gone);
        if let Some(skip) = skip {
            // Of the join's blocks, those that came or went here changed for the blocks around them.
            changed.extend(dirty.iter().filter(|k| skip.contains(k)));
        }
        let extract = match skip {
            Some(skip) => vol.extract_set_where(&changed, &self.view.show(self.cap), &|k| !skip.contains(&k)),
            None => vol.extract_set_in(&changed, &self.view.show(self.cap)),
        };
        lap("extract_set");
        if skip.is_some() {
            // The blocks around the join's read its vertices: extracted with it, after its samples are written.
            let ex: Vec<u32> = extract.iter().filter_map(|k| self.slot_of.get(k).copied()).collect();
            stats.extracted += ex.len();
            stats.released += released.len();
            self.deferred = Some((ex, released));
        } else {
            self.extract(&extract, &released, &mut stats);
        }
        lap("extract");
        stats.cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        if log {
            let mut prev = 0.0;
            let parts: Vec<String> = laps.iter().map(|(n, tt)| { let s = format!("{n} {:.1}", tt - prev); prev = *tt; s }).collect();
            eprintln!("sync: {n_dirty} dirty, {n_up} upload candidates, {} sent, {} extracted: {}", uploads.len(), extract.len(), parts.join(", "));
        }
        stats
    }

    /// Record the cull pass: the pages of the clusters in view, facing the eye, as indirect draws. The status (draws
    /// written among it) is read back after it when no other read is under way.
    pub fn cull(&self, enc: &mut wgpu::CommandEncoder, view_proj: Mat4, eye: Vec3, cull: bool) {
        let mut p = self.params(0, 0);
        p.cull = cull as u32;
        p.eye = eye.extend(1.0).to_array();
        // Frustum planes from the matrix (rows combined), each pointing inward; depth 0..1.
        let m = view_proj.transpose();
        let (r0, r1, r2, r3) = (m.x_axis, m.y_axis, m.z_axis, m.w_axis);
        let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        for (k, pl) in planes.iter().enumerate() {
            let n = pl.truncate().length().max(1e-20);
            p.planes[k] = (*pl / n).to_array();
        }
        self.queue.write_buffer(&self.buf.cull_params, 0, bytemuck::bytes_of(&p));
        enc.clear_buffer(&self.buf.status, 8, Some(4));
        if !self.indirect_count {
            enc.clear_buffer(&self.buf.draws, 0, None);
        }
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("levels cull"), timestamp_writes: None });
        cp.set_bind_group(0, &self.buf.cull_group, &[]);
        cp.set_pipeline(&self.cull);
        let (x, y) = grid(self.high.div_ceil(64).max(1));
        cp.dispatch_workgroups(x, y, 1);
        drop(cp);
        if self.status_state.load(Ordering::Acquire) == 0 {
            enc.copy_buffer_to_buffer(&self.buf.status, 0, &self.status_staging, 0, 16);
            self.status_copy.store(true, Ordering::Release);
        }
    }

    /// After the frame holding [`Self::cull`] was submitted: map the status it copied, to read on a later sync.
    pub fn after_submit(&self) {
        if self.status_copy.swap(false, Ordering::AcqRel) && self.status_state.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire).is_ok() {
            let state = self.status_state.clone();
            self.status_staging.slice(..).map_async(wgpu::MapMode::Read, move |r| state.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release));
        }
    }

    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    pub fn voxel0(&self) -> f32 {
        self.voxel0
    }
}

/// Workgroups as a 2D grid within the 65535 per dimension limit.
fn grid(n: u32) -> (u32, u32) {
    let x = n.clamp(1, 65535);
    (x, n.div_ceil(x).max(1))
}
