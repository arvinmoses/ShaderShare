//! The extracted surface kept block by block, so that after a volume edit only the blocks it reached are extracted
//! again (as the view's GPU surface is kept), and the mesh is assembled from the kept blocks.

use std::time::Instant;

use super::extract::{Extracted, assemble, extract_blocks};
use super::ms;
use crate::generator::level_volume::{BlockSurface, ExtractStats, HashMap, Key, LevelVolume, Show};
use crate::Result;

/// Every shown block's surface (vertices, cells and quads named by corner), as of the volume it last saw.
#[derive(Clone)]
pub struct BlockCache {
    cap: usize,
    surf: HashMap<Key, BlockSurface>,
    stats: ExtractStats,
}

/// What one [`BlockCache::update`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheUpdate {
    /// Finding the blocks to extract again (`extract_set_in`).
    pub set_ms: f64,
    /// Extracting them (`extract_in`).
    pub extract_ms: f64,
    /// Blocks changed, and blocks extracted again.
    pub changed: usize,
    pub extracted: usize,
    /// Blocks dropped (no longer stored or shown).
    pub removed: usize,
}

impl BlockCache {
    /// Every shown block of `vol` up to level `cap`.
    pub fn build(vol: &LevelVolume, cap: usize) -> BlockCache {
        let (surf, stats) = extract_blocks(vol, cap);
        BlockCache { cap, surf, stats }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Blocks held.
    pub fn len(&self) -> usize {
        self.surf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.surf.is_empty()
    }

    /// The kept surface of block `k`.
    pub fn block(&self, k: &Key) -> Option<&BlockSurface> {
        self.surf.get(k)
    }

    /// Bring the kept blocks up to `vol` after the blocks `changed` changed (samples or existence, as
    /// [`LevelVolume::drain_dirty`] gives them): only the blocks whose surface that can reach are extracted again.
    pub fn update(&mut self, vol: &LevelVolume, changed: &[Key]) -> CacheUpdate {
        let mut st = CacheUpdate { changed: changed.len(), ..Default::default() };
        if changed.is_empty() {
            return st;
        }
        let show = Show::upto(self.cap);
        let t = Instant::now();
        let work = vol.extract_set_in(changed, &show);
        st.set_ms = ms(t);
        let t = Instant::now();
        let (fresh, _) = vol.extract_in(&work, &show);
        st.extract_ms = ms(t);
        st.extracted = fresh.len();
        // Blocks gone (or no longer shown): those changed, and any listed that the extraction passed over.
        for k in changed.iter().chain(work.iter()) {
            if !fresh.contains_key(k) && self.surf.remove(k).is_some() {
                st.removed += 1;
            }
        }
        self.surf.extend(fresh);
        st
    }

    /// The whole surface as one named mesh, bit for bit what [`super::extract_named`] gives for the same volume. Its
    /// stats count the blocks and vertices held; the rest (time, quads, dropped) are the first build's.
    pub fn assemble(&self) -> Result<Extracted> {
        let mut stats = self.stats;
        stats.blocks = self.surf.len();
        stats.verts = self.surf.values().map(|s| s.verts.len()).sum();
        assemble(&self.surf, stats)
    }
}
