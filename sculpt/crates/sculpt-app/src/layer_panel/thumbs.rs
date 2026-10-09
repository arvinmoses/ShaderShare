//! Lazy, cached thumbnails for the layer list.
//!
//! A thumbnail is rebuilt only when the document's edit counter has moved *and* it is old enough,
//! never while a stroke is in progress, and at most `BUDGET` per frame. So the sculpt path never waits
//! on a preview, and a stale one is shown until its replacement is ready.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use egui::{Color32, ColorImage, Context, TextureHandle, TextureId, TextureOptions};
use sculpt_core::{Document, LayerId, Preview};

const RES: usize = 24;
const BUDGET: usize = 2;
const MIN_AGE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Delta,
    Mask,
}

struct Entry {
    tex: TextureHandle,
    serial: u64,
    built: Instant,
}

#[derive(Default)]
pub struct ThumbCache {
    entries: HashMap<(LayerId, Kind), Entry>,
    spent: usize,
}

/// Colours a thumbnail is painted with, from the theme.
#[derive(Clone, Copy)]
pub struct Palette {
    pub background: Color32,
    pub delta: Color32,
}

impl ThumbCache {
    /// Call once per frame before any `texture` calls.
    pub fn begin_frame(&mut self) {
        self.spent = 0;
    }

    /// The thumbnail for a layer, building or refreshing it if allowed. `None` when there is nothing to
    /// show (a folder, or a mask with no per-vertex values), so the caller draws its plain icon.
    pub fn texture(&mut self, ctx: &Context, doc: &Document, id: LayerId, kind: Kind, stroking: bool, palette: Palette) -> Option<TextureId> {
        let serial = doc.edit_serial();
        let now = Instant::now();
        let key = (id, kind);
        let stale = self.entries.get(&key).is_none_or(|e| e.serial != serial);
        let old_enough = self.entries.get(&key).is_none_or(|e| now.duration_since(e.built) >= MIN_AGE);
        if stale && old_enough && !stroking && self.spent < BUDGET {
            self.spent += 1;
            let preview = match kind {
                Kind::Delta => doc.delta_preview(id, RES),
                Kind::Mask => doc.mask_preview(id, RES),
            };
            match preview {
                Some(p) => {
                    let image = paint(&p, kind, palette);
                    match self.entries.get_mut(&key) {
                        Some(e) => {
                            e.tex.set(image, TextureOptions::LINEAR);
                            e.serial = serial;
                            e.built = now;
                        }
                        None => {
                            let tex = ctx.load_texture(format!("layer_thumb_{}_{kind:?}", id.0), image, TextureOptions::LINEAR);
                            self.entries.insert(key, Entry { tex, serial, built: now });
                        }
                    }
                }
                None => {
                    self.entries.remove(&key);
                }
            }
        } else if stale && !stroking {
            // Wanted but over budget or too soon: come back shortly.
            ctx.request_repaint_after(MIN_AGE);
        }
        self.entries.get(&key).map(|e| e.tex.id())
    }

    /// Forget layers that no longer exist.
    pub fn retain(&mut self, doc: &Document) {
        self.entries.retain(|(id, _), _| doc.layer(*id).is_some());
    }
}

fn paint(p: &Preview, kind: Kind, pal: Palette) -> ColorImage {
    let mut image = ColorImage::filled([p.res, p.res], pal.background);
    for (i, px) in image.pixels.iter_mut().enumerate() {
        let v = p.values[i];
        *px = match kind {
            Kind::Mask => {
                let g = (v * 255.0) as u8;
                Color32::from_gray(g)
            }
            Kind::Delta if p.covered[i] => lerp(pal.background, pal.delta, 0.15 + 0.85 * v),
            Kind::Delta => pal.background,
        };
    }
    image
}

fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_ramp_runs_from_background_to_accent() {
        let pal = Palette { background: Color32::from_rgb(10, 10, 10), delta: Color32::from_rgb(250, 130, 40) };
        let p = Preview { res: 2, values: vec![0.0, 1.0, 0.5, 0.0], covered: vec![true, true, true, false] };
        let img = paint(&p, Kind::Delta, pal);
        assert_eq!(img.pixels[3], pal.background, "uncovered cells stay background");
        assert!(img.pixels[1].r() > img.pixels[2].r() && img.pixels[2].r() > img.pixels[0].r());
        assert_eq!(img.pixels[1], pal.delta);
    }

    #[test]
    fn mask_is_plain_grey() {
        let pal = Palette { background: Color32::BLACK, delta: Color32::WHITE };
        let p = Preview { res: 1, values: vec![0.5], covered: vec![true] };
        let img = paint(&p, Kind::Mask, pal);
        assert_eq!(img.pixels[0], Color32::from_gray(127));
    }
}
