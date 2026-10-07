//! Drop-zone math for dragging rows in the layer list. Pure, so it is unit tested.

use egui::Rect;
use sculpt_core::{Document, LayerId, Placement};

/// Where a dragged row would land relative to the row under the pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zone {
    Above,
    Below,
    Into,
}

/// Geometry of one visible row, recorded while the list is drawn.
#[derive(Clone, Copy, Debug)]
pub struct RowGeom {
    pub id: LayerId,
    pub rect: Rect,
    pub is_folder: bool,
}

/// `frac` is the pointer's height inside the row, 0 at the top and 1 at the bottom.
/// Folders have an "into" band in the middle half; other rows split at the middle.
pub fn zone(frac: f32, is_folder: bool) -> Zone {
    if is_folder {
        match frac {
            f if f < 0.25 => Zone::Above,
            f if f > 0.75 => Zone::Below,
            _ => Zone::Into,
        }
    } else if frac < 0.5 {
        Zone::Above
    } else {
        Zone::Below
    }
}

pub fn placement(target: LayerId, zone: Zone) -> Placement {
    match zone {
        Zone::Above => Placement::Above(target),
        Zone::Below => Placement::Below(target),
        Zone::Into => Placement::Into(target),
    }
}

/// A drop is refused when it would put a folder inside itself, or onto itself.
pub fn is_valid(doc: &Document, dragged: &[LayerId], target: LayerId) -> bool {
    dragged.iter().all(|d| *d != target && !doc.is_ancestor(*d, target))
}

/// The drop under the pointer, if it is over any row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropSpot {
    pub target: LayerId,
    pub zone: Zone,
    pub rect: Rect,
    pub valid: bool,
}

pub fn find_spot(doc: &Document, rows: &[RowGeom], dragged: &[LayerId], pointer: egui::Pos2) -> Option<DropSpot> {
    let row = rows.iter().find(|r| r.rect.y_range().contains(pointer.y))?;
    let frac = ((pointer.y - row.rect.top()) / row.rect.height()).clamp(0.0, 1.0);
    let zone = zone(frac, row.is_folder);
    Some(DropSpot { target: row.id, zone, rect: row.rect, valid: is_valid(doc, dragged, row.id) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sculpt_core::primitives::quad_sphere;

    #[test]
    fn zones_split_rows() {
        assert_eq!(zone(0.1, false), Zone::Above);
        assert_eq!(zone(0.6, false), Zone::Below);
        assert_eq!(zone(0.1, true), Zone::Above);
        assert_eq!(zone(0.5, true), Zone::Into);
        assert_eq!(zone(0.9, true), Zone::Below);
        // Band edges belong to the middle (into) zone.
        assert_eq!(zone(0.25, true), Zone::Into);
        assert_eq!(zone(0.75, true), Zone::Into);
    }

    #[test]
    fn placements_map_zones() {
        let t = LayerId(7);
        assert_eq!(placement(t, Zone::Above), Placement::Above(t));
        assert_eq!(placement(t, Zone::Below), Placement::Below(t));
        assert_eq!(placement(t, Zone::Into), Placement::Into(t));
    }

    #[test]
    fn folders_cannot_drop_into_themselves() {
        let mut d = Document::from_mesh(quad_sphere(1, 1.0)).unwrap();
        let outer = d.insert_folder("Outer", Placement::Top).unwrap();
        let inner = d.insert_folder("Inner", Placement::Into(outer)).unwrap();
        let leaf = d.insert_layer("Leaf", Placement::Into(inner)).unwrap();
        assert!(!is_valid(&d, &[outer], inner));
        assert!(!is_valid(&d, &[outer], leaf));
        assert!(!is_valid(&d, &[outer], outer));
        assert!(is_valid(&d, &[leaf], outer));
        assert!(is_valid(&d, &[inner], outer), "moving within its own parent is fine");
    }

    #[test]
    fn spot_uses_the_row_under_the_pointer() {
        let mut d = Document::from_mesh(quad_sphere(1, 1.0)).unwrap();
        let a = d.insert_layer("A", Placement::Top).unwrap();
        let f = d.insert_folder("F", Placement::Top).unwrap();
        let rows = [
            RowGeom { id: f, rect: Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(100.0, 36.0)), is_folder: true },
            RowGeom { id: a, rect: Rect::from_min_max(egui::pos2(0.0, 36.0), egui::pos2(100.0, 72.0)), is_folder: false },
        ];
        let s = find_spot(&d, &rows, &[a], egui::pos2(10.0, 18.0)).unwrap();
        assert_eq!((s.target, s.zone, s.valid), (f, Zone::Into, true));
        let s = find_spot(&d, &rows, &[a], egui::pos2(10.0, 60.0)).unwrap();
        assert_eq!((s.target, s.zone, s.valid), (a, Zone::Below, false), "onto itself");
        assert!(find_spot(&d, &rows, &[a], egui::pos2(10.0, 90.0)).is_none());
    }
}
