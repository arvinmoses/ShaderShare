//! Multi-selection of layer-list rows. Pure data, no UI.

use std::collections::BTreeSet;

use sculpt_core::LayerId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClickMods {
    /// Ctrl / Cmd: toggle membership.
    pub toggle: bool,
    /// Shift: select the range from the anchor.
    pub range: bool,
}

/// Which rows are selected. `primary` is the row Properties shows and
/// commands without an explicit target act on; `anchor` starts Shift ranges.
#[derive(Clone, Debug, Default)]
pub struct LayerSelection {
    ids: BTreeSet<LayerId>,
    primary: Option<LayerId>,
    anchor: Option<LayerId>,
}

impl LayerSelection {
    pub fn primary(&self) -> Option<LayerId> {
        self.primary
    }

    pub fn contains(&self, id: LayerId) -> bool {
        self.ids.contains(&id)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn select_only(&mut self, id: LayerId) {
        self.ids.clear();
        self.ids.insert(id);
        self.primary = Some(id);
        self.anchor = Some(id);
    }

    pub fn clear(&mut self) {
        *self = LayerSelection::default();
    }

    /// Apply a click on `id`. `order` is the visible rows, top first.
    pub fn click(&mut self, id: LayerId, mods: ClickMods, order: &[LayerId]) {
        if mods.range
            && let Some(anchor) = self.anchor
            && let (Some(a), Some(b)) = (order.iter().position(|r| *r == anchor), order.iter().position(|r| *r == id))
        {
            let (lo, hi) = (a.min(b), a.max(b));
            if !mods.toggle {
                self.ids.clear();
            }
            self.ids.extend(order[lo..=hi].iter().copied());
            self.primary = Some(id);
        } else if mods.toggle {
            if !self.ids.remove(&id) {
                self.ids.insert(id);
                self.primary = Some(id);
            } else if self.primary == Some(id) {
                self.primary = self.ids.iter().next().copied();
            }
            self.anchor = Some(id);
        } else {
            self.select_only(id);
        }
    }

    /// Selected ids in display order (top first), so commands keep the stack's order.
    pub fn in_order(&self, order: &[LayerId]) -> Vec<LayerId> {
        order.iter().copied().filter(|id| self.ids.contains(id)).collect()
    }

    /// Drop ids that no longer exist (after delete or undo).
    pub fn retain(&mut self, alive: impl Fn(LayerId) -> bool) {
        self.ids.retain(|id| alive(*id));
        if self.primary.is_some_and(|p| !alive(p)) {
            self.primary = self.ids.iter().next().copied();
        }
        if self.anchor.is_some_and(|a| !alive(a)) {
            self.anchor = self.primary;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: u32) -> Vec<LayerId> {
        (1..=n).map(LayerId).collect()
    }

    #[test]
    fn plain_click_selects_only() {
        let order = ids(5);
        let mut s = LayerSelection::default();
        s.click(LayerId(2), ClickMods::default(), &order);
        s.click(LayerId(4), ClickMods::default(), &order);
        assert_eq!(s.in_order(&order), [LayerId(4)]);
        assert_eq!(s.primary(), Some(LayerId(4)));
    }

    #[test]
    fn ctrl_click_toggles_and_moves_primary() {
        let order = ids(5);
        let mut s = LayerSelection::default();
        s.click(LayerId(1), ClickMods::default(), &order);
        s.click(LayerId(3), ClickMods { toggle: true, range: false }, &order);
        assert_eq!(s.in_order(&order), [LayerId(1), LayerId(3)]);
        assert_eq!(s.primary(), Some(LayerId(3)));
        s.click(LayerId(3), ClickMods { toggle: true, range: false }, &order);
        assert_eq!(s.in_order(&order), [LayerId(1)]);
        assert_eq!(s.primary(), Some(LayerId(1)));
    }

    #[test]
    fn shift_click_selects_range_from_anchor() {
        let order = ids(6);
        let mut s = LayerSelection::default();
        s.click(LayerId(5), ClickMods::default(), &order);
        s.click(LayerId(2), ClickMods { toggle: false, range: true }, &order);
        assert_eq!(s.in_order(&order), [LayerId(2), LayerId(3), LayerId(4), LayerId(5)]);
        // The anchor stays put, so a second shift-click re-ranges from it.
        s.click(LayerId(6), ClickMods { toggle: false, range: true }, &order);
        assert_eq!(s.in_order(&order), [LayerId(5), LayerId(6)]);
    }

    #[test]
    fn retain_repairs_primary_and_anchor() {
        let order = ids(3);
        let mut s = LayerSelection::default();
        s.click(LayerId(1), ClickMods::default(), &order);
        s.click(LayerId(2), ClickMods { toggle: true, range: false }, &order);
        s.retain(|id| id != LayerId(2));
        assert_eq!(s.primary(), Some(LayerId(1)));
        s.retain(|_| false);
        assert!(s.is_empty());
        assert_eq!(s.primary(), None);
    }
}
