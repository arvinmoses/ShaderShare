//! Pen pressure with graceful fallback:
//!
//! 1. `octotablet` (feature `tablet`): Windows Ink and Wayland tablet protocol.
//! 2. Pointer force reported through winit/egui touch events (Windows pen,
//!    iPad-style devices).
//! 3. Mouse: pressure 1.0.
//!
//! The HUD shows which source is live, so pressure support can be verified on
//! any machine at a glance.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "tablet"), allow(dead_code))]
pub enum PenSource {
    Mouse,
    PointerForce,
    Tablet,
}

impl PenSource {
    pub fn label(self) -> &'static str {
        match self {
            PenSource::Mouse => "mouse (no pressure)",
            PenSource::PointerForce => "pen (pointer force)",
            PenSource::Tablet => "tablet (octotablet)",
        }
    }
}

pub struct Pen {
    pub source: PenSource,
    pub pressure: f32,
    pub status: String,
    #[cfg(feature = "tablet")]
    tablet: Option<octotablet::Manager>,
}

impl Pen {
    #[cfg(feature = "tablet")]
    pub fn new(cc: &eframe::CreationContext) -> Pen {
        // SAFETY: eframe keeps the window and display alive for the lifetime
        // of the app, which owns this manager.
        let (tablet, status) = match unsafe { octotablet::Builder::new().build_raw(cc) } {
            Ok(m) => (Some(m), "octotablet connected".to_string()),
            Err(e) => (None, format!("octotablet unavailable: {e}")),
        };
        Pen { source: PenSource::Mouse, pressure: 1.0, status, tablet }
    }

    #[cfg(not(feature = "tablet"))]
    pub fn new(_cc: &eframe::CreationContext) -> Pen {
        Pen { source: PenSource::Mouse, pressure: 1.0, status: "built without the `tablet` feature".into() }
    }

    /// Update from this frame's input. Returns per-event pressure overrides
    /// in arrival order via `self.pressure`.
    pub fn update(&mut self, events: &[egui::Event]) {
        #[cfg(feature = "tablet")]
        if let Some(m) = &mut self.tablet {
            if let Ok(evs) = m.pump() {
                for e in evs {
                    if let octotablet::events::Event::Tool { event: octotablet::events::ToolEvent::Pose(pose), .. } = e {
                        if let Some(p) = pose.pressure.get() {
                            self.pressure = p.clamp(0.0, 1.0);
                            self.source = PenSource::Tablet;
                        }
                    }
                }
            }
            if self.source == PenSource::Tablet {
                return;
            }
        }
        for e in events {
            if let egui::Event::Touch { force: Some(f), .. } = e {
                self.pressure = f.clamp(0.0, 1.0);
                self.source = PenSource::PointerForce;
            }
        }
        if self.source == PenSource::Mouse {
            self.pressure = 1.0;
        }
    }
}
