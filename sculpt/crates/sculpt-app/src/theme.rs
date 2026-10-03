//! Data-driven themes.
//!
//! A theme is a JSON file (see `themes/*.json`): UI colors, viewport colors
//! (background gradient, clay, mask overlays, brush cursor) and metrics. Built-in
//! themes are compiled in; user themes are loaded from a themes directory and
//! hot-reloaded when the file changes, so a theme can be tweaked live in any
//! text editor or in the in-app theme editor.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use egui::{Color32, CornerRadius, Stroke, Visuals};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// `#rrggbb` or `#rrggbbaa` in JSON.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hex(pub Color32);

impl Serialize for Hex {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let [r, g, b, a] = self.0.to_srgba_unmultiplied();
        let text = if a == 255 { format!("#{r:02x}{g:02x}{b:02x}") } else { format!("#{r:02x}{g:02x}{b:02x}{a:02x}") };
        s.serialize_str(&text)
    }
}

impl<'de> Deserialize<'de> for Hex {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        parse_hex(&s).map(Hex).ok_or_else(|| serde::de::Error::custom(format!("bad color '{s}', expected #rrggbb or #rrggbbaa")))
    }
}

pub fn parse_hex(s: &str) -> Option<Color32> {
    let h = s.strip_prefix('#')?;
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    match h.len() {
        6 => Some(Color32::from_rgb(byte(0)?, byte(2)?, byte(4)?)),
        8 => Some(Color32::from_rgba_unmultiplied(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiColors {
    pub window: Hex,
    pub panel: Hex,
    pub tray: Hex,
    pub widget: Hex,
    pub widget_hover: Hex,
    pub widget_active: Hex,
    pub text: Hex,
    pub text_weak: Hex,
    pub accent: Hex,
    pub separator: Hex,
    /// Panel title bars ("LAYERS", "PROPERTIES"). Defaults to `window`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<Hex>,
    /// Selected row in lists (layer stack). Defaults to a dimmed accent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_selected: Option<Hex>,
}

impl UiColors {
    pub fn header(&self) -> Color32 {
        self.header.map_or(self.window.0, |h| h.0)
    }
    pub fn row_selected(&self) -> Color32 {
        self.row_selected.map_or(self.accent.0.gamma_multiply(0.45), |h| h.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ViewportColors {
    pub background_top: Hex,
    pub background_bottom: Hex,
    pub clay: Hex,
    /// Mask / channel overlay tint.
    pub overlay: Hex,
    /// Freeze overlay tint (Mudbox shows frozen areas in blue).
    pub freeze: Hex,
    pub cursor: Hex,
    pub hud_text: Hex,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Metrics {
    pub corner_radius: u8,
    pub spacing: f32,
    pub font_size: f32,
    pub tray_tile: f32,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics { corner_radius: 2, spacing: 6.0, font_size: 13.0, tray_tile: 64.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    pub name: String,
    pub dark: bool,
    pub ui: UiColors,
    pub viewport: ViewportColors,
    #[serde(default)]
    pub metrics: Metrics,
}

impl Theme {
    pub fn apply(&self, ctx: &egui::Context) {
        let c = &self.ui;
        let mut v = if self.dark { Visuals::dark() } else { Visuals::light() };
        let r = CornerRadius::same(self.metrics.corner_radius);
        v.panel_fill = c.panel.0;
        v.window_fill = c.window.0;
        v.extreme_bg_color = c.window.0;
        v.faint_bg_color = c.widget.0;
        v.window_corner_radius = r;
        v.window_stroke = Stroke::new(1.0, c.separator.0);
        v.selection.bg_fill = c.accent.0;
        v.selection.stroke = Stroke::new(1.0, c.text.0);
        v.hyperlink_color = c.accent.0;
        v.override_text_color = Some(c.text.0);
        let w = &mut v.widgets;
        w.noninteractive.bg_fill = c.panel.0;
        w.noninteractive.weak_bg_fill = c.panel.0;
        w.noninteractive.bg_stroke = Stroke::new(1.0, c.separator.0);
        w.noninteractive.fg_stroke = Stroke::new(1.0, c.text.0);
        for (state, bg) in [(&mut w.inactive, c.widget.0), (&mut w.hovered, c.widget_hover.0), (&mut w.active, c.widget_active.0), (&mut w.open, c.widget_active.0)] {
            state.bg_fill = bg;
            state.weak_bg_fill = bg;
            state.corner_radius = r;
            state.fg_stroke = Stroke::new(1.0, c.text.0);
        }
        w.hovered.bg_stroke = Stroke::new(1.0, c.accent.0);
        w.active.bg_stroke = Stroke::new(1.0, c.accent.0);
        ctx.set_visuals(v);

        ctx.global_style_mut(|style| {
            style.spacing.item_spacing = egui::vec2(self.metrics.spacing, self.metrics.spacing * 0.66);
            for (text_style, font) in style.text_styles.iter_mut() {
                font.size = match text_style {
                    egui::TextStyle::Heading => self.metrics.font_size * 1.35,
                    egui::TextStyle::Small => self.metrics.font_size * 0.8,
                    _ => self.metrics.font_size,
                };
            }
        });
    }

    pub fn weak_text(&self) -> Color32 {
        self.ui.text_weak.0
    }
}

const BUILTIN: [&str; 4] = [
    include_str!("../themes/mudbox_dark.json"),
    include_str!("../themes/painter_dark.json"),
    include_str!("../themes/studio_light.json"),
    include_str!("../themes/high_contrast.json"),
];

pub struct ThemeLibrary {
    pub themes: Vec<Theme>,
    pub dir: PathBuf,
    /// (file, mtime) for user themes, used for hot reload.
    files: Vec<(PathBuf, Option<SystemTime>)>,
    pub errors: Vec<String>,
}

impl ThemeLibrary {
    pub fn load(dir: &Path) -> ThemeLibrary {
        let mut lib = ThemeLibrary { themes: Vec::new(), dir: dir.to_path_buf(), files: Vec::new(), errors: Vec::new() };
        lib.reload();
        lib
    }

    pub fn reload(&mut self) {
        self.themes = BUILTIN.iter().map(|s| serde_json::from_str(s).expect("built-in theme parses")).collect();
        self.errors.clear();
        self.files.clear();
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
        paths.sort();
        for p in paths {
            let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
            match std::fs::read_to_string(&p).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str::<Theme>(&t).map_err(|e| e.to_string())) {
                Ok(t) => {
                    // A user theme with a built-in's name overrides it.
                    self.themes.retain(|x| x.name != t.name);
                    self.themes.push(t);
                }
                Err(e) => self.errors.push(format!("{}: {e}", p.display())),
            }
            self.files.push((p, mtime));
        }
    }

    /// Re-read the themes directory if any file was added or changed.
    pub fn poll_changes(&mut self) -> bool {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return false };
        let mut now: Vec<(PathBuf, Option<SystemTime>)> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .map(|p| {
                let m = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
                (p, m)
            })
            .collect();
        now.sort();
        if now != self.files {
            self.reload();
            return true;
        }
        false
    }

    pub fn get(&self, name: &str) -> Theme {
        self.themes.iter().find(|t| t.name == name).cloned().unwrap_or_else(|| self.themes[0].clone())
    }

    pub fn save(&mut self, theme: &Theme) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        let file: String = theme.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
        let path = self.dir.join(format!("{file}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(theme).unwrap())?;
        self.reload();
        Ok(path)
    }
}
