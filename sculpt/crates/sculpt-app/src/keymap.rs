//! Data-driven commands and hotkeys.
//!
//! Every action the UI can perform is a [`Command`]. Menus, tray buttons and
//! hotkeys all dispatch commands, so adding a feature means adding one enum
//! variant and one handler; rebinding is editing `keymap.json`.

use std::path::Path;

use egui::{Key, KeyboardShortcut, Modifiers};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Undo,
    Redo,
    Save,
    Open,
    ToolClayBuildup,
    ToolTrimDynamic,
    ToolMove,
    ToolSmooth,
    ToolFreeze,
    ToolMaskPaint,
    ToolPose,
    BrushSizeUp,
    BrushSizeDown,
    StrengthUp,
    StrengthDown,
    FrameMesh,
    Subdivide,
    NewLayer,
    NewFolder,
    LayerDuplicate,
    LayerMergeDown,
    LayerDelete,
    LayerGroup,
    LayerUngroup,
    LayerRename,
    LayerSolo,
    LayerLock,
    LayerHide,
    ToggleHud,
    CycleOverlay,
    InvertFreeze,
    ClearFreeze,
}

impl Command {
    pub fn label(self) -> &'static str {
        match self {
            Command::Undo => "Undo",
            Command::Redo => "Redo",
            Command::Save => "Save",
            Command::Open => "Open…",
            Command::ToolClayBuildup => "Clay Buildup",
            Command::ToolTrimDynamic => "Trim Dynamic",
            Command::ToolMove => "Move",
            Command::ToolSmooth => "Smooth",
            Command::ToolFreeze => "Freeze",
            Command::ToolMaskPaint => "Mask Paint",
            Command::ToolPose => "Pose",
            Command::BrushSizeUp => "Brush size +",
            Command::BrushSizeDown => "Brush size −",
            Command::StrengthUp => "Strength +",
            Command::StrengthDown => "Strength −",
            Command::FrameMesh => "Frame mesh",
            Command::Subdivide => "Subdivide",
            Command::NewLayer => "New sculpt layer",
            Command::NewFolder => "New folder",
            Command::LayerDuplicate => "Duplicate layer",
            Command::LayerMergeDown => "Merge down",
            Command::LayerDelete => "Delete layer",
            Command::LayerGroup => "Group into folder",
            Command::LayerUngroup => "Ungroup folder",
            Command::LayerRename => "Rename layer",
            Command::LayerSolo => "Solo layer",
            Command::LayerLock => "Lock layer",
            Command::LayerHide => "Hide layer",
            Command::ToggleHud => "Toggle HUD",
            Command::CycleOverlay => "Cycle overlay",
            Command::InvertFreeze => "Invert freeze",
            Command::ClearFreeze => "Clear freeze",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    /// e.g. `"Ctrl+Shift+Z"`, `"3"`, `"F"`, `"OpenBracket"`.
    pub keys: String,
    pub command: Command,
}

pub struct Keymap {
    pub bindings: Vec<(KeyboardShortcut, Binding)>,
    pub errors: Vec<String>,
}

pub fn parse_shortcut(s: &str) -> Option<KeyboardShortcut> {
    let mut mods = Modifiers::NONE;
    let mut key = None;
    for part in s.split('+').map(str::trim) {
        match part.to_ascii_lowercase().as_str() {
            "ctrl" | "cmd" | "command" => mods.command = true,
            "shift" => mods.shift = true,
            "alt" | "option" => mods.alt = true,
            _ => key = Key::from_name(part),
        }
    }
    if mods.command {
        mods.ctrl = !cfg!(target_os = "macos");
        mods.mac_cmd = cfg!(target_os = "macos");
    }
    key.map(|k| KeyboardShortcut::new(mods, k))
}

const DEFAULT: &str = include_str!("../keymap.json");

impl Keymap {
    /// Built-in bindings, overridden per command by `user` if it exists.
    pub fn load(user: &Path) -> Keymap {
        let mut bindings: Vec<Binding> = serde_json::from_str(DEFAULT).expect("default keymap parses");
        let mut errors = Vec::new();
        if let Ok(text) = std::fs::read_to_string(user) {
            match serde_json::from_str::<Vec<Binding>>(&text) {
                Ok(overrides) => {
                    for o in overrides {
                        bindings.retain(|b| b.command != o.command);
                        bindings.push(o);
                    }
                }
                Err(e) => errors.push(format!("{}: {e}", user.display())),
            }
        }
        let bindings = bindings
            .into_iter()
            .filter_map(|b| match parse_shortcut(&b.keys) {
                Some(s) => Some((s, b)),
                None => {
                    errors.push(format!("unknown key combination '{}'", b.keys));
                    None
                }
            })
            .collect();
        Keymap { bindings, errors }
    }

    pub fn shortcut_text(&self, ctx: &egui::Context, cmd: Command) -> Option<String> {
        self.bindings.iter().find(|(_, b)| b.command == cmd).map(|(s, _)| ctx.format_shortcut(s))
    }

    /// Commands triggered this frame. Unmodified keys are ignored while a
    /// text field has focus.
    pub fn poll(&self, ctx: &egui::Context) -> Vec<Command> {
        let typing = ctx.egui_wants_keyboard_input();
        let mut out = Vec::new();
        ctx.input_mut(|i| {
            // Longest modifier sets first so Ctrl+Shift+Z wins over Ctrl+Z.
            let mut order: Vec<&(KeyboardShortcut, Binding)> = self.bindings.iter().collect();
            order.sort_by_key(|(s, _)| std::cmp::Reverse(s.modifiers.ctrl as u8 + s.modifiers.shift as u8 + s.modifiers.alt as u8 + s.modifiers.mac_cmd as u8));
            for (s, b) in order {
                if typing && s.modifiers.is_none() {
                    continue;
                }
                if i.consume_shortcut(s) {
                    out.push(b.command);
                }
            }
        });
        out
    }
}
