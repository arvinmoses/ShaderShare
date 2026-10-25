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
    BrushSizeUp,
    BrushSizeDown,
    StrengthUp,
    StrengthDown,
    /// Named for the mesh it once framed, so keymaps that bind it keep working.
    #[serde(rename = "frame_mesh")]
    Frame,
    ToggleHud,
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
            Command::BrushSizeUp => "Brush size +",
            Command::BrushSizeDown => "Brush size −",
            Command::StrengthUp => "Strength +",
            Command::StrengthDown => "Strength −",
            Command::Frame => "Frame volume",
            Command::ToggleHud => "Toggle readout",
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
    /// Built-in bindings, overridden per command by `user` if it exists. Entries the app does not know (commands of
    /// older versions) are skipped, not the whole file.
    pub fn load(user: &Path) -> Keymap {
        let mut bindings: Vec<Binding> = serde_json::from_str(DEFAULT).expect("default keymap parses");
        let mut errors = Vec::new();
        if let Ok(text) = std::fs::read_to_string(user) {
            match serde_json::from_str::<Vec<serde_json::Value>>(&text) {
                Ok(entries) => {
                    for e in entries {
                        match serde_json::from_value::<Binding>(e.clone()) {
                            Ok(o) => {
                                bindings.retain(|b| b.command != o.command);
                                bindings.push(o);
                            }
                            Err(_) => errors.push(format!("{}: skipped {e} (no such command)", user.display())),
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_keymap_parses_and_has_no_clashes() {
        let km = Keymap::load(std::path::Path::new("/nonexistent/keymap.json"));
        assert!(km.errors.is_empty(), "{:?}", km.errors);
        let mut seen = std::collections::HashMap::new();
        for (shortcut, b) in &km.bindings {
            if let Some(prev) = seen.insert(format!("{shortcut:?}"), b.command) {
                assert!(prev == b.command, "{:?} and {:?} share {}", prev, b.command, b.keys);
            }
        }
        for cmd in [Command::Undo, Command::Redo, Command::Save, Command::Open, Command::ToolClayBuildup, Command::ToolTrimDynamic, Command::Frame] {
            assert!(km.bindings.iter().any(|(_, b)| b.command == cmd), "{cmd:?} has no default key");
        }
    }

    #[test]
    fn a_user_keymap_with_old_commands_keeps_the_rest() {
        let dir = std::env::temp_dir().join(format!("sculpt-keymap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("keymap.json");
        std::fs::write(&path, r#"[{ "keys": "3", "command": "tool_pose" }, { "keys": "G", "command": "frame_mesh" }]"#).unwrap();
        let km = Keymap::load(&path);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(km.errors.len(), 1, "{:?}", km.errors);
        let frame: Vec<&str> = km.bindings.iter().filter(|(_, b)| b.command == Command::Frame).map(|(_, b)| b.keys.as_str()).collect();
        assert_eq!(frame, ["G"]);
    }
}
