//! Sculpt — Mudbox-style sculpting UI on the sculpt-core engine.
//!
//! ```text
//! sculpt-app [project.sculpt] [--level N] [--quads N] [--theme NAME]
//!            [--test-strokes FRAMES] [--bench-orbit FRAMES] [--screenshot out.png] [--size WxH]
//!            [--demo-layers]
//! ```

mod app;
mod camera;
mod icons;
mod layer_panel;
mod keymap;
mod panels;
mod pen;
mod test_driver;
mod theme;
mod tools;
mod viewport;

use std::path::PathBuf;

use eframe::egui_wgpu::{SurfaceConfig, WgpuConfiguration, WgpuSetup};

/// The default device, except that it may allocate the adapter's largest buffer. The level-of-detail
/// index pool of a 20M-triangle mesh is several hundred MB, past wgpu's conservative 256 MiB default.
fn gpu_setup() -> WgpuSetup {
    let WgpuSetup::CreateNew(mut setup) = WgpuConfiguration::default().wgpu_setup else { unreachable!("the default creates its own device") };
    let base = setup.device_descriptor.clone();
    setup.device_descriptor = std::sync::Arc::new(move |adapter| {
        let mut desc = base(adapter);
        let have = adapter.limits();
        desc.required_limits.max_buffer_size = have.max_buffer_size.min(4 << 30);
        desc.required_limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size.min(2 << 30);
        desc
    });
    WgpuSetup::CreateNew(setup)
}

fn main() -> eframe::Result {
    let mut args = std::env::args().skip(1);
    let mut opts = app::Options { project: None, level: 7, sphere_res: None, theme: None, test_frames: None, screenshot: None, demo_layers: false, bench_orbit: None };
    let mut size = [1600.0f32, 1000.0];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" => opts.level = args.next().and_then(|v| v.parse().ok()).unwrap_or(7).clamp(1, 11),
            "--quads" => opts.sphere_res = args.next().and_then(|v| v.parse().ok()).map(|n: u32| n.clamp(2, 2000)),
            "--bench-orbit" => opts.bench_orbit = args.next().and_then(|v| v.parse().ok()),
            "--theme" => opts.theme = args.next(),
            "--test-strokes" => opts.test_frames = args.next().and_then(|v| v.parse().ok()),
            "--screenshot" => opts.screenshot = args.next().map(PathBuf::from),
            "--demo-layers" => opts.demo_layers = true,
            "--size" => {
                if let Some((w, h)) = args.next().as_deref().and_then(|s| s.split_once('x')) {
                    size = [w.parse().unwrap_or(size[0]), h.parse().unwrap_or(size[1])];
                }
            }
            "-h" | "--help" => {
                println!("sculpt-app [project.sculpt] [--level N] [--theme NAME] [--test-strokes FRAMES] [--bench-orbit FRAMES] [--screenshot out.png] [--size WxH]");
                return Ok(());
            }
            p => opts.project = Some(PathBuf::from(p)),
        }
    }

    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Sculpt").with_inner_size(size).with_min_inner_size([900.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        // Sculpting wants the shortest input-to-photon path.
        wgpu_options: WgpuConfiguration { surface: SurfaceConfig::LOW_LATENCY, wgpu_setup: gpu_setup(), ..Default::default() },
        ..Default::default()
    };
    eframe::run_native("Sculpt", native, Box::new(|cc| Ok(Box::new(app::SculptApp::new(cc, opts)))))
}
