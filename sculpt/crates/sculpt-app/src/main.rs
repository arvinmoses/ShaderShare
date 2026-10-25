//! Sculpt — a layered volume sculptor on the sculpt-core engine.
//!
//! ```text
//! sculpt-app [file.svol] [--triangles M] [--theme NAME] [--zoom F] [--size WxH] [--screenshot out.png] [--demo-volume DIR]
//! sculpt-app --gpu-levels-check [...]                                    (the GPU surface against the CPU reference)
//! sculpt-app --gpu-bench [...] | --gpu-volume-bench [...] | --gpu-sdf-bench | --gpu-sdf-check | --gpu-sdf-png <dir>
//!                                                                         (headless GPU spikes, no window)
//! ```

mod app;
mod camera;
mod generator_ui;
mod gpu_boolean;
mod gpu_levels;
mod gpu_sdf;
mod gpu_volume;
mod icons;
mod keymap;
mod panels;
mod pen;
mod theme;
mod tools;
mod viewport;
mod volume_demo;
mod volume_mode;

use std::path::PathBuf;

use eframe::egui_wgpu::{SurfaceConfig, WgpuConfiguration, WgpuSetup};

/// The default device, except that it may allocate the adapter's largest buffer and bind the storage buffers the
/// volume's surface needs.
fn gpu_setup() -> WgpuSetup {
    let WgpuSetup::CreateNew(mut setup) = WgpuConfiguration::default().wgpu_setup else { unreachable!("the default creates its own device") };
    let base = setup.device_descriptor.clone();
    setup.device_descriptor = std::sync::Arc::new(move |adapter| {
        let mut desc = base(adapter);
        let have = adapter.limits();
        desc.required_limits.max_buffer_size = have.max_buffer_size.min(4 << 30);
        desc.required_limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size.min(2 << 30);
        // The volume's GPU surface binds 13 storage buffers in one pass and draws with an indirect count.
        desc.required_limits.max_storage_buffers_per_shader_stage = have.max_storage_buffers_per_shader_stage.min(16);
        desc.required_features |= adapter.features() & (eframe::egui_wgpu::wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | eframe::egui_wgpu::wgpu::Features::TIMESTAMP_QUERY);
        desc
    });
    WgpuSetup::CreateNew(setup)
}

const HELP: &str = "sculpt-app [file.svol] [--triangles M] [--theme NAME] [--zoom F] [--size WxH] [--screenshot out.png] [--demo-volume DIR]
  file.svol        open this volume (otherwise a new ball)
  --triangles M    the new ball's size, in millions of triangles (default 1)
  --screenshot P   save the window once the volume has settled, then close
  --demo-volume D  run the scripted session (SCULPT_DEMO_SCRIPT picks one), saving every frame into D
sculpt-app --gpu-levels-check            the GPU surface against the CPU reference";

fn main() -> eframe::Result {
    // Headless GPU checks and benchmarks: no window, their own wgpu device.
    let first = std::env::args().nth(1);
    if first.as_deref() == Some("--gpu-bench") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        if let Err(e) = gpu_boolean::bench::run(&args) {
            eprintln!("gpu-bench: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if first.as_deref() == Some("--gpu-levels-check") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        if let Err(e) = gpu_levels::check::run(&args) {
            eprintln!("gpu-levels-check: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if first.as_deref() == Some("--gpu-volume-bench") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        if let Err(e) = gpu_volume::bench(&args) {
            eprintln!("gpu-volume-bench: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let all: Vec<String> = std::env::args().skip(1).collect();
    if all.first().is_some_and(|a| gpu_sdf::is_gpu_sdf_command(a)) {
        if let Err(e) = gpu_sdf::cli(&all) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let mut args = std::env::args().skip(1);
    let mut opts = app::Options { file: None, triangles: 1e6, theme: None, zoom: 1.0, screenshot: None, demo_volume: None };
    let mut size = [1600.0f32, 1000.0];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--triangles" => opts.triangles = args.next().and_then(|v| v.parse::<f32>().ok()).map_or(1e6, |m| m.clamp(0.05, 40.0) * 1e6),
            "--zoom" => opts.zoom = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0f32).clamp(0.05, 20.0),
            "--theme" => opts.theme = args.next(),
            "--screenshot" => opts.screenshot = args.next().map(PathBuf::from),
            "--demo-volume" => opts.demo_volume = args.next().map(PathBuf::from),
            // The volume is all there is; accepted so older scripts still run.
            "--volume" => {}
            "--size" => {
                if let Some((w, h)) = args.next().as_deref().and_then(|s| s.split_once('x')) {
                    size = [w.parse().unwrap_or(size[0]), h.parse().unwrap_or(size[1])];
                }
            }
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            p if p.starts_with("--") => {
                eprintln!("unknown option {p}\n{HELP}");
                std::process::exit(2);
            }
            p => opts.file = Some(PathBuf::from(p)),
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
