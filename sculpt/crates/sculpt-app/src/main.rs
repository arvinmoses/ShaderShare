//! Sculpt — Mudbox-style sculpting UI on the sculpt-core engine.
//!
//! ```text
//! sculpt-app [project.sculpt] [--level N] [--quads N] [--theme NAME]
//!            [--test-strokes FRAMES] [--bench-orbit FRAMES] [--zoom F] [--screenshot out.png] [--demo-generator op:primitive:count:size:lift:seed] [--size WxH]
//!            [--demo-layers]
//! sculpt-app --gpu-bench [LEVELS] [COUNTS] [--op subtract|add|intersect|all] [--mixed] [--reps N] [--cpu] [--check] [--png out.png]
//! sculpt-app --gpu-sdf-bench | --gpu-sdf-check | --gpu-sdf-png <dir>      (headless GPU SDF boolean spike)
//! ```

mod app;
mod camera;
mod gpu_boolean;
mod gpu_sdf;
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

/// `op:primitive:count:size:lift:seed`, for example `subtract:sphere:60:0.12:0.6:7`. Trailing fields are optional.
fn parse_generator_demo(text: &str) -> Option<sculpt_core::generator::layer::GeneratorSpec> {
    use sculpt_core::generator::{BoolOp, Primitive, layer::GeneratorSpec};
    let mut f = text.split(':');
    let op = match f.next()? {
        "add" => BoolOp::Add,
        "subtract" | "sub" => BoolOp::Subtract,
        "intersect" => BoolOp::Intersect,
        other => {
            eprintln!("--demo-generator: unknown operation '{other}'");
            return None;
        }
    };
    let mut spec = GeneratorSpec { op, ..GeneratorSpec::default() };
    if let Some(p) = f.next() {
        spec.primitive = match p {
            "box" => Primitive::Box,
            "sphere" => Primitive::Sphere,
            "cylinder" => Primitive::Cylinder,
            "cone" => Primitive::Cone,
            "wedge" => Primitive::Wedge,
            "chamfer" => Primitive::ChamferCutter,
            "torus" => Primitive::Torus,
            other => {
                eprintln!("--demo-generator: unknown primitive '{other}'");
                return None;
            }
        };
    }
    spec.count = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.count);
    spec.size = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.size);
    spec.lift = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.lift);
    spec.seed = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.seed);
    Some(spec)
}

fn main() -> eframe::Result {
    // Headless GPU boolean benchmark: no window, own wgpu device.
    if std::env::args().nth(1).as_deref() == Some("--gpu-bench") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        if let Err(e) = gpu_boolean::bench::run(&args) {
            eprintln!("gpu-bench: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    // Headless GPU SDF boolean spike (own device, no window): see gpu_sdf/mod.rs.
    let all: Vec<String> = std::env::args().skip(1).collect();
    if all.first().is_some_and(|a| gpu_sdf::is_gpu_sdf_command(a)) {
        if let Err(e) = gpu_sdf::cli(&all) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let mut args = std::env::args().skip(1);
    let mut opts = app::Options { project: None, level: 7, sphere_res: None, theme: None, test_frames: None, screenshot: None, demo_layers: false, demo_generator: None, bench_orbit: None, zoom: 1.0 };
    let mut size = [1600.0f32, 1000.0];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--level" => opts.level = args.next().and_then(|v| v.parse().ok()).unwrap_or(7).clamp(1, 11),
            "--quads" => opts.sphere_res = args.next().and_then(|v| v.parse().ok()).map(|n: u32| n.clamp(2, 2000)),
            "--zoom" => opts.zoom = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.0f32).clamp(0.05, 20.0),
            "--bench-orbit" => opts.bench_orbit = args.next().and_then(|v| v.parse().ok()),
            "--theme" => opts.theme = args.next(),
            "--test-strokes" => opts.test_frames = args.next().and_then(|v| v.parse().ok()),
            "--screenshot" => opts.screenshot = args.next().map(PathBuf::from),
            "--demo-layers" => opts.demo_layers = true,
            "--demo-generator" => opts.demo_generator = args.next().and_then(|v| parse_generator_demo(&v)),
            "--size" => {
                if let Some((w, h)) = args.next().as_deref().and_then(|s| s.split_once('x')) {
                    size = [w.parse().unwrap_or(size[0]), h.parse().unwrap_or(size[1])];
                }
            }
            "-h" | "--help" => {
                println!("sculpt-app [project.sculpt] [--level N] [--theme NAME] [--test-strokes FRAMES] [--bench-orbit FRAMES] [--zoom F] [--screenshot out.png] [--demo-generator op:primitive:count:size:lift:seed] [--size WxH]");
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
