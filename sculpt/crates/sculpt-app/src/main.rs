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
mod generator_ui;
mod gpu_boolean;
mod gpu_sdf;
mod gpu_volume;
mod icons;
mod layer_panel;
mod keymap;
mod panels;
mod pen;
mod remesh_ui;
mod solid_preview;
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

/// `source>target:strength[:invert]`, for example `cavity>size:1` or `slope>size:-0.6:invert`. Sources: slope, height, noise,
/// cavity, curvature, ao, thickness, channel, random, index. Targets: size, height, lift, tilt.
fn parse_map_row(text: &str) -> Option<sculpt_core::generator::mapping::AttributeRow> {
    use sculpt_core::bake::MeshAttribute;
    use sculpt_core::generator::mapping::{AttributeRow, AttributeSource, AttributeTarget};
    let (path, rest) = text.split_once(':')?;
    let (source, target) = path.split_once('>')?;
    let source = match source {
        "slope" => AttributeSource::Slope { axis: [0.0, 1.0, 0.0] },
        "height" => AttributeSource::Height { axis: [0.0, 1.0, 0.0], from: -1.0, to: 1.0 },
        "noise" => AttributeSource::Noise { scale: 3.0, seed: 1 },
        "cavity" => AttributeSource::Mesh { attribute: MeshAttribute::Cavity },
        "curvature" => AttributeSource::Mesh { attribute: MeshAttribute::Curvature },
        "ao" => AttributeSource::Mesh { attribute: MeshAttribute::AmbientOcclusion },
        "thickness" => AttributeSource::Mesh { attribute: MeshAttribute::Thickness },
        "channel" => AttributeSource::Channel { name: "paint.mask".into() },
        "random" => AttributeSource::Random { seed: 1 },
        "index" => AttributeSource::Index,
        other => {
            eprintln!("--demo-map: unknown source '{other}'");
            return None;
        }
    };
    let target = match target {
        "size" => AttributeTarget::Size,
        "height" => AttributeTarget::Stretch,
        "lift" => AttributeTarget::Lift,
        "tilt" => AttributeTarget::Tilt,
        other => {
            eprintln!("--demo-map: unknown target '{other}'");
            return None;
        }
    };
    let mut f = rest.split(':');
    let mut row = AttributeRow::new(source, target, f.next()?.parse().ok()?);
    row.remap.invert = f.next() == Some("invert");
    Some(row)
}

/// `even`, `clusters[:scale:coverage:softness:seed]`, `facing[:down|up|x|-x|z|-z[:tightness]]` or `painted[:channel]`.
fn parse_density(text: &str) -> Option<sculpt_core::generator::density::Density> {
    use sculpt_core::generator::density::Density;
    let mut f = text.split(':');
    let num = |v: Option<&str>, d: f32| v.and_then(|v| v.parse().ok()).unwrap_or(d);
    match f.next()? {
        "even" => Some(Density::Even),
        "clusters" => {
            let Density::Clusters { scale, coverage, softness, seed } = Density::default_clusters() else { return None };
            Some(Density::Clusters { scale: num(f.next(), scale), coverage: num(f.next(), coverage), softness: num(f.next(), softness), seed: num(f.next(), seed as f32) as u32 })
        }
        "facing" => {
            let axis = match f.next().unwrap_or("down") {
                "up" => [0.0, 1.0, 0.0],
                "x" => [1.0, 0.0, 0.0],
                "-x" => [-1.0, 0.0, 0.0],
                "z" => [0.0, 0.0, 1.0],
                "-z" => [0.0, 0.0, -1.0],
                _ => [0.0, -1.0, 0.0],
            };
            Some(Density::Facing { axis, sharpness: num(f.next(), 1.5) })
        }
        "painted" => Some(Density::Painted { channel: f.next().unwrap_or("paint.mask").to_string() }),
        other => {
            eprintln!("--demo-density: unknown density '{other}'");
            None
        }
    }
}

/// `op:primitive:count:size:lift:seed[:spacing:size_var:stretch:tilt:lift_var]`, for example `subtract:sphere:60:0.12:0.6:7`.
/// Trailing fields are optional.
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
    spec.min_spacing = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.min_spacing);
    spec.size_var = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.size_var);
    spec.stretch_var = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.stretch_var);
    spec.tilt = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.tilt);
    spec.lift_var = f.next().and_then(|v| v.parse().ok()).unwrap_or(spec.lift_var);
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
    // Headless GPU volume sculpting spike: see gpu_volume/mod.rs.
    if std::env::args().nth(1).as_deref() == Some("--gpu-volume-bench") {
        let args: Vec<String> = std::env::args().skip(2).collect();
        if let Err(e) = gpu_volume::bench(&args) {
            eprintln!("gpu-volume-bench: {e}");
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
    let mut opts = app::Options { project: None, level: 7, sphere_res: None, theme: None, test_frames: None, screenshot: None, demo_layers: false, demo_generator: None, demo_sweep: false, sweep_shots: None, bench_orbit: None, zoom: 1.0 };
    let mut size = [1600.0f32, 1000.0];
    let (mut shots_dir, mut shot_times) = (None::<PathBuf>, Vec::<f32>::new());
    let mut demo_density = None::<sculpt_core::generator::density::Density>;
    let mut demo_rows: Vec<sculpt_core::generator::mapping::AttributeRow> = Vec::new();
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
            "--demo-sweep" => opts.demo_sweep = true,
            "--demo-map" => {
                if let Some(row) = args.next().and_then(|v| parse_map_row(&v)) {
                    demo_rows.push(row);
                }
            }
            "--demo-density" => demo_density = args.next().and_then(|v| parse_density(&v)),
            "--demo-sweep-shots" => shots_dir = args.next().map(PathBuf::from),
            "--demo-sweep-times" => shot_times = args.next().map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect()).unwrap_or_default(),
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

    if let Some(dir) = shots_dir.filter(|_| !shot_times.is_empty()) {
        let _ = std::fs::create_dir_all(&dir);
        opts.sweep_shots = Some((dir, shot_times));
        opts.demo_sweep = true;
    }
    if !demo_rows.is_empty() {
        opts.demo_generator.get_or_insert_with(sculpt_core::generator::layer::GeneratorSpec::default).mapping = demo_rows;
    }
    if let Some(d) = demo_density {
        opts.demo_generator.get_or_insert_with(sculpt_core::generator::layer::GeneratorSpec::default).density = d;
    }
    if opts.demo_sweep && opts.demo_generator.is_none() {
        opts.demo_generator = Some(sculpt_core::generator::layer::GeneratorSpec { count: 60, size: 0.10, seed: 7, ..Default::default() });
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
