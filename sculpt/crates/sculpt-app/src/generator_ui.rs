//! Properties sections for the volume's generator (rocks) and surface noise layers. A slider drag is one edit: the
//! volume previews it while dragging and composes it exactly when the drag ends.

use egui::{RichText, Ui};
use sculpt_core::generator::density::Density;
use sculpt_core::generator::layer::{EdgeSide, PlaceOn};
use sculpt_core::generator::{BoolOp, Primitive};

use crate::app::SculptApp;
use crate::panels::{prop, section};

const AXES: [(&str, [f32; 3]); 6] = [("Down (ground)", [0.0, -1.0, 0.0]), ("Up (top)", [0.0, 1.0, 0.0]), ("+X side", [1.0, 0.0, 0.0]), ("-X side", [-1.0, 0.0, 0.0]), ("+Z side", [0.0, 0.0, 1.0]), ("-Z side", [0.0, 0.0, -1.0])];

/// A drop-down for a direction, from the six axes. Returns true if the choice changed.
fn axis_combo(ui: &mut Ui, id: egui::Id, axis: &mut [f32; 3]) -> bool {
    let current = AXES.iter().find(|(_, a)| a == axis).map_or("Custom", |(n, _)| *n);
    let mut changed = false;
    egui::ComboBox::from_id_salt(id).selected_text(current).show_ui(ui, |ui| {
        for (name, a) in AXES {
            if ui.selectable_label(current == name, name).clicked() {
                *axis = a;
                changed = true;
            }
        }
    });
    changed
}

fn op_label(op: BoolOp) -> &'static str {
    match op {
        BoolOp::Add => "Add",
        BoolOp::Subtract => "Subtract",
        BoolOp::Intersect => "Intersect",
    }
}

fn primitive_label(p: Primitive) -> &'static str {
    match p {
        Primitive::Box => "Box",
        Primitive::Sphere => "Sphere",
        Primitive::Cylinder => "Cylinder",
        Primitive::Cone => "Cone",
        Primitive::Wedge => "Wedge",
        Primitive::ChamferCutter => "Chamfer cutter",
        Primitive::Torus => "Torus",
    }
}

/// True while the user is actually working the widget. egui widgets can report a change on their own (a slider rounding the
/// value it is shown), and that must not become an edit: it would fight whatever else set the value.
fn touched(r: &egui::Response) -> bool {
    r.dragged() || r.drag_started() || r.drag_stopped() || r.has_focus() || r.lost_focus() || r.clicked() || r.is_pointer_button_down_on()
}

/// Record a widget edit. A drag in progress coalesces into one undo step.
fn note(r: &egui::Response, changed: &mut bool, coalesce: &mut bool) {
    if r.changed() && touched(r) {
        *changed = true;
        *coalesce = r.dragged() && !r.drag_started();
    }
}

/// One slider row of the volume generator's section, noting a change and whether it is being dragged.
fn vslider(ui: &mut Ui, label: &str, w: egui::Slider, hover: Option<&str>, changed: &mut bool, coalesce: &mut bool, dragging: &mut bool) {
    let r = prop(ui, label, |ui| ui.add(w));
    let r = match hover {
        Some(h) => r.on_hover_text(h),
        None => r,
    };
    *dragging |= r.dragged();
    note(&r, changed, coalesce);
}

/// The Properties section for the volume's generator layer selected, if one is.
pub(crate) fn volume_generator_props(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let m = &mut app.volume;
    let Some(layer) = m.layer.as_ref() else { return };
    let k = layer.active();
    let Some(spec) = layer.generator(k).cloned() else { return };
    let placed = layer.generator_instances(k).len();
    let (ms, full, join) = (m.generator_ms, m.generator_full_ms, m.join_ms);
    let on_gpu = m.layer.as_ref().is_some_and(|l| l.gpu_can_join());
    let mut next = spec.clone();
    let (mut changed, mut coalesce, mut dragging) = (false, false, false);
    section(ui, fs, "volume_generator", "GENERATOR", |ui| {
        prop(ui, "Operation", |ui| {
            egui::ComboBox::from_id_salt(("vgen_op", k))
                .selected_text(op_label(next.op))
                .show_ui(ui, |ui| {
                    for op in [BoolOp::Add, BoolOp::Subtract] {
                        if ui.selectable_label(next.op == op, op_label(op)).clicked() {
                            next.op = op;
                            changed = true;
                        }
                    }
                })
                .response
        });
        prop(ui, "Primitive", |ui| {
            egui::ComboBox::from_id_salt(("vgen_prim", k)).selected_text(primitive_label(next.primitive)).show_ui(ui, |ui| {
                for p in Primitive::ALL {
                    if ui.selectable_label(next.primitive == p, primitive_label(p)).clicked() {
                        next.primitive = p;
                        changed = true;
                    }
                }
            })
        });
        let (c, co, d) = (&mut changed, &mut coalesce, &mut dragging);
        vslider(ui, "Count", egui::Slider::new(&mut next.count, 1..=2000).logarithmic(true), None, c, co, d);
        vslider(ui, "Size", egui::Slider::new(&mut next.size, 0.005..=0.5).logarithmic(true).max_decimals(3), None, c, co, d);
        vslider(ui, "Lift", egui::Slider::new(&mut next.lift, -1.0..=1.5).max_decimals(2), Some("How far each instance's centre stands out of the surface, in its size."), c, co, d);
        let r = prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut next.seed).speed(0.2)));
        *d |= r.dragged();
        note(&r, c, co);
        vslider(ui, "Blend", egui::Slider::new(&mut next.blend, 0.0..=1.0).max_decimals(2), Some("How softly it meets the surface, and the instances one another, as a fraction of their size."), c, co, d);
        vslider(ui, "Spacing", egui::Slider::new(&mut next.min_spacing, 0.0..=1.0).max_decimals(3), Some("Smallest distance between two instances. 0 places them independently; if the surface is full, fewer are placed."), c, co, d);
        vslider(ui, "Size variation", egui::Slider::new(&mut next.size_var, 0.0..=1.0).max_decimals(2), None, c, co, d);
        vslider(ui, "Stretch", egui::Slider::new(&mut next.stretch_var, 0.0..=1.0).max_decimals(2), Some("Varies each axis separately, so instances come out stretched or squashed."), c, co, d);
        vslider(ui, "Tilt", egui::Slider::new(&mut next.tilt, 0.0..=90.0).suffix(" deg").max_decimals(0), Some("Largest random lean away from the surface normal."), c, co, d);
        vslider(ui, "Lift jitter", egui::Slider::new(&mut next.lift_var, 0.0..=1.0).max_decimals(2), None, c, co, d);
        let r = prop(ui, "Spin", |ui| ui.checkbox(&mut next.random_spin, "Random"));
        note(&r, c, co);
        ui.add_space(4.0);
        // Where they go: the surface or its sharp edges, and a density that gathers them.
        prop(ui, "Place on", |ui| {
            let label = |p: &PlaceOn| if matches!(p, PlaceOn::Edges { .. }) { "Edges" } else { "Surface" };
            egui::ComboBox::from_id_salt(("vgen_place", k))
                .selected_text(label(&next.place_on))
                .show_ui(ui, |ui| {
                    for option in [PlaceOn::Surface, PlaceOn::default_edges()] {
                        if ui.selectable_label(label(&next.place_on) == label(&option), label(&option)).clicked() && label(&next.place_on) != label(&option) {
                            next.place_on = option;
                            *c = true;
                        }
                    }
                })
                .response
                .on_hover_text("Anywhere on the surface, or along its sharp edges: chips off a ridge, rubble at the foot of a wall.")
        });
        if let PlaceOn::Edges { angle, side } = &mut next.place_on {
            vslider(ui, "Edge angle", egui::Slider::new(angle, 10.0..=120.0).suffix(" deg").max_decimals(0), Some("How sharply the surface has to turn to count as an edge. Lower takes rounder ridges."), c, co, d);
            prop(ui, "Edges", |ui| {
                let name = |s: EdgeSide| match s {
                    EdgeSide::Both => "Ridges and creases",
                    EdgeSide::Ridges => "Ridges",
                    EdgeSide::Creases => "Creases",
                };
                egui::ComboBox::from_id_salt(("vgen_side", k)).selected_text(name(*side)).show_ui(ui, |ui| {
                    for s in [EdgeSide::Both, EdgeSide::Ridges, EdgeSide::Creases] {
                        if ui.selectable_label(*side == s, name(s)).clicked() && *side != s {
                            *side = s;
                            *c = true;
                        }
                    }
                })
            });
        }
        prop(ui, "Density", |ui| {
            egui::ComboBox::from_id_salt(("vgen_density", k))
                .selected_text(next.density.label())
                .show_ui(ui, |ui| {
                    for (name, option) in [("Even", Density::Even), ("Clusters", Density::default_clusters()), ("Facing", Density::default_facing()), ("Painted", Density::default_painted())] {
                        if ui.selectable_label(next.density.label() == name, name).clicked() && next.density.label() != name {
                            next.density = option;
                            *c = true;
                        }
                    }
                })
                .response
                .on_hover_text("Where the instances gather. The count stays; the density decides where it goes.")
        });
        match &mut next.density {
            Density::Clusters { scale, coverage, softness, seed } => {
                vslider(ui, "Patch size", egui::Slider::new(scale, 0.5..=12.0).logarithmic(true).max_decimals(1), Some("Larger is bigger patches. Smaller is finer, more frequent clumps."), c, co, d);
                vslider(ui, "Coverage", egui::Slider::new(coverage, 0.0..=1.0).max_decimals(2), Some("How much of the surface is dense."), c, co, d);
                vslider(ui, "Softness", egui::Slider::new(softness, 0.02..=1.0).max_decimals(2), None, c, co, d);
                let r = prop(ui, "Pattern seed", |ui| ui.add(egui::DragValue::new(seed).speed(0.2)));
                *d |= r.dragged();
                note(&r, c, co);
            }
            Density::Facing { axis, sharpness } => {
                *c |= prop(ui, "Faces", |ui| axis_combo(ui, egui::Id::new(("vgen_facing", k)), axis));
                vslider(ui, "Tightness", egui::Slider::new(sharpness, 0.3..=6.0).logarithmic(true).max_decimals(1), Some("Higher means the surface has to face that way more squarely."), c, co, d);
            }
            Density::Painted { .. } => {
                ui.label(RichText::new("Paint it: pick Density: less / more under this layer in the Layers panel, then drag on the object. The rocks scatter again when the stroke ends.").small().color(weak));
            }
            Density::Even => {}
        }
        if placed < spec.count as usize {
            ui.label(RichText::new(format!("{placed} of {} placed: the places that take them are full or few.", spec.count)).small().color(weak));
        }
        ui.add_space(4.0);
        vslider(ui, "Roughness", egui::Slider::new(&mut next.rough, 0.0..=0.6).max_decimals(2), Some("Noise on each instance's surface, up to this fraction of its size: rocks become stones."), c, co, d);
        vslider(ui, "Bump size", egui::Slider::new(&mut next.rough_scale, 0.3..=8.0).logarithmic(true).max_decimals(1), Some("Bumps of the roughness across one instance."), c, co, d);
        prop(ui, "Rough kind", |ui| {
            use sculpt_core::noise::NoiseKind;
            let name = |k: NoiseKind| match k {
                NoiseKind::Ridged => "Ridged",
                NoiseKind::Cellular => "Pitted",
                _ => "Smooth",
            };
            egui::ComboBox::from_id_salt(("vgen_rough", k)).selected_text(name(next.rough_kind)).show_ui(ui, |ui| {
                for kind in [NoiseKind::Fbm, NoiseKind::Ridged, NoiseKind::Cellular] {
                    if ui.selectable_label(next.rough_kind == kind, name(kind)).clicked() {
                        next.rough_kind = kind;
                        *c = true;
                    }
                }
            })
        });
        ui.add_space(4.0);
        let how = if on_gpu {
            format!("Last edit shown at its own level by the GPU ({ms:.1} + {join:.1} ms on the CPU); the exact field {full:.0} ms later, composed off the frame.")
        } else {
            format!("Last edit shown in {ms:.1} ms, no finer than level 1; at its own level {full:.0} ms later, composed off the frame.")
        };
        ui.label(RichText::new(format!("{how} Sculpting the form carries it along; its mask (Layers panel) fades it.")).small().color(weak));
    });
    let m = &mut app.volume;
    if changed && next != spec {
        m.set_generator(k, &next, dragging || coalesce);
    } else if !dragging {
        m.end_generator_edit();
    }
}

/// The Properties section for the volume's surface noise layer selected, if one is.
pub(crate) fn volume_noise_props(app: &mut SculptApp, ui: &mut Ui) {
    let fs = app.theme.metrics.font_size;
    let weak = app.theme.weak_text();
    let m = &mut app.volume;
    let Some(layer) = m.layer.as_ref() else { return };
    let k = layer.active();
    let Some(n0) = layer.noise(k).copied() else { return };
    let (lo, hi) = m.vol.bounds();
    let extent = (hi - lo).max_element().max(1e-3);
    let mut n = n0;
    let (mut changed, mut coalesce, mut dragging) = (false, false, false);
    section(ui, fs, "volume_noise", "SURFACE NOISE", |ui| {
        let (c, co, d) = (&mut changed, &mut coalesce, &mut dragging);
        vslider(ui, "Height", egui::Slider::new(&mut n.amplitude, 0.0001 * extent..=0.05 * extent).logarithmic(true).max_decimals(4), Some("The most it moves the surface."), c, co, d);
        vslider(ui, "Bumps per unit", egui::Slider::new(&mut n.scale, 0.5 / extent..=400.0 / extent).logarithmic(true).max_decimals(1), Some("Of its first octave; each next one is twice as fine."), c, co, d);
        vslider(ui, "Octaves", egui::Slider::new(&mut n.octaves, 1..=8), None, c, co, d);
        prop(ui, "Kind", |ui| {
            ui.horizontal(|ui| {
                *c |= ui.selectable_value(&mut n.kind, 0, "Smooth").changed();
                *c |= ui.selectable_value(&mut n.kind, 1, "Ridged").changed();
            })
            .response
        });
        let r = prop(ui, "Seed", |ui| ui.add(egui::DragValue::new(&mut n.seed).speed(0.2)));
        *d |= r.dragged();
        note(&r, c, co);
        ui.add_space(4.0);
        ui.label(RichText::new("Drawn by the GPU over the whole surface, finer than any level; not in the field, so brushes work on the surface under it. Its mask (Layers panel) fades it.").small().color(weak));
    });
    // Baking it into a detail layer, where its mask lets it show, at a level chosen beside an estimate of the cost.
    let m = &mut app.volume;
    let Some(layer) = m.layer.as_ref() else { return };
    let (from, top) = (layer.from(), m.vol.levels() - 1);
    let id = ui.id().with(("bake_level", k));
    let mut level = ui.data_mut(|d| *d.get_temp_mut_or(id, from)).clamp(from, top);
    let est_id = ui.id().with(("bake_estimate", k, level, layer.noise_epoch()));
    let estimate = match ui.data(|d| d.get_temp::<usize>(est_id)) {
        Some(e) => e,
        None => {
            let e = layer.noise_estimate(k, level);
            ui.data_mut(|d| d.insert_temp(est_id, e));
            e
        }
    };
    let mut bake = false;
    ui.horizontal(|ui| {
        bake = ui.button("Bake into a layer").on_hover_text("Puts the noise in the field as a detail layer at the level beside (the octaves it holds), where its mask lets it show, and hides this layer. One undo.").clicked();
        egui::ComboBox::from_id_salt(("bake_level_combo", k)).selected_text(format!("level {level}")).show_ui(ui, |ui| {
            for l in from..=top {
                ui.selectable_value(&mut level, l, format!("level {l}"));
            }
        });
    });
    ui.data_mut(|d| d.insert_temp(id, level));
    ui.label(RichText::new(format!("About {} blocks{}", estimate, if estimate > 200_000 { ": a lot; mask it to where it is wanted, or bake coarser" } else { "" })).small().color(weak));
    if bake {
        m.bake_noise(k, level);
        return;
    }
    if changed && n != n0 {
        m.set_noise(k, n, dragging || coalesce);
    } else if !dragging {
        m.set_noise(k, n0, false);
    }
}