//! The plugin window: DRC/routing settings, live preview, progress and "apply to KiCad".

use crate::board::Board;
use crate::drc::{Seg, ViaOut};
use crate::geom::{BBox, P2};
use crate::kicad::{BoardCtx, Kicad};
use crate::render::layer_color;
use crate::router::{Progress, RouteResult};
use crate::settings::{BackendChoice, CornerStyle, LayerDir, Settings};
use eframe::egui::{self, Color32, Pos2, RichText, Sense, Stroke, Vec2};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

pub const PLUGIN_ID: &str = "com.tracewright.router";

enum Msg {
    Loaded(Result<(Board, BoardCtx, String), String>),
    Gpu(String),
    Progress(Progress),
    /// Result, clearance violations, unconnected on-board, unconnected because of off-board pads.
    Routed(Result<(RouteResult, usize, usize, usize), String>),
    Applied(Result<(), String>),
}

#[derive(PartialEq)]
enum State {
    Loading,
    Ready,
    Routing,
    Routed,
    Applying,
    Applied,
    Failed,
}

pub struct App {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    state: State,
    status: String,
    board: Option<Arc<Board>>,
    ctx: Option<BoardCtx>,
    kicad_version: String,
    gpu: String,
    settings: Settings,
    settings_path: Option<PathBuf>,
    cancel: Arc<AtomicBool>,
    progress: Option<Progress>,
    preview: (Vec<Seg>, Vec<ViaOut>),
    result: Option<RouteResult>,
    drc_violations: usize,
    unconnected: usize,
    unconnected_offboard: usize,
    log: Vec<String>,
    started: Option<std::time::Instant>,
    // View
    zoom: f32,
    pan: Vec2,
    layer_visible: Vec<bool>,
    show_existing: bool,
    skip_text: String,
    only_text: String,
    /// Window resized to fit the monitor yet.
    fitted: bool,
}

fn settings_file(dir: Option<String>) -> Option<PathBuf> {
    let base = dir.map(PathBuf::from).or_else(|| std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("tracewright")))?;
    Some(base.join("settings.json"))
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> App {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        install_fonts(&cc.egui_ctx);
        let (tx, rx) = channel();
        let ectx = cc.egui_ctx.clone();
        // Read the board from KiCad.
        {
            let tx = tx.clone();
            let ectx = ectx.clone();
            std::thread::spawn(move || {
                let r = (|| -> anyhow::Result<(Board, BoardCtx, String)> {
                    let mut k = Kicad::connect("tracewright")?;
                    let v = k.version().unwrap_or_default();
                    let (b, c) = k.read_board()?;
                    Ok((b, c, v))
                })();
                let _ = tx.send(Msg::Loaded(r.map_err(|e| format!("{e:#}"))));
                ectx.request_repaint();
            });
        }
        // Probe the GPU in parallel.
        {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let s = match crate::search::gpu::GpuBackend::new() {
                    Ok(g) => g.adapter_name(),
                    Err(e) => format!("none ({e})"),
                };
                let _ = tx.send(Msg::Gpu(s));
                ectx.request_repaint();
            });
        }
        App {
            tx,
            rx,
            state: State::Loading,
            status: "Reading the board from KiCad…".into(),
            board: None,
            ctx: None,
            kicad_version: String::new(),
            gpu: "detecting…".into(),
            settings: Settings::default(),
            settings_path: None,
            cancel: Arc::new(AtomicBool::new(false)),
            progress: None,
            preview: (vec![], vec![]),
            result: None,
            drc_violations: 0,
            unconnected: 0,
            unconnected_offboard: 0,
            log: vec![],
            started: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            layer_visible: vec![true; 32],
            show_existing: true,
            skip_text: String::new(),
            only_text: String::new(),
            fitted: false,
        }
    }

    fn poll(&mut self) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Loaded(Ok((b, c, v))) => {
                    // Saved settings, reconciled with this board.
                    let path = Kicad::connect("tracewright").ok().and_then(|mut k| k.plugin_settings_path(PLUGIN_ID).ok());
                    self.settings_path = settings_file(path);
                    let saved = self.settings_path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str::<Settings>(&s).ok());
                    self.settings = match saved {
                        Some(mut s) => {
                            let fresh = Settings::from_board(&b);
                            s.reconcile(&b);
                            // Board rules always win for the DRC minimums.
                            s.edge_clearance = s.edge_clearance.max(fresh.edge_clearance);
                            s.hole_clearance = s.hole_clearance.max(fresh.hole_clearance);
                            s.hole_to_hole = s.hole_to_hole.max(fresh.hole_to_hole);
                            for (a, f) in s.netclasses.iter_mut().zip(&fresh.netclasses) {
                                a.clearance = a.clearance.max(f.clearance);
                                a.track_width = a.track_width.max(b.rules.min_track_width);
                                a.via_diameter = a.via_diameter.max(f.via_diameter.min(a.via_diameter.max(b.rules.min_via_diameter)));
                                a.via_drill = a.via_drill.max(b.rules.min_via_drill);
                            }
                            s
                        }
                        None => Settings::from_board(&b),
                    };
                    self.skip_text = self.settings.skip_nets.join(", ");
                    self.only_text = self.settings.only_nets.join(", ");
                    self.status = format!("{} — {} copper layers, {} nets, {} pads", short_name(&b.name), b.layers.len(), b.nets.len().saturating_sub(1), b.pads.len());
                    self.kicad_version = v;
                    self.board = Some(Arc::new(b));
                    self.ctx = Some(c);
                    self.state = State::Ready;
                    self.zoom = 1.0;
                    self.pan = Vec2::ZERO;
                }
                Msg::Loaded(Err(e)) => {
                    self.state = State::Failed;
                    self.status = format!("Could not read the board: {e}");
                }
                Msg::Gpu(s) => self.gpu = s,
                Msg::Progress(p) => {
                    if let Some(pv) = p.preview.clone() {
                        self.preview = pv;
                    }
                    self.status = p.message.clone();
                    self.progress = Some(p);
                }
                Msg::Routed(Ok((r, viol, unconn, unconn_off))) => {
                    self.preview = (r.segs.clone(), r.vias.clone());
                    let s = &r.stats;
                    let mut status = format!(
                        "Routed {}/{} nets in {:.2}s — {} vias, {:.0} mm of track",
                        s.nets_complete, s.nets_total, s.seconds, s.vias, s.wirelength_mm
                    );
                    if s.connections_failed > 0 {
                        status += &format!(" · {} connection(s) could not be routed", s.connections_failed);
                    }
                    if !r.offboard_pads.is_empty() {
                        status += &format!(" · {} skipped: pads outside the board outline", r.offboard_pads.len());
                    }
                    self.status = status;
                    self.log = r.log.clone();
                    self.drc_violations = viol;
                    self.unconnected = unconn;
                    self.unconnected_offboard = unconn_off;
                    self.result = Some(r);
                    self.state = State::Routed;
                }
                Msg::Routed(Err(e)) => {
                    self.state = State::Ready;
                    self.status = format!("Routing failed: {e}");
                }
                Msg::Applied(Ok(())) => {
                    self.state = State::Applied;
                    self.status = "Applied to KiCad as one undo step (Ctrl+Z in KiCad to revert).".into();
                }
                Msg::Applied(Err(e)) => {
                    self.state = State::Routed;
                    self.status = format!("Could not apply to KiCad: {e}");
                }
            }
        }
    }

    fn save_settings(&self) {
        if let Some(p) = &self.settings_path {
            if let Some(d) = p.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::write(p, serde_json::to_string_pretty(&self.settings).unwrap_or_default());
        }
    }

    fn start_route(&mut self, ectx: &egui::Context) {
        let Some(board) = self.board.clone() else { return };
        self.settings.skip_nets = split_patterns(&self.skip_text);
        self.settings.only_nets = split_patterns(&self.only_text);
        self.save_settings();
        let settings = self.settings.clone();
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        let ectx = ectx.clone();
        self.state = State::Routing;
        self.result = None;
        self.preview = (vec![], vec![]);
        self.progress = None;
        self.started = Some(std::time::Instant::now());
        self.status = "Preparing…".into();
        std::thread::spawn(move || {
            let r = (|| -> anyhow::Result<(RouteResult, usize, usize, usize)> {
                let mut backend = crate::search::make_backend(settings.backend);
                let txp = tx.clone();
                let ep = ectx.clone();
                let res = crate::router::route_board(&board, &settings, backend.as_mut(), &cancel, &mut |p| {
                    let _ = txp.send(Msg::Progress(p));
                    ep.request_repaint();
                })?;
                let viol = crate::drc::check_clearance(&board, &settings, &res.segs, &res.vias, 1e-6).len();
                let planes: Vec<(usize, u32)> = settings
                    .layers
                    .iter()
                    .enumerate()
                    .filter_map(|(l, c)| c.plane_net.as_ref().and_then(|n| board.net_by_name(n)).map(|n| (l, n)))
                    .collect();
                // Split unconnected items into on-board problems and those caused by pads that
                // sit outside the board outline.
                let off_nets: std::collections::HashSet<String> = res.offboard_nets.iter().cloned().collect();
                let mut unconn = 0usize;
                let mut unconn_off = 0usize;
                for (n, c) in crate::drc::unconnected_per_net(&board, &settings, &res.segs, &res.vias, &planes) {
                    let name = board.net_name(n);
                    if !settings.net_selected(name) {
                        continue;
                    }
                    if off_nets.contains(name) {
                        unconn_off += c;
                    } else {
                        unconn += c;
                    }
                }
                Ok((res, viol, unconn, unconn_off))
            })();
            let _ = tx.send(Msg::Routed(r.map_err(|e| format!("{e:#}"))));
            ectx.request_repaint();
        });
    }

    fn apply(&mut self, ectx: &egui::Context) {
        let (Some(board), Some(ctx), Some(res)) = (self.board.clone(), self.ctx.clone(), self.result.clone()) else { return };
        self.state = State::Applying;
        self.status = "Writing tracks and vias into KiCad…".into();
        let tx = self.tx.clone();
        let ectx = ectx.clone();
        std::thread::spawn(move || {
            let r = (|| -> anyhow::Result<()> {
                let mut k = Kicad::connect("tracewright")?;
                k.commit_routes(&board, &ctx, &res, true)
            })();
            let _ = tx.send(Msg::Applied(r.map_err(|e| format!("{e:#}"))));
            ectx.request_repaint();
        });
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        let busy = matches!(self.state, State::Routing | State::Applying | State::Loading);
        let Some(board) = self.board.clone() else {
            ui.label("Waiting for the board…");
            return;
        };
        ui.add_enabled_ui(!busy, |ui| {
            let s = &mut self.settings;
            egui::CollapsingHeader::new(RichText::new("Design rules").strong()).default_open(true).show(ui, |ui| {
                egui::Grid::new("netclasses").striped(true).num_columns(5).spacing([6.0, 4.0]).show(ui, |ui| {
                    ui.label("Net class");
                    ui.label("Clear.");
                    ui.label("Track");
                    ui.label("Via dia.");
                    ui.label("Drill");
                    ui.end_row();
                    for nc in s.netclasses.iter_mut() {
                        ui.label(&nc.name);
                        mm(ui, &mut nc.clearance, 0.05..=5.0);
                        mm(ui, &mut nc.track_width, 0.05..=10.0);
                        mm(ui, &mut nc.via_diameter, 0.1..=5.0);
                        mm(ui, &mut nc.via_drill, 0.05..=4.0);
                        ui.end_row();
                    }
                });
                egui::Grid::new("board_rules").num_columns(2).show(ui, |ui| {
                    ui.label("Copper to edge");
                    mm(ui, &mut s.edge_clearance, 0.0..=10.0);
                    ui.end_row();
                    ui.label("Hole clearance");
                    mm(ui, &mut s.hole_clearance, 0.0..=5.0);
                    ui.end_row();
                    ui.label("Hole to hole");
                    mm(ui, &mut s.hole_to_hole, 0.0..=5.0);
                    ui.end_row();
                });
            });
            egui::CollapsingHeader::new(RichText::new(format!("Layers ({})", board.layers.len())).strong()).default_open(true).show(ui, |ui| {
                let net_names: Vec<String> = {
                    let mut v: Vec<String> = board.zones.iter().map(|z| board.net_name(z.net).to_string()).collect();
                    for n in ["GND", "+3V3", "+5V", "VCC"] {
                        if board.net_by_name(n).is_some() {
                            v.push(n.to_string());
                        }
                    }
                    v.sort();
                    v.dedup();
                    v
                };
                egui::Grid::new("layers").striped(true).num_columns(4).show(ui, |ui| {
                    for (i, lc) in s.layers.iter_mut().enumerate() {
                        let c = layer_color(i, board.layers.len());
                        ui.label(RichText::new("■").color(hex(c)));
                        ui.checkbox(&mut lc.routable, &lc.name);
                        egui::ComboBox::from_id_salt(("dir", i))
                            .width(72.0)
                            .selected_text(match lc.dir {
                                LayerDir::Horizontal => "↔ Horiz",
                                LayerDir::Vertical => "↕ Vert",
                                LayerDir::Any => "Any",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut lc.dir, LayerDir::Horizontal, "↔ Horizontal");
                                ui.selectable_value(&mut lc.dir, LayerDir::Vertical, "↕ Vertical");
                                ui.selectable_value(&mut lc.dir, LayerDir::Any, "Any direction");
                            });
                        let label = lc.plane_net.clone().map(|n| format!("plane: {n}")).unwrap_or_else(|| "signal".into());
                        egui::ComboBox::from_id_salt(("plane", i)).width(96.0).selected_text(label).show_ui(ui, |ui| {
                            ui.selectable_value(&mut lc.plane_net, None, "signal");
                            for n in &net_names {
                                ui.selectable_value(&mut lc.plane_net, Some(n.clone()), format!("plane: {n}"));
                            }
                        });
                        ui.end_row();
                    }
                });
                ui.small("Plane layers carry no tracks; their net is fanned out to them with vias.");
            });
            egui::CollapsingHeader::new(RichText::new("Routing quality").strong()).default_open(true).show(ui, |ui| {
                egui::Grid::new("quality").num_columns(2).show(ui, |ui| {
                    ui.label("Grid pitch");
                    ui.horizontal(|ui| {
                        let mut auto = s.grid_pitch <= 0.0;
                        if ui.checkbox(&mut auto, "auto").changed() {
                            s.grid_pitch = if auto { 0.0 } else { s.effective_pitch(&board) };
                        }
                        if auto {
                            ui.label(RichText::new(format!("{:.4} mm", s.effective_pitch(&board))).weak());
                        } else {
                            mm(ui, &mut s.grid_pitch, 0.01..=1.0);
                        }
                    });
                    ui.end_row();
                    ui.label("Via cost").on_hover_text("How many grid steps of track one via is worth. Higher = fewer vias.");
                    ui.add(egui::Slider::new(&mut s.via_cost, 2.0..=200.0).logarithmic(true));
                    ui.end_row();
                    ui.label("Bend cost").on_hover_text("Penalty per 90° corner, in grid steps. Higher = straighter, cleaner tracks.");
                    ui.add(egui::Slider::new(&mut s.bend_cost, 0.1..=50.0).logarithmic(true));
                    ui.end_row();
                    ui.label("Wrong-way factor").on_hover_text("Cost multiplier for running against a layer's preferred direction.");
                    ui.add(egui::Slider::new(&mut s.wrong_way_factor, 1.0..=10.0));
                    ui.end_row();
                    ui.label("Comfort margin").on_hover_text("Extra spacing from pads that tracks prefer when there is room.");
                    mm(ui, &mut s.comfort_margin, 0.0..=2.0);
                    ui.end_row();
                    ui.label("Corners");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut s.corner_style, CornerStyle::Square, "90°");
                        ui.selectable_value(&mut s.corner_style, CornerStyle::Miter45, "45° chamfer");
                        if s.corner_style == CornerStyle::Miter45 {
                            mm(ui, &mut s.miter, 0.05..=5.0);
                        }
                    });
                    ui.end_row();
                });
                ui.checkbox(&mut s.snap_to_pad_center, "End tracks at pad centres");
                ui.checkbox(&mut s.allow_via_in_pad, "Allow vias in pads");
            });
            egui::CollapsingHeader::new(RichText::new("Nets").strong()).default_open(false).show(ui, |ui| {
                ui.checkbox(&mut s.keep_existing, "Keep existing tracks and complete the rest");
                ui.label("Skip nets (globs, comma separated)");
                ui.text_edit_singleline(&mut self.skip_text);
                ui.label("Only route nets (empty = all)");
                ui.text_edit_singleline(&mut self.only_text);
            });
            let s = &mut self.settings;
            egui::CollapsingHeader::new(RichText::new("Engine").strong()).default_open(false).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut s.backend, BackendChoice::Auto, "Auto");
                    ui.selectable_value(&mut s.backend, BackendChoice::Gpu, "GPU");
                    ui.selectable_value(&mut s.backend, BackendChoice::Cpu, "CPU");
                });
                egui::Grid::new("engine").num_columns(2).show(ui, |ui| {
                    ui.label("Max iterations");
                    ui.add(egui::DragValue::new(&mut s.max_iterations).range(1..=500));
                    ui.end_row();
                    ui.label("Search window margin");
                    mm(ui, &mut s.window_margin, 0.5..=50.0);
                    ui.end_row();
                    ui.label("Congestion growth");
                    ui.add(egui::DragValue::new(&mut s.pres_fac_mult).speed(0.01).range(1.05..=4.0));
                    ui.end_row();
                    ui.label("History weight");
                    ui.add(egui::DragValue::new(&mut s.hist_fac).speed(0.01).range(0.0..=5.0));
                    ui.end_row();
                });
            });
            ui.add_space(6.0);
            if ui.button("Reset to board defaults").clicked() {
                self.settings = Settings::from_board(&board);
                self.skip_text.clear();
                self.only_text.clear();
            }
        });
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        let Some(board) = self.board.clone() else {
            ui.centered_and_justified(|ui| {
                if self.state == State::Failed {
                    ui.label(RichText::new(&self.status).color(Color32::from_rgb(240, 120, 110)));
                } else {
                    ui.spinner();
                }
            });
            return;
        };
        let nl = board.layers.len();
        ui.horizontal_wrapped(|ui| {
            for l in 0..nl {
                let c = hex(layer_color(l, nl));
                let mut v = self.layer_visible[l];
                let text = RichText::new(&board.layers[l]).color(if v { c } else { Color32::GRAY });
                if ui.toggle_value(&mut v, text).changed() {
                    self.layer_visible[l] = v;
                }
            }
            ui.separator();
            ui.toggle_value(&mut self.show_existing, "existing copper");
            if ui.button("Fit").clicked() {
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
            }
        });
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 0.0, Color32::from_rgb(16, 20, 24));
        // Zoom around the cursor; drag to pan.
        if let Some(hp) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let f = (scroll * 0.0025).exp();
                let c = rect.center() + self.pan;
                self.pan += (hp - c) * (1.0 - f);
                self.zoom = (self.zoom * f).clamp(0.2, 200.0);
            }
        }
        if resp.dragged() {
            self.pan += resp.drag_delta();
        }
        if resp.double_clicked() {
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }
        let bb: BBox = board.bbox().expand(1.0);
        let base = (rect.width() / bb.width() as f32).min(rect.height() / bb.height() as f32);
        let scale = base * self.zoom;
        let center = rect.center() + self.pan;
        let bc = bb.center();
        let to = |p: P2| Pos2::new(center.x + (p.x - bc.x) as f32 * scale, center.y + (p.y - bc.y) as f32 * scale);
        let painter = painter.with_clip_rect(rect);
        for o in &board.outline {
            let pts: Vec<Pos2> = o.outline.iter().map(|&p| to(p)).collect();
            painter.add(egui::Shape::closed_line(pts, Stroke::new(1.5, Color32::from_rgb(208, 208, 96))));
            for h in &o.holes {
                painter.add(egui::Shape::closed_line(h.iter().map(|&p| to(p)).collect(), Stroke::new(1.0, Color32::from_rgb(208, 208, 96))));
            }
        }
        let alpha = |c: Color32, a: u8| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a);
        for l in (0..nl).rev() {
            if !self.layer_visible[l] {
                continue;
            }
            let c = hex(layer_color(l, nl));
            for p in &board.pads {
                if let Some(poly) = p.shape_on(l) {
                    let pts: Vec<Pos2> = poly.outline.iter().map(|&q| to(q)).collect();
                    painter.add(egui::Shape::convex_polygon(pts, alpha(c, 120), Stroke::NONE));
                }
            }
            if self.show_existing {
                for t in board.tracks.iter().filter(|t| t.layer == l) {
                    painter.line_segment([to(t.a), to(t.b)], Stroke::new((t.width as f32 * scale).max(1.0), alpha(c, 110)));
                }
            }
            for s in self.preview.0.iter().filter(|s| s.layer == l) {
                let w = (s.width as f32 * scale).max(1.0);
                painter.line_segment([to(s.a), to(s.b)], Stroke::new(w, alpha(c, 235)));
                if w > 2.5 {
                    painter.circle_filled(to(s.a), w / 2.0, alpha(c, 235));
                    painter.circle_filled(to(s.b), w / 2.0, alpha(c, 235));
                }
            }
        }
        let bg = Color32::from_rgb(16, 20, 24);
        for p in &board.pads {
            if let Some(d) = p.drill {
                painter.circle_filled(to(p.pos), (d as f32 * scale / 2.0).max(0.5), bg);
            }
        }
        let vias = board.vias.iter().filter(|_| self.show_existing).map(|v| (v.pos, v.diameter, v.drill)).chain(self.preview.1.iter().map(|v| (v.pos, v.diameter, v.drill)));
        for (p, d, dr) in vias {
            painter.circle_filled(to(p), (d as f32 * scale / 2.0).max(1.5), Color32::from_rgb(200, 200, 200));
            painter.circle_filled(to(p), (dr as f32 * scale / 2.0).max(0.5), bg);
        }
        if let Some(hp) = resp.hover_pos() {
            let x = bc.x + ((hp.x - center.x) / scale) as f64;
            let y = bc.y + ((hp.y - center.y) / scale) as f64;
            painter.text(rect.left_bottom() + Vec2::new(8.0, -8.0), egui::Align2::LEFT_BOTTOM, format!("{x:.3}, {y:.3} mm"), egui::FontId::monospace(11.0), Color32::GRAY);
        }
    }

    fn bottom_ui(&mut self, ui: &mut egui::Ui) {
        let ectx = ui.ctx().clone();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let can_route = matches!(self.state, State::Ready | State::Routed | State::Applied);
            if ui.add_enabled(can_route, egui::Button::new(RichText::new("▶  Route").strong()).min_size(Vec2::new(96.0, 28.0))).clicked() {
                self.start_route(&ectx);
            }
            if self.state == State::Routing && ui.add(egui::Button::new("■  Stop").min_size(Vec2::new(80.0, 28.0))).clicked() {
                self.cancel.store(true, Ordering::Relaxed);
                self.status = "Stopping after the current pass (keeping only conflict-free routes)…".into();
            }
            let can_apply = self.state == State::Routed && self.result.as_ref().is_some_and(|r| !r.segs.is_empty() || !r.vias.is_empty());
            if ui
                .add_enabled(can_apply, egui::Button::new(RichText::new("✔  Apply to KiCad").strong()).min_size(Vec2::new(140.0, 28.0)))
                .on_hover_text("Adds the routed tracks and vias to the board as a single undo step")
                .clicked()
            {
                self.apply(&ectx);
            }
            if ui.add(egui::Button::new("Close").min_size(Vec2::new(70.0, 28.0))).clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
            ui.separator();
            if self.state == State::Routing {
                ui.spinner();
                if let Some(p) = &self.progress {
                    let f = if p.max_iterations > 0 { p.iteration as f32 / p.max_iterations as f32 } else { 0.0 };
                    ui.add(egui::ProgressBar::new(f).desired_width(160.0).text(format!("pass {}  ·  {} in conflict", p.iteration, p.conflicts)));
                }
                if let Some(t) = self.started {
                    ui.label(RichText::new(format!("{:.1}s", t.elapsed().as_secs_f32())).monospace());
                }
            }
            ui.label(&self.status);
        });
        if let Some(r) = &self.result {
            ui.horizontal(|ui| {
                let ok = Color32::from_rgb(120, 200, 130);
                let bad = Color32::from_rgb(240, 150, 100);
                ui.label(RichText::new(format!("DRC: {} clearance violations", self.drc_violations)).color(if self.drc_violations == 0 { ok } else { bad }));
                ui.separator();
                if self.unconnected == 0 {
                    ui.label(RichText::new("All on-board connections made").color(ok));
                } else {
                    ui.label(RichText::new(format!("{} unconnected", self.unconnected)).color(bad));
                }
                if !r.offboard_pads.is_empty() {
                    ui.separator();
                    let warn = Color32::from_rgb(220, 190, 110);
                    let parts = part_list(&r.offboard_pads);
                    ui.label(RichText::new(format!("{} pads outside the board outline skipped: {}", r.offboard_pads.len(), parts)).color(warn)).on_hover_text(format!(
                        "These pads are outside the board outline, so no copper may reach them (copper-to-edge rule).\nMove the parts onto the board and route again.\n\nPads: {}\n\nNets left incomplete because of them ({} unconnected items): {}",
                        r.offboard_pads.join(", "),
                        self.unconnected_offboard,
                        r.offboard_nets.join(", ")
                    ));
                }
                ui.separator();
                ui.label(format!("{} segments · {} vias · grid {:.3} mm · {} passes · {} searches", r.segs.len(), r.stats.vias, r.stats.pitch, r.stats.iterations, r.stats.searches));
                if !r.incomplete_nets.is_empty() {
                    ui.separator();
                    ui.label(RichText::new(format!("could not route: {}", r.incomplete_nets.iter().take(6).cloned().collect::<Vec<_>>().join(", "))).color(bad))
                        .on_hover_text(r.incomplete_nets.join("\n") + "\n\n" + &self.log.join("\n"));
                }
            });
        }
        ui.add_space(2.0);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        // Fit the window to the monitor (high-DPI laptops can be shorter than the default).
        if !self.fitted {
            if let Some(ms) = ui.ctx().input(|i| i.viewport().monitor_size) {
                let size = Vec2::new((ms.x * 0.92).min(1360.0), (ms.y * 0.86).min(860.0));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::InnerSize(size));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::OuterPosition(Pos2::new(((ms.x - size.x) / 2.0).max(0.0), ((ms.y - size.y) / 2.0 - 16.0).max(0.0))));
                self.fitted = true;
            }
        }
        if matches!(self.state, State::Routing | State::Loading | State::Applying) {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        egui::Panel::top("header").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("Tracewright").size(18.0).strong());
                ui.label(RichText::new("GPU orthogonal autorouter").weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("⚡ {}", self.gpu)).small());
                    if !self.kicad_version.is_empty() {
                        ui.separator();
                        ui.label(RichText::new(format!("KiCad {}", self.kicad_version)).small());
                    }
                });
            });
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| self.bottom_ui(ui));
        egui::Panel::left("settings").resizable(true).default_size(390.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.settings_ui(ui));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| self.view_ui(ui));
    }
}

/// Add the system UI and symbol fonts as fallbacks so arrows, check marks and other symbols
/// never render as boxes.
fn install_fonts(ctx: &egui::Context) {
    use std::path::PathBuf;
    let mut fonts = egui::FontDefinitions::default();
    let files: Vec<(&str, PathBuf)> = if cfg!(windows) {
        let dir = std::env::var_os("WINDIR").map(|w| PathBuf::from(w).join("Fonts"));
        let Some(dir) = dir else { return };
        [("segoe_ui", "segoeui.ttf"), ("segoe_symbol", "seguisym.ttf"), ("segoe_emoji", "seguiemj.ttf")].into_iter().map(|(n, f)| (n, dir.join(f))).collect()
    } else if cfg!(target_os = "macos") {
        // Apple Color Emoji is a bitmap (sbix) font egui cannot draw, so symbols come from these.
        vec![
            ("apple_symbols", "/System/Library/Fonts/Apple Symbols.ttf".into()),
            ("arial_unicode", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf".into()),
        ]
    } else {
        vec![("dejavu_sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf".into()), ("noto_symbols", "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf".into())]
    };
    for (name, path) in files {
        if let Ok(bytes) = std::fs::read(path) {
            fonts.font_data.insert(name.to_owned(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
            for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.entry(fam).or_default().push(name.to_owned());
            }
        }
    }
    ctx.set_fonts(fonts);
}

fn mm(ui: &mut egui::Ui, v: &mut f64, range: std::ops::RangeInclusive<f64>) {
    ui.add(egui::DragValue::new(v).speed(0.005).range(range).max_decimals(4).suffix(" mm"));
}

/// "IC3, C15, C16, … (+n more)" from pad names like "IC3.4", in natural order.
fn part_list(pads: &[String]) -> String {
    let mut refs: Vec<String> = pads.iter().map(|p| p.split('.').next().unwrap_or(p).to_string()).collect();
    let key = |r: &String| {
        let split = r.find(|c: char| c.is_ascii_digit()).unwrap_or(r.len());
        (r[..split].to_string(), r[split..].parse::<u64>().unwrap_or(0), r.clone())
    };
    refs.sort_by_key(key);
    refs.dedup();
    if refs.len() > 8 {
        format!("{} … (+{} more)", refs[..8].join(", "), refs.len() - 8)
    } else {
        refs.join(", ")
    }
}

fn hex(s: &str) -> Color32 {
    let v = u32::from_str_radix(s.trim_start_matches('#'), 16).unwrap_or(0xffffff);
    Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

fn split_patterns(s: &str) -> Vec<String> {
    s.split([',', ';']).map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
}

fn short_name(path: &str) -> String {
    std::path::Path::new(path).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_else(|| path.to_string())
}

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 720.0]).with_min_inner_size([820.0, 520.0]).with_title("Tracewright — GPU autorouter"),
        ..Default::default()
    };
    eframe::run_native("Tracewright", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
