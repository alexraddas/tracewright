//! Router settings: DRC rules plus routing-quality knobs. Serialisable so the plugin can
//! remember them between runs.

use crate::board::{Board, NetClass, PadKind};
use crate::geom::P2;
use serde::{Deserialize, Serialize};

pub const MAX_LAYERS: usize = 8;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerDir {
    Horizontal,
    Vertical,
    Any,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LayerCfg {
    pub name: String,
    pub routable: bool,
    pub dir: LayerDir,
    /// When set, this layer is a plane for the named net: no signal routing on it, and the
    /// net's pads are fanned out to it with vias instead of being routed with tracks.
    #[serde(default)]
    pub plane_net: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CornerStyle {
    Square,
    Miter45,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendChoice {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Routing grid pitch in mm; 0 selects automatically from the design rules.
    pub grid_pitch: f64,
    /// Effective DRC values per net class (pre-filled from the board).
    pub netclasses: Vec<NetClass>,
    pub edge_clearance: f64,
    pub hole_clearance: f64,
    pub hole_to_hole: f64,
    pub layers: Vec<LayerCfg>,
    /// Cost of one via, in grid steps of track.
    pub via_cost: f64,
    /// Cost of one 90° bend, in grid steps.
    pub bend_cost: f64,
    /// Cost multiplier for travelling against a layer's preferred direction.
    pub wrong_way_factor: f64,
    /// Extra distance (mm) beyond the clearance in which tracks pay `comfort_cost`; keeps
    /// tracks centred in channels and away from pads when there is room.
    pub comfort_margin: f64,
    pub comfort_cost: f64,
    /// Negotiated-congestion (PathFinder) schedule.
    pub max_iterations: u32,
    pub pres_fac_init: f64,
    pub pres_fac_mult: f64,
    pub hist_fac: f64,
    /// Glob patterns of nets not to route.
    pub skip_nets: Vec<String>,
    /// If non-empty, only nets matching these patterns are routed.
    pub only_nets: Vec<String>,
    /// Keep existing tracks/vias (as fixed copper) and complete the remaining connections.
    pub keep_existing: bool,
    pub corner_style: CornerStyle,
    /// Leg length (mm) of 45° corner chamfers when `corner_style` is `Miter45`.
    pub miter: f64,
    /// Extend track ends to the pad centre when that is DRC-clean.
    pub snap_to_pad_center: bool,
    pub allow_via_in_pad: bool,
    pub backend: BackendChoice,
    /// Search window margin around each net's bounding box (mm).
    pub window_margin: f64,
    /// Cap on grid nodes per layer (memory/speed guard for auto pitch).
    pub max_nodes_per_layer: usize,
    /// Track width used only next to pads a net class's normal width cannot reach legally
    /// (fine-pitch ICs). 0 = the board's minimum track width.
    pub neck_width: f64,
    /// Board minimum track width (copied from the board rules; used for automatic neck-down).
    pub min_track_width: f64,
    /// Route nets with overlapping search windows in the same GPU batch during negotiation
    /// (much higher GPU throughput; the final legalisation pass is always exact).
    pub parallel_nets: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            grid_pitch: 0.0,
            netclasses: vec![NetClass::default()],
            edge_clearance: 0.5,
            hole_clearance: 0.25,
            hole_to_hole: 0.25,
            layers: vec![
                LayerCfg { name: "F.Cu".into(), routable: true, dir: LayerDir::Horizontal, plane_net: None },
                LayerCfg { name: "B.Cu".into(), routable: true, dir: LayerDir::Vertical, plane_net: None },
            ],
            via_cost: 25.0,
            bend_cost: 3.0,
            wrong_way_factor: 2.5,
            comfort_margin: 0.1,
            comfort_cost: 0.5,
            max_iterations: 120,
            pres_fac_init: 0.6,
            pres_fac_mult: 1.3,
            hist_fac: 2.0,
            skip_nets: vec![],
            only_nets: vec![],
            keep_existing: true,
            corner_style: CornerStyle::Square,
            miter: 0.3,
            snap_to_pad_center: true,
            allow_via_in_pad: false,
            backend: BackendChoice::Auto,
            window_margin: 3.0,
            max_nodes_per_layer: 4_000_000,
            neck_width: 0.0,
            min_track_width: 0.0,
            parallel_nets: true,
        }
    }
}

impl Settings {
    /// Defaults adapted to a board: its net classes, layer stack, rules and planes.
    pub fn from_board(board: &Board) -> Settings {
        let mut s = Settings { netclasses: board.netclasses.clone(), ..Settings::default() };
        if s.netclasses.is_empty() {
            s.netclasses.push(NetClass::default());
        }
        let r = &board.rules;
        s.min_track_width = r.min_track_width.max(0.0);
        s.edge_clearance = r.edge_clearance.max(0.0);
        s.hole_clearance = r.hole_clearance.max(0.0);
        s.hole_to_hole = r.hole_to_hole.max(0.0);
        for nc in &mut s.netclasses {
            nc.clearance = nc.clearance.max(r.min_clearance);
            nc.track_width = nc.track_width.max(r.min_track_width);
            nc.via_diameter = nc.via_diameter.max(r.min_via_diameter);
            nc.via_drill = nc.via_drill.max(r.min_via_drill);
            nc.via_diameter = nc.via_diameter.max(nc.via_drill + 2.0 * r.min_via_annular);
        }
        let n = board.layers.len();
        s.layers = board
            .layers
            .iter()
            .enumerate()
            .map(|(i, name)| LayerCfg {
                name: name.clone(),
                routable: true,
                dir: if i % 2 == 0 { LayerDir::Horizontal } else { LayerDir::Vertical },
                plane_net: None,
            })
            .collect();
        // A copper zone covering most of the board on an inner layer is treated as a plane.
        let board_area: f64 = board.outline.iter().map(|o| o.area()).sum::<f64>().max(board.bbox().width() * board.bbox().height());
        for z in &board.zones {
            for &l in &z.layers {
                if l == 0 || l + 1 >= n || z.net == 0 {
                    continue;
                }
                if z.poly.area() >= 0.5 * board_area && s.layers[l].plane_net.is_none() {
                    s.layers[l].plane_net = Some(board.net_name(z.net).to_string());
                    s.layers[l].routable = false;
                }
            }
        }
        s
    }

    /// Ensure settings match the board (layer count, net class list) after loading saved values.
    pub fn reconcile(&mut self, board: &Board) {
        let fresh = Settings::from_board(board);
        if self.layers.len() != board.layers.len() || self.layers.iter().zip(&board.layers).any(|(a, b)| &a.name != b) {
            self.layers = fresh.layers;
        }
        let mut ncs = Vec::new();
        for nc in &board.netclasses {
            ncs.push(self.netclasses.iter().find(|x| x.name == nc.name).cloned().unwrap_or_else(|| {
                fresh.netclasses.iter().find(|x| x.name == nc.name).cloned().unwrap_or_else(|| nc.clone())
            }));
        }
        if ncs.is_empty() {
            ncs.push(NetClass::default());
        }
        self.netclasses = ncs;
    }

    pub fn netclass(&self, idx: usize) -> &NetClass {
        &self.netclasses[idx.min(self.netclasses.len() - 1)]
    }

    /// Pads (index, layer) whose nearest pad of another net on the same layer is closer than
    /// two class-width tracks need side by side; these get neck-down.
    pub fn tight_pads(&self, board: &Board) -> std::collections::HashSet<(usize, usize)> {
        let mut hash: std::collections::HashMap<(i64, i64, usize), Vec<usize>> = std::collections::HashMap::new();
        let key = |p: P2, l: usize| ((p.x).floor() as i64, (p.y).floor() as i64, l);
        for (i, p) in board.pads.iter().enumerate() {
            for s in &p.shapes {
                hash.entry(key(p.pos, s.layer)).or_default().push(i);
            }
        }
        let cls = |net: u32| self.netclass(board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0));
        let mut out = std::collections::HashSet::new();
        for (i, p) in board.pads.iter().enumerate() {
            if p.net == 0 || p.kind == PadKind::Npth {
                continue;
            }
            let a = cls(p.net);
            for s in &p.shapes {
                let (kx, ky, _) = key(p.pos, s.layer);
                'search: for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(v) = hash.get(&(kx + dx, ky + dy, s.layer)) else { continue };
                        for &j in v {
                            let q = &board.pads[j];
                            if j == i || q.net == p.net {
                                continue;
                            }
                            let b = cls(q.net);
                            let need = (a.track_width + b.track_width) / 2.0 + a.clearance.max(b.clearance);
                            if p.pos.dist(q.pos) < need - 1e-9 {
                                out.insert((i, s.layer));
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Neck-down width per net class: the user's value, or automatically the board's minimum
    /// track width (at least 0.1 mm), never wider than the class width. Necks are only used in
    /// small regions around pads that need them, so the narrow end is the safe choice.
    pub fn neck_widths(&self, _board: &Board) -> Vec<f64> {
        self.netclasses
            .iter()
            .map(|nc| {
                let w = if self.neck_width > 0.0 { self.neck_width } else { self.min_track_width.max(0.1) };
                ((w * 1e6).round() / 1e6).clamp(0.05, nc.track_width)
            })
            .collect()
    }

    /// Grid pitch to use for this board.
    pub fn effective_pitch(&self, board: &Board) -> f64 {
        if self.grid_pitch > 0.0 {
            return self.grid_pitch;
        }
        // Track pitch of the tightest class: tracks can then pack at exactly the minimum spacing.
        let used: Vec<&NetClass> = {
            let mut used_idx: Vec<usize> = board.nets.iter().skip(1).map(|n| n.netclass).collect();
            used_idx.sort_unstable();
            used_idx.dedup();
            let v: Vec<&NetClass> = used_idx.iter().filter_map(|&i| self.netclasses.get(i)).collect();
            if v.is_empty() { self.netclasses.iter().collect() } else { v }
        };
        let tp = used.iter().map(|c| c.track_width + c.clearance).fold(f64::INFINITY, f64::min);
        // Small pads need grid nodes inside them.
        let mut min_pad = f64::INFINITY;
        for p in &board.pads {
            if p.kind == PadKind::Npth || p.net == 0 {
                continue;
            }
            for s in &p.shapes {
                let b = s.poly.bbox();
                min_pad = min_pad.min(b.width().min(b.height()));
            }
        }
        let target = 0.15f64.min(if min_pad.is_finite() { (min_pad * 0.7).max(0.05) } else { 0.15 });
        let bb = board.bbox();
        let area = bb.width() * bb.height();
        let fits = |p: f64| area / (p * p) <= self.max_nodes_per_layer as f64;
        // Candidates: fractions of the track pitch (rounded up to whole nm so k steps never fall
        // short of the clearance) and common placement-grid divisors.
        let tp_nm = (tp * 1e6).round();
        let mut cands: Vec<f64> = (1..=8).map(|k| (tp_nm / k as f64).ceil() / 1e6).collect();
        cands.extend([0.025, 0.05, 0.0635, 0.1, 0.125, 0.127, 0.15, 0.2, 0.25]);
        let mut cands: Vec<f64> = cands.into_iter().filter(|&p| p <= target + 1e-9 && fits(p)).collect();
        if cands.is_empty() {
            let mut p = target;
            while !fits(p) {
                p *= 1.25;
            }
            return (p * 1e6).round() / 1e6;
        }
        cands.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cands.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let pads: Vec<P2> = board.pads.iter().filter(|p| p.kind != PadKind::Npth && p.net != 0).map(|p| p.pos).collect();
        let mut best = (f64::NEG_INFINITY, cands[0]);
        for &p in &cands {
            let aligned = if pads.is_empty() { 0.0 } else { grid_alignment(&pads, p).1 };
            let steps = (tp / p - 1e-9).ceil();
            let pack = tp / (steps * p);
            let score = 0.55 * aligned + 0.45 * pack + 0.04 * (p / target);
            if score > best.0 {
                best = (score, p);
            }
        }
        best.1
    }

    /// Grid origin offset (in [0, pitch)) that puts the most pad centres on grid nodes.
    pub fn grid_offset(&self, board: &Board, pitch: f64) -> P2 {
        let pads: Vec<P2> = board.pads.iter().filter(|p| p.kind != PadKind::Npth && p.net != 0).map(|p| p.pos).collect();
        if pads.is_empty() { P2::default() } else { grid_alignment(&pads, pitch).0 }
    }

    pub fn net_selected(&self, name: &str) -> bool {
        if self.skip_nets.iter().any(|p| glob_match(p, name)) {
            return false;
        }
        self.only_nets.is_empty() || self.only_nets.iter().any(|p| glob_match(p, name))
    }
}

/// Best grid offset for a pitch and the fraction of points that land exactly on nodes
/// (coordinates compared in whole nanometres, as KiCad stores them).
pub fn grid_alignment(pts: &[P2], pitch: f64) -> (P2, f64) {
    let p_nm = ((pitch * 1e6).round() as i64).max(1);
    let mut hist: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
    for q in pts {
        let rx = ((q.x * 1e6).round() as i64).rem_euclid(p_nm);
        let ry = ((q.y * 1e6).round() as i64).rem_euclid(p_nm);
        *hist.entry((rx, ry)).or_default() += 1;
    }
    let (&(rx, ry), &n) = hist.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).unwrap();
    (P2::new(rx as f64 / 1e6, ry as f64 / 1e6), n as f64 / pts.len() as f64)
}

/// Case-insensitive glob match supporting `*` and `?`.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.trim().to_lowercase().chars().collect();
    let t: Vec<char> = text.to_lowercase().chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = pi;
            mark = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn globbing() {
        assert!(glob_match("GND", "gnd"));
        assert!(glob_match("/USB_*", "/USB_D+"));
        assert!(glob_match("*VCC*", "+3V3_VCC_IO"));
        assert!(!glob_match("VCC", "VCC2"));
        assert!(glob_match("Net-(U?-*)", "Net-(U1-Pad3)"));
    }
}
