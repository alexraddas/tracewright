//! Negotiated-congestion (PathFinder) router on top of the batched maze search.
//!
//! * Nets are split into terminal groups (pads already joined by existing copper form one
//!   group). Each net is a Steiner-tree job: grow from one group, repeatedly connect the
//!   nearest remaining group, warm-starting the search with the new path as extra sources.
//! * Plane nets (layers reserved as planes) are fanned out instead: every pad group gets the
//!   cheapest via-and-stub to the plane layer.
//! * Nets whose windows are far enough apart never interact, so they run together in one GPU
//!   batch; ordering among overlapping nets is preserved.
//! * Overlaps between nets are allowed but priced (present + history congestion); conflicted
//!   nets are ripped up and rerouted until none remain. Leftovers are legalised with hard
//!   blocking so the output is always DRC-clean.

use crate::board::{Board, NetId};
use crate::drc::{Index, Item, ItemKind, Seg, ViaOut, collect_items, copper_components, plane_polys};
use crate::geom::{P2, Shape};
use crate::grid::{FREE, SAFETY, StaticMaps, pad_terminal_nodes};
use crate::search::{Backend, INF, MAX_CELL_COST, SearchJob};
use crate::settings::{CornerStyle, LayerDir, Settings};
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Cost of one grid step of preferred-direction track.
pub const UNIT: f64 = 16.0;
/// Upper bound on distance values held by one GPU batch.
const BATCH_DIST_BUDGET: usize = 48 << 20;
const MAX_BATCH_JOBS: usize = 512;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub backend: String,
    pub pitch: f64,
    pub grid: (usize, usize, usize),
    pub nets_total: usize,
    pub nets_complete: usize,
    pub connections_total: usize,
    /// Connections that could not be routed (pads on the board).
    pub connections_failed: usize,
    /// Connections skipped because their pads lie outside the board outline.
    #[serde(default)]
    pub connections_offboard: usize,
    pub vias: usize,
    pub wirelength_mm: f64,
    pub iterations: u32,
    pub searches: u64,
    pub batches: u64,
    pub seconds: f64,
    /// Time breakdown (seconds): cost-map build, GPU/CPU relax, readback, backtrace+bookkeeping.
    #[serde(default)]
    pub t_build: f64,
    #[serde(default)]
    pub t_relax: f64,
    #[serde(default)]
    pub t_read: f64,
    #[serde(default)]
    pub t_trace: f64,
    /// Total grid nodes searched (sum of job window sizes x relax rounds).
    #[serde(default)]
    pub nodes_searched: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RouteResult {
    pub segs: Vec<Seg>,
    pub vias: Vec<ViaOut>,
    pub stats: Stats,
    /// Nets with at least one connection left unrouted.
    pub incomplete_nets: Vec<String>,
    /// Pads (e.g. "IC3.4") skipped because they lie outside the board outline.
    #[serde(default)]
    pub offboard_pads: Vec<String>,
    /// Nets left incomplete only because of off-board pads.
    #[serde(default)]
    pub offboard_nets: Vec<String>,
    /// Nets whose existing copper was ripped up and re-routed (`force_reroute`).
    #[serde(default)]
    pub rerouted_nets: Vec<String>,
    /// KiCad ids of the existing tracks/vias of `rerouted_nets`, to delete when applying.
    #[serde(default)]
    pub removed_ids: Vec<String>,
    pub log: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub phase: String,
    pub iteration: u32,
    pub max_iterations: u32,
    pub nets_total: usize,
    pub conflicts: usize,
    pub message: String,
    /// Current geometry for previews (centre lines, no stubs).
    pub preview: Option<(Vec<Seg>, Vec<ViaOut>)>,
}

#[derive(Clone, Copy, Debug)]
struct Win {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

impl Win {
    fn overlaps(&self, o: &Win, gap: usize) -> bool {
        self.x0 < o.x0 + o.w + gap && o.x0 < self.x0 + self.w + gap && self.y0 < o.y0 + o.h + gap && o.y0 < self.y0 + self.h + gap
    }
}

#[derive(Clone, Debug)]
struct Group {
    /// Terminal nodes (global index).
    nodes: Vec<u32>,
    /// Pad index per node (usize::MAX for existing copper), parallel to `nodes`.
    node_pad: Vec<usize>,
    /// Bounding node box (x0, y0, x1, y1).
    bb: (usize, usize, usize, usize),
}

#[derive(Clone, Debug, Default)]
struct NetRoute {
    nodes: Vec<u32>,
    edges: Vec<(u32, u32)>,
    /// Through vias, as plane indices.
    vias: Vec<u32>,
    /// Ordered node sequence of each connection (for pad stubs).
    paths: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
struct JobSpec {
    win: Win,
    /// Root: group index, or the plane layer (fanout).
    root: Root,
    /// Groups this job must reach.
    targets: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
enum Root {
    Group(usize),
    Plane(usize),
}

struct Task {
    net: NetId,
    class: usize,
    tp: usize,
    vp: usize,
    /// Nodes (sorted) where the net may neck down, and the rung (index into the class's neck
    /// ladder) for each: the widest that reaches the pad the node belongs to.
    neck: Vec<u32>,
    neck_rung: Vec<u8>,
    /// Via sites this net may not use (sorted plane indices), learned from hole-spacing
    /// conflicts between its own vias.
    forbid_vias: Vec<u32>,
    via_retries: u32,
    groups: Vec<Group>,
    plane: Option<usize>,
    margin: usize,
    hpwl: f64,
    route: NetRoute,
    /// Connections impossible before routing (pads without any legal node).
    static_failed: usize,
    /// Pad groups skipped because they lie outside the board outline.
    offboard: usize,
    failed: usize,
    committed: bool,
}

struct Occ {
    count: Vec<Vec<u16>>,
    hist: Vec<f32>,
    marks: Vec<u32>,
    epoch: u32,
}

pub struct Router<'a> {
    board: &'a Board,
    settings: &'a Settings,
    pub maps: StaticMaps,
    tasks: Vec<Task>,
    occ: Occ,
    /// Disk offsets: [class][probe] for track nodes and via nodes.
    trace_disk: Vec<Vec<Arc<Vec<(i32, i32)>>>>,
    via_disk: Vec<Vec<Arc<Vec<(i32, i32)>>>>,
    /// Per class, per neck rung, per probe.
    neck_disk: Vec<Vec<Vec<Arc<Vec<(i32, i32)>>>>>,
    max_reach: usize,
    /// Pads outside the board outline (skipped).
    offboard_pads: Vec<usize>,
    /// On-board pads without any legal grid node.
    unreachable_pads: Vec<usize>,
    /// Default search-window margin in cells.
    base_margin: usize,
    pub stats: Stats,
    pub log: Vec<String>,
}

fn disk(r_mm: f64, pitch: f64) -> Vec<(i32, i32)> {
    let rc = r_mm / pitch;
    let m = rc.ceil() as i32 + 1;
    let mut v = Vec::new();
    for dy in -m..=m {
        for dx in -m..=m {
            if ((dx * dx + dy * dy) as f64) < rc * rc - 1e-9 {
                v.push((dx, dy));
            }
        }
    }
    v
}

impl<'a> Router<'a> {
    pub fn new(board: &'a Board, settings: &'a Settings) -> Router<'a> {
        let t0 = Instant::now();
        let maps = StaticMaps::build(board, settings);
        let g = &maps.grid;
        let n = g.nl * g.plane();
        let pitch = g.pitch;
        let nprobe = maps.probes.len();
        let mut cache: HashMap<i64, Arc<Vec<(i32, i32)>>> = HashMap::new();
        let mut get = |r: f64| cache.entry((r * 1e6) as i64).or_insert_with(|| Arc::new(disk(r, pitch))).clone();
        let mut max_reach = 0usize;
        let mut trace_disk = Vec::new();
        let mut via_disk = Vec::new();
        let mut neck_disk = Vec::new();
        for (ci, nc) in settings.netclasses.iter().enumerate() {
            let mut td = Vec::new();
            let mut vd = Vec::new();
            let mut nd: Vec<Vec<Arc<Vec<(i32, i32)>>>> = vec![Vec::new(); maps.neck_ladder[ci].len().max(1)];
            for p in &maps.probes {
                // Routed copper sits exactly on (whole-nm) grid nodes, so no safety margin:
                // nodes at exactly the clearance distance are legal, as in KiCad's DRC.
                let rt = nc.track_width / 2.0 + nc.clearance.max(p.clearance) + p.half;
                // A via also needs hole clearance from its drill to foreign copper.
                let rv = (nc.via_diameter / 2.0 + nc.clearance.max(p.clearance)).max(nc.via_drill / 2.0 + settings.hole_clearance) + p.half;
                max_reach = max_reach.max((rt.max(rv) / pitch).ceil() as usize + 1);
                td.push(get(rt));
                vd.push(get(rv));
                for (ri, rung) in nd.iter_mut().enumerate() {
                    let w = maps.neck_ladder[ci].get(ri).map(|r| r.1).unwrap_or(maps.neck_w[ci]);
                    rung.push(get(w / 2.0 + nc.clearance.max(p.clearance) + p.half));
                }
            }
            trace_disk.push(td);
            via_disk.push(vd);
            neck_disk.push(nd);
        }
        let occ = Occ { count: (0..nprobe).map(|_| vec![0u16; n]).collect(), hist: vec![0.0; n], marks: vec![0; n], epoch: 0 };
        let stats = Stats { pitch, grid: (g.nx, g.ny, g.nl), ..Default::default() };
        let base_margin = ((settings.window_margin / pitch).ceil() as usize).max(4);
        let mut r = Router { board, settings, maps, tasks: vec![], occ, trace_disk, via_disk, neck_disk, max_reach, offboard_pads: vec![], unreachable_pads: vec![], base_margin, stats, log: vec![] };
        r.log.push(format!(
            "grid {}x{}x{} at {:.4} mm, {} probe classes, rasterised in {:.2}s",
            r.maps.grid.nx,
            r.maps.grid.ny,
            r.maps.grid.nl,
            pitch,
            r.maps.probes.len(),
            t0.elapsed().as_secs_f64()
        ));
        r.build_tasks();
        r
    }

    /// Build routing tasks: terminal groups per net from pads and existing copper.
    fn build_tasks(&mut self) {
        let board = self.board;
        let settings = self.settings;
        let g = self.maps.grid.clone();
        let nl = g.nl;
        let plane_layers: Vec<(usize, NetId)> = self.maps.plane_net.iter().enumerate().filter_map(|(l, n)| n.map(|n| (l, n))).collect();
        let items = collect_items(board, settings, &[], &[], settings.keep_existing);
        let mut dsu = copper_components(&items, &plane_layers, &plane_polys(board, &plane_layers));
        // Pads with several layer shapes are one item set.
        let mut first: HashMap<usize, usize> = HashMap::new();
        for (i, it) in items.iter().enumerate() {
            if it.kind == ItemKind::Pad {
                if let Some(&f) = first.get(&it.src) {
                    dsu.union(i, f);
                } else {
                    first.insert(it.src, i);
                }
            }
        }
        let plane_root: HashMap<NetId, usize> = plane_layers.iter().enumerate().map(|(pi, &(_, n))| (n, items.len() + pi)).collect();

        // Copper items per net, grouped by component.
        let mut by_net: HashMap<NetId, HashMap<usize, Vec<usize>>> = HashMap::new();
        for (i, it) in items.iter().enumerate() {
            if it.net == 0 || !matches!(it.kind, ItemKind::Pad | ItemKind::Track | ItemKind::Via) {
                continue;
            }
            let root = dsu.find(i);
            by_net.entry(it.net).or_default().entry(root).or_default().push(i);
        }
        let mut nets: Vec<NetId> = by_net.keys().copied().collect();
        nets.sort_unstable();
        let tight = settings.tight_pads(board);
        let margin_cells = ((settings.window_margin / g.pitch).ceil() as usize).max(4);
        for net in nets {
            let name = board.net_name(net);
            if !settings.net_selected(name) {
                continue;
            }
            let comps = &by_net[&net];
            let class = board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0).min(settings.netclasses.len() - 1);
            let cp = self.maps.class_probes[class];
            let plane = plane_layers.iter().find(|&&(_, n)| n == net).map(|&(l, _)| l);
            let mut roots: Vec<usize> = comps.keys().copied().collect();
            roots.sort_unstable();
            let mut groups = Vec::new();
            let mut unreachable = 0usize;
            let mut offboard = 0usize;
            let mut neck: Vec<(u32, u8)> = Vec::new();
            let ladder = self.maps.neck_ladder[class].clone();
            for root in roots {
                if plane.is_some() && plane_root.get(&net).is_some_and(|&pr| dsu.find(pr) == root) {
                    continue; // already on the plane
                }
                let members = &comps[&root];
                if !members.iter().any(|&i| items[i].kind == ItemKind::Pad) {
                    continue; // dangling existing copper
                }
                let mut nodes = Vec::new();
                let mut node_pad = Vec::new();
                for &i in members {
                    let it = &items[i];
                    match it.kind {
                        ItemKind::Pad => {
                            let pad = it.src;
                            let layer = it.layers.trailing_zeros() as usize;
                            let before = nodes.len();
                            for e in pad_terminal_nodes(board, &self.maps, pad, cp.trace) {
                                if e as usize / g.plane() == layer {
                                    nodes.push(e as u32);
                                    node_pad.push(pad);
                                }
                            }
                            if !ladder.is_empty() && (nodes.len() == before || tight.contains(&(pad, layer))) {
                                // The class width cannot reach this pad, or cannot enter it
                                // next to a neighbouring pad's track: neck down next to it, only
                                // as far as needed (widest rung that reaches it).
                                let mut found: Vec<u32> = Vec::new();
                                let mut rung = 0u8;
                                for (ri, &(probe, _)) in ladder.iter().enumerate() {
                                    found = pad_terminal_nodes(board, &self.maps, pad, probe).into_iter().filter(|&e| e as usize / g.plane() == layer).map(|e| e as u32).collect();
                                    if !found.is_empty() {
                                        rung = ri as u8;
                                        break;
                                    }
                                }
                                if !found.is_empty() {
                                    nodes.truncate(before);
                                    node_pad.truncate(before);
                                    neck.extend(self.neck_region(pad).into_iter().map(|e| (e, rung)));
                                    for e in found {
                                        nodes.push(e);
                                        node_pad.push(pad);
                                    }
                                }
                            }
                            if nodes.len() == before {
                                // Fallback: legal nodes just outside the pad, joined by a stub.
                                for e in self.ring_nodes(pad, layer, cp.trace) {
                                    nodes.push(e);
                                    node_pad.push(pad);
                                }
                            }
                        }
                        ItemKind::Track | ItemKind::Via => {
                            for l in 0..nl {
                                if it.layers & (1 << l) == 0 || !self.maps.routable[l] {
                                    continue;
                                }
                                let Some((x0, y0, x1, y1)) = g.range(&it.shape.bbox()) else { continue };
                                for y in y0..=y1 {
                                    for x in x0..=x1 {
                                        let e = g.idx(l, y, x);
                                        if it.shape.contains(g.pos(x, y)) && self.maps.trace_ok(cp.trace, e, net) {
                                            nodes.push(e as u32);
                                            node_pad.push(usize::MAX);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if nodes.is_empty() {
                    let mut pads: Vec<usize> = members.iter().filter(|&&i| items[i].kind == ItemKind::Pad).map(|&i| items[i].src).collect();
                    pads.dedup();
                    let names = |v: &[usize]| v.iter().map(|&p| board.pads[p].name.as_str()).collect::<Vec<_>>().join(", ");
                    let (off, on): (Vec<usize>, Vec<usize>) = pads.iter().partition(|&&p| !board.inside_outline(board.pads[p].pos));
                    if on.is_empty() {
                        offboard += 1;
                    } else {
                        unreachable += 1;
                    }
                    self.offboard_pads.extend(off.iter().copied());
                    if !off.is_empty() {
                        self.log.push(format!("net {name}: pad(s) {} lie outside the board outline", names(&off)));
                    }
                    if !on.is_empty() {
                        self.log.push(format!("net {name}: no legal grid node at pad(s) {} (blocked by clearance; try a finer grid)", names(&on)));
                        self.unreachable_pads.extend(on.iter().copied());
                    }
                    continue;
                }
                let mut bb = (usize::MAX, usize::MAX, 0, 0);
                for &e in &nodes {
                    let p = e as usize % g.plane();
                    let (y, x) = (p / g.nx, p % g.nx);
                    bb = (bb.0.min(x), bb.1.min(y), bb.2.max(x), bb.3.max(y));
                }
                groups.push(Group { nodes, node_pad, bb });
            }
            let needed = if plane.is_some() { groups.len() } else { groups.len().saturating_sub(1) };
            if needed == 0 && unreachable == 0 && offboard == 0 {
                continue;
            }
            if plane.is_none() && groups.len() + unreachable + offboard <= 1 {
                // A single pad group (e.g. a no-connect pin's own net): nothing to connect.
                self.log.retain(|m| !m.starts_with(&format!("net {name}: ")));
                self.unreachable_pads.retain(|&p| board.pads[p].net != net);
                continue;
            }
            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
            for gr in &groups {
                x0 = x0.min(gr.bb.0);
                y0 = y0.min(gr.bb.1);
                x1 = x1.max(gr.bb.2);
                y1 = y1.max(gr.bb.3);
            }
            let hpwl = if groups.is_empty() { 0.0 } else { ((x1 - x0) + (y1 - y0)) as f64 * g.pitch };
            self.stats.connections_total += needed + unreachable + offboard;
            // Where two pads' neck regions overlap, the narrower rung applies.
            neck.sort_unstable();
            let mut neck_nodes: Vec<u32> = Vec::with_capacity(neck.len());
            let mut neck_rung: Vec<u8> = Vec::with_capacity(neck.len());
            for (e, r) in neck {
                if neck_nodes.last() == Some(&e) {
                    *neck_rung.last_mut().unwrap() = r;
                } else {
                    neck_nodes.push(e);
                    neck_rung.push(r);
                }
            }
            self.tasks.push(Task {
                net,
                class,
                tp: cp.trace,
                vp: cp.via,
                neck: neck_nodes,
                neck_rung,
                forbid_vias: Vec::new(),
                via_retries: 0,
                groups,
                plane,
                margin: margin_cells,
                hpwl,
                route: NetRoute::default(),
                static_failed: unreachable,
                offboard,
                failed: unreachable,
                committed: false,
            });
        }
        // Fan-outs first (they claim via sites next to pads), then short nets first.
        self.tasks.sort_by(|a, b| b.plane.is_some().cmp(&a.plane.is_some()).then(a.hpwl.partial_cmp(&b.hpwl).unwrap()).then(a.net.cmp(&b.net)));
        self.stats.nets_total = self.tasks.len();
    }

    /// Legal nodes in a one-pitch ring around a pad that has no grid node inside it.
    fn ring_nodes(&self, pad: usize, layer: usize, probe: usize) -> Vec<u32> {
        let g = &self.maps.grid;
        let p = &self.board.pads[pad];
        let Some(poly) = p.shape_on(layer) else { return vec![] };
        let Some((x0, y0, x1, y1)) = g.range(&poly.bbox().expand(g.pitch * 1.5)) else { return vec![] };
        let mut v = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let e = g.idx(layer, y, x);
                if poly.dist(g.pos(x, y)) <= g.pitch * 1.5 && self.maps.trace_ok(probe, e, p.net) {
                    v.push(e as u32);
                }
            }
        }
        v
    }

    /// Nodes around a pad (on its layers) where neck-down width is allowed.
    fn neck_region(&self, pad: usize) -> Vec<u32> {
        let g = &self.maps.grid;
        let p = &self.board.pads[pad];
        let mut v = Vec::new();
        for s in &p.shapes {
            if !self.maps.routable[s.layer] {
                continue;
            }
            let bb = s.poly.bbox();
            let reach = bb.width().max(bb.height()).max(1.0);
            let Some((x0, y0, x1, y1)) = g.range(&bb.expand(reach)) else { continue };
            for y in y0..=y1 {
                for x in x0..=x1 {
                    v.push(g.idx(s.layer, y, x) as u32);
                }
            }
        }
        v
    }

    fn job_specs(&self, t: &Task) -> Vec<JobSpec> {
        let g = &self.maps.grid;
        let grow = |bb: (usize, usize, usize, usize), m: usize| {
            let x0 = bb.0.saturating_sub(m);
            let y0 = bb.1.saturating_sub(m);
            let x1 = (bb.2 + m).min(g.nx - 1);
            let y1 = (bb.3 + m).min(g.ny - 1);
            Win { x0, y0, w: x1 - x0 + 1, h: y1 - y0 + 1 }
        };
        if let Some(pl) = t.plane {
            // Fan-out. Pads close together (e.g. neighbouring IC pins) share one job, so they
            // can share a via and their vias are placed one after another. Any remaining
            // hole-spacing clash between jobs is caught by the own-via check and retried.
            let m = (t.margin / 2).clamp(4, self.base_margin);
            let wins: Vec<Win> = t.groups.iter().map(|gr| grow(gr.bb, m)).collect();
            let near: Vec<Win> = t.groups.iter().map(|gr| grow(gr.bb, 0)).collect();
            let gap = self.max_reach + 2;
            let mut dsu = crate::drc::Dsu::new(wins.len());
            for i in 0..wins.len() {
                for j in i + 1..wins.len() {
                    if near[i].overlaps(&near[j], gap) {
                        dsu.union(i, j);
                    }
                }
            }
            let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in 0..wins.len() {
                let r = dsu.find(i);
                clusters.entry(r).or_default().push(i);
            }
            let mut keys: Vec<usize> = clusters.keys().copied().collect();
            keys.sort_unstable();
            keys.into_iter()
                .map(|k| {
                    let members = clusters.remove(&k).unwrap();
                    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
                    for &i in &members {
                        let w = wins[i];
                        x0 = x0.min(w.x0);
                        y0 = y0.min(w.y0);
                        x1 = x1.max(w.x0 + w.w - 1);
                        y1 = y1.max(w.y0 + w.h - 1);
                    }
                    JobSpec { win: Win { x0, y0, w: x1 - x0 + 1, h: y1 - y0 + 1 }, root: Root::Plane(pl), targets: members }
                })
                .collect()
        } else {
            if t.groups.len() < 2 {
                return vec![];
            }
            let mut bb = t.groups[0].bb;
            for gr in &t.groups[1..] {
                bb = (bb.0.min(gr.bb.0), bb.1.min(gr.bb.1), bb.2.max(gr.bb.2), bb.3.max(gr.bb.3));
            }
            // Start from the group nearest the centre: shorter trees and nicer topology.
            let cx = (bb.0 + bb.2) as f64 / 2.0;
            let cy = (bb.1 + bb.3) as f64 / 2.0;
            let start = (0..t.groups.len())
                .min_by(|&a, &b| {
                    let d = |i: usize| {
                        let b = t.groups[i].bb;
                        ((b.0 + b.2) as f64 / 2.0 - cx).powi(2) + ((b.1 + b.3) as f64 / 2.0 - cy).powi(2)
                    };
                    d(a).partial_cmp(&d(b)).unwrap()
                })
                .unwrap();
            vec![JobSpec { win: grow(bb, t.margin), root: Root::Group(start), targets: (0..t.groups.len()).filter(|&i| i != start).collect() }]
        }
    }

    fn full_grid(&self, w: &Win) -> bool {
        w.x0 == 0 && w.y0 == 0 && w.w == self.maps.grid.nx && w.h == self.maps.grid.ny
    }

    /// Cost arrays for one job window.
    fn build_job(&self, t: &Task, win: &Win, pres: f64, hard: bool) -> SearchJob {
        let s = self.settings;
        let g = &self.maps.grid;
        let nl = g.nl;
        let (w, h) = (win.w, win.h);
        let plane = w * h;
        let net = t.net;
        let tp = t.tp;
        let vp = t.vp;
        let comfort_cost = s.comfort_cost.max(0.0) as f32;
        let owner = &self.maps.owner[tp];
        let comfort = &self.maps.comfort[tp];
        let count = &self.occ.count[tp];
        // Owner and occupancy maps per neck rung.
        let nmaps: Vec<(&Vec<u16>, &Vec<u16>)> = self.maps.neck_ladder[t.class].iter().map(|&(p, _)| (&self.maps.owner[p], &self.occ.count[p])).collect();
        // Neck-down nodes inside this window: their rung + 1 (0 = not a neck node).
        let mut neck_here = vec![0u8; nl * plane];
        for (&e, &r) in t.neck.iter().zip(&t.neck_rung) {
            let e = e as usize;
            let (l, p) = (e / g.plane(), e % g.plane());
            let (y, x) = (p / g.nx, p % g.nx);
            if x >= win.x0 && y >= win.y0 && x < win.x0 + w && y < win.y0 + h {
                neck_here[l * plane + (y - win.y0) * w + (x - win.x0)] = r + 1;
            }
        }
        let neck_here = &neck_here;
        let nmaps = &nmaps;
        let hist = &self.occ.hist;
        let routable = &self.maps.routable;
        let mut cell = vec![INF; nl * plane];
        cell.par_chunks_mut(w).enumerate().for_each(|(row, out)| {
            let l = row / h;
            let y = win.y0 + row % h;
            if !routable[l] {
                return;
            }
            for (lx, o) in out.iter_mut().enumerate() {
                let gi = g.idx(l, y, win.x0 + lx);
                let r = neck_here[row * w + lx];
                let is_neck = r > 0;
                let (ow, occ) = if is_neck {
                    let (no, nc) = nmaps[r as usize - 1];
                    (no[gi], nc[gi])
                } else {
                    (owner[gi], count[gi])
                };
                if !(ow == FREE || ow as u32 == net) {
                    continue;
                }
                let occ = occ as f64;
                if hard && occ > 0.0 {
                    continue;
                }
                let cw = comfort[gi];
                let mut base = 1.0 + hist[gi];
                if !is_neck && !(cw == FREE || cw as u32 == net) {
                    base += comfort_cost;
                }
                let c = UNIT * base as f64 * (1.0 + pres * occ);
                *o = (c.round() as u32).clamp(UNIT as u32, MAX_CELL_COST);
            }
        });
        let bend = ((UNIT * s.bend_cost).round() as u32).max(1);
        let via_unit = (UNIT * s.via_cost.max(s.bend_cost)).max(bend as f64);
        let vowner = &self.maps.owner[vp];
        let vcount = &self.occ.count[vp];
        let vin_pad = &self.maps.via_in_pad[vp];
        let allow_vip = s.allow_via_in_pad;
        let forbid = &t.forbid_vias;
        let mut via = vec![INF; plane];
        via.par_chunks_mut(w).enumerate().for_each(|(ly, out)| {
            let y = win.y0 + ly;
            for (lx, o) in out.iter_mut().enumerate() {
                let p = y * g.nx + win.x0 + lx;
                if !forbid.is_empty() && forbid.binary_search(&(p as u32)).is_ok() {
                    continue;
                }
                let mut ok = true;
                let mut occ = 0u16;
                let mut hmax = 0f32;
                for l in 0..nl {
                    let gi = l * g.plane() + p;
                    let ow = vowner[gi];
                    if !(ow == FREE || ow as u32 == net) || (!allow_vip && vin_pad[gi] != 0) {
                        ok = false;
                        break;
                    }
                    occ = occ.max(vcount[gi]);
                    hmax = hmax.max(hist[gi]);
                }
                if !ok || (hard && occ > 0) {
                    continue;
                }
                let c = via_unit * (1.0 + hmax as f64) * (1.0 + pres * occ as f64);
                *o = (c.round() as u32).clamp(bend, INF - 1);
            }
        });
        let dirfac = (0..nl)
            .map(|l| {
                let ww = ((16.0 * s.wrong_way_factor.max(1.0)).round() as u32).clamp(16, 1023);
                match s.layers.get(l).map(|c| c.dir).unwrap_or(LayerDir::Any) {
                    LayerDir::Horizontal => [16, ww],
                    LayerDir::Vertical => [ww, 16],
                    LayerDir::Any => [16, 16],
                }
            })
            .collect();
        SearchJob {
            w: w as u32,
            h: h as u32,
            nl: nl as u32,
            cell,
            via,
            via_spans: vec![(0, nl as u32 - 1)],
            dirfac,
            bend,
            sweep_layer: routable.clone(),
        }
    }

    #[inline]
    fn to_local(&self, win: &Win, gi: u32) -> Option<u32> {
        let g = &self.maps.grid;
        let gi = gi as usize;
        let l = gi / g.plane();
        let p = gi % g.plane();
        let (y, x) = (p / g.nx, p % g.nx);
        if x < win.x0 || y < win.y0 || x >= win.x0 + win.w || y >= win.y0 + win.h {
            return None;
        }
        Some((l * win.w * win.h + (y - win.y0) * win.w + (x - win.x0)) as u32)
    }


    /// Add (sign = +1) or remove (-1) a task's footprint from the congestion maps.
    fn stamp(&mut self, ti: usize, sign: i32) {
        let g = self.maps.grid.clone();
        let (nx, ny, plane) = (g.nx as i32, g.ny as i32, g.plane());
        let class = self.tasks[ti].class;
        let route = std::mem::take(&mut self.tasks[ti].route);
        for k in 0..self.maps.probes.len() {
            self.occ.epoch = self.occ.epoch.wrapping_add(1);
            if self.occ.epoch == 0 {
                self.occ.marks.iter_mut().for_each(|m| *m = 0);
                self.occ.epoch = 1;
            }
            let epoch = self.occ.epoch;
            let td = self.trace_disk[class][k].clone();
            let vd = self.via_disk[class][k].clone();
            let nds: Vec<Arc<Vec<(i32, i32)>>> = self.neck_disk[class].iter().map(|r| r[k].clone()).collect();
            let neck = std::mem::take(&mut self.tasks[ti].neck);
            let neck_rung = std::mem::take(&mut self.tasks[ti].neck_rung);
            let Occ { count, marks, .. } = &mut self.occ;
            let cnt = &mut count[k];
            let mut apply = |l: usize, y: i32, x: i32, d: &[(i32, i32)]| {
                for &(dx, dy) in d {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= nx || yy >= ny {
                        continue;
                    }
                    let i = l * plane + yy as usize * nx as usize + xx as usize;
                    if marks[i] != epoch {
                        marks[i] = epoch;
                        cnt[i] = if sign > 0 { cnt[i].saturating_add(1) } else { cnt[i].saturating_sub(1) };
                    }
                }
            };
            for &e in &route.nodes {
                let d: &[(i32, i32)] = match neck.binary_search(&e) {
                    Ok(i) => &nds[neck_rung[i] as usize],
                    Err(_) => &td,
                };
                let e = e as usize;
                let (l, p) = (e / plane, e % plane);
                apply(l, (p / g.nx) as i32, (p % g.nx) as i32, d);
            }
            for &p in &route.vias {
                let p = p as usize;
                for l in 0..g.nl {
                    apply(l, (p / g.nx) as i32, (p % g.nx) as i32, &vd);
                }
            }
            self.tasks[ti].neck = neck;
            self.tasks[ti].neck_rung = neck_rung;
        }
        self.tasks[ti].route = route;
        self.tasks[ti].committed = sign > 0;
    }

    fn rip_up(&mut self, ti: usize) {
        if self.tasks[ti].committed {
            self.stamp(ti, -1);
        }
        let t = &mut self.tasks[ti];
        t.route = NetRoute::default();
    }

    /// Trace probe that governs node `e` of task `t`: its neck rung's probe inside a neck region,
    /// otherwise the class trace probe.
    fn node_probe(&self, t: &Task, e: u32) -> usize {
        match t.neck.binary_search(&e) {
            Ok(i) => self.maps.neck_ladder[t.class][t.neck_rung[i] as usize].0,
            Err(_) => t.tp,
        }
    }

    /// Neck-down width at node `e` of task `t`, if it lies in a neck region.
    fn node_neck_w(&self, t: &Task, e: u32) -> Option<f64> {
        t.neck.binary_search(&e).ok().map(|i| self.maps.neck_ladder[t.class][t.neck_rung[i] as usize].1)
    }

    fn conflicted(&self, ti: usize) -> bool {
        let t = &self.tasks[ti];
        let plane = self.maps.grid.plane();
        t.route.nodes.iter().any(|&e| self.occ.count[self.node_probe(t, e)][e as usize] >= 2)
            || t.route.vias.iter().any(|&p| (0..self.maps.grid.nl).any(|l| self.occ.count[t.vp][l * plane + p as usize] >= 2))
    }

    fn add_history(&mut self, ti: usize) {
        let s = self.settings;
        let plane = self.maps.grid.plane();
        let t = &self.tasks[ti];
        let mut bump: Vec<usize> = Vec::new();
        for &e in &t.route.nodes {
            let k = self.node_probe(t, e);
            if self.occ.count[k][e as usize] >= 2 {
                bump.push(e as usize);
            }
        }
        for &p in &t.route.vias {
            for l in 0..self.maps.grid.nl {
                let gi = l * plane + p as usize;
                if self.occ.count[t.vp][gi] >= 2 {
                    bump.push(gi);
                }
            }
        }
        for gi in bump {
            self.occ.hist[gi] += s.hist_fac as f32;
        }
    }

    /// Route a set of tasks in batches of non-interacting windows.
    fn run_tasks(&mut self, mut queue: Vec<usize>, backend: &mut dyn Backend, pres: f64, hard: bool, cancel: &AtomicBool) -> Result<()> {
        let gap = 2 * self.max_reach + 2;
        // Nets in one batch never overlap (so they cannot race for the same space). In strict
        // order mode a net also waits for every earlier overlapping net; with `parallel_nets`
        // any non-overlapping net may join the batch, which gives far larger GPU batches.
        // Legalisation (hard) always keeps the strict order.
        let parallel = !hard && self.settings.parallel_nets;
        while !queue.is_empty() {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            // Pick a batch.
            let mut picked: Vec<(usize, Vec<JobSpec>)> = Vec::new();
            let mut blocked: Vec<Win> = Vec::new();
            let mut budget = 0usize;
            let mut rest = Vec::new();
            for &ti in &queue {
                let specs = self.job_specs(&self.tasks[ti]);
                let wins: Vec<Win> = specs.iter().map(|s| s.win).collect();
                let clash = wins.iter().any(|w| (!parallel && blocked.iter().any(|b| w.overlaps(b, gap))) || picked.iter().any(|(_, ps)| ps.iter().any(|p| w.overlaps(&p.win, gap))));
                let cost: usize = specs.iter().map(|s| 2 * s.win.w * s.win.h * self.maps.grid.nl).sum();
                let fits = picked.is_empty() || (budget + cost <= BATCH_DIST_BUDGET && picked.iter().map(|p| p.1.len()).sum::<usize>() + specs.len() <= MAX_BATCH_JOBS);
                if clash || !fits {
                    blocked.extend(wins);
                    rest.push(ti);
                } else {
                    budget += cost;
                    picked.push((ti, specs));
                }
            }
            queue = rest;
            let retry = self.run_batch(&picked, backend, pres, hard)?;
            for ti in retry {
                queue.push(ti);
            }
        }
        Ok(())
    }

    /// Run one batch. Returns tasks that need a larger window.
    fn run_batch(&mut self, picked: &[(usize, Vec<JobSpec>)], backend: &mut dyn Backend, pres: f64, hard: bool) -> Result<Vec<usize>> {
        struct Live {
            task: usize,
            win: Win,
            job: Arc<SearchJob>,
            remaining: Vec<usize>,
            paths: Vec<Vec<u32>>,
            vias: Vec<u32>,
            done: bool,
            grow: bool,
            failed: usize,
        }
        let mut lives: Vec<Live> = Vec::new();
        let mut jobs: Vec<Arc<SearchJob>> = Vec::new();
        let tb = Instant::now();
        let this = &*self;
        let built: Vec<(usize, JobSpec, SearchJob)> = picked
            .par_iter()
            .flat_map(|(ti, specs)| specs.par_iter().map(move |s| (*ti, s.clone(), this.build_job(&this.tasks[*ti], &s.win, pres, hard))))
            .collect();
        let mut seeds_per: Vec<Vec<u32>> = Vec::new();
        for (ti, spec, job) in built {
            let job = Arc::new(job);
            let seeds: Vec<u32> = match spec.root {
                Root::Group(gi) => self.tasks[ti].groups[gi].nodes.iter().filter_map(|&e| self.to_local(&spec.win, e)).collect(),
                Root::Plane(pl) => {
                    // Only where a via actually lands on the plane's copper.
                    let pw = spec.win.w * spec.win.h;
                    let g = &self.maps.grid;
                    (0..pw)
                        .filter(|&p| {
                            let (y, x) = (spec.win.y0 + p / spec.win.w, spec.win.x0 + p % spec.win.w);
                            self.maps.on_plane(pl, y * g.nx + x)
                        })
                        .map(|p| (pl * pw + p) as u32)
                        .collect()
                }
            };
            jobs.push(job.clone());
            seeds_per.push(seeds);
            lives.push(Live { task: ti, win: spec.win, job, remaining: spec.targets.clone(), paths: vec![], vias: vec![], done: false, grow: false, failed: 0 });
        }
        if lives.is_empty() {
            return Ok(vec![]);
        }
        backend.load(jobs)?;
        self.stats.t_build += tb.elapsed().as_secs_f64();
        self.stats.batches += 1;
        for (slot, s) in seeds_per.iter().enumerate() {
            if s.is_empty() {
                lives[slot].done = true;
                lives[slot].grow = true;
            } else {
                backend.seed(slot, s);
            }
        }
        loop {
            let active: Vec<usize> = (0..lives.len()).filter(|&i| !lives[i].done).collect();
            if active.is_empty() {
                break;
            }
            // Targets: local nodes of every remaining group, with the group each belongs to.
            let mut tlists: Vec<Vec<u32>> = Vec::with_capacity(active.len());
            let mut tgroups: Vec<HashMap<u32, usize>> = Vec::with_capacity(active.len());
            for &slot in &active {
                let lv = &lives[slot];
                let task = &self.tasks[lv.task];
                let g = &self.maps.grid;
                let w = lv.win;
                let mut list = Vec::new();
                let mut map = HashMap::new();
                for &gi in &lv.remaining {
                    for &e in &task.groups[gi].nodes {
                        let e = e as usize;
                        let (l, p) = (e / g.plane(), e % g.plane());
                        let (y, x) = (p / g.nx, p % g.nx);
                        if x >= w.x0 && y >= w.y0 && x < w.x0 + w.w && y < w.y0 + w.h {
                            let le = (l * w.w * w.h + (y - w.y0) * w.w + (x - w.x0)) as u32;
                            list.push(le);
                            map.insert(le, gi);
                        }
                    }
                }
                tlists.push(list);
                tgroups.push(map);
            }
            let t1 = Instant::now();
            backend.relax(&active)?;
            let t2 = Instant::now();
            // Pick the nearest target and trace the path (on the GPU when available).
            let picks = backend.pick(&active, &tlists)?;
            let t3 = Instant::now();
            self.stats.t_relax += (t2 - t1).as_secs_f64();
            self.stats.t_read += (t3 - t2).as_secs_f64();
            self.stats.searches += active.len() as u64;
            self.stats.nodes_searched += active.iter().map(|&s| lives[s].job.n() as u64).sum::<u64>();
            for (k, &slot) in active.iter().enumerate() {
                let lv = &mut lives[slot];
                let task = &self.tasks[lv.task];
                let Some(pk) = picks[k].clone() else {
                    let full = lv.win.x0 == 0 && lv.win.y0 == 0 && lv.win.w == self.maps.grid.nx && lv.win.h == self.maps.grid.ny;
                    if full {
                        lv.failed += lv.remaining.len();
                        lv.remaining.clear();
                    } else {
                        lv.grow = true;
                    }
                    lv.done = true;
                    continue;
                };
                let gi = tgroups[k][&pk.node];
                let steps = pk.steps;
                let win = lv.win;
                let wp = (win.w * win.h) as u32;
                let mut path: Vec<u32> = Vec::with_capacity(steps.len());
                let mut new_seeds: Vec<u32> = Vec::with_capacity(steps.len());
                for (i, st) in steps.iter().enumerate() {
                    let g = &self.maps.grid;
                    let (ly, lx) = (st.p as usize / win.w, st.p as usize % win.w);
                    let gidx = g.idx(st.layer as usize, win.y0 + ly, win.x0 + lx) as u32;
                    if i > 0 && steps[i - 1].layer != st.layer {
                        lv.vias.push(((win.y0 + ly) * g.nx + win.x0 + lx) as u32);
                    }
                    path.push(gidx);
                    new_seeds.push(st.layer * wp + st.p);
                }
                // Groups touched by the path are connected too.
                let on_path: HashSet<u32> = path.iter().copied().collect();
                lv.remaining.retain(|&r| r != gi && !task.groups[r].nodes.iter().any(|n| on_path.contains(n)));
                lv.paths.push(path);
                if lv.remaining.is_empty() {
                    lv.done = true;
                } else {
                    backend.seed(slot, &new_seeds);
                }
            }
            self.stats.t_trace += t3.elapsed().as_secs_f64();
        }
        // Commit per task.
        let mut by_task: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, lv) in lives.iter().enumerate() {
            by_task.entry(lv.task).or_default().push(i);
        }
        let mut retry = Vec::new();
        let mut tasks_sorted: Vec<usize> = by_task.keys().copied().collect();
        tasks_sorted.sort_unstable();
        for ti in tasks_sorted {
            let idx = &by_task[&ti];
            if idx.iter().any(|&i| lives[i].grow) {
                let can_grow = idx.iter().any(|&i| lives[i].grow && !self.full_grid(&lives[i].win));
                if can_grow {
                    self.tasks[ti].margin = (self.tasks[ti].margin * 3).max(8);
                    retry.push(ti);
                    continue;
                }
            }
            let mut route = NetRoute::default();
            let mut failed = 0usize;
            for &i in idx {
                let lv = &lives[i];
                failed += lv.failed + if lv.grow { lv.remaining.len().max(1) } else { 0 };
                for p in &lv.paths {
                    route.nodes.extend_from_slice(p);
                    for wv in p.windows(2) {
                        let (a, b) = (wv[0], wv[1]);
                        let plane = self.maps.grid.plane() as u32;
                        if a / plane == b / plane && a != b {
                            route.edges.push((a.min(b), a.max(b)));
                        }
                    }
                }
                route.vias.extend_from_slice(&lv.vias);
                route.paths.extend(lv.paths.iter().cloned());
            }
            route.nodes.sort_unstable();
            route.nodes.dedup();
            route.edges.sort_unstable();
            route.edges.dedup();
            // Hole-to-hole spacing between this net's own vias (other nets are kept apart by
            // the congestion maps). Vias in path order: the earlier one stays.
            let ordered: Vec<u32> = {
                let plane = self.maps.grid.plane() as u32;
                let mut v: Vec<u32> = Vec::new();
                for p in &route.paths {
                    for w in p.windows(2) {
                        if w[0] / plane != w[1] / plane && !v.contains(&(w[1] % plane)) {
                            v.push(w[1] % plane);
                        }
                    }
                }
                v
            };
            let bad = self.own_via_conflicts(self.tasks[ti].class, &ordered);
            if !bad.is_empty() && self.tasks[ti].via_retries < 8 {
                let mut f = std::mem::take(&mut self.tasks[ti].forbid_vias);
                for keep in bad {
                    f.extend(self.via_keepout_around(self.tasks[ti].class, keep));
                }
                f.sort_unstable();
                f.dedup();
                self.tasks[ti].forbid_vias = f;
                self.tasks[ti].via_retries += 1;
                retry.push(ti);
                continue;
            }
            route.vias.sort_unstable();
            route.vias.dedup();
            self.tasks[ti].route = route;
            self.tasks[ti].failed = self.tasks[ti].static_failed + failed;
            self.stamp(ti, 1);
        }
        Ok(retry)
    }

    /// Minimum centre distance between two vias of one net class (hole-to-hole rule).
    fn via_spacing(&self, class: usize) -> f64 {
        let nc = self.settings.netclass(class);
        nc.via_drill + self.settings.hole_to_hole
    }

    /// Earlier vias (plane indices) that a later via of the same net sits too close to.
    fn own_via_conflicts(&self, class: usize, vias: &[u32]) -> Vec<u32> {
        let g = &self.maps.grid;
        let min = self.via_spacing(class);
        let pos = |p: u32| g.pos(p as usize % g.nx, p as usize / g.nx);
        let mut keep = Vec::new();
        for j in 1..vias.len() {
            for i in 0..j {
                if pos(vias[i]).dist(pos(vias[j])) < min - 1e-9 && !keep.contains(&vias[i]) {
                    keep.push(vias[i]);
                }
            }
        }
        keep
    }

    /// Via sites around `keep` (excluding it) that would violate hole-to-hole spacing.
    fn via_keepout_around(&self, class: usize, keep: u32) -> Vec<u32> {
        let g = &self.maps.grid;
        let min = self.via_spacing(class);
        let r = (min / g.pitch).ceil() as i64;
        let (kx, ky) = ((keep as usize % g.nx) as i64, (keep as usize / g.nx) as i64);
        let mut v = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (kx + dx, ky + dy);
                if (dx, dy) == (0, 0) || x < 0 || y < 0 || x >= g.nx as i64 || y >= g.ny as i64 {
                    continue;
                }
                if (((dx * dx + dy * dy) as f64).sqrt() * g.pitch) < min - 1e-9 {
                    v.push((y as usize * g.nx + x as usize) as u32);
                }
            }
        }
        v
    }

    /// Debug: for each conflicted net, where it conflicts and with which net.
    fn report_conflicts(&self, conflicts: &[usize]) {
        let g = &self.maps.grid;
        let plane = g.plane();
        for &ti in conflicts.iter().take(25) {
            let t = &self.tasks[ti];
            let bad: Vec<u32> = t
                .route
                .nodes
                .iter()
                .copied()
                .filter(|&e| self.occ.count[self.node_probe(t, e)][e as usize] >= 2)
                .collect();
            let bad_vias: Vec<u32> = t.route.vias.iter().copied().filter(|&p| (0..g.nl).any(|l| self.occ.count[t.vp][l * plane + p as usize] >= 2)).collect();
            let Some(&e) = bad.first() else {
                if let Some(&p) = bad_vias.first() {
                    let q = g.pos(p as usize % g.nx, p as usize / g.nx);
                    eprintln!("conflict {}: via at ({:.3},{:.3}) ({} vias)", self.board.net_name(t.net), q.x, q.y, bad_vias.len());
                }
                continue;
            };
            let (l, p) = (e as usize / plane, e as usize % plane);
            let q = g.pos(p % g.nx, p / g.nx);
            // Which other net is closest to this node?
            let mut other = (f64::INFINITY, String::new());
            for (tj, u) in self.tasks.iter().enumerate() {
                if tj == ti {
                    continue;
                }
                for &f in &u.route.nodes {
                    let (l2, p2) = (f as usize / plane, f as usize % plane);
                    if l2 == l {
                        let d = q.dist(g.pos(p2 % g.nx, p2 / g.nx));
                        if d < other.0 {
                            other = (d, format!("{} (track)", self.board.net_name(u.net)));
                        }
                    }
                }
                for &f in &u.route.vias {
                    let d = q.dist(g.pos(f as usize % g.nx, f as usize / g.nx));
                    if d < other.0 {
                        other = (d, format!("{} (via)", self.board.net_name(u.net)));
                    }
                }
            }
            let in_neck = t.neck.binary_search(&e).is_ok();
            eprintln!(
                "conflict {}: {} nodes, {} vias; first at layer {} ({:.3},{:.3}){} near {} at {:.3} mm",
                self.board.net_name(t.net),
                bad.len(),
                bad_vias.len(),
                l,
                q.x,
                q.y,
                if in_neck { " [neck]" } else { "" },
                other.1,
                other.0
            );
        }
    }

    /// Run the full negotiated-congestion flow.
    pub fn route(&mut self, backend: &mut dyn Backend, cancel: &AtomicBool, progress: &mut dyn FnMut(Progress)) -> Result<()> {
        let t0 = Instant::now();
        let s = self.settings;
        self.stats.backend = backend.name();
        let all: Vec<usize> = (0..self.tasks.len()).collect();
        let mut pres = s.pres_fac_init;
        let mut conflicts: Vec<usize> = Vec::new();
        for it in 0..s.max_iterations.max(1) {
            let todo: Vec<usize> = if it == 0 { all.clone() } else { conflicts.clone() };
            if todo.is_empty() {
                break;
            }
            for &ti in &todo {
                self.rip_up(ti);
            }
            self.run_tasks(todo, backend, pres, false, cancel)?;
            self.stats.iterations = it + 1;
            conflicts = (0..self.tasks.len()).filter(|&ti| self.conflicted(ti)).collect();
            progress(Progress {
                phase: "negotiating".into(),
                iteration: it + 1,
                max_iterations: s.max_iterations,
                nets_total: self.tasks.len(),
                conflicts: conflicts.len(),
                message: format!("iteration {}: {} nets in conflict", it + 1, conflicts.len()),
                preview: Some(self.preview()),
            });
            if conflicts.is_empty() || cancel.load(Ordering::Relaxed) {
                break;
            }
            for &ti in &conflicts {
                self.add_history(ti);
                // Nets stuck in conflict get room to detour around congestion.
                if it >= 2 {
                    let t = &mut self.tasks[ti];
                    t.margin = (t.margin + t.margin / 3 + 2).min(4 * self.base_margin);
                }
            }
            pres *= s.pres_fac_mult;
        }
        if !conflicts.is_empty() && std::env::var_os("TW_DEBUG").is_some() {
            self.report_conflicts(&conflicts);
        }
        if !conflicts.is_empty() {
            // Legalise: reroute the remaining conflicted nets against hard obstacles.
            self.log.push(format!("{} nets still conflicting after negotiation; legalising", conflicts.len()));
            for &ti in &conflicts {
                self.rip_up(ti);
            }
            if !cancel.load(Ordering::Relaxed) {
                self.run_tasks(conflicts.clone(), backend, pres, true, cancel)?;
            }
            // Anything still conflicting (only possible after cancel) is dropped.
            for ti in 0..self.tasks.len() {
                if self.conflicted(ti) {
                    self.rip_up(ti);
                    let t = &mut self.tasks[ti];
                    t.failed = t.static_failed + if t.plane.is_some() { t.groups.len() } else { t.groups.len().saturating_sub(1) };
                }
            }
        }
        self.stats.seconds = t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// Centre-line geometry of all current routes.
    pub fn preview(&self) -> (Vec<Seg>, Vec<ViaOut>) {
        let mut segs = Vec::new();
        let mut vias = Vec::new();
        for t in &self.tasks {
            let (s, v) = self.raw_geometry(t);
            segs.extend(s);
            vias.extend(v);
        }
        (segs, vias)
    }

    fn raw_geometry(&self, t: &Task) -> (Vec<Seg>, Vec<ViaOut>) {
        let g = &self.maps.grid;
        let nc = self.settings.netclass(t.class);
        let plane = g.plane() as u32;
        let width = nc.track_width;
        // Adjacency per node.
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(a, b) in &t.route.edges {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        let via_set: HashSet<u32> = t.route.vias.iter().copied().collect();
        let mut ends: HashSet<u32> = HashSet::new();
        for p in &t.route.paths {
            if let (Some(&a), Some(&b)) = (p.first(), p.last()) {
                ends.insert(a);
                ends.insert(b);
            }
        }
        let dir = |a: u32, b: u32| -> i64 { b as i64 - a as i64 };
        // A segment touching a neck node takes the narrower neck width of its two ends.
        let narrow_w = |a: u32, b: u32| -> Option<f64> {
            match (self.node_neck_w(t, a), self.node_neck_w(t, b)) {
                (None, None) => None,
                (x, y) => Some(x.unwrap_or(f64::MAX).min(y.unwrap_or(f64::MAX))),
            }
        };
        let is_break = |n: u32| -> bool {
            let Some(nb) = adj.get(&n) else { return true };
            if nb.len() != 2 || via_set.contains(&(n % plane)) || ends.contains(&n) {
                return true;
            }
            dir(nb[0], n) != dir(n, nb[1]) || narrow_w(nb[0], n) != narrow_w(n, nb[1])
        };
        let mut segs = Vec::new();
        let mut used: HashSet<(u32, u32)> = HashSet::new();
        let mut keys: Vec<u32> = adj.keys().copied().collect();
        keys.sort_unstable();
        for &start in &keys {
            if !is_break(start) {
                continue;
            }
            for &nb in &adj[&start] {
                if used.contains(&(start.min(nb), start.max(nb))) {
                    continue;
                }
                let step = dir(start, nb);
                let mut prev = start;
                let mut cur = nb;
                used.insert((prev.min(cur), prev.max(cur)));
                while !is_break(cur) {
                    let next = adj[&cur].iter().copied().find(|&x| x != prev).unwrap();
                    if dir(cur, next) != step {
                        break;
                    }
                    used.insert((cur.min(next), cur.max(next)));
                    prev = cur;
                    cur = next;
                }
                let la = start as usize;
                let lb = cur as usize;
                let pa = la % g.plane();
                let pb = lb % g.plane();
                segs.push(Seg {
                    net: t.net,
                    layer: la / g.plane(),
                    a: g.pos(pa % g.nx, pa / g.nx),
                    b: g.pos(pb % g.nx, pb / g.nx),
                    width: narrow_w(start, nb).unwrap_or(width),
                });
            }
        }
        let vias = t
            .route
            .vias
            .iter()
            .map(|&p| {
                let p = p as usize;
                ViaOut { net: t.net, pos: g.pos(p % g.nx, p / g.nx), diameter: nc.via_diameter, drill: nc.via_drill, from_layer: 0, to_layer: g.nl - 1 }
            })
            .collect();
        (segs, vias)
    }

    /// Final geometry with pad-centre stubs, merged runs and optional mitred corners.
    pub fn result(&mut self) -> RouteResult {
        let mut segs = Vec::new();
        let mut vias = Vec::new();
        for t in &self.tasks {
            let (s, v) = self.raw_geometry(t);
            segs.extend(s);
            vias.extend(v);
        }
        let s = self.settings;
        // Spatial index of everything for validating off-grid additions.
        let mut items = collect_items(self.board, s, &segs, &vias, s.keep_existing);
        let mut stubs: Vec<Seg> = Vec::new();
        {
            let index = Index::new(items.clone(), 2.0);
            for t in &self.tasks {
                let nc = s.netclass(t.class);
                let node_pad: HashMap<u32, usize> = t.groups.iter().flat_map(|g| g.nodes.iter().copied().zip(g.node_pad.iter().copied())).filter(|&(_, p)| p != usize::MAX).collect();
                let mut done: HashSet<(u32, usize)> = HashSet::new();
                for p in &t.route.paths {
                    for &e in [p.first(), p.last()].into_iter().flatten() {
                        let Some(&pad) = node_pad.get(&e) else { continue };
                        if !done.insert((e, pad)) {
                            continue;
                        }
                        let g = &self.maps.grid;
                        let layer = e as usize / g.plane();
                        let pp = e as usize % g.plane();
                        let q = g.pos(pp % g.nx, pp / g.nx);
                        let pd = &self.board.pads[pad];
                        let inside = pd.shape_on(layer).is_some_and(|poly| poly.contains(q));
                        if inside && !s.snap_to_pad_center {
                            continue;
                        }
                        let target = pd.pos;
                        if q.dist(target) < 1e-6 {
                            continue;
                        }
                        // Only axis-aligned stubs: they extend the track into the pad centre
                        // instead of adding a jog. Inside the pad the track is connected anyway.
                        let aligned = (q.x - target.x).abs() < 1e-6 || (q.y - target.y).abs() < 1e-6;
                        if inside && !aligned {
                            continue;
                        }
                        let sw = self.node_neck_w(t, e).unwrap_or(nc.track_width);
                        if segment_clear(&index, s, t.net, layer, q, target, sw / 2.0, nc.clearance) {
                            stubs.push(Seg { net: t.net, layer, a: q, b: target, width: sw });
                        } else if !inside {
                            self.log.push(format!("net {}: pad {} reached only by a node outside it; stub blocked", self.board.net_name(t.net), pd.name));
                        }
                    }
                }
            }
        }
        segs.extend(stubs);
        let mut segs = merge_collinear(segs, &vias);
        if s.corner_style == CornerStyle::Miter45 {
            items = collect_items(self.board, s, &segs, &vias, s.keep_existing);
            let index = Index::new(items, 2.0);
            segs = miter_corners(segs, &vias, &index, s, self.board);
        }
        self.stats.vias = vias.len();
        self.stats.wirelength_mm = segs.iter().map(|s| s.a.dist(s.b)).sum();
        self.stats.connections_failed = self.tasks.iter().map(|t| t.failed).sum();
        self.stats.connections_offboard = self.tasks.iter().map(|t| t.offboard).sum();
        self.stats.nets_complete = self.tasks.iter().filter(|t| t.failed == 0 && t.offboard == 0).count();
        let incomplete_nets = self.tasks.iter().filter(|t| t.failed > 0).map(|t| self.board.net_name(t.net).to_string()).collect();
        let offboard_nets = self.tasks.iter().filter(|t| t.failed == 0 && t.offboard > 0).map(|t| self.board.net_name(t.net).to_string()).collect();
        let mut pads = self.offboard_pads.clone();
        pads.sort_unstable();
        pads.dedup();
        let offboard_pads = pads.iter().map(|&p| self.board.pads[p].name.clone()).collect();
        RouteResult { segs, vias, stats: self.stats.clone(), incomplete_nets, offboard_pads, offboard_nets, rerouted_nets: vec![], removed_ids: vec![], log: self.log.clone() }
    }
}

/// True when a track segment of `net` (half-width r) keeps clearance to everything else.
pub fn segment_clear(index: &Index, s: &Settings, net: NetId, layer: usize, a: P2, b: P2, r: f64, clearance: f64) -> bool {
    let shape = Shape::Segment { a, b, r };
    let reach = clearance.max(s.edge_clearance).max(s.hole_clearance) + r + 0.5;
    let bb = crate::geom::BBox::from_points([a, b].iter()).expand(reach);
    for j in index.query(&bb) {
        let o: &Item = &index.items[j as usize];
        if o.layers & (1u64 << layer) == 0 {
            continue;
        }
        let required = match o.kind {
            ItemKind::Edge => s.edge_clearance,
            ItemKind::Hole => s.hole_clearance,
            ItemKind::Keepout => {
                if o.no_tracks && shape.shape_dist(&o.shape) <= 0.0 {
                    return false;
                }
                continue;
            }
            _ => {
                if o.net == net && net != 0 {
                    continue;
                }
                clearance.max(o.clearance)
            }
        };
        if shape.shape_dist(&o.shape) < required + SAFETY * 0.5 {
            return false;
        }
    }
    true
}

/// Merge collinear segments that meet end-to-end with nothing else at the joint.
fn merge_collinear(mut segs: Vec<Seg>, vias: &[ViaOut]) -> Vec<Seg> {
    let key = |p: P2| ((p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64);
    loop {
        let mut at: HashMap<(usize, NetId, (i64, i64)), Vec<usize>> = HashMap::new();
        for (i, s) in segs.iter().enumerate() {
            at.entry((s.layer, s.net, key(s.a))).or_default().push(i);
            at.entry((s.layer, s.net, key(s.b))).or_default().push(i);
        }
        let via_pts: HashSet<(i64, i64)> = vias.iter().map(|v| key(v.pos)).collect();
        let mut merged = false;
        let mut dead = vec![false; segs.len()];
        let mut keys: Vec<_> = at.keys().copied().collect();
        keys.sort_unstable();
        for k in keys {
            let ids = &at[&k];
            if ids.len() != 2 || via_pts.contains(&k.2) {
                continue;
            }
            let (i, j) = (ids[0], ids[1]);
            if dead[i] || dead[j] || i == j || (segs[i].width - segs[j].width).abs() > 1e-9 {
                continue;
            }
            let p = |s: &Seg| if key(s.a) == k.2 { (s.b, s.a) } else { (s.a, s.b) };
            let (a0, a1) = p(&segs[i]);
            let (b1, b0) = p(&segs[j]);
            let _ = b0;
            let d1 = a1 - a0;
            let d2 = b1 - a1;
            if d1.cross(d2).abs() > 1e-9 * (d1.len() * d2.len()).max(1e-12) || d1.dot(d2) <= 0.0 {
                continue;
            }
            segs[i] = Seg { a: a0, b: b1, ..segs[i].clone() };
            dead[j] = true;
            merged = true;
        }
        segs = segs.into_iter().zip(dead).filter(|(_, d)| !d).map(|(s, _)| s).collect();
        if !merged {
            return segs;
        }
    }
}

/// Replace 90° corners by 45° chamfers where the chamfer keeps clearance.
fn miter_corners(mut segs: Vec<Seg>, vias: &[ViaOut], index: &Index, s: &Settings, board: &Board) -> Vec<Seg> {
    let key = |p: P2| ((p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64);
    let via_pts: HashSet<(i64, i64)> = vias.iter().map(|v| key(v.pos)).collect();
    let pad_centres: HashSet<(i64, i64)> = board.pads.iter().map(|p| key(p.pos)).collect();
    let mut at: HashMap<(usize, NetId, (i64, i64)), Vec<usize>> = HashMap::new();
    for (i, sg) in segs.iter().enumerate() {
        at.entry((sg.layer, sg.net, key(sg.a))).or_default().push(i);
        at.entry((sg.layer, sg.net, key(sg.b))).or_default().push(i);
    }
    let mut extra = Vec::new();
    let mut keys: Vec<_> = at.keys().copied().collect();
    keys.sort_unstable();
    for k in keys {
        let ids = at[&k].clone();
        if ids.len() != 2 || via_pts.contains(&k.2) || pad_centres.contains(&k.2) {
            continue;
        }
        let (i, j) = (ids[0], ids[1]);
        let corner_of = |sg: &Seg| if key(sg.a) == k.2 { (sg.a, sg.b) } else { (sg.b, sg.a) };
        let (c, pi) = corner_of(&segs[i]);
        let (_, pj) = corner_of(&segs[j]);
        let (di, dj) = (pi - c, pj - c);
        if di.dot(dj).abs() > 1e-9 || di.len() < 1e-9 || dj.len() < 1e-9 {
            continue;
        }
        let m = s.miter.min(di.len() * 0.5).min(dj.len() * 0.5);
        if m < 0.05 {
            continue;
        }
        let qi = c + di * (m / di.len());
        let qj = c + dj * (m / dj.len());
        let nc = s.netclass(board.nets.get(segs[i].net as usize).map(|n| n.netclass).unwrap_or(0));
        if !segment_clear(index, s, segs[i].net, segs[i].layer, qi, qj, segs[i].width / 2.0, nc.clearance) {
            continue;
        }
        let shorten = |sg: &mut Seg, newp: P2| {
            if key(sg.a) == k.2 {
                sg.a = newp;
            } else {
                sg.b = newp;
            }
        };
        shorten(&mut segs[i], qi);
        shorten(&mut segs[j], qj);
        extra.push(Seg { net: segs[i].net, layer: segs[i].layer, a: qi, b: qj, width: segs[i].width });
    }
    segs.extend(extra);
    segs
}

/// The board as the router sees it: under `force_reroute`, each existing track narrower than
/// `min_neck_width` is removed together with the copper run it belongs to (tracks and vias of the
/// same net joined end to end or through vias, stopping at pads), so that connection is routed
/// again. The rest of the net is kept. Returns the board, the affected net names and the KiCad ids
/// of the removed items.
pub fn prepare_board(board: &Board, settings: &Settings) -> (Board, Vec<String>, Vec<String>) {
    prepare_board_with(board, settings, &[], &[])
}

/// `prepare_board`, also ripping the runs of any net with copper within `r` of the given points
/// (pads a forced net could not reach), and keeping the copper of nets in `keep`.
fn prepare_board_with(board: &Board, settings: &Settings, around: &[(P2, f64)], keep: &[NetId]) -> (Board, Vec<String>, Vec<String>) {
    let near = |net: NetId, d: &dyn Fn(P2) -> f64| net != 0 && around.iter().any(|&(c, r)| d(c) < r);
    let mut forced: Vec<NetId> = settings.forced_nets(board);
    if forced.is_empty() {
        return (board.clone(), vec![], vec![]);
    }
    for t in &board.tracks {
        if near(t.net, &|c| crate::geom::point_seg_dist(c, t.a, t.b) - t.width / 2.0) {
            forced.push(t.net);
        }
    }
    for v in &board.vias {
        if near(v.net, &|c| c.dist(v.pos) - v.diameter / 2.0) {
            forced.push(v.net);
        }
    }
    forced.retain(|n| !keep.contains(n));
    forced.sort_unstable();
    forced.dedup();
    if forced.is_empty() {
        return (board.clone(), vec![], vec![]);
    }
    let (nt, nv) = (board.tracks.len(), board.vias.len());
    let mut parent: Vec<usize> = (0..nt + nv).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    // Item endpoints keyed by (net, layer, whole-micron position); a via sits on all its layers.
    let key = |net: NetId, layer: usize, q: P2| (net, layer, (q.x * 1e3).round() as i64, (q.y * 1e3).round() as i64);
    let mut at: HashMap<(NetId, usize, i64, i64), Vec<usize>> = HashMap::new();
    for (i, t) in board.tracks.iter().enumerate() {
        if forced.binary_search(&t.net).is_ok() {
            at.entry(key(t.net, t.layer, t.a)).or_default().push(i);
            at.entry(key(t.net, t.layer, t.b)).or_default().push(i);
        }
    }
    for (j, v) in board.vias.iter().enumerate() {
        if forced.binary_search(&v.net).is_ok() {
            for l in v.from_layer..=v.to_layer {
                at.entry(key(v.net, l, v.pos)).or_default().push(nt + j);
            }
        }
    }
    for ids in at.values() {
        for w in ids.windows(2) {
            let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
            parent[a] = b;
        }
    }
    // Tracks ending on another track's interior (T-junctions) belong to its run too.
    for (i, t) in board.tracks.iter().enumerate() {
        if forced.binary_search(&t.net).is_err() {
            continue;
        }
        for (k, u) in board.tracks.iter().enumerate() {
            if k == i || u.net != t.net || u.layer != t.layer {
                continue;
            }
            if [u.a, u.b].iter().any(|&q| crate::geom::point_seg_dist(q, t.a, t.b) < 1e-3) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, k));
                parent[a] = b;
            }
        }
    }
    let seeds: Vec<usize> = (0..nt + nv)
        .filter(|&i| {
            if i < nt {
                let t = &board.tracks[i];
                forced.binary_search(&t.net).is_ok() && (t.width < settings.min_neck_width - 1e-6 || near(t.net, &|c| crate::geom::point_seg_dist(c, t.a, t.b) - t.width / 2.0))
            } else {
                let v = &board.vias[i - nt];
                forced.binary_search(&v.net).is_ok() && near(v.net, &|c| c.dist(v.pos) - v.diameter / 2.0)
            }
        })
        .collect();
    let mut gone_roots: Vec<usize> = seeds.iter().map(|&i| find(&mut parent, i)).collect();
    let forced: Vec<NetId> = {
        let mut v: Vec<NetId> = seeds.iter().map(|&i| if i < nt { board.tracks[i].net } else { board.vias[i - nt].net }).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    gone_roots.sort_unstable();
    gone_roots.dedup();
    let gone: Vec<bool> = (0..nt + nv).map(|i| gone_roots.binary_search(&find(&mut parent, i)).is_ok()).collect();
    let mut removed: Vec<String> = board.tracks.iter().enumerate().filter(|&(i, _)| gone[i]).map(|(_, t)| t.id.clone()).chain(board.vias.iter().enumerate().filter(|&(j, _)| gone[nt + j]).map(|(_, v)| v.id.clone())).filter(|id| !id.is_empty()).collect();
    removed.sort();
    removed.dedup(); // arcs come in as several segments sharing one id
    let mut b = board.clone();
    let mut i = 0;
    b.tracks.retain(|_| {
        i += 1;
        !gone[i - 1]
    });
    let mut j = 0;
    b.vias.retain(|_| {
        j += 1;
        !gone[nt + j - 1]
    });
    (b, forced.iter().map(|&n| board.net_name(n).to_string()).collect(), removed)
}

/// Mark nets that the final geometry leaves disconnected as incomplete (a terminal reached only
/// through a stub that failed validation, for example), checked on the board as it will be applied.
fn verify_connectivity(board: &Board, settings: &Settings, res: &mut RouteResult) {
    let planes: Vec<(usize, NetId)> = settings.layers.iter().enumerate().filter_map(|(l, c)| c.plane_net.as_ref().and_then(|n| board.net_by_name(n)).map(|n| (l, n))).collect();
    let after = board_after(board, res);
    let mut bad: Vec<(NetId, usize)> = crate::drc::unconnected_per_net(&after, settings, &res.segs, &res.vias, &planes)
        .into_iter()
        .filter(|&(n, c)| c > 0 && settings.net_selected(board.net_name(n)) && !res.offboard_nets.iter().any(|o| o == board.net_name(n)))
        .collect();
    bad.sort_unstable();
    for (n, c) in bad {
        let name = board.net_name(n).to_string();
        if !res.incomplete_nets.contains(&name) {
            res.log.push(format!("net {name}: {c} pad group(s) still disconnected in the final copper"));
            res.incomplete_nets.push(name);
        }
    }
}

/// The board as it will be once `res` is applied: its `removed_ids` taken out.
pub fn board_after(board: &Board, res: &RouteResult) -> Board {
    let mut b = board.clone();
    if !res.removed_ids.is_empty() {
        let gone: HashSet<&str> = res.removed_ids.iter().map(|s| s.as_str()).collect();
        b.tracks.retain(|t| !gone.contains(t.id.as_str()));
        b.vias.retain(|v| !gone.contains(v.id.as_str()));
    }
    b
}

impl Router<'_> {
    /// Pad groups on the board that have no legal grid node at the current pitch.
    fn unreachable(&self) -> usize {
        self.tasks.iter().map(|t| t.static_failed).sum()
    }
}

/// Auto pitch: when some on-board pads have no legal grid node, try finer pitches (within the
/// node budget) and keep the one that reaches the most pads.
fn choose_pitch(board: &Board, settings: &Settings, log: &mut Vec<String>) -> Settings {
    let mut s = settings.clone();
    if settings.grid_pitch > 0.0 {
        return s;
    }
    let p0 = settings.effective_pitch(board);
    let u0 = Router::new(board, &s).unreachable();
    if u0 == 0 {
        return s;
    }
    let bb = board.bbox();
    let fits = |p: f64| bb.width() * bb.height() / (p * p) <= settings.max_nodes_per_layer as f64;
    let mut best = (u0, p0);
    for p in [0.127, 0.125, 0.1, 0.0635, 0.05, 0.04] {
        if p >= p0 - 1e-9 || !fits(p) {
            continue;
        }
        s.grid_pitch = p;
        let u = Router::new(board, &s).unreachable();
        if u < best.0 {
            best = (u, p);
        }
        if u == 0 {
            break;
        }
    }
    s.grid_pitch = best.1;
    if best.1 != p0 {
        log.push(format!("auto pitch {p0:.4} mm left {u0} pad group(s) without a legal grid node; refined to {:.4} mm ({} left)", best.1, best.0));
    }
    s
}

/// Convenience: route a board start to finish.
pub fn route_board(board: &Board, settings: &Settings, backend: &mut dyn Backend, cancel: &AtomicBool, progress: &mut dyn FnMut(Progress)) -> Result<RouteResult> {
    let original = board;
    let mut around: Vec<(P2, f64)> = Vec::new();
    let mut keep: Vec<NetId> = Vec::new();
    let mut pre_log = Vec::new();
    // Force re-route: if a ripped connection's pad cannot be reached at the minimum neck-down
    // (other nets' copper in the way), rip those neighbours too, a little wider each round.
    let (mut board, mut rerouted_nets, mut removed_ids) = prepare_board_with(original, settings, &around, &keep);
    let mut settings_eff = choose_pitch(&board, settings, &mut pre_log);
    if !rerouted_nets.is_empty() {
        for round in 1..=4 {
            let probe = Router::new(&board, &settings_eff);
            let stuck: Vec<usize> = probe.unreachable_pads.iter().copied().filter(|&p| rerouted_nets.iter().any(|n| n == original.net_name(original.pads[p].net))).collect();
            if stuck.is_empty() {
                break;
            }
            let r = 0.6 * round as f64;
            around.extend(stuck.iter().map(|&p| (original.pads[p].pos, r)));
            if round == 4 {
                // Still unreachable: keep these nets' existing copper rather than lose connections.
                keep.extend(stuck.iter().map(|&p| original.pads[p].net));
                keep.sort_unstable();
                keep.dedup();
                pre_log.push(format!("force re-route: kept the existing copper of {} (no route at {:.3} mm minimum neck-down)", keep.iter().map(|&n| original.net_name(n)).collect::<Vec<_>>().join(", "), settings.min_neck_width));
            }
            (board, rerouted_nets, removed_ids) = prepare_board_with(original, settings, &around, &keep);
            settings_eff = choose_pitch(&board, settings, &mut pre_log);
        }
        pre_log.insert(0, format!("force re-route: {} net(s) ripped up and re-routed at a {:.3} mm minimum neck-down: {}", rerouted_nets.len(), settings.min_neck_width, rerouted_nets.join(", ")));
    }
    let settings = &settings_eff;
    let board = &board;
    let mut r = Router::new(board, settings);
    r.log.splice(0..0, pre_log);
    progress(Progress {
        phase: "prepared".into(),
        iteration: 0,
        max_iterations: settings.max_iterations,
        nets_total: r.stats.nets_total,
        conflicts: 0,
        message: format!("{} nets, {} connections to route", r.stats.nets_total, r.stats.connections_total),
        preview: None,
    });
    r.route(backend, cancel, progress)?;
    let mut res = r.result();
    res.removed_ids = removed_ids.clone();
    verify_connectivity(original, settings, &mut res);
    let mut log = res.log.clone();
    // A ripped net that did not complete would lose connections when its old copper is deleted:
    // keep that copper and route once more.
    for _ in 0..4 {
        let failed: Vec<NetId> = rerouted_nets.iter().filter(|n| res.incomplete_nets.contains(n)).filter_map(|n| original.net_by_name(n)).collect();
        if failed.is_empty() {
            break;
        }
        keep.extend(failed);
        keep.sort_unstable();
        keep.dedup();
        let (b2, nets2, ids2) = prepare_board_with(original, settings, &around, &keep);
        let mut r2 = Router::new(&b2, settings);
        r2.log.splice(0..0, log.iter().cloned());
        r2.log.push(format!("force re-route: {} could not be completed; kept their existing copper and routed again", keep.iter().map(|&n| original.net_name(n)).collect::<Vec<_>>().join(", ")));
        r2.route(backend, cancel, progress)?;
        res = r2.result();
        rerouted_nets = nets2;
        removed_ids = ids2;
        res.removed_ids = removed_ids.clone();
        verify_connectivity(original, settings, &mut res);
        log = res.log.clone();
    }
    res.log = log;
    res.rerouted_nets = rerouted_nets;
    res.removed_ids = removed_ids;
    Ok(res)
}

