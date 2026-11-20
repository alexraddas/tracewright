//! Routing grid and static rasterisation.
//!
//! Every routed track runs node-to-node along grid lines and every via sits on a node, so
//! clearance is checked at nodes only. For a "probe" (a track of half-width h, or a via of
//! radius r, with clearance c) we precompute, per layer and node, who may occupy that node:
//! nobody (`BLOCKED`), anybody (`FREE`) or only one net (its id).
//!
//! Exactness: two grid-aligned polylines reach their minimum distance at grid nodes, so
//! dynamic (routed-vs-routed) checks at nodes are exact. Static obstacles can be off-grid;
//! for convex shapes the distance from a unit grid segment is at least
//! sqrt(d_node^2 - p^2/4), so inflating the required radius R to sqrt(R^2 + p^2/4) makes
//! node checks also cover the segments between nodes.

use crate::board::{Board, NetId, PadKind};
use crate::geom::{BBox, P2, Polygon, Shape};
use crate::settings::Settings;
use rayon::prelude::*;

pub const FREE: u16 = 0;
pub const BLOCKED: u16 = u16::MAX;
/// Extra distance added to every clearance (absorbs KiCad's arc polygonisation error).
pub const SAFETY: f64 = 0.008;

#[derive(Clone, Debug)]
pub struct Grid {
    pub origin: P2,
    pub pitch: f64,
    pub nx: usize,
    pub ny: usize,
    pub nl: usize,
}

impl Grid {
    /// Grid covering `bb` whose nodes sit at `offset + k * pitch` (all in whole nanometres).
    pub fn new(bb: &BBox, pitch: f64, nl: usize, offset: P2) -> Grid {
        let nm = |v: f64| (v * 1e6).round() / 1e6;
        let x0 = nm(offset.x + ((bb.min.x - offset.x) / pitch).floor() * pitch);
        let y0 = nm(offset.y + ((bb.min.y - offset.y) / pitch).floor() * pitch);
        let nx = ((bb.max.x - x0) / pitch).ceil() as usize + 1;
        let ny = ((bb.max.y - y0) / pitch).ceil() as usize + 1;
        Grid { origin: P2::new(x0, y0), pitch, nx, ny, nl }
    }
    #[inline]
    pub fn plane(&self) -> usize {
        self.nx * self.ny
    }
    #[inline]
    pub fn idx(&self, l: usize, y: usize, x: usize) -> usize {
        l * self.plane() + y * self.nx + x
    }
    #[inline]
    pub fn pos(&self, x: usize, y: usize) -> P2 {
        P2::new(self.origin.x + x as f64 * self.pitch, self.origin.y + y as f64 * self.pitch)
    }
    /// Inclusive node index range covering a bbox, clipped to the grid.
    pub fn range(&self, bb: &BBox) -> Option<(usize, usize, usize, usize)> {
        let fx0 = ((bb.min.x - self.origin.x) / self.pitch).ceil();
        let fy0 = ((bb.min.y - self.origin.y) / self.pitch).ceil();
        let fx1 = ((bb.max.x - self.origin.x) / self.pitch).floor();
        let fy1 = ((bb.max.y - self.origin.y) / self.pitch).floor();
        if fx1 < 0.0 || fy1 < 0.0 || fx0 > (self.nx - 1) as f64 || fy0 > (self.ny - 1) as f64 || fx0 > fx1 || fy0 > fy1 {
            return None;
        }
        Some((
            fx0.max(0.0) as usize,
            fy0.max(0.0) as usize,
            (fx1 as usize).min(self.nx - 1),
            (fy1 as usize).min(self.ny - 1),
        ))
    }
    /// Nearest node to a point (clamped).
    pub fn nearest(&self, p: P2) -> (usize, usize) {
        let x = ((p.x - self.origin.x) / self.pitch).round().clamp(0.0, (self.nx - 1) as f64) as usize;
        let y = ((p.y - self.origin.y) / self.pitch).round().clamp(0.0, (self.ny - 1) as f64) as usize;
        (x, y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    Trace,
    Via,
}

/// A class of moving object whose legality we precompute.
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    pub kind: ProbeKind,
    /// Copper half-width (track) or radius (via).
    pub half: f64,
    pub clearance: f64,
    /// Via drill diameter (vias only).
    pub drill: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassProbes {
    pub trace: usize,
    pub via: usize,
    /// Narrow track probe used for neck-down next to fine-pitch pads.
    pub neck: usize,
}

/// Static legality maps for a board.
pub struct StaticMaps {
    pub grid: Grid,
    pub probes: Vec<Probe>,
    pub class_probes: Vec<ClassProbes>,
    /// Per probe: owner of each node (`FREE`, `BLOCKED`, or a net id).
    pub owner: Vec<Vec<u16>>,
    /// Per probe (traces only): owner map dilated by the comfort margin.
    pub comfort: Vec<Vec<u16>>,
    /// Per probe (vias only): 1 where a via would overlap pad copper (via-in-pad).
    pub via_in_pad: Vec<Vec<u8>>,
    /// Routable layers (from settings) and per-layer plane net.
    pub routable: Vec<bool>,
    pub plane_net: Vec<Option<NetId>>,
    /// Neck-down track width per net class.
    pub neck_w: Vec<f64>,
    /// Per plane layer: nodes where a via of the plane net lands fully inside the plane's
    /// copper (its zone fill on that layer, eroded by the via radius). `None` when the board has
    /// no zone for the plane net on that layer (the whole layer is then taken as copper).
    pub plane_cover: Vec<Option<Vec<bool>>>,
}

/// Nodes of one grid layer inside any of `polys` (even-odd over all rings), by scanline.
fn fill_mask(grid: &Grid, polys: &[&Polygon]) -> Vec<bool> {
    let mut mask = vec![false; grid.plane()];
    let mut xs: Vec<f64> = Vec::new();
    for y in 0..grid.ny {
        let py = grid.origin.y + y as f64 * grid.pitch;
        xs.clear();
        for poly in polys {
            for ring in std::iter::once(&poly.outline).chain(poly.holes.iter()) {
                let n = ring.len();
                for i in 0..n {
                    let (a, b) = (ring[i], ring[(i + 1) % n]);
                    if (a.y <= py && py < b.y) || (b.y <= py && py < a.y) {
                        xs.push(a.x + (py - a.y) * (b.x - a.x) / (b.y - a.y));
                    }
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in xs.chunks_exact(2) {
            let x0 = ((pair[0] - grid.origin.x) / grid.pitch).ceil().max(0.0) as usize;
            let x1 = ((pair[1] - grid.origin.x) / grid.pitch).floor();
            if x1 < 0.0 {
                continue;
            }
            for x in x0..=(x1 as usize).min(grid.nx - 1) {
                mask[y * grid.nx + x] = true;
            }
        }
    }
    mask
}

/// Square erosion by `r` nodes (conservative stand-in for a disc), separable.
fn erode(grid: &Grid, mask: &[bool], r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    let (nx, ny) = (grid.nx, grid.ny);
    let mut h = vec![false; mask.len()];
    for y in 0..ny {
        let row = &mask[y * nx..(y + 1) * nx];
        // run of consecutive true cells ending at x, from the left and from the right
        let mut left = vec![0usize; nx];
        let mut run = 0;
        for x in 0..nx {
            run = if row[x] { run + 1 } else { 0 };
            left[x] = run;
        }
        let mut run = 0;
        for x in (0..nx).rev() {
            run = if row[x] { run + 1 } else { 0 };
            h[y * nx + x] = row[x] && left[x] > r && run > r;
        }
    }
    let mut out = vec![false; mask.len()];
    for x in 0..nx {
        let mut up = vec![0usize; ny];
        let mut run = 0;
        for y in 0..ny {
            run = if h[y * nx + x] { run + 1 } else { 0 };
            up[y] = run;
        }
        let mut run = 0;
        for y in (0..ny).rev() {
            run = if h[y * nx + x] { run + 1 } else { 0 };
            out[y * nx + x] = h[y * nx + x] && up[y] > r && run > r;
        }
    }
    out
}

#[inline]
fn merge_owner(cur: u16, net: u16) -> u16 {
    if net == BLOCKED || cur == BLOCKED {
        BLOCKED
    } else if cur == FREE || cur == net {
        net
    } else {
        BLOCKED
    }
}

/// One static obstacle with its rasterisation parameters.
struct Item {
    shape: Shape,
    layers: Vec<usize>,
    /// Owner written into covered nodes.
    owner: u16,
    /// Clearance of the item itself (combined with the probe's by max).
    clearance: f64,
    /// Which probe kinds the item affects, and special rules.
    rule: ItemRule,
}

#[derive(Clone, Copy, PartialEq)]
enum ItemRule {
    /// Copper: required distance = max(c_item, c_probe) + probe half.
    Copper,
    /// Drilled hole: hole clearance for copper, hole-to-hole for vias (always blocks).
    Hole { hole_clear: f64, hole_to_hole: f64 },
    /// Keepout area: blocks tracks and/or vias whose copper would enter the area.
    Keepout { tracks: bool, vias: bool },
}

impl StaticMaps {
    pub fn build(board: &Board, settings: &Settings) -> StaticMaps {
        let nl = board.layers.len();
        let pitch = settings.effective_pitch(board);
        let bb = board.bbox().expand(pitch * 2.0);
        let grid = Grid::new(&bb, pitch, nl, settings.grid_offset(board, pitch));

        let neck_w = settings.neck_widths(board);
        // Probe classes, deduplicated.
        let mut probes: Vec<Probe> = Vec::new();
        let mut find_or_add = |p: Probe| -> usize {
            if let Some(i) = probes.iter().position(|q| {
                q.kind == p.kind && (q.half - p.half).abs() < 1e-9 && (q.clearance - p.clearance).abs() < 1e-9 && (q.drill - p.drill).abs() < 1e-9
            }) {
                i
            } else {
                probes.push(p);
                probes.len() - 1
            }
        };
        let class_probes: Vec<ClassProbes> = settings
            .netclasses
            .iter()
            .enumerate()
            .map(|(ci, nc)| ClassProbes {
                trace: find_or_add(Probe { kind: ProbeKind::Trace, half: nc.track_width / 2.0, clearance: nc.clearance, drill: 0.0 }),
                via: find_or_add(Probe { kind: ProbeKind::Via, half: nc.via_diameter / 2.0, clearance: nc.clearance, drill: nc.via_drill }),
                neck: find_or_add(Probe { kind: ProbeKind::Trace, half: neck_w[ci] / 2.0, clearance: nc.clearance, drill: 0.0 }),
            })
            .collect();

        let routable: Vec<bool> = (0..nl).map(|l| settings.layers.get(l).map(|c| c.routable && c.plane_net.is_none()).unwrap_or(true)).collect();
        let plane_net: Vec<Option<NetId>> = (0..nl)
            .map(|l| settings.layers.get(l).and_then(|c| c.plane_net.as_ref()).and_then(|n| board.net_by_name(n)))
            .collect();

        // Collect obstacles.
        let nc_clear = |net: NetId| settings.netclass(board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0)).clearance;
        let default_clear = settings.netclass(0).clearance;
        let mut items: Vec<Item> = Vec::new();
        for p in &board.pads {
            let owner = if p.net == 0 || p.kind == PadKind::Npth { BLOCKED } else { p.net.min(BLOCKED as u32 - 1) as u16 };
            let clearance = p.clearance.unwrap_or(if p.net == 0 { default_clear } else { nc_clear(p.net) });
            for s in &p.shapes {
                items.push(Item { shape: Shape::Polygon { poly: s.poly.clone() }, layers: vec![s.layer], owner, clearance, rule: ItemRule::Copper });
            }
            if let Some(d) = p.drill {
                if d > 0.0 {
                    items.push(Item {
                        shape: Shape::Circle { c: p.pos, r: d / 2.0 },
                        layers: (0..nl).collect(),
                        owner: if p.kind == PadKind::Npth { BLOCKED } else { owner },
                        clearance: 0.0,
                        // Plated holes keep hole clearance too: a THT pad may have no ring on
                        // some layers (unused pads removed) while its hole is still there.
                        rule: ItemRule::Hole { hole_clear: settings.hole_clearance, hole_to_hole: settings.hole_to_hole },
                    });
                }
            }
        }
        if settings.keep_existing {
            for t in &board.tracks {
                let owner = if t.net == 0 { BLOCKED } else { t.net as u16 };
                items.push(Item {
                    shape: Shape::Segment { a: t.a, b: t.b, r: t.width / 2.0 },
                    layers: vec![t.layer],
                    owner,
                    clearance: nc_clear(t.net),
                    rule: ItemRule::Copper,
                });
            }
            for v in &board.vias {
                let owner = if v.net == 0 { BLOCKED } else { v.net as u16 };
                items.push(Item {
                    shape: Shape::Circle { c: v.pos, r: v.diameter / 2.0 },
                    layers: (v.from_layer..=v.to_layer.min(nl - 1)).collect(),
                    owner,
                    clearance: nc_clear(v.net),
                    rule: ItemRule::Copper,
                });
                items.push(Item {
                    shape: Shape::Circle { c: v.pos, r: v.drill / 2.0 },
                    layers: (0..nl).collect(),
                    owner,
                    clearance: 0.0,
                    rule: ItemRule::Hole { hole_clear: 0.0, hole_to_hole: settings.hole_to_hole },
                });
            }
        }
        for o in &board.obstacles {
            items.push(Item {
                shape: o.shape.clone(),
                layers: vec![o.layer],
                owner: if o.net == 0 { BLOCKED } else { o.net as u16 },
                clearance: if o.net == 0 { default_clear } else { nc_clear(o.net) },
                rule: ItemRule::Copper,
            });
        }
        for k in &board.keepouts {
            items.push(Item {
                shape: Shape::Polygon { poly: k.poly.clone() },
                layers: k.layers.clone(),
                owner: BLOCKED,
                clearance: 0.0,
                rule: ItemRule::Keepout { tracks: k.no_tracks, vias: k.no_vias },
            });
        }
        // Board edge segments.
        let edge_r = settings.edge_clearance + board_edge_half_width(board);
        for o in &board.outline {
            for (a, b) in o.edges() {
                items.push(Item {
                    shape: Shape::Segment { a, b, r: 0.0 },
                    layers: (0..nl).collect(),
                    owner: BLOCKED,
                    clearance: edge_r,
                    rule: ItemRule::Keepout { tracks: true, vias: true },
                });
            }
        }

        let plane = grid.plane();
        let mut owner: Vec<Vec<u16>> = probes.iter().map(|_| vec![FREE; nl * plane]).collect();
        let mut comfort: Vec<Vec<u16>> =
            probes.iter().map(|p| if p.kind == ProbeKind::Trace { vec![FREE; nl * plane] } else { Vec::new() }).collect();
        let mut via_in_pad: Vec<Vec<u8>> =
            probes.iter().map(|p| if p.kind == ProbeKind::Via { vec![0u8; nl * plane] } else { Vec::new() }).collect();

        // Outside-of-board nodes are blocked for everything.
        if !board.outline.is_empty() {
            let inside = inside_mask(board, &grid);
            for m in owner.iter_mut() {
                for l in 0..nl {
                    let s = &mut m[l * plane..(l + 1) * plane];
                    s.par_iter_mut().zip(inside.par_iter()).for_each(|(o, &ins)| {
                        if !ins {
                            *o = BLOCKED;
                        }
                    });
                }
            }
        }

        // Rasterise items in parallel into coverage lists, then apply.
        let half_pitch2 = grid.pitch * grid.pitch / 4.0;
        let comfort_margin = settings.comfort_margin.max(0.0);
        let cover: Vec<Vec<(u8, u8, u32, u16)>> = items
            .par_iter()
            .map(|it| {
                let mut out: Vec<(u8, u8, u32, u16)> = Vec::new(); // (probe, map: 0 owner / 1 comfort / 2 via-in-pad, idx, owner)
                for (pi, pr) in probes.iter().enumerate() {
                    // Required distance from item to probe copper centre.
                    let (base, applies, owner_val) = match it.rule {
                        ItemRule::Copper => {
                            let mut r = it.clearance.max(pr.clearance) + pr.half;
                            if pr.kind == ProbeKind::Via {
                                // The via's drill also needs hole clearance to foreign copper.
                                r = r.max(settings.hole_clearance + pr.drill / 2.0);
                            }
                            (r, true, it.owner)
                        }
                        ItemRule::Hole { hole_clear, hole_to_hole } => match pr.kind {
                            ProbeKind::Via => {
                                let h2h = hole_to_hole + pr.drill / 2.0;
                                let hc = if hole_clear > 0.0 { hole_clear + pr.half } else { 0.0 };
                                (h2h.max(hc), true, BLOCKED)
                            }
                            ProbeKind::Trace => (hole_clear + pr.half, hole_clear > 0.0, it.owner),
                        },
                        ItemRule::Keepout { tracks, vias } => {
                            let applies = match pr.kind {
                                ProbeKind::Trace => tracks,
                                ProbeKind::Via => vias,
                            };
                            (it.clearance + pr.half, applies, BLOCKED)
                        }
                    };
                    if !applies {
                        continue;
                    }
                    let r = base + SAFETY;
                    let r_eff = if pr.kind == ProbeKind::Trace { (r * r + half_pitch2).sqrt() } else { r };
                    let r_comf = if pr.kind == ProbeKind::Trace && it.rule == ItemRule::Copper && it.owner != BLOCKED {
                        r_eff + comfort_margin
                    } else {
                        r_eff
                    };
                    let reach = r_comf.max(r_eff);
                    let Some((x0, y0, x1, y1)) = grid.range(&it.shape.bbox().expand(reach)) else { continue };
                    let pad_touch = pr.kind == ProbeKind::Via && it.rule == ItemRule::Copper && it.owner != BLOCKED;
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let d = it.shape.dist(grid.pos(x, y));
                            if d >= reach {
                                continue;
                            }
                            for &l in &it.layers {
                                if l >= nl {
                                    continue;
                                }
                                let idx = (l * plane + y * grid.nx + x) as u32;
                                if d < r_eff {
                                    out.push((pi as u8, 0, idx, owner_val));
                                }
                                if pr.kind == ProbeKind::Trace && d < r_comf {
                                    out.push((pi as u8, 1, idx, owner_val));
                                }
                                if pad_touch && d < pr.half + SAFETY {
                                    out.push((pi as u8, 2, idx, 1));
                                }
                            }
                        }
                    }
                }
                out
            })
            .collect();
        for list in cover {
            for (pi, map, idx, o) in list {
                let (pi, idx) = (pi as usize, idx as usize);
                match map {
                    0 => owner[pi][idx] = merge_owner(owner[pi][idx], o),
                    1 => comfort[pi][idx] = merge_owner(comfort[pi][idx], o),
                    _ => via_in_pad[pi][idx] = 1,
                }
            }
        }

        // Plane coverage from the plane net's zone fills (or outlines, if the board gave no fill).
        let plane_cover: Vec<Option<Vec<bool>>> = (0..nl)
            .map(|l| {
                let net = plane_net[l]?;
                let polys: Vec<&Polygon> = board.zones.iter().filter(|z| z.net == net && z.layers.contains(&l)).map(|z| &z.poly).collect();
                if polys.is_empty() {
                    return None;
                }
                let nc = settings.netclass(board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0));
                let r = ((nc.via_diameter / 2.0) / grid.pitch).ceil() as usize;
                Some(erode(&grid, &fill_mask(&grid, &polys), r))
            })
            .collect();

        StaticMaps { grid, probes, class_probes, owner, comfort, via_in_pad, routable, plane_net, neck_w, plane_cover }
    }

    /// True if a via of the plane net at plane-local node `p` lands on plane `layer`'s copper.
    #[inline]
    pub fn on_plane(&self, layer: usize, p: usize) -> bool {
        self.plane_cover.get(layer).and_then(|c| c.as_ref()).is_none_or(|m| m[p])
    }

    pub fn probes_for(&self, board: &Board, net: NetId) -> ClassProbes {
        let nc = board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0);
        self.class_probes[nc.min(self.class_probes.len() - 1)]
    }

    /// True when net `net` may put a track node at `idx`.
    #[inline]
    pub fn trace_ok(&self, probe: usize, idx: usize, net: NetId) -> bool {
        let o = self.owner[probe][idx];
        o == FREE || o as u32 == net
    }
}

fn board_edge_half_width(board: &Board) -> f64 {
    let _ = board;
    // KiCad measures edge clearance from the Edge.Cuts outline; allow for a typical 0.1 mm line.
    0.05
}

/// Per-node inside-board mask (one layer plane), by scanline crossings.
fn inside_mask(board: &Board, grid: &Grid) -> Vec<bool> {
    let edges: Vec<(P2, P2)> = board.outline.iter().flat_map(|o| o.edges()).collect();
    let mut mask = vec![false; grid.plane()];
    mask.par_chunks_mut(grid.nx).enumerate().for_each(|(y, row)| {
        let py = grid.origin.y + y as f64 * grid.pitch;
        let mut xs: Vec<f64> = edges
            .iter()
            .filter(|(a, b)| (a.y > py) != (b.y > py))
            .map(|(a, b)| a.x + (py - a.y) * (b.x - a.x) / (b.y - a.y))
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        let mut inside = false;
        for (x, m) in row.iter_mut().enumerate() {
            let px = grid.origin.x + x as f64 * grid.pitch;
            while k < xs.len() && xs[k] <= px {
                inside = !inside;
                k += 1;
            }
            *m = inside;
        }
    });
    mask
}

/// Grid nodes (layer, y, x) inside a pad on its layers that the pad's net may use.
pub fn pad_terminal_nodes(board: &Board, maps: &StaticMaps, pad: usize, probe: usize) -> Vec<usize> {
    let p = &board.pads[pad];
    let g = &maps.grid;
    let mut out = Vec::new();
    for s in &p.shapes {
        if !maps.routable[s.layer] && maps.plane_net[s.layer] != Some(p.net) {
            continue;
        }
        let Some((x0, y0, x1, y1)) = g.range(&s.poly.bbox()) else { continue };
        for y in y0..=y1 {
            for x in x0..=x1 {
                let q = g.pos(x, y);
                if s.poly.contains(q) {
                    let idx = g.idx(s.layer, y, x);
                    if maps.trace_ok(probe, idx, p.net) {
                        out.push(idx);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid10() -> Grid {
        Grid::new(&BBox { min: P2::new(0.0, 0.0), max: P2::new(10.0, 10.0) }, 1.0, 1, P2::new(0.0, 0.0))
    }

    #[test]
    fn scanline_fill_respects_holes() {
        let g = grid10();
        let mut poly = Polygon::rect(P2::new(0.5, 0.5), P2::new(8.5, 8.5));
        poly.holes.push(Polygon::rect(P2::new(3.5, 3.5), P2::new(5.5, 5.5)).outline);
        let m = fill_mask(&g, &[&poly]);
        let at = |x: usize, y: usize| m[y * g.nx + x];
        assert!(at(1, 1) && at(8, 8) && at(2, 6));
        assert!(!at(0, 0) && !at(9, 9), "outside the outline");
        assert!(!at(4, 4) && !at(5, 5), "inside the hole");
    }

    #[test]
    fn erosion_keeps_only_interior() {
        let g = grid10();
        let m = fill_mask(&g, &[&Polygon::rect(P2::new(0.5, 0.5), P2::new(8.5, 8.5))]);
        let e = erode(&g, &m, 2);
        let at = |x: usize, y: usize| e[y * g.nx + x];
        assert!(at(3, 3) && at(6, 6));
        assert!(!at(1, 4) && !at(7, 4) && !at(4, 2), "within 2 nodes of the edge");
    }
}
