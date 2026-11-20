//! Exact geometric checks, independent of the routing grid: copper connectivity
//! (union-find over touching copper) and clearance DRC for routed tracks and vias.

use crate::board::{Board, NetId, PadKind};
use crate::geom::{BBox, P2, Shape};
use crate::settings::Settings;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A routed track segment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Seg {
    pub net: NetId,
    pub layer: usize,
    pub a: P2,
    pub b: P2,
    pub width: f64,
}

/// A routed via.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ViaOut {
    pub net: NetId,
    pub pos: P2,
    pub diameter: f64,
    pub drill: f64,
    pub from_layer: usize,
    pub to_layer: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Pad,
    Track,
    Via,
    NewTrack,
    NewVia,
    Hole,
    Edge,
    Keepout,
    /// Copper text or graphics.
    Graphic,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub kind: ItemKind,
    pub net: NetId,
    /// Bitmask of copper layers.
    pub layers: u64,
    pub shape: Shape,
    pub clearance: f64,
    /// Index into the source collection (pads, tracks, new segments...).
    pub src: usize,
    pub no_tracks: bool,
    pub no_vias: bool,
}

/// Uniform spatial hash over item bounding boxes.
pub struct Index {
    pub items: Vec<Item>,
    cell: f64,
    map: HashMap<(i32, i32), Vec<u32>>,
}

impl Index {
    pub fn new(items: Vec<Item>, cell: f64) -> Index {
        let mut map: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (i, it) in items.iter().enumerate() {
            let b = it.shape.bbox();
            for k in Self::keys(&b, cell) {
                map.entry(k).or_default().push(i as u32);
            }
        }
        Index { items, cell, map }
    }

    fn keys(b: &BBox, cell: f64) -> impl Iterator<Item = (i32, i32)> {
        let (x0, y0) = ((b.min.x / cell).floor() as i32, (b.min.y / cell).floor() as i32);
        let (x1, y1) = ((b.max.x / cell).floor() as i32, (b.max.y / cell).floor() as i32);
        (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| (x, y)))
    }

    /// Indices of items whose bbox intersects `b`.
    pub fn query(&self, b: &BBox) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for k in Self::keys(b, self.cell) {
            if let Some(v) = self.map.get(&k) {
                for &i in v {
                    if self.items[i as usize].shape.bbox().intersects(b) {
                        out.push(i);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn layer_bits(range: impl IntoIterator<Item = usize>) -> u64 {
    range.into_iter().fold(0u64, |m, l| m | (1u64 << l))
}

/// All copper and obstacles of a board plus routed items.
pub fn collect_items(board: &Board, settings: &Settings, segs: &[Seg], vias: &[ViaOut], include_existing: bool) -> Vec<Item> {
    let nl = board.layers.len();
    let all = layer_bits(0..nl);
    let clr = |net: NetId| settings.netclass(board.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0)).clearance;
    let mut items = Vec::new();
    let base = |kind, net, layers, shape, clearance, src| Item { kind, net, layers, shape, clearance, src, no_tracks: false, no_vias: false };
    for (i, p) in board.pads.iter().enumerate() {
        let c = p.clearance.unwrap_or_else(|| clr(p.net));
        let net = if p.kind == PadKind::Npth { 0 } else { p.net };
        for s in &p.shapes {
            items.push(base(ItemKind::Pad, net, 1u64 << s.layer, Shape::Polygon { poly: s.poly.clone() }, c, i));
        }
        if p.kind == PadKind::Npth {
            if let Some(d) = p.drill {
                items.push(base(ItemKind::Hole, 0, all, Shape::Circle { c: p.pos, r: d / 2.0 }, settings.hole_clearance, i));
            }
        }
    }
    if include_existing {
        for (i, t) in board.tracks.iter().enumerate() {
            items.push(base(ItemKind::Track, t.net, 1u64 << t.layer, Shape::Segment { a: t.a, b: t.b, r: t.width / 2.0 }, clr(t.net), i));
        }
        for (i, v) in board.vias.iter().enumerate() {
            items.push(base(ItemKind::Via, v.net, layer_bits(v.from_layer..=v.to_layer.min(nl - 1)), Shape::Circle { c: v.pos, r: v.diameter / 2.0 }, clr(v.net), i));
        }
    }
    for (i, s) in segs.iter().enumerate() {
        items.push(base(ItemKind::NewTrack, s.net, 1u64 << s.layer, Shape::Segment { a: s.a, b: s.b, r: s.width / 2.0 }, clr(s.net), i));
    }
    for (i, v) in vias.iter().enumerate() {
        items.push(base(ItemKind::NewVia, v.net, layer_bits(v.from_layer..=v.to_layer), Shape::Circle { c: v.pos, r: v.diameter / 2.0 }, clr(v.net), i));
    }
    for (i, o) in board.obstacles.iter().enumerate() {
        let c = if o.net == 0 { settings.netclass(0).clearance } else { clr(o.net) };
        items.push(base(ItemKind::Graphic, o.net, 1u64 << o.layer, o.shape.clone(), c, i));
    }
    for (i, o) in board.outline.iter().enumerate() {
        for (a, b) in o.edges() {
            items.push(base(ItemKind::Edge, 0, all, Shape::Segment { a, b, r: 0.0 }, settings.edge_clearance, i));
        }
    }
    for (i, k) in board.keepouts.iter().enumerate() {
        let mut it = base(ItemKind::Keepout, 0, layer_bits(k.layers.iter().copied()), Shape::Polygon { poly: k.poly.clone() }, 0.0, i);
        it.no_tracks = k.no_tracks;
        it.no_vias = k.no_vias;
        items.push(it);
    }
    items
}

fn is_copper(k: ItemKind) -> bool {
    matches!(k, ItemKind::Pad | ItemKind::Track | ItemKind::Via | ItemKind::NewTrack | ItemKind::NewVia)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Violation {
    pub kind: String,
    pub net_a: NetId,
    pub net_b: NetId,
    pub layer: usize,
    pub at: P2,
    pub actual: f64,
    pub required: f64,
}

/// Clearance violations involving routed (new) items. `tol` absorbs float noise.
pub fn check_clearance(board: &Board, settings: &Settings, segs: &[Seg], vias: &[ViaOut], tol: f64) -> Vec<Violation> {
    let items = collect_items(board, settings, segs, vias, settings.keep_existing);
    let index = Index::new(items, 2.0);
    let mut out = Vec::new();
    let nl = board.layers.len();
    for (i, it) in index.items.iter().enumerate() {
        if !matches!(it.kind, ItemKind::NewTrack | ItemKind::NewVia) {
            continue;
        }
        let reach = it.clearance.max(settings.edge_clearance).max(settings.hole_clearance) + 1.0;
        for j in index.query(&it.shape.bbox().expand(reach)) {
            let j = j as usize;
            if j == i {
                continue;
            }
            let o = &index.items[j];
            let common = it.layers & o.layers;
            if common == 0 {
                continue;
            }
            // Each pair of new items is checked once.
            if matches!(o.kind, ItemKind::NewTrack | ItemKind::NewVia) && j < i {
                continue;
            }
            let layer = common.trailing_zeros() as usize;
            let (kind, required) = match o.kind {
                ItemKind::Edge => ("edge_clearance", settings.edge_clearance),
                ItemKind::Hole => ("hole_clearance", settings.hole_clearance),
                ItemKind::Keepout => {
                    let forbidden = (it.kind == ItemKind::NewTrack && o.no_tracks) || (it.kind == ItemKind::NewVia && o.no_vias);
                    if !forbidden {
                        continue;
                    }
                    ("keepout", 0.0)
                }
                k if is_copper(k) || k == ItemKind::Graphic => {
                    if o.net == it.net && it.net != 0 {
                        continue;
                    }
                    ("clearance", it.clearance.max(o.clearance))
                }
                _ => continue,
            };
            let d = it.shape.shape_dist(&o.shape);
            let violated = if kind == "keepout" { d <= 0.0 } else { d + tol < required };
            if violated {
                let at = it.shape.bbox().center();
                out.push(Violation { kind: kind.into(), net_a: it.net, net_b: o.net, layer: layer.min(nl.saturating_sub(1)), at, actual: d, required });
            }
        }
    }
    out
}

/// Union-find.
pub struct Dsu(Vec<usize>);
impl Dsu {
    pub fn new(n: usize) -> Self {
        Dsu((0..n).collect())
    }
    pub fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut c = x;
        while self.0[c] != r {
            let n = self.0[c];
            self.0[c] = r;
            c = n;
        }
        r
    }
    pub fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a] = b;
        }
    }
}

/// Connected copper components among the copper items. Two items connect when they share a
/// layer and touch. Items on a plane layer of their own net connect through the plane.
pub fn copper_components(items: &[Item], plane_layers: &[(usize, NetId)]) -> Dsu {
    let copper: Vec<usize> = (0..items.len()).filter(|&i| is_copper(items[i].kind) && items[i].net != 0).collect();
    let mut dsu = Dsu::new(items.len() + plane_layers.len());
    // The per-layer shapes of one pad are a single conductor.
    let mut first_of_pad: HashMap<usize, usize> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        if it.kind == ItemKind::Pad {
            match first_of_pad.get(&it.src) {
                Some(&f) => dsu.union(i, f),
                None => {
                    first_of_pad.insert(it.src, i);
                }
            }
        }
    }
    let sub: Vec<Item> = copper.iter().map(|&i| items[i].clone()).collect();
    let index = Index::new(sub, 2.0);
    for (k, it) in index.items.iter().enumerate() {
        for j in index.query(&it.shape.bbox().expand(1e-3)) {
            let j = j as usize;
            if j <= k {
                continue;
            }
            let o = &index.items[j];
            if o.net != it.net || it.layers & o.layers == 0 {
                continue;
            }
            if it.shape.shape_dist(&o.shape) <= 1e-4 {
                dsu.union(copper[k], copper[j]);
            }
        }
        for (pi, &(pl, pnet)) in plane_layers.iter().enumerate() {
            if it.net == pnet && it.layers & (1u64 << pl) != 0 {
                dsu.union(copper[k], items.len() + pi);
            }
        }
    }
    dsu
}

/// Per net: number of pad groups that are still disconnected (0 = fully connected).
pub fn unconnected_per_net(board: &Board, settings: &Settings, segs: &[Seg], vias: &[ViaOut], plane_layers: &[(usize, NetId)]) -> HashMap<NetId, usize> {
    let items = collect_items(board, settings, segs, vias, true);
    let mut dsu = copper_components(&items, plane_layers);
    let mut roots: HashMap<NetId, Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        if it.kind == ItemKind::Pad && it.net != 0 {
            let r = dsu.find(i);
            roots.entry(it.net).or_default().push(r);
        }
    }
    roots
        .into_iter()
        .map(|(net, mut r)| {
            r.sort_unstable();
            r.dedup();
            (net, r.len() - 1)
        })
        .collect()
}
