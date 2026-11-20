//! Board model: everything the router needs from a PCB, independent of KiCad.
//! All coordinates are millimetres in board space (KiCad's Y-down convention is kept).

use crate::geom::{BBox, P2, Polygon, Shape};
use serde::{Deserialize, Serialize};

/// Index into `Board::nets`. Net 0 is "no net".
pub type NetId = u32;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Net {
    pub name: String,
    /// KiCad net code (for write-back).
    pub code: i32,
    /// Index into `Board::netclasses`.
    pub netclass: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetClass {
    pub name: String,
    pub clearance: f64,
    pub track_width: f64,
    pub via_diameter: f64,
    pub via_drill: f64,
}

impl Default for NetClass {
    fn default() -> Self {
        NetClass { name: "Default".into(), clearance: 0.2, track_width: 0.25, via_diameter: 0.6, via_drill: 0.3 }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PadKind {
    Smd,
    Pth,
    Npth,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerShape {
    pub layer: usize,
    pub poly: Polygon,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pad {
    /// KiCad KIID (opaque).
    pub id: String,
    /// Human-readable name, e.g. "U3.14".
    pub name: String,
    pub net: NetId,
    pub kind: PadKind,
    pub pos: P2,
    /// Copper shape per copper layer the pad exists on.
    pub shapes: Vec<LayerShape>,
    /// Drill diameter (largest dimension for slots), if drilled.
    #[serde(default)]
    pub drill: Option<f64>,
    /// Pad-specific clearance override.
    #[serde(default)]
    pub clearance: Option<f64>,
}

impl Pad {
    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.shapes.iter().map(|s| s.layer)
    }
    pub fn shape_on(&self, layer: usize) -> Option<&Polygon> {
        self.shapes.iter().find(|s| s.layer == layer).map(|s| &s.poly)
    }
    pub fn bbox(&self) -> BBox {
        let mut b = BBox::empty();
        for s in &self.shapes {
            b = b.union(&s.poly.bbox());
        }
        if b.is_empty() {
            let r = self.drill.unwrap_or(0.1) / 2.0;
            b = BBox { min: P2::new(self.pos.x - r, self.pos.y - r), max: P2::new(self.pos.x + r, self.pos.y + r) };
        }
        b
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    #[serde(default)]
    pub id: String,
    pub net: NetId,
    pub layer: usize,
    pub a: P2,
    pub b: P2,
    pub width: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Via {
    #[serde(default)]
    pub id: String,
    pub net: NetId,
    pub pos: P2,
    pub diameter: f64,
    pub drill: f64,
    /// Inclusive copper layer span.
    pub from_layer: usize,
    pub to_layer: usize,
}

/// Rule area that forbids tracks and/or vias.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keepout {
    pub layers: Vec<usize>,
    pub poly: Polygon,
    pub no_tracks: bool,
    pub no_vias: bool,
}

/// Other copper that routes must keep clear of: copper text, graphic shapes on copper layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Obstacle {
    pub layer: usize,
    /// Net of the copper (0 = none: blocks every net).
    pub net: NetId,
    pub shape: Shape,
}

/// Copper zone (pour); used to detect power/ground planes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Zone {
    pub net: NetId,
    pub layers: Vec<usize>,
    pub poly: Polygon,
    #[serde(default)]
    pub name: String,
}

/// Board-level design rule minimums (from the KiCad project's board setup).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoardRules {
    pub min_clearance: f64,
    pub edge_clearance: f64,
    pub hole_clearance: f64,
    pub hole_to_hole: f64,
    pub min_track_width: f64,
    pub min_via_diameter: f64,
    pub min_via_drill: f64,
    #[serde(default)]
    pub min_via_annular: f64,
}

impl Default for BoardRules {
    fn default() -> Self {
        BoardRules {
            min_clearance: 0.0,
            edge_clearance: 0.5,
            hole_clearance: 0.25,
            hole_to_hole: 0.25,
            min_track_width: 0.0,
            min_via_diameter: 0.5,
            min_via_drill: 0.3,
            min_via_annular: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Board {
    pub name: String,
    /// Copper layer names, front to back (F.Cu, In1.Cu, ..., B.Cu).
    pub layers: Vec<String>,
    /// Board areas. Holes are cutouts.
    pub outline: Vec<Polygon>,
    pub nets: Vec<Net>,
    pub netclasses: Vec<NetClass>,
    pub pads: Vec<Pad>,
    #[serde(default)]
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub vias: Vec<Via>,
    #[serde(default)]
    pub keepouts: Vec<Keepout>,
    #[serde(default)]
    pub zones: Vec<Zone>,
    #[serde(default)]
    pub obstacles: Vec<Obstacle>,
    #[serde(default)]
    pub rules: BoardRules,
}

impl Board {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn netclass_of(&self, net: NetId) -> &NetClass {
        let i = self.nets.get(net as usize).map(|n| n.netclass).unwrap_or(0);
        &self.netclasses[i.min(self.netclasses.len().saturating_sub(1))]
    }

    pub fn net_name(&self, net: NetId) -> &str {
        self.nets.get(net as usize).map(|n| n.name.as_str()).unwrap_or("")
    }

    pub fn net_by_name(&self, name: &str) -> Option<NetId> {
        self.nets.iter().position(|n| n.name == name).map(|i| i as NetId)
    }

    /// Bounding box of the board (outline if present, otherwise all pads plus a margin).
    pub fn bbox(&self) -> BBox {
        let mut b = BBox::empty();
        for p in &self.outline {
            b = b.union(&p.bbox());
        }
        if b.is_empty() {
            for p in &self.pads {
                b = b.union(&p.bbox());
            }
            b = b.expand(2.0);
        }
        b
    }

    /// True if the point lies inside the board area (inside some outline, outside its cutouts).
    pub fn inside_outline(&self, p: P2) -> bool {
        self.outline.is_empty() || self.outline.iter().any(|o| o.contains(p))
    }

    /// Distance from a point to the board edge (outline boundaries including cutouts).
    pub fn edge_dist(&self, p: P2) -> f64 {
        self.outline.iter().map(|o| o.boundary_dist(p)).fold(f64::INFINITY, f64::min)
    }

    /// Basic validation and normalisation after import.
    pub fn normalize(&mut self) {
        if self.netclasses.is_empty() {
            self.netclasses.push(NetClass::default());
        }
        if self.nets.is_empty() {
            self.nets.push(Net { name: String::new(), code: 0, netclass: 0 });
        }
        let nc = self.netclasses.len();
        for n in &mut self.nets {
            if n.netclass >= nc {
                n.netclass = 0;
            }
        }
        let nl = self.layers.len();
        for p in &mut self.pads {
            p.shapes.retain(|s| s.layer < nl && s.poly.outline.len() >= 3);
        }
        self.tracks.retain(|t| t.layer < nl);
        self.obstacles.retain(|o| o.layer < nl);
        for v in &mut self.vias {
            v.to_layer = v.to_layer.min(nl.saturating_sub(1));
        }
    }

    /// Static copper shapes on a layer, with their net and clearance (for DRC and stub checks).
    pub fn static_copper(&self, layer: usize) -> Vec<(NetId, Shape, f64)> {
        let mut v = Vec::new();
        for p in &self.pads {
            if let Some(poly) = p.shape_on(layer) {
                let c = p.clearance.unwrap_or_else(|| self.netclass_of(p.net).clearance);
                v.push((p.net, Shape::Polygon { poly: poly.clone() }, c));
            }
        }
        for t in &self.tracks {
            if t.layer == layer {
                v.push((t.net, Shape::Segment { a: t.a, b: t.b, r: t.width / 2.0 }, self.netclass_of(t.net).clearance));
            }
        }
        for vi in &self.vias {
            if layer >= vi.from_layer && layer <= vi.to_layer {
                v.push((vi.net, Shape::Circle { c: vi.pos, r: vi.diameter / 2.0 }, self.netclass_of(vi.net).clearance));
            }
        }
        v
    }

    /// Number of routable connections per net (pads with a net, grouped by net).
    pub fn pads_by_net(&self) -> Vec<Vec<usize>> {
        let mut v = vec![Vec::new(); self.nets.len()];
        for (i, p) in self.pads.iter().enumerate() {
            if p.net != 0 && (p.net as usize) < v.len() && p.kind != PadKind::Npth {
                v[p.net as usize].push(i);
            }
        }
        v
    }
}
