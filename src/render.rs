//! SVG rendering of a board and its routes (for reports and visual checks).

use crate::board::Board;
use crate::drc::{Seg, ViaOut};
use crate::geom::Polygon;
use std::fmt::Write;

/// Layer colours, front to back (KiCad-like palette).
pub const LAYER_COLORS: [&str; 8] = ["#c83434", "#e0a030", "#3aa060", "#b050c0", "#20a0c0", "#d07090", "#8090d0", "#4d7fc4"];

pub fn layer_color(layer: usize, nl: usize) -> &'static str {
    if nl > 1 && layer == nl - 1 {
        return LAYER_COLORS[7];
    }
    LAYER_COLORS[layer.min(6)]
}

fn path(p: &Polygon) -> String {
    let mut s = String::new();
    for ring in p.rings() {
        for (i, v) in ring.iter().enumerate() {
            let _ = write!(s, "{}{:.4},{:.4}", if i == 0 { "M" } else { "L" }, v.x, v.y);
        }
        s.push('Z');
    }
    s
}

pub fn svg(board: &Board, segs: &[Seg], vias: &[ViaOut]) -> String {
    let bb = board.bbox().expand(1.0);
    let nl = board.layers.len();
    let mut s = String::new();
    let _ = write!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.3} {:.3} {:.3} {:.3}" width="{:.0}" height="{:.0}" style="background:#101418">"#,
        bb.min.x,
        bb.min.y,
        bb.width(),
        bb.height(),
        (bb.width() * 12.0).min(2400.0),
        (bb.height() * 12.0 * (bb.width() * 12.0).min(2400.0) / (bb.width() * 12.0)).max(100.0)
    );
    for o in &board.outline {
        let _ = write!(s, r##"<path d="{}" fill="#1a2028" stroke="#d0d060" stroke-width="0.1" fill-rule="evenodd"/>"##, path(o));
    }
    // Back layers first so the front is on top.
    for layer in (0..nl).rev() {
        let c = layer_color(layer, nl);
        let _ = write!(s, r#"<g opacity="{}">"#, if layer == 0 { "0.9" } else { "0.75" });
        for p in &board.pads {
            if let Some(poly) = p.shape_on(layer) {
                let _ = write!(s, r#"<path d="{}" fill="{}" fill-opacity="0.55"/>"#, path(poly), c);
            }
        }
        for t in board.tracks.iter().filter(|t| t.layer == layer) {
            let _ = write!(s, r#"<line x1="{:.4}" y1="{:.4}" x2="{:.4}" y2="{:.4}" stroke="{}" stroke-opacity="0.5" stroke-width="{:.4}" stroke-linecap="round"/>"#, t.a.x, t.a.y, t.b.x, t.b.y, c, t.width);
        }
        for t in segs.iter().filter(|t| t.layer == layer) {
            let _ = write!(s, r#"<line x1="{:.4}" y1="{:.4}" x2="{:.4}" y2="{:.4}" stroke="{}" stroke-width="{:.4}" stroke-linecap="round"/>"#, t.a.x, t.a.y, t.b.x, t.b.y, c, t.width);
        }
        s.push_str("</g>");
    }
    for p in &board.pads {
        if let Some(d) = p.drill {
            let _ = write!(s, r##"<circle cx="{:.4}" cy="{:.4}" r="{:.4}" fill="#101418"/>"##, p.pos.x, p.pos.y, d / 2.0);
        }
    }
    for v in board.vias.iter().map(|v| (v.pos, v.diameter, v.drill)).chain(vias.iter().map(|v| (v.pos, v.diameter, v.drill))) {
        let _ = write!(s, r##"<circle cx="{:.4}" cy="{:.4}" r="{:.4}" fill="#c8c8c8"/><circle cx="{:.4}" cy="{:.4}" r="{:.4}" fill="#101418"/>"##, v.0.x, v.0.y, v.1 / 2.0, v.0.x, v.0.y, v.2 / 2.0);
    }
    s.push_str("</svg>");
    s
}
