//! 2D geometry in millimetres: points, polygons, capsules and exact distance queries.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct P2 {
    pub x: f64,
    pub y: f64,
}

impl P2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn dot(self, o: P2) -> f64 {
        self.x * o.x + self.y * o.y
    }
    pub fn cross(self, o: P2) -> f64 {
        self.x * o.y - self.y * o.x
    }
    pub fn len2(self) -> f64 {
        self.dot(self)
    }
    pub fn len(self) -> f64 {
        self.len2().sqrt()
    }
    pub fn dist(self, o: P2) -> f64 {
        (self - o).len()
    }
}

impl Add for P2 {
    type Output = P2;
    fn add(self, o: P2) -> P2 {
        P2::new(self.x + o.x, self.y + o.y)
    }
}
impl Sub for P2 {
    type Output = P2;
    fn sub(self, o: P2) -> P2 {
        P2::new(self.x - o.x, self.y - o.y)
    }
}
impl Mul<f64> for P2 {
    type Output = P2;
    fn mul(self, s: f64) -> P2 {
        P2::new(self.x * s, self.y * s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub min: P2,
    pub max: P2,
}

impl BBox {
    pub fn empty() -> Self {
        BBox {
            min: P2::new(f64::INFINITY, f64::INFINITY),
            max: P2::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn from_points<'a>(pts: impl IntoIterator<Item = &'a P2>) -> Self {
        let mut b = BBox::empty();
        for p in pts {
            b.add_point(*p);
        }
        b
    }
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }
    pub fn add_point(&mut self, p: P2) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
    }
    pub fn union(&self, o: &BBox) -> BBox {
        BBox {
            min: P2::new(self.min.x.min(o.min.x), self.min.y.min(o.min.y)),
            max: P2::new(self.max.x.max(o.max.x), self.max.y.max(o.max.y)),
        }
    }
    pub fn expand(&self, d: f64) -> BBox {
        BBox {
            min: P2::new(self.min.x - d, self.min.y - d),
            max: P2::new(self.max.x + d, self.max.y + d),
        }
    }
    pub fn intersects(&self, o: &BBox) -> bool {
        self.min.x <= o.max.x && o.min.x <= self.max.x && self.min.y <= o.max.y && o.min.y <= self.max.y
    }
    pub fn contains(&self, p: P2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
    pub fn center(&self) -> P2 {
        (self.min + self.max) * 0.5
    }
}

/// Squared distance from point `p` to segment `a`-`b`.
pub fn point_seg_dist2(p: P2, a: P2, b: P2) -> f64 {
    let ab = b - a;
    let l2 = ab.len2();
    if l2 <= 1e-24 {
        return (p - a).len2();
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    (p - (a + ab * t)).len2()
}

pub fn point_seg_dist(p: P2, a: P2, b: P2) -> f64 {
    point_seg_dist2(p, a, b).sqrt()
}

fn orient(a: P2, b: P2, c: P2) -> f64 {
    (b - a).cross(c - a)
}

/// True when closed segments `a`-`b` and `c`-`d` intersect (including touching).
pub fn segments_intersect(a: P2, b: P2, c: P2, d: P2) -> bool {
    let d1 = orient(c, d, a);
    let d2 = orient(c, d, b);
    let d3 = orient(a, b, c);
    let d4 = orient(a, b, d);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    let on = |p: P2, q: P2, r: P2, o: f64| {
        o.abs() <= 1e-12 && r.x >= p.x.min(q.x) - 1e-12 && r.x <= p.x.max(q.x) + 1e-12 && r.y >= p.y.min(q.y) - 1e-12 && r.y <= p.y.max(q.y) + 1e-12
    };
    on(c, d, a, d1) || on(c, d, b, d2) || on(a, b, c, d3) || on(a, b, d, d4)
}

/// Distance between segments `a`-`b` and `c`-`d` (0 when they intersect).
pub fn seg_seg_dist(a: P2, b: P2, c: P2, d: P2) -> f64 {
    if segments_intersect(a, b, c, d) {
        return 0.0;
    }
    point_seg_dist2(a, c, d)
        .min(point_seg_dist2(b, c, d))
        .min(point_seg_dist2(c, a, b))
        .min(point_seg_dist2(d, a, b))
        .sqrt()
}

/// Even-odd point-in-ring test.
pub fn point_in_ring(p: P2, ring: &[P2]) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (pi, pj) = (ring[i], ring[j]);
        if (pi.y > p.y) != (pj.y > p.y) {
            let x = pj.x + (p.y - pj.y) * (pi.x - pj.x) / (pi.y - pj.y);
            if p.x < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn ring_edges(ring: &[P2]) -> impl Iterator<Item = (P2, P2)> + '_ {
    let n = ring.len();
    (0..n).map(move |i| (ring[i], ring[(i + 1) % n]))
}

/// Polygon with optional holes. Rings are implicitly closed.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Polygon {
    pub outline: Vec<P2>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub holes: Vec<Vec<P2>>,
}

impl Polygon {
    pub fn new(outline: Vec<P2>) -> Self {
        Polygon { outline, holes: vec![] }
    }

    pub fn rect(min: P2, max: P2) -> Self {
        Polygon::new(vec![min, P2::new(max.x, min.y), max, P2::new(min.x, max.y)])
    }

    pub fn circle(c: P2, r: f64, segments: usize) -> Self {
        let n = segments.max(8);
        Polygon::new(
            (0..n)
                .map(|i| {
                    let a = i as f64 / n as f64 * std::f64::consts::TAU;
                    P2::new(c.x + r * a.cos(), c.y + r * a.sin())
                })
                .collect(),
        )
    }

    pub fn bbox(&self) -> BBox {
        BBox::from_points(self.outline.iter())
    }

    pub fn rings(&self) -> impl Iterator<Item = &Vec<P2>> {
        std::iter::once(&self.outline).chain(self.holes.iter())
    }

    pub fn edges(&self) -> impl Iterator<Item = (P2, P2)> + '_ {
        self.rings().flat_map(|r| ring_edges(r))
    }

    pub fn contains(&self, p: P2) -> bool {
        point_in_ring(p, &self.outline) && !self.holes.iter().any(|h| point_in_ring(p, h))
    }

    /// Distance from `p` to the polygon boundary.
    pub fn boundary_dist(&self, p: P2) -> f64 {
        let mut best = f64::INFINITY;
        for (a, b) in self.edges() {
            best = best.min(point_seg_dist2(p, a, b));
        }
        best.sqrt()
    }

    /// Distance from `p` to the filled polygon (0 inside).
    pub fn dist(&self, p: P2) -> f64 {
        if self.contains(p) { 0.0 } else { self.boundary_dist(p) }
    }

    /// Signed distance: negative inside.
    pub fn signed_dist(&self, p: P2) -> f64 {
        let d = self.boundary_dist(p);
        if self.contains(p) { -d } else { d }
    }

    /// Distance from segment `a`-`b` to the filled polygon (0 if they overlap).
    pub fn seg_dist(&self, a: P2, b: P2) -> f64 {
        if self.contains(a) || self.contains(b) {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        for (c, d) in self.edges() {
            best = best.min(seg_seg_dist(a, b, c, d));
            if best == 0.0 {
                break;
            }
        }
        best
    }

    pub fn area(&self) -> f64 {
        let ring_area = |r: &Vec<P2>| {
            let mut s = 0.0;
            for (a, b) in ring_edges(r) {
                s += a.cross(b);
            }
            (s * 0.5).abs()
        };
        ring_area(&self.outline) - self.holes.iter().map(ring_area).sum::<f64>()
    }

    pub fn centroid(&self) -> P2 {
        let mut a = 0.0;
        let mut c = P2::default();
        for (p, q) in ring_edges(&self.outline) {
            let w = p.cross(q);
            a += w;
            c = c + (p + q) * w;
        }
        if a.abs() < 1e-18 {
            return self.bbox().center();
        }
        c * (1.0 / (3.0 * a))
    }
}

/// A copper or obstacle shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    Circle { c: P2, r: f64 },
    /// Capsule: segment with round caps, `r` = half width.
    Segment { a: P2, b: P2, r: f64 },
    Polygon { poly: Polygon },
}

impl Shape {
    pub fn bbox(&self) -> BBox {
        match self {
            Shape::Circle { c, r } => BBox { min: P2::new(c.x - r, c.y - r), max: P2::new(c.x + r, c.y + r) },
            Shape::Segment { a, b, r } => BBox::from_points([*a, *b].iter()).expand(*r),
            Shape::Polygon { poly } => poly.bbox(),
        }
    }

    /// Distance from a point to the filled shape (0 inside).
    pub fn dist(&self, p: P2) -> f64 {
        match self {
            Shape::Circle { c, r } => (p.dist(*c) - r).max(0.0),
            Shape::Segment { a, b, r } => (point_seg_dist(p, *a, *b) - r).max(0.0),
            Shape::Polygon { poly } => poly.dist(p),
        }
    }

    /// Signed distance (negative inside); approximate for nothing - exact for all variants.
    pub fn signed_dist(&self, p: P2) -> f64 {
        match self {
            Shape::Circle { c, r } => p.dist(*c) - r,
            Shape::Segment { a, b, r } => point_seg_dist(p, *a, *b) - r,
            Shape::Polygon { poly } => poly.signed_dist(p),
        }
    }

    /// Distance from segment a-b (zero width) to the filled shape.
    pub fn seg_dist(&self, a: P2, b: P2) -> f64 {
        match self {
            Shape::Circle { c, r } => (point_seg_dist(*c, a, b) - r).max(0.0),
            Shape::Segment { a: c, b: d, r } => (seg_seg_dist(a, b, *c, *d) - r).max(0.0),
            Shape::Polygon { poly } => poly.seg_dist(a, b),
        }
    }

    /// Distance between two shapes (0 if overlapping).
    pub fn shape_dist(&self, o: &Shape) -> f64 {
        match (self, o) {
            (Shape::Circle { c, r }, _) => (o.dist(*c) - r).max(0.0),
            (_, Shape::Circle { c, r }) => (self.dist(*c) - r).max(0.0),
            (Shape::Segment { a, b, r }, _) => (o.seg_dist(*a, *b) - r).max(0.0),
            (_, Shape::Segment { a, b, r }) => (self.seg_dist(*a, *b) - r).max(0.0),
            (Shape::Polygon { poly: p }, Shape::Polygon { poly: q }) => {
                if p.outline.first().is_some_and(|v| q.contains(*v)) || q.outline.first().is_some_and(|v| p.contains(*v)) {
                    return 0.0;
                }
                let mut best = f64::INFINITY;
                for (a, b) in p.edges() {
                    for (c, d) in q.edges() {
                        best = best.min(seg_seg_dist(a, b, c, d));
                    }
                }
                best
            }
        }
    }

    pub fn contains(&self, p: P2) -> bool {
        self.signed_dist(p) <= 0.0
    }
}

/// Points along the circular arc start→mid→end, inclusive of both ends, with chord error <= `tol`.
pub fn arc_points(start: P2, mid: P2, end: P2, tol: f64) -> Vec<P2> {
    // Circumcircle of the three points.
    let (ax, ay, bx, by, cx, cy) = (start.x, start.y, mid.x, mid.y, end.x, end.y);
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        return vec![start, end];
    }
    let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    let center = P2::new(ux, uy);
    let r = center.dist(start);
    let ang = |p: P2| (p.y - uy).atan2(p.x - ux);
    let a0 = ang(start);
    let mut am = ang(mid);
    let mut a1 = ang(end);
    // Determine sweep direction so that mid lies between start and end.
    let norm = |mut a: f64, base: f64| {
        while a < base {
            a += std::f64::consts::TAU;
        }
        while a >= base + std::f64::consts::TAU {
            a -= std::f64::consts::TAU;
        }
        a
    };
    am = norm(am, a0);
    a1 = norm(a1, a0);
    let sweep = if am <= a1 { a1 - a0 } else { a1 - a0 - std::f64::consts::TAU };
    let step = if r > tol { 2.0 * (1.0 - tol / r).clamp(-1.0, 1.0).acos() } else { 0.5 };
    let n = ((sweep.abs() / step.max(1e-3)).ceil() as usize).clamp(2, 720);
    (0..=n)
        .map(|i| {
            let a = a0 + sweep * i as f64 / n as f64;
            P2::new(ux + r * a.cos(), uy + r * a.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seg_distances() {
        let d = seg_seg_dist(P2::new(0., 0.), P2::new(2., 0.), P2::new(1., 1.), P2::new(1., 3.));
        assert!((d - 1.0).abs() < 1e-12);
        assert_eq!(seg_seg_dist(P2::new(0., 0.), P2::new(2., 2.), P2::new(0., 2.), P2::new(2., 0.)), 0.0);
    }

    #[test]
    fn polygon_queries() {
        let mut p = Polygon::rect(P2::new(0., 0.), P2::new(4., 4.));
        p.holes.push(Polygon::rect(P2::new(1., 1.), P2::new(3., 3.)).outline);
        assert!(p.contains(P2::new(0.5, 0.5)));
        assert!(!p.contains(P2::new(2., 2.)));
        assert!((p.signed_dist(P2::new(2., 2.)) - 1.0).abs() < 1e-12);
        assert!((p.area() - 12.0).abs() < 1e-12);
        assert!((p.seg_dist(P2::new(5., 0.), P2::new(5., 4.)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn arc_flattening() {
        let pts = arc_points(P2::new(1., 0.), P2::new(0., 1.), P2::new(-1., 0.), 0.001);
        assert!(pts.len() > 10);
        for p in &pts {
            assert!((p.len() - 1.0).abs() < 1e-9);
            assert!(p.y >= -1e-9);
        }
    }
}
