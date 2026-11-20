//! KiCad IPC API client: read the open board into the router's model and commit routes back.

pub mod nng;

use crate::board::{Board, BoardRules, Keepout, LayerShape, Net, NetClass, Obstacle, Pad, PadKind, Track, Via, Zone};
use crate::geom::{P2, Polygon, Shape, arc_points, point_in_ring};
use crate::proto::kiapi::board::commands as bcmd;
use crate::proto::kiapi::board::types as bt;
use crate::proto::kiapi::common::commands as ccmd;
use crate::proto::kiapi::common::types as ct;
use crate::proto::kiapi::common::{ApiRequest, ApiRequestHeader, ApiResponse, ApiStatusCode};
use crate::router::RouteResult;
use anyhow::{Context, Result, anyhow, bail};
use prost::{Message, Name};
use std::collections::HashMap;
use std::path::PathBuf;

const NM: f64 = 1e-6;

/// What the client needs to map the router's indices back to KiCad.
#[derive(Clone, Debug)]
pub struct BoardCtx {
    pub doc: ct::DocumentSpecifier,
    /// KiCad `BoardLayer` value per router layer index.
    pub layer_ids: Vec<i32>,
    /// (net code, net name) per router `NetId`.
    pub nets: Vec<(i32, String)>,
    pub project_file: Option<PathBuf>,
    pub board_file: Option<String>,
}

pub struct Kicad {
    sock: nng::ReqSocket,
    token: String,
    client: String,
}

fn default_socket() -> String {
    if let Ok(s) = std::env::var("KICAD_API_SOCKET") {
        return s;
    }
    if cfg!(windows) {
        format!("ipc://{}\\kicad\\api.sock", std::env::temp_dir().display().to_string().trim_end_matches('\\'))
    } else if cfg!(target_os = "macos") {
        // KiCad uses /tmp on macOS, not $TMPDIR (whose long path would exceed the socket limit).
        "ipc:///tmp/kicad/api.sock".to_string()
    } else {
        // Elsewhere KiCad uses wxStandardPaths::GetTempDir(): $TMPDIR, else /tmp.
        format!("ipc://{}/kicad/api.sock", std::env::temp_dir().display().to_string().trim_end_matches('/'))
    }
}

fn v2(p: &Option<ct::Vector2>) -> P2 {
    p.as_ref().map(|v| P2::new(v.x_nm as f64 * NM, v.y_nm as f64 * NM)).unwrap_or_default()
}

fn vec2(p: P2) -> ct::Vector2 {
    ct::Vector2 { x_nm: (p.x / NM).round() as i64, y_nm: (p.y / NM).round() as i64 }
}

fn dist(d: &Option<ct::Distance>) -> Option<f64> {
    d.as_ref().map(|d| d.value_nm as f64 * NM)
}

/// Flatten a KiCad polyline (points and arcs) into points.
fn polyline(pl: &ct::PolyLine) -> Vec<P2> {
    let mut out: Vec<P2> = Vec::new();
    for n in &pl.nodes {
        match &n.geometry {
            Some(ct::poly_line_node::Geometry::Point(p)) => out.push(v2(&Some(*p))),
            Some(ct::poly_line_node::Geometry::Arc(a)) => {
                let pts = arc_points(v2(&a.start), v2(&a.mid), v2(&a.end), 0.002);
                for (i, p) in pts.into_iter().enumerate() {
                    if i == 0 && out.last().is_some_and(|q| q.dist(p) < 1e-6) {
                        continue;
                    }
                    out.push(p);
                }
            }
            None => {}
        }
    }
    if out.len() > 1 && out[0].dist(*out.last().unwrap()) < 1e-6 {
        out.pop();
    }
    out
}

fn polygon(p: &ct::PolygonWithHoles) -> Polygon {
    Polygon { outline: p.outline.as_ref().map(polyline).unwrap_or_default(), holes: p.holes.iter().map(polyline).filter(|h| h.len() >= 3).collect() }
}

impl Kicad {
    pub fn connect(client_name: &str) -> Result<Kicad> {
        let url = default_socket();
        let sock = nng::ReqSocket::connect(&url).with_context(|| {
            "could not connect to KiCad. Is the PCB editor open, and is Preferences > Plugins > \"Enable KiCad API\" turned on?".to_string()
        })?;
        let token = std::env::var("KICAD_API_TOKEN").unwrap_or_default();
        Ok(Kicad { sock, token, client: format!("{client_name}-{}", std::process::id()) })
    }

    fn send<Q: Message + Name, R: Message + Name + Default>(&mut self, req: &Q) -> Result<R> {
        let env = ApiRequest {
            header: Some(ApiRequestHeader { kicad_token: self.token.clone(), client_name: self.client.clone() }),
            message: Some(prost_types::Any { type_url: Q::type_url(), value: req.encode_to_vec() }),
        };
        let bytes = env.encode_to_vec();
        // KiCad answers AS_BUSY while it is doing something else (e.g. refilling zones).
        let start = std::time::Instant::now();
        let (resp, status) = loop {
            let reply = self.sock.request(&bytes)?;
            let resp = ApiResponse::decode(reply.as_slice()).context("decode API response")?;
            let status = resp.status.clone().unwrap_or_default();
            if status.status == ApiStatusCode::AsBusy as i32 && start.elapsed() < std::time::Duration::from_secs(120) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            break (resp, status);
        };
        if status.status != ApiStatusCode::AsOk as i32 {
            let code = ApiStatusCode::try_from(status.status).map(|c| c.as_str_name()).unwrap_or("?");
            bail!("KiCad API error {code} for {}: {}", Q::NAME, status.error_message);
        }
        if self.token.is_empty() {
            if let Some(h) = &resp.header {
                self.token = h.kicad_token.clone();
            }
        }
        let any = resp.message.ok_or_else(|| anyhow!("empty reply to {}", Q::NAME))?;
        R::decode(any.value.as_slice()).with_context(|| format!("decode {}", R::NAME))
    }

    pub fn version(&mut self) -> Result<String> {
        let r: ccmd::GetVersionResponse = self.send(&ccmd::GetVersion {})?;
        Ok(r.version.map(|v| v.full_version).unwrap_or_default())
    }

    pub fn plugin_settings_path(&mut self, id: &str) -> Result<String> {
        let r: ccmd::StringResponse = self.send(&ccmd::GetPluginSettingsPath { identifier: id.into() })?;
        Ok(r.response)
    }

    fn get_items(&mut self, doc: &ct::DocumentSpecifier, types: &[ct::KiCadObjectType]) -> Result<Vec<prost_types::Any>> {
        let r: ccmd::GetItemsResponse = self.send(&ccmd::GetItems {
            header: Some(ct::ItemHeader { document: Some(doc.clone()), container: None, field_mask: None }),
            types: types.iter().map(|t| *t as i32).collect(),
        })?;
        Ok(r.items)
    }

    /// Read the open PCB.
    pub fn read_board(&mut self) -> Result<(Board, BoardCtx)> {
        let docs: ccmd::GetOpenDocumentsResponse = self.send(&ccmd::GetOpenDocuments { r#type: ct::DocumentType::DoctypePcb as i32 })?;
        let doc = docs.documents.into_iter().next().ok_or_else(|| anyhow!("no PCB is open in KiCad"))?;
        let board_file = match &doc.identifier {
            Some(ct::document_specifier::Identifier::BoardFilename(f)) => Some(f.clone()),
            _ => None,
        };

        // Copper layers, front to back.
        let en: bcmd::BoardEnabledLayersResponse = self.send(&bcmd::GetBoardEnabledLayers { board: Some(doc.clone()) })?;
        let f = bt::BoardLayer::BlFCu as i32;
        let b = bt::BoardLayer::BlBCu as i32;
        let mut cu: Vec<i32> = en.layers.iter().copied().filter(|&l| l > f && l < b).collect();
        cu.sort_unstable();
        cu.insert(0, f);
        cu.push(b);
        cu.dedup();
        if cu.len() as u32 != en.copper_layer_count.max(1) && en.copper_layer_count > 0 {
            // Fall back to the canonical stack for the reported count.
            let n = en.copper_layer_count as i32;
            cu = std::iter::once(f).chain((0..n - 2).map(|i| bt::BoardLayer::BlIn1Cu as i32 + i)).chain(std::iter::once(b)).collect();
        }
        let lidx: HashMap<i32, usize> = cu.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let mut layer_names = Vec::new();
        for &l in &cu {
            let r: Result<bcmd::BoardLayerNameResponse> = self.send(&bcmd::GetBoardLayerName { board: Some(doc.clone()), layer: l });
            layer_names.push(match r {
                Ok(r) if !r.name.is_empty() => r.name,
                _ => bt::BoardLayer::try_from(l).map(|x| x.as_str_name().trim_start_matches("BL_").replace('_', ".")).unwrap_or_default(),
            });
        }

        // Nets and their classes.
        let nets: bcmd::NetsResponse = self.send(&bcmd::GetNets { board: Some(doc.clone()), netclass_filter: vec![] })?;
        let mut net_list: Vec<(i32, String)> = vec![(0, String::new())];
        let mut kn: Vec<bt::Net> = nets.nets.into_iter().filter(|n| n.code.as_ref().is_some_and(|c| c.value > 0)).collect();
        kn.sort_by_key(|n| n.code.as_ref().map(|c| c.value).unwrap_or(0));
        for n in &kn {
            net_list.push((n.code.as_ref().map(|c| c.value).unwrap_or(0), n.name.clone()));
        }
        let code_to_id: HashMap<i32, u32> = net_list.iter().enumerate().map(|(i, (c, _))| (*c, i as u32)).collect();
        let classes: HashMap<String, crate::proto::kiapi::common::project::NetClass> = if kn.is_empty() {
            HashMap::new()
        } else {
            let r: bcmd::NetClassForNetsResponse = self.send(&bcmd::GetNetClassForNets { net: kn.clone() })?;
            r.classes
        };
        let mut netclasses: Vec<NetClass> = vec![NetClass::default()];
        let mut nc_index: HashMap<String, usize> = HashMap::new();
        let mut nets_out: Vec<Net> = vec![Net { name: String::new(), code: 0, netclass: 0 }];
        for (code, name) in net_list.iter().skip(1) {
            let mut idx = 0;
            if let Some(c) = classes.get(name) {
                let key = c.name.clone();
                idx = match nc_index.get(&key) {
                    Some(&i) => i,
                    None => {
                        let mut nc = NetClass { name: key.clone(), ..NetClass::default() };
                        if let Some(bs) = &c.board {
                            nc.clearance = dist(&bs.clearance).unwrap_or(nc.clearance);
                            nc.track_width = dist(&bs.track_width).unwrap_or(nc.track_width);
                            if let Some(vs) = &bs.via_stack {
                                if let Some(sz) = vs.copper_layers.first().and_then(|l| l.size.as_ref()) {
                                    if sz.x_nm > 0 {
                                        nc.via_diameter = sz.x_nm as f64 * NM;
                                    }
                                }
                                if let Some(d) = vs.drill.as_ref().and_then(|d| d.diameter.as_ref()) {
                                    if d.x_nm > 0 {
                                        nc.via_drill = d.x_nm as f64 * NM;
                                    }
                                }
                            }
                        }
                        if key == "Default" {
                            netclasses[0] = nc;
                            nc_index.insert(key, 0);
                            0
                        } else {
                            netclasses.push(nc);
                            nc_index.insert(key, netclasses.len() - 1);
                            netclasses.len() - 1
                        }
                    }
                };
            }
            nets_out.push(Net { name: name.clone(), code: *code, netclass: idx });
        }
        // Items may carry only the net name (KiCad 10 identifies nets by name), so match by
        // name first and fall back to the code.
        let name_to_id: HashMap<String, u32> = net_list.iter().enumerate().skip(1).map(|(i, (_, n))| (n.clone(), i as u32)).collect();
        let net_of = |n: &Option<bt::Net>| -> u32 {
            let Some(n) = n.as_ref() else { return 0 };
            if !n.name.is_empty() {
                if let Some(&id) = name_to_id.get(&n.name) {
                    return id;
                }
            }
            n.code.as_ref().filter(|c| c.value > 0).and_then(|c| code_to_id.get(&c.value).copied()).unwrap_or(0)
        };

        // Footprint references for readable pad names.
        let mut fp_ref: HashMap<String, String> = HashMap::new();
        for any in self.get_items(&doc, &[ct::KiCadObjectType::KotPcbFootprint])? {
            if let Ok(fp) = bt::FootprintInstance::decode(any.value.as_slice()) {
                let r = fp.reference_field.as_ref().and_then(|f| f.text.as_ref()).and_then(|t| t.text.as_ref()).map(|t| t.text.clone()).unwrap_or_default();
                if let Some(id) = fp.id {
                    fp_ref.insert(id.value, r);
                }
            }
        }

        // Pads, with exact copper polygons from KiCad.
        let mut kpads: Vec<bt::Pad> = Vec::new();
        for any in self.get_items(&doc, &[ct::KiCadObjectType::KotPcbPad])? {
            kpads.push(bt::Pad::decode(any.value.as_slice()).context("decode pad")?);
        }
        let ids: Vec<ct::Kiid> = kpads.iter().filter_map(|p| p.id.clone()).collect();
        // Which copper layers each pad really has copper on. (KiCad returns a pad polygon for
        // any layer it is asked about, and THT pads may have unused rings removed.)
        let mut present: std::collections::HashSet<(String, i32)> = std::collections::HashSet::new();
        let mut presence_ok = true;
        for chunk in ids.chunks(1000) {
            let r: Result<bcmd::PadstackPresenceResponse> = self.send(&bcmd::CheckPadstackPresenceOnLayers { board: Some(doc.clone()), items: chunk.to_vec(), layers: cu.clone() });
            match r {
                Ok(r) => {
                    for e in r.entries {
                        if e.presence == bcmd::PadstackPresence::PspPresent as i32 {
                            if let Some(id) = e.item {
                                present.insert((id.value, e.layer));
                            }
                        }
                    }
                }
                Err(_) => {
                    presence_ok = false;
                    break;
                }
            }
        }
        if !presence_ok {
            // Older API: fall back to the pad stack's layer list.
            present.clear();
            for p in &kpads {
                let id = p.id.as_ref().map(|i| i.value.clone()).unwrap_or_default();
                let ls = p.pad_stack.as_ref().map(|s| s.layers.clone()).unwrap_or_default();
                let tht = matches!(bt::PadType::try_from(p.r#type), Ok(bt::PadType::PtPth));
                for &l in &cu {
                    if tht || ls.contains(&l) {
                        present.insert((id.clone(), l));
                    }
                }
            }
        }
        let mut shapes: HashMap<String, Vec<LayerShape>> = HashMap::new();
        for (li, &l) in cu.iter().enumerate() {
            for chunk in ids.chunks(2000) {
                let r: bcmd::PadShapeAsPolygonResponse = self.send(&bcmd::GetPadShapeAsPolygon { board: Some(doc.clone()), pads: chunk.to_vec(), layer: l })?;
                for (id, poly) in r.pads.iter().zip(r.polygons.iter()) {
                    if !present.contains(&(id.value.clone(), l)) {
                        continue;
                    }
                    let p = polygon(poly);
                    if p.outline.len() >= 3 {
                        shapes.entry(id.value.clone()).or_default().push(LayerShape { layer: li, poly: p });
                    }
                }
            }
        }
        let mut pads = Vec::new();
        for p in &kpads {
            let id = p.id.as_ref().map(|i| i.value.clone()).unwrap_or_default();
            let kind = match bt::PadType::try_from(p.r#type).unwrap_or(bt::PadType::PtSmd) {
                bt::PadType::PtPth => PadKind::Pth,
                bt::PadType::PtNpth => PadKind::Npth,
                _ => PadKind::Smd,
            };
            let drill = p.pad_stack.as_ref().and_then(|s| s.drill.as_ref()).and_then(|d| d.diameter.as_ref()).map(|d| d.x_nm.max(d.y_nm) as f64 * NM).filter(|&d| d > 0.0);
            let fpr = p.parent.as_ref().and_then(|k| fp_ref.get(&k.value)).cloned().unwrap_or_default();
            pads.push(Pad {
                name: if fpr.is_empty() { p.number.clone() } else { format!("{fpr}.{}", p.number) },
                net: net_of(&p.net),
                kind,
                pos: v2(&p.position),
                shapes: shapes.remove(&id).unwrap_or_default(),
                drill: if kind == PadKind::Smd { None } else { drill },
                clearance: dist(&p.copper_clearance_override).filter(|&c| c > 0.0),
                id,
            });
        }

        // Existing tracks, arcs and vias.
        let mut tracks = Vec::new();
        let mut vias = Vec::new();
        let tv = [ct::KiCadObjectType::KotPcbTrace, ct::KiCadObjectType::KotPcbArc, ct::KiCadObjectType::KotPcbVia];
        for any in self.get_items(&doc, &tv)? {
            if any.type_url.ends_with(".Track") {
                let t = bt::Track::decode(any.value.as_slice())?;
                if let Some(&l) = lidx.get(&t.layer) {
                    tracks.push(Track { id: t.id.map(|i| i.value).unwrap_or_default(), net: net_of(&t.net), layer: l, a: v2(&t.start), b: v2(&t.end), width: dist(&t.width).unwrap_or(0.2) });
                }
            } else if any.type_url.ends_with(".Arc") {
                let a = bt::Arc::decode(any.value.as_slice())?;
                if let Some(&l) = lidx.get(&a.layer) {
                    let pts = arc_points(v2(&a.start), v2(&a.mid), v2(&a.end), 0.002);
                    for w in pts.windows(2) {
                        tracks.push(Track { id: a.id.as_ref().map(|i| i.value.clone()).unwrap_or_default(), net: net_of(&a.net), layer: l, a: w[0], b: w[1], width: dist(&a.width).unwrap_or(0.2) });
                    }
                }
            } else if any.type_url.ends_with(".Via") {
                let v = bt::Via::decode(any.value.as_slice())?;
                let ps = v.pad_stack.clone().unwrap_or_default();
                let dia = ps.copper_layers.iter().filter_map(|l| l.size.as_ref()).map(|s| s.x_nm).max().unwrap_or(0) as f64 * NM;
                let d = ps.drill.clone().unwrap_or_default();
                let drill = d.diameter.as_ref().map(|x| x.x_nm as f64 * NM).unwrap_or(0.3);
                let (a, b2) = (lidx.get(&d.start_layer).copied().unwrap_or(0), lidx.get(&d.end_layer).copied().unwrap_or(cu.len() - 1));
                vias.push(Via { id: v.id.map(|i| i.value).unwrap_or_default(), net: net_of(&v.net), pos: v2(&v.position), diameter: dia.max(drill), drill, from_layer: a.min(b2), to_layer: a.max(b2) });
            }
        }

        // Board outline from Edge.Cuts graphics.
        let mut pieces: Vec<(Vec<P2>, bool)> = Vec::new();
        let mut obstacles: Vec<Obstacle> = Vec::new();
        let edge = bt::BoardLayer::BlEdgeCuts as i32;
        for any in self.get_items(&doc, &[ct::KiCadObjectType::KotPcbShape])? {
            let Ok(s) = bt::BoardGraphicShape::decode(any.value.as_slice()) else { continue };
            if let Some(&l) = lidx.get(&s.layer) {
                // Graphic copper: an obstacle for other nets.
                let net = net_of(&s.net);
                let attrs = s.shape.as_ref().and_then(|g| g.attributes.clone()).unwrap_or_default();
                let r = attrs.stroke.as_ref().and_then(|st| dist(&st.width)).unwrap_or(0.0) / 2.0;
                let filled = attrs.fill.as_ref().is_some_and(|f| f.fill_type == ct::GraphicFillType::GftFilled as i32);
                if let Some(geom) = s.shape.as_ref().and_then(|g| g.geometry.clone()) {
                    for shape in graphic_obstacles(geom, r, filled) {
                        obstacles.push(Obstacle { layer: l, net, shape });
                    }
                }
                continue;
            }
            if s.layer != edge {
                continue;
            }
            let Some(geom) = s.shape.and_then(|g| g.geometry) else { continue };
            use ct::graphic_shape::Geometry as G;
            match geom {
                G::Segment(g) => pieces.push((vec![v2(&g.start), v2(&g.end)], false)),
                G::Arc(g) => pieces.push((arc_points(v2(&g.start), v2(&g.mid), v2(&g.end), 0.002), false)),
                G::Rectangle(g) => {
                    let (a, c) = (v2(&g.top_left), v2(&g.bottom_right));
                    pieces.push((Polygon::rect(P2::new(a.x.min(c.x), a.y.min(c.y)), P2::new(a.x.max(c.x), a.y.max(c.y))).outline, true));
                }
                G::Circle(g) => {
                    let c = v2(&g.center);
                    let r = c.dist(v2(&g.radius_point));
                    pieces.push((Polygon::circle(c, r, ((r * 40.0) as usize).clamp(32, 720)).outline, true));
                }
                G::Polygon(ps) => {
                    for p in &ps.polygons {
                        let pg = polygon(p);
                        pieces.push((pg.outline, true));
                        for h in pg.holes {
                            pieces.push((h, true));
                        }
                    }
                }
                G::Bezier(g) => {
                    let (p0, p1, p2, p3) = (v2(&g.start), v2(&g.control1), v2(&g.control2), v2(&g.end));
                    let pts = (0..=32)
                        .map(|i| {
                            let t = i as f64 / 32.0;
                            let u = 1.0 - t;
                            p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
                        })
                        .collect();
                    pieces.push((pts, false));
                }
            }
        }
        let outline = assemble_outline(pieces);

        // Copper text: its bounding box is an obstacle.
        let mut text_ids: Vec<(ct::Kiid, usize)> = Vec::new();
        for any in self.get_items(&doc, &[ct::KiCadObjectType::KotPcbText, ct::KiCadObjectType::KotPcbTextbox])? {
            let (id, layer) = if any.type_url.ends_with(".BoardTextBox") {
                match bt::BoardTextBox::decode(any.value.as_slice()) {
                    Ok(t) => (t.id, t.layer),
                    Err(_) => continue,
                }
            } else {
                match bt::BoardText::decode(any.value.as_slice()) {
                    Ok(t) => (t.id, t.layer),
                    Err(_) => continue,
                }
            };
            if let (Some(id), Some(&l)) = (id, lidx.get(&layer)) {
                text_ids.push((id, l));
            }
        }
        if !text_ids.is_empty() {
            let r: Result<ccmd::GetBoundingBoxResponse> = self.send(&ccmd::GetBoundingBox {
                header: Some(ct::ItemHeader { document: Some(doc.clone()), container: None, field_mask: None }),
                items: text_ids.iter().map(|t| t.0.clone()).collect(),
                mode: ccmd::BoundingBoxMode::BbmItemOnly as i32,
            });
            if let Ok(r) = r {
                for (id, b) in r.items.iter().zip(r.boxes.iter()) {
                    let Some(&(_, l)) = text_ids.iter().find(|t| t.0.value == id.value) else { continue };
                    let p = v2(&b.position);
                    let sz = v2(&b.size);
                    let a = P2::new(p.x.min(p.x + sz.x), p.y.min(p.y + sz.y));
                    let c = P2::new(p.x.max(p.x + sz.x), p.y.max(p.y + sz.y));
                    obstacles.push(Obstacle { layer: l, net: 0, shape: Shape::Polygon { poly: Polygon::rect(a, c) } });
                }
            }
        }

        // Zones: copper pours (plane detection) and rule areas (keepouts).
        let mut zones = Vec::new();
        let mut keepouts = Vec::new();
        for any in self.get_items(&doc, &[ct::KiCadObjectType::KotPcbZone])? {
            let Ok(z) = bt::Zone::decode(any.value.as_slice()) else { continue };
            let layers: Vec<usize> = z.layers.iter().filter_map(|l| lidx.get(l).copied()).collect();
            let polys: Vec<Polygon> = z.outline.as_ref().map(|o| o.polygons.iter().map(polygon).collect()).unwrap_or_default();
            match z.settings {
                Some(bt::zone::Settings::RuleAreaSettings(r)) => {
                    let (nt, nv) = (r.keepout_tracks || r.keepout_copper, r.keepout_vias || r.keepout_copper);
                    if nt || nv {
                        for p in polys {
                            keepouts.push(Keepout { layers: layers.clone(), poly: p, no_tracks: nt, no_vias: nv });
                        }
                    }
                }
                Some(bt::zone::Settings::CopperSettings(c)) => {
                    let net = net_of(&c.net);
                    if net != 0 {
                        for p in polys {
                            zones.push(Zone { net, layers: layers.clone(), poly: p, name: z.name.clone() });
                        }
                    }
                }
                None => {}
            }
        }

        let project_file = find_project(&doc, board_file.as_deref());
        let rules = project_file.as_ref().and_then(|p| read_rules(p).ok()).unwrap_or_default();
        let mut board = Board {
            name: board_file.clone().unwrap_or_else(|| "board".into()),
            layers: layer_names,
            outline,
            nets: nets_out,
            netclasses,
            pads,
            tracks,
            vias,
            keepouts,
            zones,
            obstacles,
            rules,
        };
        board.normalize();
        Ok((board, BoardCtx { doc, layer_ids: cu, nets: net_list, project_file, board_file }))
    }

    /// Save the board document in KiCad.
    pub fn save(&mut self, ctx: &BoardCtx) -> Result<()> {
        let _: prost_types::Any = self.send(&ccmd::SaveDocument { document: Some(ctx.doc.clone()) })?;
        Ok(())
    }

    /// Add routed tracks and vias as a single undoable commit, then optionally refill zones.
    pub fn commit_routes(&mut self, board: &Board, ctx: &BoardCtx, res: &RouteResult, refill_zones: bool) -> Result<()> {
        let knet = |id: u32| {
            let (code, name) = ctx.nets.get(id as usize).cloned().unwrap_or((0, String::new()));
            Some(bt::Net { code: Some(bt::NetCode { value: code }), name })
        };
        let mut items: Vec<prost_types::Any> = Vec::new();
        for s in &res.segs {
            let t = bt::Track {
                id: None,
                start: Some(vec2(s.a)),
                end: Some(vec2(s.b)),
                width: Some(ct::Distance { value_nm: (s.width / NM).round() as i64 }),
                locked: 0,
                layer: ctx.layer_ids[s.layer],
                net: knet(s.net),
                parent: None,
            };
            items.push(prost_types::Any { type_url: bt::Track::type_url(), value: t.encode_to_vec() });
        }
        for v in &res.vias {
            let d = (v.diameter / NM).round() as i64;
            let dr = (v.drill / NM).round() as i64;
            let (from, to) = (ctx.layer_ids[v.from_layer], ctx.layer_ids[v.to_layer]);
            let through = v.from_layer == 0 && v.to_layer + 1 == board.layers.len();
            let via = bt::Via {
                id: None,
                position: Some(vec2(v.pos)),
                pad_stack: Some(bt::PadStack {
                    r#type: bt::PadStackType::PstNormal as i32,
                    layers: vec![from, to],
                    drill: Some(bt::DrillProperties {
                        start_layer: from,
                        end_layer: to,
                        diameter: Some(ct::Vector2 { x_nm: dr, y_nm: dr }),
                        shape: bt::DrillShape::DsCircle as i32,
                        ..Default::default()
                    }),
                    copper_layers: vec![bt::PadStackLayer {
                        layer: bt::BoardLayer::BlFCu as i32,
                        shape: bt::PadStackShape::PssCircle as i32,
                        size: Some(ct::Vector2 { x_nm: d, y_nm: d }),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                locked: 0,
                net: knet(v.net),
                r#type: if through { bt::ViaType::VtThrough as i32 } else { bt::ViaType::VtBlindBuried as i32 },
                parent: None,
            };
            items.push(prost_types::Any { type_url: bt::Via::type_url(), value: via.encode_to_vec() });
        }
        let commit: ccmd::BeginCommitResponse = self.send(&ccmd::BeginCommit {})?;
        let result = (|| -> Result<()> {
            for chunk in items.chunks(500) {
                let r: ccmd::CreateItemsResponse = self.send(&ccmd::CreateItems {
                    header: Some(ct::ItemHeader { document: Some(ctx.doc.clone()), container: None, field_mask: None }),
                    items: chunk.to_vec(),
                    container: None,
                })?;
                if let Some(bad) = r.created_items.iter().find(|c| c.status.as_ref().is_some_and(|s| s.code != ccmd::ItemStatusCode::IscOk as i32)) {
                    bail!("KiCad rejected an item: {}", bad.status.as_ref().map(|s| s.error_message.clone()).unwrap_or_default());
                }
            }
            Ok(())
        })();
        let action = if result.is_ok() { ccmd::CommitAction::CmaCommit } else { ccmd::CommitAction::CmaDrop };
        let end: Result<ccmd::EndCommitResponse> = self.send(&ccmd::EndCommit { id: commit.id, action: action as i32, message: "Tracewright autoroute".into() });
        result?;
        end?;
        if refill_zones && !board.zones.is_empty() {
            let _: Result<prost_types::Any> = self.send(&bcmd::RefillZones { board: Some(ctx.doc.clone()), zones: vec![] });
        }
        Ok(())
    }
}

/// Obstacle shapes for a graphic item on a copper layer (stroke half-width `r`).
fn graphic_obstacles(geom: ct::graphic_shape::Geometry, r: f64, filled: bool) -> Vec<Shape> {
    use ct::graphic_shape::Geometry as G;
    let r = r.max(0.001);
    let chain = |pts: &[P2], closed: bool| -> Vec<Shape> {
        let mut v: Vec<Shape> = pts.windows(2).map(|w| Shape::Segment { a: w[0], b: w[1], r }).collect();
        if closed && pts.len() > 2 {
            v.push(Shape::Segment { a: *pts.last().unwrap(), b: pts[0], r });
        }
        v
    };
    let ring = |pts: Vec<P2>| -> Vec<Shape> {
        let mut v = chain(&pts, true);
        if filled && pts.len() >= 3 {
            v.push(Shape::Polygon { poly: Polygon::new(pts) });
        }
        v
    };
    match geom {
        G::Segment(g) => vec![Shape::Segment { a: v2(&g.start), b: v2(&g.end), r }],
        G::Arc(g) => chain(&arc_points(v2(&g.start), v2(&g.mid), v2(&g.end), 0.002), false),
        G::Rectangle(g) => {
            let (a, c) = (v2(&g.top_left), v2(&g.bottom_right));
            ring(Polygon::rect(P2::new(a.x.min(c.x), a.y.min(c.y)), P2::new(a.x.max(c.x), a.y.max(c.y))).outline)
        }
        G::Circle(g) => {
            let c = v2(&g.center);
            let rad = c.dist(v2(&g.radius_point));
            if filled { vec![Shape::Circle { c, r: rad + r }] } else { ring(Polygon::circle(c, rad, 64).outline) }
        }
        G::Polygon(ps) => ps.polygons.iter().flat_map(|p| ring(polygon(p).outline)).collect(),
        G::Bezier(g) => {
            let (p0, p1, p2, p3) = (v2(&g.start), v2(&g.control1), v2(&g.control2), v2(&g.end));
            let pts: Vec<P2> = (0..=32)
                .map(|i| {
                    let t = i as f64 / 32.0;
                    let u = 1.0 - t;
                    p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
                })
                .collect();
            chain(&pts, false)
        }
    }
}

/// Chain Edge.Cuts pieces into closed loops and nest them into outlines with cutouts.
pub fn assemble_outline(pieces: Vec<(Vec<P2>, bool)>) -> Vec<Polygon> {
    let tol = 0.01;
    let mut rings: Vec<Vec<P2>> = pieces.iter().filter(|(p, closed)| *closed && p.len() >= 3).map(|(p, _)| p.clone()).collect();
    let mut open: Vec<Vec<P2>> = pieces.into_iter().filter(|(p, closed)| !*closed && p.len() >= 2).map(|(p, _)| p).collect();
    while let Some(mut cur) = open.pop() {
        loop {
            let end = *cur.last().unwrap();
            if cur.len() > 2 && end.dist(cur[0]) < tol {
                cur.pop();
                rings.push(cur);
                break;
            }
            let next = open.iter().position(|p| p[0].dist(end) < tol || p.last().unwrap().dist(end) < tol);
            match next {
                Some(i) => {
                    let mut p = open.swap_remove(i);
                    if p[0].dist(end) >= tol {
                        p.reverse();
                    }
                    cur.extend(p.into_iter().skip(1));
                }
                None => break, // open chain: not part of a closed outline
            }
        }
    }
    // Nesting depth decides outline (even) vs cutout (odd).
    let depth: Vec<usize> = rings.iter().enumerate().map(|(i, r)| rings.iter().enumerate().filter(|&(j, o)| j != i && point_in_ring(r[0], o)).count()).collect();
    let mut out: Vec<(usize, Polygon)> = Vec::new();
    for (i, r) in rings.iter().enumerate() {
        if depth[i] % 2 == 0 {
            out.push((i, Polygon::new(r.clone())));
        }
    }
    for (i, r) in rings.iter().enumerate() {
        if depth[i] % 2 == 1 {
            // Smallest enclosing outline of one lower depth.
            if let Some((_, poly)) = out.iter_mut().filter(|(j, _)| depth[*j] + 1 == depth[i] && point_in_ring(r[0], &rings[*j])).min_by(|a, b| a.1.area().partial_cmp(&b.1.area()).unwrap()) {
                poly.holes.push(r.clone());
            }
        }
    }
    out.into_iter().map(|(_, p)| p).collect()
}

fn find_project(doc: &ct::DocumentSpecifier, board_file: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = &doc.project {
        let path = PathBuf::from(&p.path);
        if path.extension().is_some_and(|e| e == "kicad_pro") && path.exists() {
            return Some(path);
        }
        let cand = path.join(format!("{}.kicad_pro", p.name));
        if cand.exists() {
            return Some(cand);
        }
        if let Some(bf) = board_file {
            let cand = path.join(bf).with_extension("kicad_pro");
            if cand.exists() {
                return Some(cand);
            }
        }
        if let Ok(rd) = std::fs::read_dir(&path) {
            for e in rd.flatten() {
                if e.path().extension().is_some_and(|x| x == "kicad_pro") {
                    return Some(e.path());
                }
            }
        }
    }
    None
}

/// Board design-rule minimums from a .kicad_pro file.
pub fn read_rules(path: &PathBuf) -> Result<BoardRules> {
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let r = &j["board"]["design_settings"]["rules"];
    let g = |k: &str, d: f64| r[k].as_f64().unwrap_or(d);
    let def = BoardRules::default();
    Ok(BoardRules {
        min_clearance: g("min_clearance", def.min_clearance),
        edge_clearance: g("min_copper_edge_clearance", def.edge_clearance),
        hole_clearance: g("min_hole_clearance", def.hole_clearance),
        hole_to_hole: g("min_hole_to_hole", def.hole_to_hole),
        min_track_width: g("min_track_width", def.min_track_width),
        min_via_diameter: g("min_via_diameter", def.min_via_diameter),
        min_via_drill: g("min_through_hole_diameter", def.min_via_drill),
        min_via_annular: g("min_via_annular_width", def.min_via_annular),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn outline_with_cutout() {
        let sq = |a: f64, b: f64| vec![(vec![P2::new(a, a), P2::new(b, a)], false), (vec![P2::new(b, a), P2::new(b, b)], false), (vec![P2::new(a, b), P2::new(b, b)], false), (vec![P2::new(a, b), P2::new(a, a)], false)];
        let mut pieces = sq(0.0, 10.0);
        pieces.extend(sq(4.0, 6.0));
        let o = assemble_outline(pieces);
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].holes.len(), 1);
        assert!((o[0].area() - 96.0).abs() < 1e-9);
    }
}
