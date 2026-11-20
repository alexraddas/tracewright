"""Dev/test tool: dump a .kicad_pcb into Tracewright's board JSON using KiCad's own geometry.

Runs with KiCad's bundled Python (it needs the SWIG `pcbnew` module):

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" tools/kicad_dump.py board.kicad_pcb out.json [--strip]

--strip drops existing tracks and vias so the whole board is routed from scratch.
The plugin itself reads boards live over KiCad's IPC API; this tool exists so the router can
be tested headlessly against real boards and checked with `kicad-cli pcb drc`.
"""
import json
import sys

import pcbnew

NM = 1e-6


def pt(v):
    return {"x": v.x * NM, "y": v.y * NM}


def chain(c):
    return [pt(c.CPoint(i)) for i in range(c.PointCount())]


def polys(ps):
    out = []
    for i in range(ps.OutlineCount()):
        out.append({"outline": chain(ps.Outline(i)), "holes": [chain(ps.Hole(i, h)) for h in range(ps.HoleCount(i))]})
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    strip = "--strip" in sys.argv
    b = pcbnew.LoadBoard(src)
    cu = list(b.GetEnabledLayers().CuStack())
    lidx = {l: i for i, l in enumerate(cu)}
    ds = b.GetDesignSettings()
    ns = ds.m_NetSettings

    # Net classes.
    by_code = b.GetNetsByNetcode()
    class_names = ["Default"]
    for code in sorted(by_code.keys()):
        n = by_code[code].GetNetClassName() or "Default"
        for part in n.split(","):
            part = part.strip()
            if part and part not in class_names:
                class_names.append(part)
    netclasses = []
    for name in class_names:
        nc = ns.GetNetClassByName(name) if name != "Default" else ns.GetDefaultNetclass()
        netclasses.append({
            "name": name,
            "clearance": nc.GetClearance() * NM,
            "track_width": nc.GetTrackWidth() * NM,
            "via_diameter": nc.GetViaDiameter() * NM,
            "via_drill": nc.GetViaDrill() * NM,
        })

    codes = sorted(by_code.keys())
    net_index = {c: i for i, c in enumerate(codes)}
    nets = []
    for c in codes:
        cls = (by_code[c].GetNetClassName() or "Default").split(",")[0].strip() or "Default"
        nets.append({"name": by_code[c].GetNetname(), "code": c, "netclass": class_names.index(cls) if cls in class_names else 0})

    outline = pcbnew.SHAPE_POLY_SET()
    b.GetBoardPolygonOutlines(outline, True)

    pads = []
    for fp in b.GetFootprints():
        for pad in fp.Pads():
            attr = pad.GetAttribute()
            kind = "npth" if attr == pcbnew.PAD_ATTRIB_NPTH else ("pth" if attr == pcbnew.PAD_ATTRIB_PTH else "smd")
            shapes = []
            for l in cu:
                if not pad.IsOnLayer(l):
                    continue
                if hasattr(pad, "FlashLayer") and not pad.FlashLayer(l):
                    continue
                ps = pcbnew.SHAPE_POLY_SET()
                pad.TransformShapeToPolygon(ps, l, 0, 2000, pcbnew.ERROR_OUTSIDE)
                for p in polys(ps):
                    shapes.append({"layer": lidx[l], "poly": p})
            drill = None
            if attr in (pcbnew.PAD_ATTRIB_PTH, pcbnew.PAD_ATTRIB_NPTH):
                d = pad.GetDrillSize()
                drill = max(d.x, d.y) * NM
            lc = pad.GetLocalClearance()
            pads.append({
                "id": pad.m_Uuid.AsString(),
                "name": "%s.%s" % (fp.GetReference(), pad.GetNumber()),
                "net": net_index.get(pad.GetNetCode(), 0),
                "kind": kind,
                "pos": pt(pad.GetPosition()),
                "shapes": shapes,
                "drill": drill,
                "clearance": (lc * NM) if isinstance(lc, int) and lc > 0 else None,
            })

    tracks, vias = [], []
    if not strip:
        for t in b.GetTracks():
            if t.Type() == pcbnew.PCB_VIA_T:
                top, bot = lidx.get(t.TopLayer(), 0), lidx.get(t.BottomLayer(), len(cu) - 1)
                vias.append({"id": t.m_Uuid.AsString(), "net": net_index.get(t.GetNetCode(), 0), "pos": pt(t.GetPosition()),
                             "diameter": t.GetWidth(pcbnew.F_Cu) * NM, "drill": t.GetDrillValue() * NM,
                             "from_layer": min(top, bot), "to_layer": max(top, bot)})
            elif t.Type() in (pcbnew.PCB_TRACE_T, pcbnew.PCB_ARC_T) and t.GetLayer() in lidx:
                tracks.append({"id": t.m_Uuid.AsString(), "net": net_index.get(t.GetNetCode(), 0), "layer": lidx[t.GetLayer()],
                               "a": pt(t.GetStart()), "b": pt(t.GetEnd()), "width": t.GetWidth() * NM})

    keepouts, zones = [], []
    # Fill first so plane zones are exported as the copper that actually exists
    # (no-pour rule areas, clearances and islands removed), not their outline.
    pcbnew.ZONE_FILLER(b).Fill(b.Zones())
    all_zones = list(b.Zones()) + [z for f in b.GetFootprints() for z in f.Zones()]   # footprint keepouts too
    for z in all_zones:
        layers = [lidx[l] for l in z.GetLayerSet().CuStack() if l in lidx]
        if z.GetIsRuleArea():
            if z.GetDoNotAllowTracks() or z.GetDoNotAllowVias():
                for p in polys(z.Outline()):
                    keepouts.append({"layers": layers, "poly": p, "no_tracks": bool(z.GetDoNotAllowTracks()), "no_vias": bool(z.GetDoNotAllowVias())})
        elif z.GetNetCode() > 0:
            for l in z.GetLayerSet().CuStack():
                if l not in lidx: continue
                for p in polys(z.GetFilledPolysList(l)):
                    zones.append({"net": net_index.get(z.GetNetCode(), 0), "layers": [lidx[l]], "poly": p, "name": z.GetZoneName()})

    # Copper text and graphics (board-level and inside footprints) as polygons.
    obstacles = []

    def add_obstacle(item):
        l = item.GetLayer()
        if l not in lidx:
            return
        ps = pcbnew.SHAPE_POLY_SET()
        try:
            item.TransformShapeToPolygon(ps, l, 0, 2000, pcbnew.ERROR_OUTSIDE)
        except Exception:
            return
        net = net_index.get(item.GetNetCode(), 0) if hasattr(item, "GetNetCode") else 0
        for p in polys(ps):
            obstacles.append({"layer": lidx[l], "net": net, "shape": {"kind": "polygon", "poly": p}})

    for d in b.GetDrawings():
        add_obstacle(d)
    for fp in b.GetFootprints():
        for d in fp.GraphicalItems():
            add_obstacle(d)
        for t in (fp.Reference(), fp.Value()):
            if t.IsVisible():
                add_obstacle(t)

    board = {
        "name": src,
        "layers": [b.GetLayerName(l) for l in cu],
        "outline": polys(outline),
        "nets": nets,
        "netclasses": netclasses,
        "pads": pads,
        "tracks": tracks,
        "vias": vias,
        "keepouts": keepouts,
        "zones": zones,
        "obstacles": obstacles,
        "rules": {
            "min_clearance": ds.m_MinClearance * NM,
            "edge_clearance": ds.m_CopperEdgeClearance * NM,
            "hole_clearance": ds.m_HoleClearance * NM,
            "hole_to_hole": ds.m_HoleToHoleMin * NM,
            "min_track_width": ds.m_TrackMinWidth * NM,
            "min_via_diameter": ds.m_ViasMinSize * NM,
            "min_via_drill": ds.m_MinThroughDrill * NM,
            "min_via_annular": ds.m_ViasMinAnnularWidth * NM,
        },
    }
    with open(dst, "w") as f:
        json.dump(board, f)
    print("%s: %d layers, %d nets, %d pads, %d tracks, %d vias, %d keepouts, %d zones, %d copper graphics"
          % (dst, len(cu), len(nets), len(pads), len(tracks), len(vias), len(keepouts), len(zones), len(obstacles)))


if __name__ == "__main__":
    main()
