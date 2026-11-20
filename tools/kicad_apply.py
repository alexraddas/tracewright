"""Dev/test tool: write a Tracewright route result into a copy of a .kicad_pcb.

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" tools/kicad_apply.py board.kicad_pcb board.json result.json out.kicad_pcb [--strip]

`board.json` is the dump the route was made from (for net codes and layer order).
--strip removes existing tracks/vias first (use when the dump was made with --strip).
Without it, existing items listed in the result's `removed_ids` (force re-route) are removed.
"""
import json
import sys

import pcbnew


def main():
    src, board_json, result_json, dst = sys.argv[1:5]
    strip = "--strip" in sys.argv
    b = pcbnew.LoadBoard(src)
    cu = list(b.GetEnabledLayers().CuStack())
    info = json.load(open(board_json))
    res = json.load(open(result_json))
    codes = [n["code"] for n in info["nets"]]
    if strip:
        for t in list(b.GetTracks()):
            b.Remove(t)
    else:
        # Force re-route: drop the old copper of re-routed nets before adding the new routes.
        gone = set(res.get("removed_ids", []))
        if gone:
            dead = [t for t in b.GetTracks() if t.m_Uuid.AsString() in gone]
            for t in dead:
                b.Remove(t)
            print("removed %d existing tracks/vias of %d re-routed net(s)" % (len(dead), len(res.get("rerouted_nets", []))))
    mm = lambda v: int(round(v * 1e6))
    vec = lambda p: pcbnew.VECTOR2I(mm(p["x"]), mm(p["y"]))
    for s in res["segs"]:
        t = pcbnew.PCB_TRACK(b)
        t.SetStart(vec(s["a"]))
        t.SetEnd(vec(s["b"]))
        t.SetWidth(mm(s["width"]))
        t.SetLayer(cu[s["layer"]])
        t.SetNetCode(codes[s["net"]])
        b.Add(t)
    for v in res["vias"]:
        via = pcbnew.PCB_VIA(b)
        via.SetPosition(vec(v["pos"]))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        via.SetLayerPair(cu[v["from_layer"]], cu[v["to_layer"]])
        via.SetDrill(mm(v["drill"]))
        try:
            via.SetWidth(mm(v["diameter"]))
        except TypeError:
            via.SetWidth(pcbnew.F_Cu, mm(v["diameter"]))
        via.SetNetCode(codes[v["net"]])
        b.Add(via)
    # Zone fills in the file predate the new copper: run DRC with `kicad-cli pcb drc --refill-zones`.
    pcbnew.SaveBoard(dst, b)
    print("wrote %s: %d tracks, %d vias" % (dst, len(res["segs"]), len(res["vias"])))


if __name__ == "__main__":
    main()
