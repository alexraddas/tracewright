"""Summarise a kicad-cli DRC JSON report: violation counts by type and unconnected nets.

    python tools/drc_summary.py drc.json [board.json]

With board.json (a kicad_dump), unconnected items whose nets have pads outside the board
outline are reported separately (they cannot be routed by anyone).
"""
import collections
import json
import re
import sys


def inside(pt, ring):
    x, y, c = pt["x"], pt["y"], False
    j = len(ring) - 1
    for i in range(len(ring)):
        a, b = ring[i], ring[j]
        if (a["y"] > y) != (b["y"] > y) and x < b["x"] + (y - b["y"]) * (a["x"] - b["x"]) / (a["y"] - b["y"]):
            c = not c
        j = i
    return c


def main():
    d = json.load(open(sys.argv[1]))
    off_nets = set()
    if len(sys.argv) > 2:
        b = json.load(open(sys.argv[2]))
        for p in b["pads"]:
            if b["outline"] and not any(inside(p["pos"], o["outline"]) for o in b["outline"]):
                off_nets.add(b["nets"][p["net"]]["name"])
    v = d.get("violations", [])
    print("violations: %d %s" % (len(v), dict(collections.Counter(x["type"] for x in v))))
    for x in v[:12]:
        print("   ", x["type"], "|", x["description"], "|", " / ".join(i["description"][:70] for i in x["items"]))
    nets = collections.Counter()
    for x in d.get("unconnected_items", []):
        m = re.search(r"\[([^\]]*)\]", x["items"][0]["description"])
        nets[m.group(1) if m else "?"] += 1
    real = {n: c for n, c in nets.items() if n not in off_nets}
    print("unconnected: %d items in %d nets (%d nets have off-board pads)" % (sum(nets.values()), len(nets), len(nets) - len(real)))
    if real:
        print("    unconnected nets with all pads on the board:", real)


if __name__ == "__main__":
    main()
