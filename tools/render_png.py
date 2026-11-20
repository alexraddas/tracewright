"""Dev tool: render a board JSON plus a route result to PNG (needs Pillow; KiCad's Python has it).

    python tools/render_png.py board.json result.json out.png [--scale 12] [--crop x0,y0,x1,y1]
"""
import json
import sys

from PIL import Image, ImageDraw

COLORS = ["#c83434", "#e0a030", "#3aa060", "#b050c0", "#20a0c0", "#d07090", "#8090d0", "#4d7fc4"]


def color(layer, nl):
    return COLORS[7] if nl > 1 and layer == nl - 1 else COLORS[min(layer, 6)]


def main():
    board = json.load(open(sys.argv[1]))
    res = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 12.0
    nl = len(board["layers"])
    if "--crop" in sys.argv:
        x0, y0, x1, y1 = map(float, sys.argv[sys.argv.index("--crop") + 1].split(","))
    else:
        pts = [v for o in board["outline"] for v in o["outline"]] or [p["pos"] for p in board["pads"]]
        x0, y0 = min(p["x"] for p in pts) - 1, min(p["y"] for p in pts) - 1
        x1, y1 = max(p["x"] for p in pts) + 1, max(p["y"] for p in pts) + 1
    W, H = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGB", (W, H), "#101418")
    d = ImageDraw.Draw(img, "RGBA")
    tx = lambda p: ((p["x"] - x0) * scale, (p["y"] - y0) * scale)
    for o in board["outline"]:
        d.polygon([tx(v) for v in o["outline"]], fill="#1a2028", outline="#d0d060")
        for h in o.get("holes", []):
            d.polygon([tx(v) for v in h], fill="#101418", outline="#d0d060")

    def rgba(c, a):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (a,)

    for layer in reversed(range(nl)):
        c = color(layer, nl)
        for p in board["pads"]:
            for s in p["shapes"]:
                if s["layer"] == layer and len(s["poly"]["outline"]) >= 3:
                    d.polygon([tx(v) for v in s["poly"]["outline"]], fill=rgba(c, 140))
        for s in res["segs"]:
            if s["layer"] == layer:
                w = max(1, int(round(s["width"] * scale)))
                a, b = tx(s["a"]), tx(s["b"])
                d.line([a, b], fill=rgba(c, 235), width=w)
                r = w / 2
                for q in (a, b):
                    d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=rgba(c, 235))
    for p in board["pads"]:
        if p.get("drill"):
            q, r = tx(p["pos"]), p["drill"] / 2 * scale
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill="#101418")
    for v in res["vias"]:
        q = tx(v["pos"])
        r, rd = v["diameter"] / 2 * scale, v["drill"] / 2 * scale
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill="#c8c8c8")
        d.ellipse([q[0] - rd, q[1] - rd, q[0] + rd, q[1] + rd], fill="#101418")
    img.save(out)
    print(out, W, H)


if __name__ == "__main__":
    main()
