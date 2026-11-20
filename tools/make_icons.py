"""Generate the toolbar icons (light/dark, 24 and 48 px) for the KiCad plugin.

    python tools/make_icons.py plugin/icons
"""
import os
import sys

from PIL import Image, ImageDraw


def icon(size, fg, accent):
    s = size / 24.0
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(2, round(2.2 * s))
    # Two orthogonal traces, like a routed bus.
    d.line([(3 * s, 7 * s), (13 * s, 7 * s), (13 * s, 19 * s), (21 * s, 19 * s)], fill=fg, width=w, joint="curve")
    d.line([(3 * s, 13 * s), (8 * s, 13 * s), (8 * s, 21 * s)], fill=fg, width=w, joint="curve")
    # Pads and a via.
    for (x, y) in [(3, 7), (3, 13), (21, 19), (8, 21)]:
        r = 2.0 * s
        d.rectangle([x * s - r, y * s - r, x * s + r, y * s + r], fill=fg)
    r = 3.6 * s
    d.ellipse([18 * s - r, 6 * s - r, 18 * s + r, 6 * s + r], fill=accent)
    r = 1.5 * s
    d.ellipse([18 * s - r, 6 * s - r, 18 * s + r, 6 * s + r], fill=(0, 0, 0, 0))
    d.line([(13 * s, 7 * s), (15 * s, 7 * s)], fill=fg, width=w)
    return img


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for theme, fg, accent in [("light", (40, 60, 90, 255), (200, 60, 50, 255)), ("dark", (220, 225, 235, 255), (255, 120, 90, 255))]:
        for size in (24, 48):
            icon(size, fg, accent).save(os.path.join(out, "icon-%s-%d.png" % (theme, size)))
    print("icons written to", out)


if __name__ == "__main__":
    main()
