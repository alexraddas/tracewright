#!/usr/bin/env python3
"""Build KiCad Plugin and Content Manager (PCM) packages and a PCM repository for Tracewright.

  package   one platform's install zip (KiCad: Plugin and Content Manager > Install from File...)
      tools/package_pcm.py package --platform macos --exe target/release/tracewright \
          --version 0.2.0 --out dist/
  repo      packages.json + repository.json describing every platform zip, for
            Plugin and Content Manager > Manage... > Add repository
      tools/package_pcm.py repo --version 0.2.0 --base-url https://github.com/<owner>/tracewright/releases/download/v0.2.0 \
          --out dist/ dist/*.zip

Zip layout (KiCad extracts each top-level folder to <3rdparty>/<folder>/com_tracewright_router/):
  metadata.json            package metadata, one version, restricted to the zip's platform
  plugins/plugin.json      API plugin manifest; entrypoint relative to it (KiCad skips absolute ones)
  plugins/tracewright[.exe]  stored with mode 0755 so KiCad restores the execute bit on macOS/Linux
  plugins/icons/*.png
  resources/icon.png       icon shown in the Plugin and Content Manager
Standard library only, so it runs on every CI runner.
"""
import argparse, hashlib, json, os, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENTIFIER = "com.tracewright.router"
PLATFORMS = ("macos", "linux", "windows")
KICAD_MIN = "10.0"
HOMEPAGE = "https://github.com/alexraddas/tracewright"


def package_meta(version_entries):
    manifest = json.load(open(os.path.join(ROOT, "plugin", "plugin.json")))
    return {
        "$schema": "https://go.kicad.org/pcm/schemas/v1",
        "name": manifest["name"],
        "description": manifest["description"],
        "description_full": (
            "Tracewright routes the open board orthogonally on the GPU (Metal, Vulkan or DX12) or CPU, "
            "using the board's own design rules, net classes, keepouts and planes, with a live preview. "
            "The result is applied as a single undo step.\n\n"
            "Requires KiCad 10 with Preferences > Plugins > Enable KiCad API turned on. "
            "After installing, restart the PCB Editor; the 'Tracewright: Autoroute' button appears in its toolbar."
        ),
        "identifier": IDENTIFIER,
        "type": "plugin",
        "author": {"name": "Alex Raddas", "contact": {"github": HOMEPAGE}},
        "license": "MIT",
        "resources": {"homepage": HOMEPAGE},
        "tags": ["autorouter", "routing", "gpu", "pcb"],
        "versions": version_entries,
    }


def version_entry(version, platform):
    return {"version": version, "status": "stable", "kicad_version": KICAD_MIN,
            "platforms": [platform], "runtime": "ipc"}


def add_file(z, arcname, data, executable=False):
    info = zipfile.ZipInfo(arcname, date_time=(2024, 1, 1, 0, 0, 0))   # fixed time: reproducible zips
    info.create_system = 3   # unix, so KiCad reads the mode bits
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def cmd_package(a):
    if a.platform not in PLATFORMS: sys.exit(f"platform must be one of {PLATFORMS}")
    exe_name = "tracewright.exe" if a.platform == "windows" else "tracewright"
    manifest = json.load(open(os.path.join(ROOT, "plugin", "plugin.json")))
    for action in manifest["actions"]:
        action["entrypoint"] = exe_name
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"tracewright-{a.version}-{a.platform}.zip")
    icons = os.path.join(ROOT, "plugin", "icons")
    with zipfile.ZipFile(path, "w") as z:
        add_file(z, "metadata.json", json.dumps(package_meta([version_entry(a.version, a.platform)]), indent=2) + "\n")
        add_file(z, "plugins/plugin.json", json.dumps(manifest, indent=2) + "\n")
        add_file(z, f"plugins/{exe_name}", open(a.exe, "rb").read(), executable=True)
        for name in sorted(os.listdir(icons)):
            if name.endswith(".png"):
                add_file(z, f"plugins/icons/{name}", open(os.path.join(icons, name), "rb").read())
        add_file(z, "resources/icon.png", open(os.path.join(icons, "icon-light-48.png"), "rb").read())
    print(path)


def cmd_repo(a):
    entries = []
    for zpath in a.zips:
        with zipfile.ZipFile(zpath) as z:
            meta = json.loads(z.read("metadata.json"))
            install_size = sum(i.file_size for i in z.infolist())
        v = meta["versions"][0]
        if v["version"] != a.version: sys.exit(f"{zpath} is version {v['version']}, expected {a.version}")
        data = open(zpath, "rb").read()
        v.update({"download_url": f"{a.base_url.rstrip('/')}/{os.path.basename(zpath)}",
                  "download_sha256": hashlib.sha256(data).hexdigest(),
                  "download_size": len(data), "install_size": install_size})
        entries.append(v)
    if not entries: sys.exit("no package zips given")
    os.makedirs(a.out, exist_ok=True)
    packages = json.dumps({"packages": [package_meta(entries)]}, indent=2) + "\n"
    open(os.path.join(a.out, "packages.json"), "w").write(packages)
    now = int(time.time())
    repository = {
        "$schema": "https://go.kicad.org/pcm/schemas/v1#/definitions/Repository",
        "name": "Tracewright",
        "maintainer": {"name": "Alex Raddas", "contact": {"github": HOMEPAGE}},
        "packages": {"url": a.packages_url or f"{a.base_url.rstrip('/')}/packages.json",
                     "sha256": hashlib.sha256(packages.encode()).hexdigest(),
                     "update_timestamp": now,
                     "update_time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))},
    }
    open(os.path.join(a.out, "repository.json"), "w").write(json.dumps(repository, indent=2) + "\n")
    print("platforms:", ", ".join(e["platforms"][0] for e in entries))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
pk = sub.add_parser("package"); pk.add_argument("--platform", required=True); pk.add_argument("--exe", required=True)
pk.add_argument("--version", required=True); pk.add_argument("--out", default="dist")
rp = sub.add_parser("repo"); rp.add_argument("--version", required=True); rp.add_argument("--base-url", required=True)
rp.add_argument("--packages-url", help="where packages.json is served (default: next to the zips)")
rp.add_argument("--out", default="dist"); rp.add_argument("zips", nargs="+")
a = p.parse_args()
cmd_package(a) if a.cmd == "package" else cmd_repo(a)
