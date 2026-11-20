# Tracewright — GPU orthogonal autorouter for KiCad

A native KiCad 10 plugin (Rust) that routes a board's nets with a GPU maze search and
negotiated congestion. Orthogonal (Manhattan) routing on up to 8 copper layers (the engine
handles 32), driven by the board's own design rules, with output that is DRC-clean by
construction.

## Using it

1. In KiCad: **Preferences → Plugins → Enable KiCad API** (must be on).
2. Install, then restart the PCB Editor:
   * Windows: `tools\install_plugin.ps1` (copies `dist\tracewright.exe`, the manifest and icons
     to `Documents\KiCad\10.0\plugins\com.tracewright.router`).
   * macOS / Linux: `tools/install_plugin.sh` (copies `target/release/tracewright` to
     `~/Documents/KiCad/10.0/plugins/com.tracewright.router` on macOS, or
     `~/.local/share/kicad/10.0/plugins/...` on Linux). The binary is unsigned: a copy you
     downloaded rather than built carries Gatekeeper's quarantine flag, which the script removes
     (`xattr -d com.apple.quarantine`); otherwise macOS refuses to launch it from KiCad.
3. Click **Tracewright: Autoroute** in the PCB Editor toolbar. The window shows:
   * **Design rules** per net class (clearance, track width, via size/drill), copper-to-edge,
     hole clearance, hole-to-hole, pre-filled from the board and its `.kicad_pro`.
   * **Layers**: routable on/off, preferred direction (H/V/any), or *plane* for a net
     (inner layers covering most of the board with a zone are detected as planes; their net is
     fanned out to them with vias).
   * **Routing quality**: grid pitch (auto), via cost, bend cost, wrong-way factor, comfort
     margin, 90° or 45° corners, end tracks at pad centres, via-in-pad.
   * **Nets**: keep existing tracks and finish the rest, skip / only-route globs.
   * **Engine**: Auto / GPU / CPU, iteration cap, congestion schedule.
4. **Route** shows the routing converge live; **Apply to KiCad** adds all tracks and vias as one
   undo step and refills zones. Settings are remembered per user.

Headless: `tracewright-cli route board.json --svg out.svg`, `tracewright-cli route-live`
(route the board open in KiCad), `tracewright-cli dump`, `tracewright-cli gpu-info`.

## How it works

* **Grid + exact clearance.** Tracks run node-to-node on a grid whose pitch and origin are
  chosen so most pad centres sit on nodes and tracks can pack at exactly minimum spacing
  (whole-nanometre pitch). For each probe class (track or via of a net class) every node is
  pre-classified as free / owned by one net / blocked. Static obstacles (pads from KiCad's own
  polygons, holes, copper text and graphics, keepouts, board edge) are inflated by
  `sqrt(R² + p²/4)`, which makes node checks also cover the segments between nodes; routed copper
  is exact at nodes. An independent geometric DRC re-checks every result.
* **GPU maze search** (`src/search`). State = (layer, node, facing H/V). One min-plus prefix
  scan per row and column line (a 256-thread workgroup each, Hillis–Steele with carry) relaxes
  bends, per-layer direction costs and vias; a via pass connects layer spans. Integer saturating
  costs make the wgpu (Vulkan/DX12/Metal) backend bit-identical to the multithreaded CPU
  fallback. Target selection and the path trace also run on the GPU, so only paths come back.
* **Router** (`src/router.rs`). Steiner growth per net with warm-started searches; plane nets
  fan out in clusters; neck-down to the board's minimum track width only next to pads the class
  width cannot reach or enter side by side; PathFinder negotiation (present + history
  congestion, growing windows for stuck nets); a final hard legalisation pass; own-net via
  hole spacing; segments merged, axis-aligned pad-centre stubs, optional 45° mitres — every
  off-grid addition validated geometrically.
* **KiCad link** (`src/kicad`). A small NNG REQ client (named pipe / Unix socket) speaking the
  KiCad IPC protobuf API; bindings are generated from a descriptor set extracted from kipy
  (`tools/extract_kicad_proto.py`), so no `protoc` is needed.

## Results (KiCad demo boards, all tracks stripped and rerouted, checked with `kicad-cli pcb drc --refill-zones`)

| Board | Layers | Nets | Result | Time (RTX 5080) |
|---|---|---|---|---|
| multichannel_mixer | 2 | 79 | 0 violations; all on-board nets connected (20 nets have parts parked off the board) | 0.8 s |
| kit-dev-coldfire-xilinx | 4 (2 planes) | 209 | 209/209 routed, 0 violations, 0 unconnected | ~16 s |
| RoyalBlue54L-Feather | 8 (6 planes) | 68 | **not solved yet**: ~44 nets stay in conflict around the WLCSP/QFN area | — |

Live test: board read over IPC matches KiCad's own geometry for all 327 pads; routes committed
into the running PCB editor as one undo step and saved; KiCad DRC clean.

## Building

Needs Rust 1.95 or newer (egui/eframe 0.37). CI (`.github/workflows/ci.yml`) builds and runs
`cargo test --lib` on macOS arm64 (Metal, GPU tests required to run), Linux and Windows (no GPU on
those runners, so only the CPU tests run there).

macOS (Apple Silicon): `cargo build --release`; the GPU backend uses Metal. Linux:
`cargo build --release` (needs `libfontconfig1-dev`, `libwayland-dev`, `libxkbcommon-dev`);
the GPU backend uses Vulkan.

Rust (GNU toolchain) on Windows without Visual Studio: `~/.cargo/config.toml` points rustc at
LLVM's dlltool and links with LLD (`lld-as-ld\ld.exe`, a copy of `rust-lld.exe`) because GNU ld
mis-links LLVM import libraries. Then `cargo build --release` and copy `tracewright.exe` to
`dist\`. `cargo test --lib` runs the CPU-vs-Dijkstra, GPU-vs-CPU (distances, pick and trace) and
geometry tests.

Dev tools (KiCad's Python): `tools/kicad_dump.py` (board → JSON), `tools/kicad_apply.py`
(result → .kicad_pcb), `tools/verify_board.ps1` (dump, route, apply, kicad-cli DRC, PNG),
`tools/render_png.py`, `tools/drc_summary.py`.

## Known limitations

* **Toolbar button not confirmed yet.** After installing, the "Tracewright: Autoroute" button
  did not show in the KiCad 10.0.8 PCB Editor toolbar during testing (`pcbnew.json` →
  `plugins.actions` stayed empty), so KiCad may not be registering the plugin. Running the
  installed `tracewright.exe` directly while the PCB Editor is open works (it connects over the
  default API socket). Probable cause: KiCad 10 skips any action whose `entrypoint` is an
  absolute path (`common/api/api_plugin.cpp`, "action contains abs path; skipping"), and
  `install_plugin.ps1` writes an absolute one. `install_plugin.sh` keeps it relative; the Windows
  script is unchanged until this is confirmed there.
* Pads outside the board outline are skipped and reported separately ("pads outside the board
  outline skipped"); they are not counted as routing failures.
* macOS: the engine, tests and headless CLI are verified on Apple M4 Pro (Metal):
  kit-dev-coldfire-xilinx routes 209/209 with output byte-identical to the CPU backend. The
  plugin has not yet been launched from KiCad's toolbar on macOS. Throughput is about 3.5× below
  the RTX 5080 (that board takes ~57 s on Metal vs ~37 s on the M4 Pro's CPU backend, all in the
  relaxation kernels, not CPU-GPU round trips), so on Apple Silicon **Engine: CPU** can be
  faster. Linux builds in CI but is untested on a real GPU.
* Through vias only in the router (the engine supports blind/buried spans; not exposed yet).
* Very dense boards with 2 signal layers and fine-pitch parts (RoyalBlue54L) do not converge yet.
* Net-class widths are used as given: boards whose classes keep KiCad's stock 0.25 mm / 0.8 mm
  values but were designed with smaller sizes need the sizes set in the dialog.
* No diff-pair, length matching or custom DRC rule (`.kicad_dru`) support.
