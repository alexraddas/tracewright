//! Headless command-line front end: route board JSON files, benchmark, inspect the GPU.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use tracewright::board::Board;
use tracewright::settings::{BackendChoice, Settings};

#[derive(Parser)]
#[command(name = "tracewright-cli", version, about = "GPU-accelerated orthogonal PCB autorouter")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Route a board JSON (from tools/kicad_dump.py or `dump`).
    Route {
        board: PathBuf,
        /// Settings JSON (defaults derived from the board when omitted).
        #[arg(long)]
        settings: Option<PathBuf>,
        #[arg(long, default_value = "result.json")]
        out: PathBuf,
        #[arg(long)]
        svg: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "auto")]
        backend: Backend,
        /// Override the grid pitch (mm).
        #[arg(long)]
        pitch: Option<f64>,
        /// Write the effective settings used.
        #[arg(long)]
        dump_settings: Option<PathBuf>,
    },
    /// Show the GPU adapter the router would use.
    GpuInfo,
    /// Connect to a running KiCad (IPC API) and dump the open board to JSON.
    Dump { out: PathBuf },
    /// Low-level check of the NNG transport against an NNG REP echo server.
    #[command(hide = true)]
    NngTest { url: String, msg: String },
    /// Print the running KiCad's version over the IPC API.
    Ping,
    /// Save the board open in KiCad.
    Save,
    /// Connect to a running KiCad and route the open board, committing the result.
    RouteLive {
        #[arg(long)]
        settings: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "auto")]
        backend: Backend,
        /// Route and report without changing the board.
        #[arg(long)]
        dry_run: bool,
        /// Save the board in KiCad after committing.
        #[arg(long)]
        save: bool,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Backend {
    Auto,
    Gpu,
    Cpu,
}

impl From<Backend> for BackendChoice {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Auto => BackendChoice::Auto,
            Backend::Gpu => BackendChoice::Gpu,
            Backend::Cpu => BackendChoice::Cpu,
        }
    }
}

fn load_settings(board: &Board, path: &Option<PathBuf>) -> Result<Settings> {
    let mut s = match path {
        Some(p) => serde_json::from_str::<Settings>(&std::fs::read_to_string(p).context("read settings")?)?,
        None => Settings::from_board(board),
    };
    if path.is_some() {
        s.reconcile(board);
    }
    Ok(s)
}

fn run_route(board: &Board, settings: &Settings) -> Result<tracewright::router::RouteResult> {
    let mut backend = tracewright::search::make_backend(settings.backend);
    eprintln!("backend: {}", backend.name());
    let cancel = AtomicBool::new(false);
    let mut last = std::time::Instant::now();
    let res = tracewright::router::route_board(board, settings, backend.as_mut(), &cancel, &mut |p| {
        if p.iteration <= 1 || last.elapsed().as_secs_f64() > 0.5 || p.conflicts == 0 {
            eprintln!("[{}] {}", p.phase, p.message);
            last = std::time::Instant::now();
        }
    })?;
    Ok(res)
}

fn report(res: &tracewright::router::RouteResult) {
    let s = &res.stats;
    for l in &res.log {
        eprintln!("  {l}");
    }
    eprintln!(
        "routed {}/{} nets ({} of {} connections failed, {} skipped: pads off the board), {} segments, {} vias, {:.1} mm, {} iterations, {} searches in {} batches, {:.2}s",
        s.nets_complete,
        s.nets_total,
        s.connections_failed,
        s.connections_total,
        s.connections_offboard,
        res.segs.len(),
        s.vias,
        s.wirelength_mm,
        s.iterations,
        s.searches,
        s.batches,
        s.seconds
    );
    eprintln!(
        "time: build {:.2}s, relax {:.2}s, readback {:.2}s, trace {:.2}s; {:.1} M nodes searched",
        s.t_build, s.t_relax, s.t_read, s.t_trace, s.nodes_searched as f64 / 1e6
    );
    if !res.incomplete_nets.is_empty() {
        eprintln!("could not route: {}", res.incomplete_nets.join(", "));
    }
    if !res.offboard_pads.is_empty() {
        eprintln!("skipped {} pads outside the board outline: {}", res.offboard_pads.len(), res.offboard_pads.join(", "));
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Route { board, settings, out, svg, backend, pitch, dump_settings } => {
            let mut b: Board = serde_json::from_str(&std::fs::read_to_string(&board).context("read board")?).context("parse board")?;
            b.normalize();
            let mut s = load_settings(&b, &settings)?;
            s.backend = backend.into();
            if let Some(p) = pitch {
                s.grid_pitch = p;
            }
            if let Some(p) = dump_settings {
                std::fs::write(p, serde_json::to_string_pretty(&s)?)?;
            }
            let res = run_route(&b, &s)?;
            report(&res);
            let viol = tracewright::drc::check_clearance(&b, &s, &res.segs, &res.vias, 1e-6);
            eprintln!("geometric DRC: {} clearance violations", viol.len());
            for v in viol.iter().take(10) {
                eprintln!("  {} nets {}/{} layer {} at ({:.3},{:.3}): {:.4} < {:.4}", v.kind, b.net_name(v.net_a), b.net_name(v.net_b), v.layer, v.at.x, v.at.y, v.actual, v.required);
            }
            std::fs::write(&out, serde_json::to_string(&res)?)?;
            if let Some(p) = svg {
                std::fs::write(p, tracewright::render::svg(&b, &res.segs, &res.vias))?;
            }
        }
        Cmd::NngTest { url, msg } => {
            let mut s = tracewright::kicad::nng::ReqSocket::connect(&url)?;
            for i in 0..3 {
                let r = s.request(format!("{msg}{i}").as_bytes())?;
                println!("{}", String::from_utf8_lossy(&r));
            }
        }
        Cmd::Ping => {
            let mut k = tracewright::kicad::Kicad::connect("tracewright-cli")?;
            println!("KiCad {}", k.version()?);
        }
        Cmd::Save => {
            let mut k = tracewright::kicad::Kicad::connect("tracewright-cli")?;
            let (_, ctx) = k.read_board()?;
            k.save(&ctx)?;
            eprintln!("saved");
        }
        Cmd::GpuInfo => match tracewright::search::gpu::GpuBackend::new() {
            Ok(g) => println!("{}", g.adapter_name()),
            Err(e) => println!("no usable GPU: {e:#}"),
        },
        Cmd::Dump { out } => {
            let mut k = tracewright::kicad::Kicad::connect("tracewright-cli")?;
            let (b, _) = k.read_board()?;
            std::fs::write(&out, serde_json::to_string(&b)?)?;
            eprintln!("{}: {} layers, {} nets, {} pads", out.display(), b.layers.len(), b.nets.len(), b.pads.len());
        }
        Cmd::RouteLive { settings, backend, dry_run, save } => {
            let mut k = tracewright::kicad::Kicad::connect("tracewright-cli")?;
            let (b, ctx) = k.read_board()?;
            let mut s = load_settings(&b, &settings)?;
            s.backend = backend.into();
            let res = run_route(&b, &s)?;
            report(&res);
            if !dry_run {
                k.commit_routes(&b, &ctx, &res, true)?;
                eprintln!("committed to KiCad (one undo step)");
                if save {
                    k.save(&ctx)?;
                    eprintln!("saved");
                }
            }
        }
    }
    Ok(())
}
