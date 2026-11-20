//! Multi-layer, direction-aware maze search with bend and via costs.
//!
//! State per node: `dist[o][l][y][x]`, the cheapest cost of standing on node (l, y, x) facing
//! orientation `o` (0 = horizontal, 1 = vertical). Moving one node along `o` costs the
//! entering cost of the next node (per-layer direction factor applied), turning costs `bend`,
//! and a via of type v at a node costs `via[v]` and connects every layer in its span.
//!
//! The solver is a fixed-point relaxation built from four sweeps (rows forward/backward,
//! columns forward/backward) and a via pass. Each sweep is a min-plus prefix scan along a line,
//! which the GPU runs as a parallel workgroup scan and the CPU runs sequentially. All costs are
//! saturating u32 integers, so both backends reach bit-identical distances.

pub mod cpu;
pub mod gpu;

use anyhow::{Result, bail};
use std::sync::Arc;

/// Unreachable / blocked.
pub const INF: u32 = 1 << 30;
/// Maximum copper layers in one job (KiCad supports 32).
pub const MAX_LAYERS: usize = 32;
/// Maximum via types per job.
pub const MAX_VIA_TYPES: usize = 8;
/// Fixed-point scale of direction factors (16 = 1.0).
pub const DIR_ONE: u32 = 16;
/// Largest finite cell cost (keeps cost * factor within u32).
pub const MAX_CELL_COST: u32 = (1 << 20) - 1;

#[inline]
pub fn sat(a: u32, b: u32) -> u32 {
    (a + b).min(INF)
}

/// One search problem on a rectangular window of the board grid.
#[derive(Clone, Debug)]
pub struct SearchJob {
    pub w: u32,
    pub h: u32,
    pub nl: u32,
    /// Entering cost per node, `nl*h*w`; `INF` = blocked; finite values in 1..=MAX_CELL_COST.
    pub cell: Vec<u32>,
    /// Via cost per via type per node, `nv*h*w`; `INF` = via not allowed.
    pub via: Vec<u32>,
    /// Inclusive layer span of each via type.
    pub via_spans: Vec<(u32, u32)>,
    /// Per layer [horizontal, vertical] multipliers in 1/16 units (>= 16).
    pub dirfac: Vec<[u32; 2]>,
    pub bend: u32,
    /// Layers that need sweeping (others only receive vias, e.g. plane layers).
    pub sweep_layer: Vec<bool>,
}

impl SearchJob {
    #[inline]
    pub fn plane(&self) -> usize {
        (self.w * self.h) as usize
    }
    #[inline]
    pub fn n(&self) -> usize {
        self.plane() * self.nl as usize
    }
    pub fn nv(&self) -> usize {
        self.via_spans.len()
    }
    /// Cost of entering node `e` (layer-major index) moving along orientation `o`.
    #[inline]
    pub fn enter(&self, e: usize, o: usize) -> u32 {
        let c = self.cell[e];
        if c >= INF {
            return INF;
        }
        let l = e / self.plane();
        ((c * self.dirfac[l][o]) >> 4).min(INF)
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.n();
        if self.nl as usize > MAX_LAYERS || self.nv() > MAX_VIA_TYPES {
            bail!("job exceeds layer/via-type limits");
        }
        if self.cell.len() != n || self.via.len() != self.nv() * self.plane() || self.dirfac.len() != self.nl as usize {
            bail!("job array sizes do not match its dimensions");
        }
        if self.bend == 0 || self.dirfac.iter().any(|d| d[0] < DIR_ONE || d[1] < DIR_ONE) {
            bail!("bend and direction factors must be positive");
        }
        if self.cell.iter().any(|&c| c == 0 || (c > MAX_CELL_COST && c < INF)) || self.via.iter().any(|&c| c == 0) {
            bail!("costs must be in 1..=MAX_CELL_COST or INF");
        }
        // A via also connects its own layer to itself, so a via cheaper than a bend would act
        // as a "turn" that has no physical meaning. With via >= bend, a plain turn is never
        // worse, and backtracing never needs such a via.
        if self.via.iter().any(|&c| c < self.bend) {
            bail!("via costs must be >= bend cost");
        }
        for &(a, b) in &self.via_spans {
            if a > b || b >= self.nl {
                bail!("bad via span");
            }
        }
        Ok(())
    }
}

/// A search backend holds one batch of jobs at a time.
pub trait Backend: Send {
    fn name(&self) -> String;
    /// Load a batch of jobs, replacing the previous one. All distances start at `INF`.
    fn load(&mut self, jobs: Vec<Arc<SearchJob>>) -> Result<()>;
    /// Set distance 0 (both orientations) at the given node indices of job `slot`.
    fn seed(&mut self, slot: usize, nodes: &[u32]);
    /// Relax the given slots to their fixed point (applies pending seeds first).
    fn relax(&mut self, slots: &[usize]) -> Result<()>;
    /// Read back `dist` (2 * n values, orientation-major) of the given slots.
    fn read(&mut self, slots: &[usize]) -> Result<Vec<Vec<u32>>>;
    /// For each slot, find the cheapest of its target nodes (ties: lowest node index) and
    /// trace the path back to a seed. `None` when no target is reachable.
    fn pick(&mut self, slots: &[usize], targets: &[Vec<u32>]) -> Result<Vec<Option<Pick>>>;
}

/// Result of `Backend::pick`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    pub cost: u32,
    /// Target node reached (layer-major index).
    pub node: u32,
    /// Path from a seed to the target.
    pub steps: Vec<Step>,
}

/// CPU reference for `Backend::pick` on one job.
pub fn pick_cpu(job: &SearchJob, dist: &[u32], targets: &[u32]) -> Result<Option<Pick>> {
    let Some((cost, node, o)) = best_target(job, dist, targets) else { return Ok(None) };
    let steps = backtrace(job, dist, node, o)?;
    Ok(Some(Pick { cost, node, steps }))
}

/// One node on a traced path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub layer: u32,
    /// Index within the layer plane (y * w + x).
    pub p: u32,
    /// Via type used to arrive at this node from the previous step (layer change).
    pub via: Option<u8>,
}

/// Best (cost, node, orientation) among candidate nodes.
pub fn best_target(job: &SearchJob, dist: &[u32], nodes: &[u32]) -> Option<(u32, u32, usize)> {
    let n = job.n();
    let mut best: Option<(u32, u32, usize)> = None;
    for &e in nodes {
        for o in 0..2 {
            let d = dist[o * n + e as usize];
            if d < INF && best.is_none_or(|b| d < b.0 || (d == b.0 && e < b.1)) {
                best = Some((d, e, o));
            }
        }
    }
    best
}

/// Trace the optimal path back from node `e` (orientation `o`) to a seed. Returns the steps
/// from seed to target.
pub fn backtrace(job: &SearchJob, dist: &[u32], e: u32, o: usize) -> Result<Vec<Step>> {
    let n = job.n();
    let plane = job.plane();
    let w = job.w as usize;
    let h = job.h as usize;
    let (mut e, mut o) = (e as usize, o);
    let mut d = dist[o * n + e];
    if d >= INF {
        bail!("target unreachable");
    }
    let mut steps = vec![Step { layer: (e / plane) as u32, p: (e % plane) as u32, via: None }];
    // Direction of the last move (+1/-1 along the orientation) to keep straight runs straight.
    let mut last_dir: i32 = 0;
    let mut guard = 0usize;
    'walk: while d > 0 {
        guard += 1;
        if guard > 8 * n + 16 {
            bail!("backtrace did not terminate");
        }
        let l = e / plane;
        let p = e % plane;
        let (y, x) = (p / w, p % w);
        // Moves and turns only happen on swept layers; other layers are reached by vias only.
        let swept = job.sweep_layer[l];
        // 1) Straight move from a neighbour along orientation o.
        let c = if swept { job.enter(e, o) } else { INF };
        if c < INF {
            let mut cand: [(i32, Option<usize>); 2] = [(-1, None), (1, None)];
            for item in cand.iter_mut() {
                let s = item.0;
                item.1 = if o == 0 {
                    let nx = x as i64 - s as i64;
                    (nx >= 0 && (nx as usize) < w).then(|| l * plane + y * w + nx as usize)
                } else {
                    let ny = y as i64 - s as i64;
                    (ny >= 0 && (ny as usize) < h).then(|| l * plane + ny as usize * w + x)
                };
            }
            if last_dir == 1 {
                cand.swap(0, 1);
            }
            for (s, q) in cand {
                let Some(q) = q else { continue };
                let dq = dist[o * n + q];
                if dq < INF && sat(dq, c) == d {
                    e = q;
                    d = dq;
                    last_dir = s;
                    steps.push(Step { layer: l as u32, p: (q % plane) as u32, via: None });
                    continue 'walk;
                }
            }
        }
        // 2) Via from another layer.
        for (v, &(a, b)) in job.via_spans.iter().enumerate() {
            if (l as u32) < a || (l as u32) > b {
                continue;
            }
            let cv = job.via[v * plane + p];
            if cv >= INF {
                continue;
            }
            for l2 in a..=b {
                let l2 = l2 as usize;
                if l2 == l {
                    continue;
                }
                let e2 = l2 * plane + p;
                let (dh, dv) = (dist[e2], dist[n + e2]);
                let (m, o2) = if dh <= dv { (dh, 0) } else { (dv, 1) };
                if m < INF && sat(m, cv) == d {
                    if let Some(last) = steps.last_mut() {
                        last.via = Some(v as u8);
                    }
                    e = e2;
                    o = o2;
                    d = m;
                    last_dir = 0;
                    steps.push(Step { layer: l2 as u32, p: p as u32, via: None });
                    continue 'walk;
                }
            }
        }
        // 3) Turn in place.
        let other = dist[(1 - o) * n + e];
        if swept && other < INF && sat(other, job.bend) == d {
            o = 1 - o;
            d = other;
            last_dir = 0;
            continue 'walk;
        }
        bail!("backtrace found no predecessor (inconsistent distances)");
    }
    // Walking backwards, a via mark was put on the node reached *after* the via in forward
    // order, so after reversing, `via` on step i means "arrived from step i-1 through a via".
    steps.reverse();
    Ok(steps)
}

/// Pick the best available backend.
pub fn make_backend(choice: crate::settings::BackendChoice) -> Box<dyn Backend> {
    use crate::settings::BackendChoice;
    match choice {
        BackendChoice::Cpu => Box::new(cpu::CpuBackend::new()),
        BackendChoice::Gpu | BackendChoice::Auto => match gpu::GpuBackend::new() {
            Ok(g) => Box::new(g),
            Err(e) => {
                if choice == BackendChoice::Gpu {
                    eprintln!("GPU backend unavailable ({e:#}); falling back to CPU");
                }
                Box::new(cpu::CpuBackend::new())
            }
        },
    }
}

#[cfg(test)]
mod tests;
