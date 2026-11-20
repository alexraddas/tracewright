//! Multithreaded CPU reference backend. Same sweep order and integer arithmetic as the GPU
//! shader, so results are bit-identical.

use super::{Backend, INF, SearchJob, sat};
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
struct SharedMut(*mut u32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}
impl SharedMut {
    #[inline]
    unsafe fn get(self, i: usize) -> u32 {
        unsafe { *self.0.add(i) }
    }
    #[inline]
    unsafe fn set(self, i: usize, v: u32) {
        unsafe { *self.0.add(i) = v }
    }
}

pub struct CpuBackend {
    jobs: Vec<Arc<SearchJob>>,
    dist: Vec<Vec<u32>>,
    pending: Vec<Vec<u32>>,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend { jobs: vec![], dist: vec![], pending: vec![] }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Scan one line: `base + k*stride` for k in 0..len, forward or backward.
#[inline]
fn scan_line(job: &SearchJob, dist: SharedMut, n: usize, o: usize, base: usize, len: usize, stride: usize, backward: bool) -> bool {
    let own_off = o * n;
    let oth_off = (1 - o) * n;
    let mut running = INF;
    let mut changed = false;
    for i in 0..len {
        let k = if backward { len - 1 - i } else { i };
        let e = base + k * stride;
        unsafe {
            let own = dist.get(own_off + e);
            let oth = dist.get(oth_off + e);
            let v = own.min(sat(oth, job.bend));
            let a = sat(running, job.enter(e, o)).min(v);
            if a < own {
                dist.set(own_off + e, a);
                changed = true;
            }
            running = a;
        }
    }
    changed
}

/// One full relaxation iteration (rows fwd/bwd, columns fwd/bwd, vias). Returns whether any
/// distance decreased.
pub fn relax_iteration(job: &SearchJob, dist: &mut [u32]) -> bool {
    let n = job.n();
    let plane = job.plane();
    let (w, h) = (job.w as usize, job.h as usize);
    let ptr = SharedMut(dist.as_mut_ptr());
    let changed = AtomicBool::new(false);
    let layers: Vec<usize> = (0..job.nl as usize).filter(|&l| job.sweep_layer[l]).collect();
    for o in 0..2 {
        let lines: Vec<(usize, usize, usize)> = if o == 0 {
            layers.iter().flat_map(|&l| (0..h).map(move |y| (l * plane + y * w, w, 1))).collect()
        } else {
            layers.iter().flat_map(|&l| (0..w).map(move |x| (l * plane + x, h, w))).collect()
        };
        for backward in [false, true] {
            let any = lines.par_iter().map(|&(base, len, stride)| scan_line(job, ptr, n, o, base, len, stride, backward)).reduce(|| false, |a, b| a | b);
            if any {
                changed.store(true, Ordering::Relaxed);
            }
        }
    }
    if job.nv() > 0 {
        let any = (0..plane)
            .into_par_iter()
            .map(|p| {
                let mut ch = false;
                for (v, &(a, b)) in job.via_spans.iter().enumerate() {
                    let cv = job.via[v * plane + p];
                    if cv >= INF {
                        continue;
                    }
                    let mut m = INF;
                    for l in a..=b {
                        let e = l as usize * plane + p;
                        unsafe {
                            m = m.min(ptr.get(e)).min(ptr.get(n + e));
                        }
                    }
                    let t = sat(m, cv);
                    if t >= INF {
                        continue;
                    }
                    for l in a..=b {
                        let e = l as usize * plane + p;
                        unsafe {
                            if t < ptr.get(e) {
                                ptr.set(e, t);
                                ch = true;
                            }
                            if t < ptr.get(n + e) {
                                ptr.set(n + e, t);
                                ch = true;
                            }
                        }
                    }
                }
                ch
            })
            .reduce(|| false, |a, b| a | b);
        if any {
            changed.store(true, Ordering::Relaxed);
        }
    }
    changed.load(Ordering::Relaxed)
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        format!("CPU ({} threads)", rayon::current_num_threads())
    }

    fn load(&mut self, jobs: Vec<Arc<SearchJob>>) -> Result<()> {
        for j in &jobs {
            j.validate()?;
        }
        self.dist = jobs.iter().map(|j| vec![INF; 2 * j.n()]).collect();
        self.pending = jobs.iter().map(|_| Vec::new()).collect();
        self.jobs = jobs;
        Ok(())
    }

    fn seed(&mut self, slot: usize, nodes: &[u32]) {
        self.pending[slot].extend_from_slice(nodes);
    }

    fn relax(&mut self, slots: &[usize]) -> Result<()> {
        for &s in slots {
            let job = self.jobs[s].clone();
            let n = job.n();
            let dist = &mut self.dist[s];
            for &e in &self.pending[s] {
                dist[e as usize] = 0;
                dist[n + e as usize] = 0;
            }
            self.pending[s].clear();
            while relax_iteration(&job, dist) {}
        }
        Ok(())
    }

    fn read(&mut self, slots: &[usize]) -> Result<Vec<Vec<u32>>> {
        Ok(slots.iter().map(|&s| self.dist[s].clone()).collect())
    }

    fn pick(&mut self, slots: &[usize], targets: &[Vec<u32>]) -> Result<Vec<Option<super::Pick>>> {
        slots.iter().zip(targets).map(|(&s, t)| super::pick_cpu(&self.jobs[s], &self.dist[s], t)).collect()
    }
}
