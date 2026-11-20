use super::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Tiny deterministic PRNG for tests.
pub struct Rng(u64);
impl Rng {
    pub fn new(s: u64) -> Self {
        Rng(s.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo + 1) as u64) as u32
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() % 1_000_000) as f64 / 1e6 < p
    }
}

fn random_job(r: &mut Rng) -> (SearchJob, Vec<u32>) {
    let w = r.range(3, 45);
    let h = r.range(3, 45);
    let nl = r.range(1, 4);
    let plane = (w * h) as usize;
    let n = plane * nl as usize;
    let block = r.range(0, 35) as f64 / 100.0;
    let cell: Vec<u32> = (0..n).map(|_| if r.chance(block) { INF } else { r.range(16, 80) }).collect();
    let mut via_spans = vec![(0, nl - 1)];
    if nl >= 2 && r.chance(0.5) {
        via_spans.push((0, 1));
    }
    let bend = r.range(1, 200);
    let via: Vec<u32> = (0..via_spans.len() * plane).map(|_| if r.chance(0.3) { INF } else { r.range(bend, bend + 600) }).collect();
    let dirfac = (0..nl).map(|_| [r.range(16, 60), r.range(16, 60)]).collect();
    let sweep_layer = (0..nl).map(|_| !r.chance(0.15)).collect();
    let job = SearchJob { w, h, nl, cell, via, via_spans, dirfac, bend, sweep_layer };
    let seeds: Vec<u32> = (0..r.range(1, 3)).map(|_| r.range(0, n as u32 - 1)).collect();
    (job, seeds)
}

/// Plain Dijkstra over (orientation, node) states with the same cost model.
fn dijkstra(job: &SearchJob, seeds: &[u32]) -> Vec<u32> {
    let n = job.n();
    let plane = job.plane();
    let (w, h) = (job.w as usize, job.h as usize);
    let mut dist = vec![INF; 2 * n];
    let mut heap = BinaryHeap::new();
    for &s in seeds {
        for o in 0..2 {
            dist[o * n + s as usize] = 0;
            heap.push(Reverse((0u32, o * n + s as usize)));
        }
    }
    while let Some(Reverse((d, st))) = heap.pop() {
        if d > dist[st] {
            continue;
        }
        let (o, e) = (st / n, st % n);
        let l = e / plane;
        let p = e % plane;
        let (y, x) = (p / w, p % w);
        let mut relax = |to: usize, c: u32, heap: &mut BinaryHeap<Reverse<(u32, usize)>>| {
            let nd = sat(d, c);
            if nd < dist[to] {
                dist[to] = nd;
                heap.push(Reverse((nd, to)));
            }
        };
        if job.sweep_layer[l] {
            relax((1 - o) * n + e, job.bend, &mut heap);
            let nbrs: Vec<usize> = if o == 0 {
                [(x > 0).then(|| e - 1), (x + 1 < w).then(|| e + 1)].into_iter().flatten().collect()
            } else {
                [(y > 0).then(|| e - w), (y + 1 < h).then(|| e + w)].into_iter().flatten().collect()
            };
            for q in nbrs {
                let c = job.enter(q, o);
                if c < INF {
                    relax(o * n + q, c, &mut heap);
                }
            }
        }
        for (v, &(a, b)) in job.via_spans.iter().enumerate() {
            if (l as u32) < a || (l as u32) > b {
                continue;
            }
            let cv = job.via[v * plane + p];
            if cv >= INF {
                continue;
            }
            for l2 in a..=b {
                for o2 in 0..2 {
                    relax(o2 * n + l2 as usize * plane + p, cv, &mut heap);
                }
            }
        }
    }
    dist
}

fn run_backend(b: &mut dyn Backend, jobs: &[(SearchJob, Vec<u32>)]) -> Vec<Vec<u32>> {
    b.load(jobs.iter().map(|(j, _)| Arc::new(j.clone())).collect()).unwrap();
    for (i, (_, s)) in jobs.iter().enumerate() {
        b.seed(i, s);
    }
    let slots: Vec<usize> = (0..jobs.len()).collect();
    b.relax(&slots).unwrap();
    b.read(&slots).unwrap()
}

/// Re-cost a traced path independently.
fn path_cost(job: &SearchJob, steps: &[Step]) -> u32 {
    let plane = job.plane();
    let w = job.w as usize;
    let mut cost = 0u32;
    let mut orient: Option<usize> = None;
    for i in 1..steps.len() {
        let (a, b) = (steps[i - 1], steps[i]);
        if a.layer != b.layer {
            assert_eq!(a.p, b.p, "via must not move");
            let v = b.via.expect("via step marked") as usize;
            cost = sat(cost, job.via[v * plane + a.p as usize]);
            orient = None;
            continue;
        }
        let (ay, ax) = (a.p as usize / w, a.p as usize % w);
        let (by, bx) = (b.p as usize / w, b.p as usize % w);
        let o = if ay == by { 0 } else { 1 };
        assert_eq!(ay.abs_diff(by) + ax.abs_diff(bx), 1, "unit steps only");
        if let Some(prev) = orient {
            if prev != o {
                cost = sat(cost, job.bend);
            }
        }
        orient = Some(o);
        cost = sat(cost, job.enter(b.layer as usize * plane + b.p as usize, o));
    }
    cost
}

#[test]
fn cpu_matches_dijkstra_and_backtrace_is_optimal() {
    let mut r = Rng::new(7);
    for case in 0..250 {
        let (job, seeds) = random_job(&mut r);
        let reference = dijkstra(&job, &seeds);
        let mut cpu = cpu::CpuBackend::new();
        let got = run_backend(&mut cpu, &[(job.clone(), seeds.clone())]).pop().unwrap();
        assert_eq!(got, reference, "case {case}: CPU sweep differs from Dijkstra");
        // Backtrace from a few reachable nodes.
        let n = job.n();
        for _ in 0..5 {
            let e = r.range(0, n as u32 - 1);
            let o = if got[e as usize] <= got[n + e as usize] { 0 } else { 1 };
            let d = got[o * n + e as usize];
            if d >= INF {
                continue;
            }
            let path = backtrace(&job, &got, e, o).unwrap();
            assert!(seeds.contains(&(path[0].layer * job.plane() as u32 + path[0].p)), "case {case}: path must start at a seed");
            let last = path.last().unwrap();
            assert_eq!(last.layer * job.plane() as u32 + last.p, e);
            assert!(path_cost(&job, &path) <= d, "case {case}: traced path costs more than its distance");
        }
    }
}

#[test]
fn gpu_matches_cpu_bit_for_bit() {
    let Ok(mut gpu) = gpu::GpuBackend::new() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut r = Rng::new(99);
    for round in 0..12 {
        let jobs: Vec<(SearchJob, Vec<u32>)> = (0..r.range(1, 12)).map(|_| random_job(&mut r)).collect();
        let g = run_backend(&mut gpu, &jobs);
        let mut cpu = cpu::CpuBackend::new();
        let c = run_backend(&mut cpu, &jobs);
        for (i, (a, b)) in g.iter().zip(&c).enumerate() {
            assert!(a == b, "round {round} job {i}: GPU and CPU distances differ");
        }
    }
    // A long line (multiple 256-element chunks) exercises the scan carry.
    let w = 1500;
    let job = SearchJob {
        w,
        h: 3,
        nl: 2,
        cell: (0..w * 3 * 2).map(|i| 16 + (i * 7919 % 13)).collect(),
        via: vec![400; (w * 3) as usize],
        via_spans: vec![(0, 1)],
        dirfac: vec![[16, 40], [40, 16]],
        bend: 30,
        sweep_layer: vec![true, true],
    };
    let jobs = vec![(job, vec![0u32, 2 * w * 3 - 1])];
    let g = run_backend(&mut gpu, &jobs);
    let mut cpu = cpu::CpuBackend::new();
    assert!(g == run_backend(&mut cpu, &jobs), "long-line GPU result differs");
}

#[test]
fn gpu_pick_and_trace_match_cpu() {
    let Ok(mut gpu) = gpu::GpuBackend::new() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut r = Rng::new(4242);
    for round in 0..20 {
        let jobs: Vec<(SearchJob, Vec<u32>)> = (0..r.range(1, 10)).map(|_| random_job(&mut r)).collect();
        let targets: Vec<Vec<u32>> = jobs.iter().map(|(j, _)| (0..r.range(1, 40)).map(|_| r.range(0, j.n() as u32 - 1)).collect()).collect();
        let slots: Vec<usize> = (0..jobs.len()).collect();
        run_backend(&mut gpu, &jobs);
        let g = gpu.pick(&slots, &targets).unwrap();
        let mut cpu = cpu::CpuBackend::new();
        run_backend(&mut cpu, &jobs);
        let c = cpu.pick(&slots, &targets).unwrap();
        assert_eq!(g, c, "round {round}: GPU pick/trace differs from CPU");
    }
}

#[test]
fn warm_start_with_added_seeds() {
    // Relax, add seeds, relax again: must equal a fresh solve with all seeds.
    let mut r = Rng::new(3);
    let backends: Vec<Box<dyn Backend>> = {
        let mut v: Vec<Box<dyn Backend>> = vec![Box::new(cpu::CpuBackend::new())];
        if let Ok(g) = gpu::GpuBackend::new() {
            v.push(Box::new(g));
        }
        v
    };
    for mut b in backends {
        for _ in 0..20 {
            let (job, seeds) = random_job(&mut r);
            let extra: Vec<u32> = (0..3).map(|_| r.range(0, job.n() as u32 - 1)).collect();
            b.load(vec![Arc::new(job.clone())]).unwrap();
            b.seed(0, &seeds);
            b.relax(&[0]).unwrap();
            b.seed(0, &extra);
            b.relax(&[0]).unwrap();
            let warm = b.read(&[0]).unwrap().pop().unwrap();
            let all: Vec<u32> = seeds.iter().chain(&extra).copied().collect();
            assert_eq!(warm, dijkstra(&job, &all), "{}: warm start differs", b.name());
        }
    }
}
