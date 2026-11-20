//! wgpu compute backend (Vulkan / DX12 / Metal). Runs a whole batch of search jobs together:
//! every row/column line of every active job is one workgroup in a single dispatch.

use super::{Backend, INF, MAX_LAYERS, MAX_VIA_TYPES, Pick, SearchJob, Step};
use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::sync::Arc;

const WG: u32 = 256;
const ROW: u32 = 65535;
/// Relaxation iterations in the first GPU submission before checking convergence; later
/// submissions double this (up to `MAX_ITERS_PER_SUBMIT`) to cut CPU-GPU round trips.
const ITERS_PER_SUBMIT: usize = 4;
const MAX_ITERS_PER_SUBMIT: usize = 32;
/// Maximum traced path length (must match CAP in sweep.wgsl).
const TRACE_CAP: usize = 4096;
const NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GJob {
    cell_off: u32,
    dist_off: u32,
    via_off: u32,
    n: u32,
    plane: u32,
    nl: u32,
    nv: u32,
    bend: u32,
    dirfac: [[u32; 2]; MAX_LAYERS],
    spans: [[u32; 2]; MAX_VIA_TYPES],
    w: u32,
    sweep_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GLine {
    base: u32,
    len: u32,
    stride: u32,
    job: u32,
    layer: u32,
    orient: u32,
    pad0: u32,
    pad1: u32,
}

struct Buf {
    buf: wgpu::Buffer,
    cap: u64,
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    max_binding: u64,
    bgl: wgpu::BindGroupLayout,
    p_fwd: wgpu::ComputePipeline,
    p_bwd: wgpu::ComputePipeline,
    p_via: wgpu::ComputePipeline,
    p_seed: wgpu::ComputePipeline,
    p_fill: wgpu::ComputePipeline,
    bgl_t: wgpu::BindGroupLayout,
    p_pick_cost: wgpu::ComputePipeline,
    p_pick_node: wgpu::ComputePipeline,
    p_trace: wgpu::ComputePipeline,
    targets: Option<Buf>,
    best: Option<Buf>,
    paths: Option<Buf>,
    tmeta: Option<Buf>,
    staging_t: Option<Buf>,
    dist: Option<Buf>,
    cell: Option<Buf>,
    via: Option<Buf>,
    jobs_buf: Option<Buf>,
    lines_h: Option<Buf>,
    lines_v: Option<Buf>,
    tiles: Option<Buf>,
    seeds: Option<Buf>,
    flags: Option<Buf>,
    staging: Option<Buf>,
    flags_staging: Option<Buf>,
    jobs: Vec<Arc<SearchJob>>,
    gjobs: Vec<GJob>,
    dist_used: u64,
    line_cache: Vec<(Vec<GLine>, Vec<GLine>)>,
    tile_cache: Vec<Vec<[u32; 2]>>,
    pending: Vec<Vec<u32>>,
    /// Total relaxation iterations run (for statistics).
    pub iterations: u64,
}

fn ensure(device: &wgpu::Device, slot: &mut Option<Buf>, bytes: u64, usage: wgpu::BufferUsages, label: &str) {
    let bytes = bytes.max(16).div_ceil(256) * 256;
    if slot.as_ref().is_some_and(|b| b.cap >= bytes) {
        return;
    }
    let cap = (bytes + bytes / 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: cap, usage, mapped_at_creation: false });
    *slot = Some(Buf { buf, cap });
}

fn groups(count: u32) -> (u32, u32) {
    (count.clamp(1, ROW), count.div_ceil(ROW).max(1))
}

impl GpuBackend {
    pub fn new() -> Result<GpuBackend> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("only a software adapter ({}) is available", info.name);
        }
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 8 || limits.max_compute_workgroup_size_x < WG || limits.max_compute_invocations_per_workgroup < WG {
            bail!("GPU limits too small for the router kernels");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tracewright"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("request GPU device")?;
        device.on_uncaptured_error(Arc::new(|e| eprintln!("wgpu error: {e}")));

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sweep.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sweep.wgsl").into()),
        });
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sweep"),
            entries: &[entry(0, false), entry(1, true), entry(2, true), entry(3, true), entry(4, true), entry(5, true), entry(6, true), entry(7, false)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sweep"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (p_fwd, p_bwd, p_via, p_seed, p_fill) = (pipe("scan_fwd"), pipe("scan_bwd"), pipe("via_relax"), pipe("seed"), pipe("fill"));
        // Target pick + trace pipelines (their own layout keeps each within 8 storage buffers).
        let bgl_t = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace"),
            entries: &[entry(0, false), entry(1, true), entry(2, true), entry(4, true), entry(8, true), entry(9, false), entry(10, false), entry(11, false)],
        });
        let layout_t = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("trace"), bind_group_layouts: &[Some(&bgl_t)], immediate_size: 0 });
        let pipe_t = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout_t),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (p_pick_cost, p_pick_node, p_trace) = (pipe_t("pick_cost"), pipe_t("pick_node"), pipe_t("trace"));
        if let Some(err) = pollster::block_on(scope.pop()) {
            bail!("shader/pipeline creation failed: {err}");
        }
        let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        Ok(GpuBackend {
            device,
            queue,
            info,
            max_binding,
            bgl,
            p_fwd,
            p_bwd,
            p_via,
            p_seed,
            p_fill,
            bgl_t,
            p_pick_cost,
            p_pick_node,
            p_trace,
            targets: None,
            best: None,
            paths: None,
            tmeta: None,
            staging_t: None,
            dist: None,
            cell: None,
            via: None,
            jobs_buf: None,
            lines_h: None,
            lines_v: None,
            tiles: None,
            seeds: None,
            flags: None,
            staging: None,
            flags_staging: None,
            jobs: vec![],
            gjobs: vec![],
            dist_used: 0,
            line_cache: vec![],
            tile_cache: vec![],
            pending: vec![],
            iterations: 0,
        })
    }

    pub fn adapter_name(&self) -> String {
        format!("{} ({:?})", self.info.name, self.info.backend)
    }

    /// Largest number of u32 distance values one batch may hold.
    pub fn max_batch_dist(&self) -> u64 {
        self.max_binding / 4
    }

    fn binding<'a>(buf: &'a Option<Buf>, bytes: u64) -> wgpu::BindingResource<'a> {
        let b = buf.as_ref().expect("buffer allocated");
        wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &b.buf, offset: 0, size: NonZeroU64::new(bytes.max(4).min(b.cap)) })
    }

    fn bind_group(&self, lines: &Option<Buf>, n_lines: usize, n_tiles: usize, n_seeds: usize) -> wgpu::BindGroup {
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: Self::binding(&self.dist, self.dist_used * 4) },
            wgpu::BindGroupEntry { binding: 1, resource: Self::binding(&self.cell, self.cell.as_ref().unwrap().cap) },
            wgpu::BindGroupEntry { binding: 2, resource: Self::binding(&self.via, self.via.as_ref().unwrap().cap) },
            wgpu::BindGroupEntry { binding: 3, resource: Self::binding(lines, n_lines.max(1) as u64 * 32) },
            wgpu::BindGroupEntry { binding: 4, resource: Self::binding(&self.jobs_buf, self.gjobs.len().max(1) as u64 * std::mem::size_of::<GJob>() as u64) },
            wgpu::BindGroupEntry { binding: 5, resource: Self::binding(&self.tiles, n_tiles.max(1) as u64 * 8) },
            wgpu::BindGroupEntry { binding: 6, resource: Self::binding(&self.seeds, n_seeds.max(1) as u64 * 8) },
            wgpu::BindGroupEntry { binding: 7, resource: Self::binding(&self.flags, self.gjobs.len().max(1) as u64 * 4) },
        ];
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("sweep"), layout: &self.bgl, entries: &entries })
    }

    fn map_read(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let slice = buf.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU poll failed: {e}"))?;
        rx.recv().context("map callback dropped")?.map_err(|e| anyhow!("buffer map failed: {e}"))?;
        let data = buf.get_mapped_range(..bytes).map_err(|e| anyhow!("mapped range: {e:?}"))?.to_vec();
        buf.unmap();
        Ok(data)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> String {
        format!("GPU: {}", self.adapter_name())
    }

    fn load(&mut self, jobs: Vec<Arc<SearchJob>>) -> Result<()> {
        let mut gjobs = Vec::with_capacity(jobs.len());
        let (mut cell_tot, mut via_tot, mut dist_tot) = (0u64, 0u64, 0u64);
        for j in &jobs {
            j.validate()?;
            if j.dirfac.iter().any(|d| d[0] > 1023 || d[1] > 1023) {
                bail!("direction factor too large for GPU arithmetic");
            }
            let mut g = GJob::zeroed();
            g.cell_off = cell_tot as u32;
            g.dist_off = dist_tot as u32;
            g.via_off = via_tot as u32;
            g.n = j.n() as u32;
            g.plane = j.plane() as u32;
            g.nl = j.nl;
            g.nv = j.nv() as u32;
            g.bend = j.bend;
            g.w = j.w;
            g.sweep_mask = j.sweep_layer.iter().enumerate().filter(|(_, s)| **s).fold(0u32, |m, (l, _)| m | (1 << l));
            for (l, d) in j.dirfac.iter().enumerate() {
                g.dirfac[l] = *d;
            }
            for (v, &(a, b)) in j.via_spans.iter().enumerate() {
                g.spans[v] = [a, b];
            }
            gjobs.push(g);
            cell_tot += j.n() as u64;
            via_tot += (j.nv() * j.plane()) as u64;
            dist_tot += 2 * j.n() as u64;
        }
        if dist_tot * 4 > self.max_binding || cell_tot * 4 > self.max_binding || dist_tot > u32::MAX as u64 {
            bail!("search batch too large for the GPU ({} MB)", dist_tot * 4 >> 20);
        }
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        ensure(&self.device, &mut self.dist, dist_tot * 4, st | wgpu::BufferUsages::COPY_SRC, "dist");
        ensure(&self.device, &mut self.cell, cell_tot * 4, st, "cell");
        ensure(&self.device, &mut self.via, via_tot * 4, st, "via");
        ensure(&self.device, &mut self.jobs_buf, (gjobs.len() * std::mem::size_of::<GJob>()) as u64, st, "jobs");
        ensure(&self.device, &mut self.flags, gjobs.len() as u64 * 4, st | wgpu::BufferUsages::COPY_SRC, "flags");
        ensure(&self.device, &mut self.seeds, 8, st, "seeds");
        ensure(&self.device, &mut self.tiles, 8, st, "tiles");
        ensure(&self.device, &mut self.lines_h, 32, st, "lines_h");
        ensure(&self.device, &mut self.lines_v, 32, st, "lines_v");
        for (j, g) in jobs.iter().zip(&gjobs) {
            self.queue.write_buffer(&self.cell.as_ref().unwrap().buf, g.cell_off as u64 * 4, bytemuck::cast_slice(&j.cell));
            if !j.via.is_empty() {
                self.queue.write_buffer(&self.via.as_ref().unwrap().buf, g.via_off as u64 * 4, bytemuck::cast_slice(&j.via));
            }
        }
        self.queue.write_buffer(&self.jobs_buf.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&gjobs));

        // Line and tile descriptors per job.
        self.line_cache = jobs
            .iter()
            .enumerate()
            .map(|(s, j)| {
                let (w, h, plane) = (j.w, j.h, j.plane() as u32);
                let mut hl = Vec::new();
                let mut vl = Vec::new();
                for l in 0..j.nl {
                    if !j.sweep_layer[l as usize] {
                        continue;
                    }
                    for y in 0..h {
                        hl.push(GLine { base: l * plane + y * w, len: w, stride: 1, job: s as u32, layer: l, orient: 0, pad0: 0, pad1: 0 });
                    }
                    for x in 0..w {
                        vl.push(GLine { base: l * plane + x, len: h, stride: w, job: s as u32, layer: l, orient: 1, pad0: 0, pad1: 0 });
                    }
                }
                (hl, vl)
            })
            .collect();
        self.tile_cache = jobs
            .iter()
            .enumerate()
            .map(|(s, j)| if j.nv() == 0 { vec![] } else { (0..j.plane() as u32).step_by(WG as usize).map(|t| [s as u32, t]).collect() })
            .collect();
        self.pending = jobs.iter().map(|_| Vec::new()).collect();
        self.dist_used = dist_tot;
        self.gjobs = gjobs;
        self.jobs = jobs;

        // Initialise all distances to INF.
        let bg = self.bind_group(&self.lines_h, 0, 0, 0);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.p_fill);
            pass.set_bind_group(0, &bg, &[]);
            let (gx, gy) = groups(dist_tot.div_ceil(WG as u64) as u32);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    fn seed(&mut self, slot: usize, nodes: &[u32]) {
        self.pending[slot].extend_from_slice(nodes);
    }

    fn relax(&mut self, slots: &[usize]) -> Result<()> {
        let mut active: Vec<usize> = slots.to_vec();
        active.sort_unstable();
        active.dedup();
        let mut seeds: Vec<[u32; 2]> = Vec::new();
        for &s in &active {
            for &e in &self.pending[s] {
                seeds.push([s as u32, e]);
            }
            self.pending[s].clear();
        }
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        ensure(&self.device, &mut self.seeds, seeds.len() as u64 * 8, st, "seeds");
        if !seeds.is_empty() {
            self.queue.write_buffer(&self.seeds.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&seeds));
        }
        let n_jobs = self.gjobs.len();
        ensure(&self.device, &mut self.flags_staging, n_jobs as u64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "flags_staging");
        let mut first = true;
        let mut rounds = 0usize;
        let mut iters = ITERS_PER_SUBMIT;
        while !active.is_empty() {
            rounds += 1;
            if rounds > 100_000 {
                bail!("GPU relaxation did not converge");
            }
            let hl: Vec<GLine> = active.iter().flat_map(|&s| self.line_cache[s].0.iter().copied()).collect();
            let vl: Vec<GLine> = active.iter().flat_map(|&s| self.line_cache[s].1.iter().copied()).collect();
            let tl: Vec<[u32; 2]> = active.iter().flat_map(|&s| self.tile_cache[s].iter().copied()).collect();
            ensure(&self.device, &mut self.lines_h, hl.len() as u64 * 32, st, "lines_h");
            ensure(&self.device, &mut self.lines_v, vl.len() as u64 * 32, st, "lines_v");
            ensure(&self.device, &mut self.tiles, tl.len() as u64 * 8, st, "tiles");
            if !hl.is_empty() {
                self.queue.write_buffer(&self.lines_h.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&hl));
            }
            if !vl.is_empty() {
                self.queue.write_buffer(&self.lines_v.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&vl));
            }
            if !tl.is_empty() {
                self.queue.write_buffer(&self.tiles.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&tl));
            }
            let n_seeds = if first { seeds.len() } else { 0 };
            let bg_h = self.bind_group(&self.lines_h, hl.len(), tl.len(), n_seeds);
            let bg_v = self.bind_group(&self.lines_v, vl.len(), tl.len(), n_seeds);
            let mut enc = self.device.create_command_encoder(&Default::default());
            let run = |enc: &mut wgpu::CommandEncoder, iters: usize, seed_first: bool| {
                let mut pass = enc.begin_compute_pass(&Default::default());
                if seed_first && n_seeds > 0 {
                    pass.set_pipeline(&self.p_seed);
                    pass.set_bind_group(0, &bg_h, &[]);
                    let (gx, gy) = groups((n_seeds as u32).div_ceil(WG));
                    pass.dispatch_workgroups(gx, gy, 1);
                }
                for _ in 0..iters {
                    for (pipe, bg, count) in [(&self.p_fwd, &bg_h, hl.len()), (&self.p_bwd, &bg_h, hl.len()), (&self.p_fwd, &bg_v, vl.len()), (&self.p_bwd, &bg_v, vl.len())] {
                        if count == 0 {
                            continue;
                        }
                        pass.set_pipeline(pipe);
                        pass.set_bind_group(0, bg, &[]);
                        let (gx, gy) = groups(count as u32);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                    if !tl.is_empty() {
                        pass.set_pipeline(&self.p_via);
                        pass.set_bind_group(0, &bg_h, &[]);
                        let (gx, gy) = groups(tl.len() as u32);
                        pass.dispatch_workgroups(gx, gy, 1);
                    }
                }
            };
            run(&mut enc, iters - 1, first);
            enc.clear_buffer(&self.flags.as_ref().unwrap().buf, 0, Some((n_jobs as u64 * 4).div_ceil(4) * 4));
            run(&mut enc, 1, false);
            enc.copy_buffer_to_buffer(&self.flags.as_ref().unwrap().buf, 0, &self.flags_staging.as_ref().unwrap().buf, 0, n_jobs as u64 * 4);
            self.queue.submit([enc.finish()]);
            self.iterations += iters as u64;
            iters = (iters * 2).min(MAX_ITERS_PER_SUBMIT);
            let flags: Vec<u32> = bytemuck::cast_slice(&self.map_read(&self.flags_staging.as_ref().unwrap().buf, n_jobs as u64 * 4)?).to_vec();
            active.retain(|&s| flags[s] != 0);
            first = false;
        }
        Ok(())
    }

    fn pick(&mut self, slots: &[usize], targets: &[Vec<u32>]) -> Result<Vec<Option<Pick>>> {
        let pairs: Vec<[u32; 2]> = slots.iter().zip(targets).flat_map(|(&s, t)| t.iter().map(move |&e| [s as u32, e])).collect();
        if pairs.is_empty() || slots.is_empty() {
            return Ok(slots.iter().map(|_| None).collect());
        }
        let n_act = slots.len();
        let n_jobs = self.gjobs.len();
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let meta_bytes = (n_act * 32) as u64;
        let path_bytes = (n_act * TRACE_CAP * 8) as u64;
        ensure(&self.device, &mut self.targets, pairs.len() as u64 * 8, st, "targets");
        ensure(&self.device, &mut self.best, n_jobs as u64 * 8, st, "best");
        ensure(&self.device, &mut self.paths, path_bytes, st | wgpu::BufferUsages::COPY_SRC, "paths");
        ensure(&self.device, &mut self.tmeta, meta_bytes, st | wgpu::BufferUsages::COPY_SRC, "tmeta");
        ensure(&self.device, &mut self.staging_t, meta_bytes + path_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "staging_t");
        self.queue.write_buffer(&self.targets.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&pairs));
        let init: Vec<u32> = (0..n_jobs).flat_map(|_| [INF, NONE]).collect();
        self.queue.write_buffer(&self.best.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&init));
        let meta: Vec<u32> = slots.iter().flat_map(|&s| [s as u32, 0, 0, 0, 0, 0, 0, 0]).collect();
        self.queue.write_buffer(&self.tmeta.as_ref().unwrap().buf, 0, bytemuck::cast_slice(&meta));
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: Self::binding(&self.dist, self.dist_used * 4) },
            wgpu::BindGroupEntry { binding: 1, resource: Self::binding(&self.cell, self.cell.as_ref().unwrap().cap) },
            wgpu::BindGroupEntry { binding: 2, resource: Self::binding(&self.via, self.via.as_ref().unwrap().cap) },
            wgpu::BindGroupEntry { binding: 4, resource: Self::binding(&self.jobs_buf, n_jobs as u64 * std::mem::size_of::<GJob>() as u64) },
            wgpu::BindGroupEntry { binding: 8, resource: Self::binding(&self.targets, pairs.len() as u64 * 8) },
            wgpu::BindGroupEntry { binding: 9, resource: Self::binding(&self.best, n_jobs as u64 * 8) },
            wgpu::BindGroupEntry { binding: 10, resource: Self::binding(&self.paths, path_bytes) },
            wgpu::BindGroupEntry { binding: 11, resource: Self::binding(&self.tmeta, meta_bytes) },
        ];
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("trace"), layout: &self.bgl_t, entries: &entries });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bg, &[]);
            let (gx, gy) = groups((pairs.len() as u32).div_ceil(WG));
            pass.set_pipeline(&self.p_pick_cost);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.p_pick_node);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.p_trace);
            pass.dispatch_workgroups((n_act as u32).div_ceil(64), 1, 1);
        }
        let stg = &self.staging_t.as_ref().unwrap().buf;
        enc.copy_buffer_to_buffer(&self.tmeta.as_ref().unwrap().buf, 0, stg, 0, meta_bytes);
        enc.copy_buffer_to_buffer(&self.paths.as_ref().unwrap().buf, 0, stg, meta_bytes, path_bytes);
        self.queue.submit([enc.finish()]);
        let data = self.map_read(stg, meta_bytes + path_bytes)?;
        let words: &[u32] = bytemuck::cast_slice(&data);
        let (meta, paths) = words.split_at(n_act * 8);
        let mut out = Vec::with_capacity(n_act);
        let mut fallback: Vec<(usize, usize)> = Vec::new();
        for (k, &slot) in slots.iter().enumerate() {
            let m = &meta[k * 8..k * 8 + 8];
            let (cost, node, len, status) = (m[1], m[2], m[4] as usize, m[5]);
            if cost >= INF {
                out.push(None);
                continue;
            }
            if status != 0 {
                // Path longer than the trace buffer (or an inconsistency): trace on the CPU.
                fallback.push((k, slot));
                out.push(None);
                continue;
            }
            let plane = self.gjobs[slot].plane;
            let p = &paths[k * TRACE_CAP * 2..k * TRACE_CAP * 2 + len * 2];
            let mut steps: Vec<Step> = p.chunks(2).map(|c| Step { layer: c[0] / plane, p: c[0] % plane, via: (c[1] != NONE).then_some(c[1] as u8) }).collect();
            steps.reverse();
            out.push(Some(Pick { cost, node, steps }));
        }
        for (k, slot) in fallback {
            let dist = self.read(&[slot])?.pop().unwrap();
            out[k] = super::pick_cpu(&self.jobs[slot], &dist, &targets[k])?;
        }
        Ok(out)
    }

    fn read(&mut self, slots: &[usize]) -> Result<Vec<Vec<u32>>> {
        let total: u64 = slots.iter().map(|&s| self.gjobs[s].n as u64 * 2 * 4).sum();
        if total == 0 {
            return Ok(slots.iter().map(|_| Vec::new()).collect());
        }
        ensure(&self.device, &mut self.staging, total, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "staging");
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut off = 0u64;
        for &s in slots {
            let g = &self.gjobs[s];
            let bytes = g.n as u64 * 2 * 4;
            enc.copy_buffer_to_buffer(&self.dist.as_ref().unwrap().buf, g.dist_off as u64 * 4, &self.staging.as_ref().unwrap().buf, off, bytes);
            off += bytes;
        }
        self.queue.submit([enc.finish()]);
        let data = self.map_read(&self.staging.as_ref().unwrap().buf, total)?;
        let all: &[u32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(slots.len());
        let mut off = 0usize;
        for &s in slots {
            let len = self.gjobs[s].n as usize * 2;
            out.push(all[off..off + len].to_vec());
            off += len;
        }
        debug_assert!(out.iter().all(|d| d.iter().all(|&v| v <= INF)));
        Ok(out)
    }
}
