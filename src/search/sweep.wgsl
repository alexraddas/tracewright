// Tracewright GPU maze-search kernels.
//
// dist layout per job: [orientation][layer][y][x], orientation 0 = horizontal, 1 = vertical.
// A line is a row (orientation 0) or column (orientation 1) of one layer of one job. Each
// workgroup scans one line with the associative operator
//     (A1, S1) (+) (A2, S2) = (min(A1 + S2, A2), S1 + S2)
// where element i is (launch_i, enter_i): launch_i = min(own_i, other_i + bend) and enter_i
// the cost of entering node i along the line's orientation. The inclusive prefix A_i is the
// best cost to stand on node i facing the line's orientation. Saturating u32 arithmetic makes
// the result identical to the sequential CPU scan.

const INF: u32 = 1073741824u; // 1 << 30
const WG: u32 = 256u;
const ROW: u32 = 65535u;

struct Job {
    cell_off: u32,
    dist_off: u32,
    via_off: u32,
    n: u32,
    plane: u32,
    nl: u32,
    nv: u32,
    bend: u32,
    dirfac: array<vec2<u32>, 32>,
    spans: array<vec2<u32>, 8>,
    // Window width and bit mask of swept layers (used by the tracer).
    w: u32,
    sweep_mask: u32,
};

struct Line {
    base: u32,
    len: u32,
    stride: u32,
    job: u32,
    layer: u32,
    orient: u32,
    pad0: u32,
    pad1: u32,
};

struct Pair {
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read_write> dist: array<u32>;
@group(0) @binding(1) var<storage, read> cell: array<u32>;
@group(0) @binding(2) var<storage, read> viac: array<u32>;
@group(0) @binding(3) var<storage, read> lines: array<Line>;
@group(0) @binding(4) var<storage, read> jobs: array<Job>;
@group(0) @binding(5) var<storage, read> tiles: array<Pair>;   // (job, first node of 256-node tile)
@group(0) @binding(6) var<storage, read> seeds: array<Pair>;   // (job, node)
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;

fn sat(a: u32, b: u32) -> u32 {
    return min(a + b, INF);
}

var<workgroup> sA: array<u32, 256>;
var<workgroup> sS: array<u32, 256>;
var<workgroup> wCarry: u32;
var<workgroup> wLen: u32;

fn scan(line_id: u32, lid: u32, backward: bool) {
    let ln = lines[line_id];
    let j = jobs[ln.job];
    if (lid == 0u) {
        wCarry = INF;
        wLen = ln.len;
    }
    let len = workgroupUniformLoad(&wLen);
    let own_off = j.dist_off + ln.orient * j.n;
    let oth_off = j.dist_off + (1u - ln.orient) * j.n;
    let fac = j.dirfac[ln.layer][ln.orient];
    var changed = false;
    let nch = (len + WG - 1u) / WG;
    for (var ch = 0u; ch < nch; ch = ch + 1u) {
        let pos = ch * WG + lid;
        let valid = pos < len;
        var a = INF;
        var s = 0u;
        var own = INF;
        var e = 0u;
        if (valid) {
            var k = pos;
            if (backward) {
                k = len - 1u - pos;
            }
            e = ln.base + k * ln.stride;
            own = dist[own_off + e];
            let oth = dist[oth_off + e];
            a = min(own, sat(oth, j.bend));
            let c = cell[j.cell_off + e];
            if (c < INF) {
                s = min((c * fac) >> 4u, INF);
            } else {
                s = INF;
            }
        }
        sA[lid] = a;
        sS[lid] = s;
        workgroupBarrier();
        for (var off = 1u; off < WG; off = off << 1u) {
            var na = sA[lid];
            var ns = sS[lid];
            if (lid >= off) {
                let pa = sA[lid - off];
                let ps = sS[lid - off];
                na = min(sat(pa, ns), na);
                ns = sat(ps, ns);
            }
            workgroupBarrier();
            sA[lid] = na;
            sS[lid] = ns;
            workgroupBarrier();
        }
        let cin = wCarry;
        let res = min(sat(cin, sS[lid]), sA[lid]);
        if (valid && res < own) {
            dist[own_off + e] = res;
            changed = true;
        }
        workgroupBarrier();
        if (lid == WG - 1u) {
            wCarry = res;
        }
        workgroupBarrier();
    }
    if (changed) {
        atomicOr(&flags[ln.job], 1u);
    }
}

@compute @workgroup_size(256)
fn scan_fwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let line_id = wid.y * ROW + wid.x;
    if (line_id >= arrayLength(&lines)) {
        return;
    }
    scan(line_id, lid, false);
}

@compute @workgroup_size(256)
fn scan_bwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let line_id = wid.y * ROW + wid.x;
    if (line_id >= arrayLength(&lines)) {
        return;
    }
    scan(line_id, lid, true);
}

@compute @workgroup_size(256)
fn via_relax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = wid.y * ROW + wid.x;
    if (t >= arrayLength(&tiles)) {
        return;
    }
    let tile = tiles[t];
    let j = jobs[tile.a];
    let p = tile.b + lid;
    if (p >= j.plane) {
        return;
    }
    var changed = false;
    for (var v = 0u; v < j.nv; v = v + 1u) {
        let cv = viac[j.via_off + v * j.plane + p];
        if (cv >= INF) {
            continue;
        }
        let sp = j.spans[v];
        var m = INF;
        for (var l = sp.x; l <= sp.y; l = l + 1u) {
            let e = l * j.plane + p;
            m = min(m, min(dist[j.dist_off + e], dist[j.dist_off + j.n + e]));
        }
        let tv = sat(m, cv);
        if (tv >= INF) {
            continue;
        }
        for (var l = sp.x; l <= sp.y; l = l + 1u) {
            let e = l * j.plane + p;
            if (tv < dist[j.dist_off + e]) {
                dist[j.dist_off + e] = tv;
                changed = true;
            }
            if (tv < dist[j.dist_off + j.n + e]) {
                dist[j.dist_off + j.n + e] = tv;
                changed = true;
            }
        }
    }
    if (changed) {
        atomicOr(&flags[tile.a], 1u);
    }
}

@compute @workgroup_size(256)
fn seed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * ROW + wid.x) * WG + lid;
    if (i >= arrayLength(&seeds)) {
        return;
    }
    let s = seeds[i];
    let j = jobs[s.a];
    dist[j.dist_off + s.b] = 0u;
    dist[j.dist_off + j.n + s.b] = 0u;
}

@compute @workgroup_size(256)
fn fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * ROW + wid.x) * WG + lid;
    if (i >= arrayLength(&dist)) {
        return;
    }
    dist[i] = INF;
}

// ---------------------------------------------------------------------------------------------
// Target selection and path tracing on the GPU (separate bind group layout, bindings 8..11),
// so only the traced path is read back instead of whole distance maps.

const NONE: u32 = 0xffffffffu;
const CAP: u32 = 4096u;


@group(0) @binding(8) var<storage, read> targets: array<Pair>;      // (job, node)
@group(0) @binding(9) var<storage, read_write> best: array<atomic<u32>>; // per job: cost, node
@group(0) @binding(10) var<storage, read_write> paths: array<u32>;  // per active job: CAP x (node, via)
@group(0) @binding(11) var<storage, read_write> tmeta: array<u32>;  // per active job: 8 words

@compute @workgroup_size(256)
fn pick_cost(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * ROW + wid.x) * WG + lid;
    if (i >= arrayLength(&targets)) {
        return;
    }
    let t = targets[i];
    let j = jobs[t.a];
    let d = min(dist[j.dist_off + t.b], dist[j.dist_off + j.n + t.b]);
    if (d < INF) {
        atomicMin(&best[2u * t.a], d);
    }
}

@compute @workgroup_size(256)
fn pick_node(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wid.y * ROW + wid.x) * WG + lid;
    if (i >= arrayLength(&targets)) {
        return;
    }
    let t = targets[i];
    let j = jobs[t.a];
    let d = min(dist[j.dist_off + t.b], dist[j.dist_off + j.n + t.b]);
    if (d < INF && d == atomicLoad(&best[2u * t.a])) {
        atomicMin(&best[2u * t.a + 1u], t.b);
    }
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= arrayLength(&tmeta) / 8u) {
        return;
    }
    let slot = tmeta[k * 8u];
    let j = jobs[slot];
    let cost = atomicLoad(&best[2u * slot]);
    let base = k * CAP * 2u;
    if (cost >= INF) {
        tmeta[k * 8u + 1u] = INF;
        tmeta[k * 8u + 4u] = 0u;
        tmeta[k * 8u + 5u] = 0u;
        return;
    }
    let n = j.n;
    let plane = j.plane;
    let w = j.w;
    let h = plane / w;
    var e = atomicLoad(&best[2u * slot + 1u]);
    var o = 0u;
    if (dist[j.dist_off + n + e] < dist[j.dist_off + e]) {
        o = 1u;
    }
    var d = dist[j.dist_off + o * n + e];
    tmeta[k * 8u + 1u] = d;
    tmeta[k * 8u + 2u] = e;
    tmeta[k * 8u + 3u] = o;
    paths[base] = e;
    paths[base + 1u] = NONE;
    var len = 1u;
    var last_dir: i32 = 0;
    var status = 0u;
    var guard = 0u;
    loop {
        if (d == 0u) {
            break;
        }
        guard = guard + 1u;
        if (len >= CAP || guard > 8u * n + 16u) {
            status = 1u;
            break;
        }
        let l = e / plane;
        let p = e % plane;
        let y = p / w;
        let x = p % w;
        let swept = ((j.sweep_mask >> l) & 1u) == 1u;
        var moved = false;
        // 1) Straight move from a neighbour along orientation o (same order as the CPU).
        var c = INF;
        if (swept) {
            let cc = cell[j.cell_off + e];
            if (cc < INF) {
                c = min((cc * j.dirfac[l][o]) >> 4u, INF);
            }
        }
        if (c < INF) {
            for (var t = 0u; t < 2u; t = t + 1u) {
                var s: i32 = -1;
                if (t == 1u) {
                    s = 1;
                }
                if (last_dir == 1) {
                    s = -s;
                }
                var ok = false;
                var q = 0u;
                if (o == 0u) {
                    let nx = i32(x) - s;
                    if (nx >= 0 && nx < i32(w)) {
                        ok = true;
                        q = l * plane + y * w + u32(nx);
                    }
                } else {
                    let ny = i32(y) - s;
                    if (ny >= 0 && ny < i32(h)) {
                        ok = true;
                        q = l * plane + u32(ny) * w + x;
                    }
                }
                if (ok) {
                    let dq = dist[j.dist_off + o * n + q];
                    if (dq < INF && sat(dq, c) == d) {
                        e = q;
                        d = dq;
                        last_dir = s;
                        paths[base + len * 2u] = q;
                        paths[base + len * 2u + 1u] = NONE;
                        len = len + 1u;
                        moved = true;
                        break;
                    }
                }
            }
        }
        if (moved) {
            continue;
        }
        // 2) Via from another layer.
        for (var v = 0u; v < j.nv; v = v + 1u) {
            if (moved) {
                break;
            }
            let sp = j.spans[v];
            if (l < sp.x || l > sp.y) {
                continue;
            }
            let cv = viac[j.via_off + v * plane + p];
            if (cv >= INF) {
                continue;
            }
            for (var l2 = sp.x; l2 <= sp.y; l2 = l2 + 1u) {
                if (l2 == l) {
                    continue;
                }
                let e2 = l2 * plane + p;
                let dh = dist[j.dist_off + e2];
                let dv = dist[j.dist_off + n + e2];
                var m = dh;
                var o2 = 0u;
                if (dv < dh) {
                    m = dv;
                    o2 = 1u;
                }
                if (m < INF && sat(m, cv) == d) {
                    paths[base + (len - 1u) * 2u + 1u] = v;
                    e = e2;
                    o = o2;
                    d = m;
                    last_dir = 0;
                    paths[base + len * 2u] = e2;
                    paths[base + len * 2u + 1u] = NONE;
                    len = len + 1u;
                    moved = true;
                    break;
                }
            }
        }
        if (moved) {
            continue;
        }
        // 3) Turn in place.
        let other = dist[j.dist_off + (1u - o) * n + e];
        if (swept && other < INF && sat(other, j.bend) == d) {
            o = 1u - o;
            d = other;
            last_dir = 0;
            continue;
        }
        status = 2u;
        break;
    }
    tmeta[k * 8u + 4u] = len;
    tmeta[k * 8u + 5u] = status;
}
