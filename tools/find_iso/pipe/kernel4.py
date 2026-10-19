"""kernel4: grid-native hyperbolic radiant kernel (no per-cell Newton).

The map x=(log rho, rhodot) -> (radiant R, v_inf, b, q, e) is smooth; evaluate it once on an adaptive grid and,
for every radiant cell whose centre falls inside the sky image of a grid quad, obtain the solution by inverting the
bilinear patch (pure arithmetic). Element derivatives w.r.t. radiant offset (A) come from the same corner values.
Rate-error sigmas and perihelion time are computed lazily in the pair stage (only for candidate pairs).
Same radiant grid, same (v_inf, q) ranges, every root: a re-organisation of the computation, not a cut."""
import numpy as np, math
from numba import njit, prange
from kernel2 import GM, CLIGHT, KMS, make_grid, cell_of, tangent_basis, radiant, finish

NOUT = 24  # +[22,23] = d tp/d(delta1,delta2);  vinf_kms, bx, by, bz, lrho, rhodot(AU/d), q, e, A[4x2] (d(vinf_kms,b)/d(delta1,delta2) in cell tangent basis)
NPT = 11   # per grid point: Rx Ry Rz vinf bx by bz q e ra tp
MSUB = 16
TWOPI = 6.283185307179586

def cell_bases(g):
    tb = np.empty((g['ncell'], 6))
    _bases(g['centers'], tb)
    return tb

@njit(cache=True)
def _bases(centers, tb):
    for k in range(centers.shape[0]):
        tangent_basis(centers[k], tb[k])

@njit(cache=True, fastmath=True, error_model='numpy', inline='always')
def fast_atan2(y, x):
    """max error ~1e-5 rad; branch-light rational approximation"""
    ax = abs(x); ay = abs(y)
    mx = max(ax, ay); mn = min(ax, ay)
    a = mn / (mx + 1e-300)
    s2 = a * a
    r = ((-0.0464964749 * s2 + 0.15931422) * s2 - 0.327622764) * s2 * a + a
    if ay > ax: r = 1.57079637 - r
    if x < 0: r = 3.14159274 - r
    if y < 0: r = -r
    return r

@njit(cache=True, fastmath=True, error_model='numpy')
def eval_pt(t, u, udot, E, Edot, rho, rhodot, o, P, idx, vlo, vhi, qhi, stats, fin):
    stats[0] += 1
    if radiant(t, u, udot, E, Edot, rho, rhodot, o) == 0: return 0
    if o[0] < vlo or o[0] > vhi or o[5] > qhi: return 0
    vinf = o[0]; vinx = -o[1]; viny = -o[2]; vinz = -o[3]
    P[idx, 0] = o[1]; P[idx, 1] = o[2]; P[idx, 2] = o[3]; P[idx, 3] = vinf
    P[idx, 4] = (viny * o[9] - vinz * o[8]) / vinf
    P[idx, 5] = (vinz * o[7] - vinx * o[9]) / vinf
    P[idx, 6] = (vinx * o[8] - viny * o[7]) / vinf
    P[idx, 7] = o[5]; P[idx, 8] = o[4]
    ra = fast_atan2(o[2], o[1])
    if ra < 0.0: ra += TWOPI
    P[idx, 9] = ra
    finish(t, u, E, rho, o, fin)
    P[idx, 10] = fin[3]
    return 1

@njit(cache=True, fastmath=True, error_model='numpy')
def do_quad(S, i0, i1, i2, i3, lr0, rd0, dl, dr, nb_, nras, starts, centers, cbases, cell_rad, vmin_ad, vmax_ad, qmax,
            tb, res_cell, res_val, n, maxc, stats, mode, t, u, udot, E, Edot, sig_rate, ub, o, o1, fin, fin1, blo, bhi, keep):
    """Quad corners S[i0]=(0,0) S[i1]=(1,0) S[i2]=(0,1) S[i3]=(1,1) in (lrho, rhodot). Appends solutions; returns n."""
    stats[1] += 1
    udp = np.empty(3)
    # whole quad outside the accepted ranges -> no interior solution can be inside (bilinear stays in corner hull)
    if S[i0, 3] < vmin_ad and S[i1, 3] < vmin_ad and S[i2, 3] < vmin_ad and S[i3, 3] < vmin_ad: return n
    if S[i0, 3] > vmax_ad and S[i1, 3] > vmax_ad and S[i2, 3] > vmax_ad and S[i3, 3] > vmax_ad: return n
    if S[i0, 7] > qmax and S[i1, 7] > qmax and S[i2, 7] > qmax and S[i3, 7] > qmax: return n
    R0x = S[i0, 0]; R0y = S[i0, 1]; R0z = S[i0, 2]
    # fold check on the diagonals
    if R0x * S[i3, 0] + R0y * S[i3, 1] + R0z * S[i3, 2] < 0.5: return n
    if S[i1, 0] * S[i2, 0] + S[i1, 1] * S[i2, 1] + S[i1, 2] * S[i2, 2] < 0.5: return n
    tangent_basis(S[i0, 0:3], tb)
    w1 = S[i1, 0] * R0x + S[i1, 1] * R0y + S[i1, 2] * R0z
    a0 = (S[i1, 0] * tb[0] + S[i1, 1] * tb[1] + S[i1, 2] * tb[2]) / w1
    a1 = (S[i1, 0] * tb[3] + S[i1, 1] * tb[4] + S[i1, 2] * tb[5]) / w1
    w2 = S[i2, 0] * R0x + S[i2, 1] * R0y + S[i2, 2] * R0z
    b0 = (S[i2, 0] * tb[0] + S[i2, 1] * tb[1] + S[i2, 2] * tb[2]) / w2
    b1 = (S[i2, 0] * tb[3] + S[i2, 1] * tb[4] + S[i2, 2] * tb[5]) / w2
    w3 = S[i3, 0] * R0x + S[i3, 1] * R0y + S[i3, 2] * R0z
    c0 = (S[i3, 0] * tb[0] + S[i3, 1] * tb[1] + S[i3, 2] * tb[2]) / w3
    c1 = (S[i3, 0] * tb[3] + S[i3, 1] * tb[4] + S[i3, 2] * tb[5]) / w3
    d0 = c0 - a0 - b0; d1 = c1 - a1 - b1
    det0 = a0 * b1 - a1 * b0
    if abs(det0) < 1e-14: return n
    xmin = min(min(0.0, a0), min(b0, c0)); xmax = max(max(0.0, a0), max(b0, c0))
    ymin = min(min(0.0, a1), min(b1, c1)); ymax = max(max(0.0, a1), max(b1, c1))
    mg = 0.1 * (xmax - xmin + ymax - ymin) + 1e-6
    xmin -= mg; xmax += mg; ymin -= mg; ymax += mg
    # index bounding box from corner (z, ra)
    zmin = min(min(R0z, S[i1, 2]), min(S[i2, 2], S[i3, 2])); zmax = max(max(R0z, S[i1, 2]), max(S[i2, 2], S[i3, 2]))
    ra0 = S[i0, 9]; dmin = 0.0; dmax = 0.0
    for a in range(3):
        ia = i1 if a == 0 else (i2 if a == 1 else i3)
        d = S[ia, 9] - ra0
        if d > math.pi: d -= TWOPI
        if d < -math.pi: d += TWOPI
        if d < dmin: dmin = d
        if d > dmax: dmax = d
    mz = 0.03 * cell_rad
    mx = R0x + S[i1, 0] + S[i2, 0] + S[i3, 0]; my = R0y + S[i1, 1] + S[i2, 1] + S[i3, 1]; mzc = R0z + S[i1, 2] + S[i2, 2] + S[i3, 2]
    mn = math.sqrt(mx * mx + my * my + mzc * mzc); mx /= mn; my /= mn; mzc /= mn
    cmin = min(min(mx * R0x + my * R0y + mzc * R0z, mx * S[i1, 0] + my * S[i1, 1] + mzc * S[i1, 2]), min(mx * S[i2, 0] + my * S[i2, 1] + mzc * S[i2, 2], mx * S[i3, 0] + my * S[i3, 1] + mzc * S[i3, 2]))
    cmin = cmin - 0.08 * cell_rad * math.sqrt(max(1.0 - cmin * cmin, 1e-6)) - 1e-6
    if mode == 4: return n
    klo = int((zmin - mz + 1) * 0.5 * nb_); khi = int((zmax + mz + 1) * 0.5 * nb_)
    if klo < blo: klo = blo
    if khi > bhi - 1: khi = bhi - 1
    for k in range(klo, khi + 1):
        nra = nras[k]
        z0 = -1.0 + 2.0 * k / nb_; z1 = -1.0 + 2.0 * (k + 1) / nb_
        zz = max(abs(z0), abs(z1)); cosd = math.sqrt(max(1.0 - zz * zz, 0.0))
        if cosd < 2.0 * cell_rad:
            jlo = 0; jhi = nra - 1
        else:
            mra = mz / cosd
            jlo = int(math.floor((ra0 + dmin - mra) / TWOPI * nra)); jhi = int(math.floor((ra0 + dmax + mra) / TWOPI * nra))
            if jhi - jlo + 1 >= nra:
                jlo = 0; jhi = nra - 1
        for jj in range(jlo, jhi + 1):
            j = jj % nra
            if j < 0: j += nra
            kc = starts[k] + j
            if keep[kc] == 0: continue   # pass-level cell mask: cells this pass never reads are not refined
            stats[2] += 1
            Cx = centers[kc, 0]; Cy = centers[kc, 1]; Cz = centers[kc, 2]
            if Cx * mx + Cy * my + Cz * mzc < cmin: continue
            wc = Cx * R0x + Cy * R0y + Cz * R0z
            if wc < 0.7: continue
            x = (Cx * tb[0] + Cy * tb[1] + Cz * tb[2]) / wc
            if x < xmin or x > xmax: continue
            y = (Cx * tb[3] + Cy * tb[4] + Cz * tb[5]) / wc
            if y < ymin or y > ymax: continue
            if mode == 5: continue
            al = (x * b1 - y * b0) / det0; be = (a0 * y - a1 * x) / det0
            j00 = a0; j01 = b0; j10 = a1; j11 = b1; det = det0
            for it in range(2):
                fx = al * a0 + be * b0 + al * be * d0 - x
                fy = al * a1 + be * b1 + al * be * d1 - y
                j00 = a0 + be * d0; j01 = b0 + al * d0; j10 = a1 + be * d1; j11 = b1 + al * d1
                det = j00 * j11 - j01 * j10
                if abs(det) < 1e-16: break
                al -= (fx * j11 - fy * j01) / det; be -= (j00 * fy - j10 * fx) / det
            if abs(det) < 1e-16: continue
            if al < 0.0 or al >= 1.0 or be < 0.0 or be >= 1.0: continue
            if mode == 6: continue
            w00 = (1 - al) * (1 - be); w10 = al * (1 - be); w01 = (1 - al) * be; w11 = al * be
            vinf = w00 * S[i0, 3] + w10 * S[i1, 3] + w01 * S[i2, 3] + w11 * S[i3, 3]
            if vinf < vmin_ad or vinf > vmax_ad: continue
            q = w00 * S[i0, 7] + w10 * S[i1, 7] + w01 * S[i2, 7] + w11 * S[i3, 7]
            if q > qmax: continue
            if n >= maxc: return n
            # A_gn = [dE/dal dE/dbe] . Jb^-1 (gnomonic coords at R0), then to cell tangent basis via T
            inv00 = j11 / det; inv01 = -j01 / det; inv10 = -j10 / det; inv11 = j00 / det
            f1R = cbases[kc, 0] * R0x + cbases[kc, 1] * R0y + cbases[kc, 2] * R0z
            f2R = cbases[kc, 3] * R0x + cbases[kc, 4] * R0y + cbases[kc, 5] * R0z
            f1e1 = cbases[kc, 0] * tb[0] + cbases[kc, 1] * tb[1] + cbases[kc, 2] * tb[2]
            f1e2 = cbases[kc, 0] * tb[3] + cbases[kc, 1] * tb[4] + cbases[kc, 2] * tb[5]
            f2e1 = cbases[kc, 3] * tb[0] + cbases[kc, 4] * tb[1] + cbases[kc, 5] * tb[2]
            f2e2 = cbases[kc, 3] * tb[3] + cbases[kc, 4] * tb[4] + cbases[kc, 5] * tb[5]
            T00 = (f1e1 - x * f1R) / wc; T10 = (f1e2 - y * f1R) / wc
            T01 = (f2e1 - x * f2R) / wc; T11 = (f2e2 - y * f2R) / wc
            # M = Jb^-1 . T  (2x2): d(al,be)/d(delta)
            M00 = inv00 * T00 + inv01 * T10; M01 = inv00 * T01 + inv01 * T11
            M10 = inv10 * T00 + inv11 * T10; M11 = inv10 * T01 + inv11 * T11
            # exact elements at the interpolated (rho, rhodot); bilinear only located the cell's solution
            lr = lr0 + al * dl; rd = rd0 + be * dr; rho = math.exp(lr)
            if radiant(t, u, udot, E, Edot, rho, rd, o) == 0: continue
            # one Newton correction on the exact map: move (al,be) so the exact radiant hits the cell centre
            wr = Cx * o[1] + Cy * o[2] + Cz * o[3]
            if wr < 0.9: continue
            p1 = (cbases[kc, 0] * o[1] + cbases[kc, 1] * o[2] + cbases[kc, 2] * o[3]) / wr
            p2 = (cbases[kc, 3] * o[1] + cbases[kc, 4] * o[2] + cbases[kc, 5] * o[3]) / wr
            stats[4] += int(math.sqrt(p1 * p1 + p2 * p2) / cell_rad * 1e6)
            al -= M00 * p1 + M01 * p2; be -= M10 * p1 + M11 * p2
            lr = lr0 + al * dl; rd = rd0 + be * dr; rho = math.exp(lr)
            if radiant(t, u, udot, E, Edot, rho, rd, o) == 0: continue
            wr = Cx * o[1] + Cy * o[2] + Cz * o[3]
            if wr < 0.9: continue
            p1 = (cbases[kc, 0] * o[1] + cbases[kc, 1] * o[2] + cbases[kc, 2] * o[3]) / wr
            p2 = (cbases[kc, 3] * o[1] + cbases[kc, 4] * o[2] + cbases[kc, 5] * o[3]) / wr
            stats[5] += int(math.sqrt(p1 * p1 + p2 * p2) / cell_rad * 1e6)
            kc2 = cell_of(o[1], o[2], o[3], nb_, nras, starts)
            if kc2 != kc:
                stats[6] += 1
                if kc2 < 0: continue
            vinf = o[0]
            if vinf < vmin_ad or vinf > vmax_ad or o[5] > qmax: continue
            finish(t, u, E, rho, o, fin)
            base = n * NOUT
            res_cell[n] = kc2
            kc = kc2
            res_val[base + 0] = vinf * KMS
            res_val[base + 1] = fin[0]; res_val[base + 2] = fin[1]; res_val[base + 3] = fin[2]
            res_val[base + 4] = fin[3]
            res_val[base + 5] = o[5]; res_val[base + 6] = o[4]
            res_val[base + 7] = lr; res_val[base + 8] = rd
            for c in range(5):
                cc = 3 + c if c < 4 else 10  # column in S: vinf, bx, by, bz, tp
                ed = S[i3, cc] - S[i1, cc] - S[i2, cc] + S[i0, cc]
                dea = (S[i1, cc] - S[i0, cc]) + be * ed
                deb = (S[i2, cc] - S[i0, cc]) + al * ed
                sc = KMS if c == 0 else 1.0
                oi = 9 + 2 * c if c < 4 else 22
                res_val[base + oi] = (dea * M00 + deb * M10) * sc
                res_val[base + oi + 1] = (dea * M01 + deb * M11) * sc
            # rate-error propagation at fixed radiant cell: dE/du' - A . dR_tan/du'  (two perpendicular rate directions)
            for c in range(5): res_val[base + 17 + c] = 0.0
            eps = 1e-5
            for d in range(2):
                for c in range(3): udp[c] = udot[c] + eps * ub[3 * d + c]
                if radiant(t, u, udp, E, Edot, rho, rd, o1) == 0: continue
                finish(t, u, E, rho, o1, fin1)
                dRx = (o1[1] - o[1]) / eps; dRy = (o1[2] - o[2]) / eps; dRz = (o1[3] - o[3]) / eps
                s1 = (cbases[kc, 0] * dRx + cbases[kc, 1] * dRy + cbases[kc, 2] * dRz)
                s2 = (cbases[kc, 3] * dRx + cbases[kc, 4] * dRy + cbases[kc, 5] * dRz)
                g = ((o1[0] - o[0]) / eps) * KMS - res_val[base + 9] * s1 - res_val[base + 10] * s2
                res_val[base + 17] += g * g
                for c in range(3):
                    g = (fin1[c] - fin[c]) / eps - res_val[base + 11 + 2 * c] * s1 - res_val[base + 12 + 2 * c] * s2
                    res_val[base + 18 + c] += g * g
                g = (fin1[3] - fin[3]) / eps - res_val[base + 22] * s1 - res_val[base + 23] * s2
                res_val[base + 21] += g * g
            for c in range(5): res_val[base + 17 + c] = math.sqrt(res_val[base + 17 + c]) * sig_rate
            n += 1
    return n

@njit(cache=True, fastmath=True, error_model='numpy')
def process_tracklet(t, u, udot, E, Edot, sig_rate, lrhos, rhodots, nb_, nras, starts, centers, cbases, cell_rad,
                     vmin, vmax, qmax, step_fac, G, V, S, VS, o, tb, res_cell, res_val, maxc, stats, mode, blo, bhi, keep):
    o1 = np.empty(17); fin = np.empty(4); fin1 = np.empty(4); ub = np.empty(6)
    # two unit directions perpendicular to u (rate-error directions)
    ax = 1.0 if abs(u[0]) < 0.9 else 0.0; ay = 1.0 - ax
    e1x = ay * u[2] - 0.0 * u[1]; e1y = 0.0 * u[0] - ax * u[2]; e1z = ax * u[1] - ay * u[0]
    nn = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z); e1x /= nn; e1y /= nn; e1z /= nn
    ub[0] = e1x; ub[1] = e1y; ub[2] = e1z
    ub[3] = u[1] * e1z - u[2] * e1y; ub[4] = u[2] * e1x - u[0] * e1z; ub[5] = u[0] * e1y - u[1] * e1x
    nl = lrhos.shape[0]; nr = rhodots.shape[0]
    dl = lrhos[1] - lrhos[0]; dr = rhodots[1] - rhodots[0]
    vmin_ad = vmin / KMS; vmax_ad = vmax / KMS
    vlo = 0.25 * vmin_ad; vhi = 1.3 * vmax_ad; qhi = 2.0 * qmax
    spare = (MSUB + 1) * (MSUB + 1)
    for i in range(nl):
        rho_i = math.exp(lrhos[i])
        for j in range(nr):
            V[i * nr + j] = eval_pt(t, u, udot, E, Edot, rho_i, rhodots[j], o, G, i * nr + j, vlo, vhi, qhi, stats, fin)
    step = step_fac * cell_rad
    n = 0
    if mode == 3: return 0
    for i in range(nl - 1):
        for j in range(nr - 1):
            q00 = i * nr + j; q10 = (i + 1) * nr + j; q01 = i * nr + j + 1; q11 = (i + 1) * nr + j + 1
            nv = V[q00] + V[q10] + V[q01] + V[q11]
            if nv == 0: continue
            # anisotropic extent: along lrho (edges q00-q10, q01-q11) and along rhodot (q00-q01, q10-q11)
            sa = 0.0; sb = 0.0
            for a in range(2):
                p0 = q00 if a == 0 else q01; p1 = q10 if a == 0 else q11
                if V[p0] == 1 and V[p1] == 1:
                    dd = 1.0 - (G[p0, 0] * G[p1, 0] + G[p0, 1] * G[p1, 1] + G[p0, 2] * G[p1, 2])
                    if dd > sa: sa = dd
                p0 = q00 if a == 0 else q10; p1 = q01 if a == 0 else q11
                if V[p0] == 1 and V[p1] == 1:
                    dd = 1.0 - (G[p0, 0] * G[p1, 0] + G[p0, 1] * G[p1, 1] + G[p0, 2] * G[p1, 2])
                    if dd > sb: sb = dd
            # all valid corners outside accepted ranges on the same side -> nothing inside can be accepted
            lo = 1; hi = 1; qq = 1
            for a in range(4):
                ia = q00 if a == 0 else (q10 if a == 1 else (q01 if a == 2 else q11))
                if V[ia] == 0: continue
                if G[ia, 3] >= vmin_ad: lo = 0
                if G[ia, 3] <= vmax_ad: hi = 0
                if G[ia, 7] <= qmax: qq = 0
            if lo == 1 or hi == 1 or qq == 1: continue
            ma = int(math.sqrt(max(2.0 * sa, 0.0)) / step + 0.5); mb = int(math.sqrt(max(2.0 * sb, 0.0)) / step + 0.5)
            if ma < 1: ma = 1
            if mb < 1: mb = 1
            if nv < 4:
                if ma < 3: ma = 3
                if mb < 3: mb = 3
            if ma > MSUB: ma = MSUB
            if mb > MSUB: mb = MSUB
            m = ma
            if ma == 1 and mb == 1:
                if mode >= 1 and mode <= 3: continue
                n = do_quad(G, q00, q10, q01, q11, lrhos[i], rhodots[j], dl, dr, nb_, nras, starts, centers, cbases, cell_rad,
                            vmin_ad, vmax_ad, qmax, tb, res_cell, res_val, n, maxc, stats, mode, t, u, udot, E, Edot, sig_rate, ub, o, o1, fin, fin1, blo, bhi, keep)
                if n >= maxc: return n
                continue
            if mode == 2 or mode == 3: continue
            mpa = ma + 1; mpb = mb + 1
            for a in range(mpa):
                rho_a = math.exp(lrhos[i] + a * dl / ma)
                for b in range(mpb):
                    si = a * mpb + b
                    if (a == 0 or a == ma) and (b == 0 or b == mb):
                        src = q00 if (a == 0 and b == 0) else (q10 if (a == ma and b == 0) else (q01 if (a == 0 and b == mb) else q11))
                        VS[si] = V[src]
                        if V[src] == 1:
                            for c in range(NPT): S[si, c] = G[src, c]
                    else:
                        VS[si] = eval_pt(t, u, udot, E, Edot, rho_a, rhodots[j] + b * dr / mb, o, S, si, vlo, vhi, qhi, stats, fin)
            for a in range(ma):
                for b in range(mb):
                    s00 = a * mpb + b; s10 = (a + 1) * mpb + b; s01 = a * mpb + b + 1; s11 = (a + 1) * mpb + b + 1
                    nvs = VS[s00] + VS[s10] + VS[s01] + VS[s11]
                    if nvs < 3:
                        stats[7] += 1; continue
                    if mode >= 1 and mode <= 3: continue
                    if nvs == 3:
                        # sheet edge: synthesize the missing corner by affine extrapolation (parallelogram) into the spare slot
                        if VS[s00] == 0:
                            pa = s10; pb = s01; pc = s11
                        elif VS[s10] == 0:
                            pa = s00; pb = s11; pc = s01
                        elif VS[s01] == 0:
                            pa = s00; pb = s11; pc = s10
                        else:
                            pa = s10; pb = s01; pc = s00
                        for c in range(NPT): S[spare, c] = S[pa, c] + S[pb, c] - S[pc, c]
                        nrm = math.sqrt(S[spare, 0] ** 2 + S[spare, 1] ** 2 + S[spare, 2] ** 2)
                        if nrm < 1e-9: continue
                        S[spare, 0] /= nrm; S[spare, 1] /= nrm; S[spare, 2] /= nrm
                        if S[spare, 3] <= 0.0: continue
                        ra = fast_atan2(S[spare, 1], S[spare, 0])
                        if ra < 0.0: ra += TWOPI
                        S[spare, 9] = ra
                        if VS[s00] == 0: s00 = spare
                        elif VS[s10] == 0: s10 = spare
                        elif VS[s01] == 0: s01 = spare
                        else: s11 = spare
                    n = do_quad(S, s00, s10, s01, s11, lrhos[i] + a * dl / ma, rhodots[j] + b * dr / mb, dl / ma, dr / mb,
                                nb_, nras, starts, centers, cbases, cell_rad, vmin_ad, vmax_ad, qmax, tb, res_cell, res_val, n, maxc, stats, mode, t, u, udot, E, Edot, sig_rate, ub, o, o1, fin, fin1, blo, bhi, keep)
                    if n >= maxc: return n
    return n

@njit(parallel=True, cache=True, error_model='numpy')
def process_chunk(T, U, UD, EP, EV, SR, lrhos, rhodots, nb_, nras, starts, centers, cbases, cell_rad, vmin, vmax, qmax, step_fac, maxc, mode, blo, bhi, keep):
    m = T.shape[0]
    nl = lrhos.shape[0]; nr = rhodots.shape[0]
    counts = np.zeros(m, np.int64)
    rc = np.empty((m, maxc), np.int32)
    rv = np.empty((m, maxc * NOUT), np.float32)
    st = np.zeros((m, 8), np.int64)
    for a in prange(m):
        G = np.empty((nl * nr, NPT)); V = np.zeros(nl * nr, np.uint8)
        S = np.empty(((MSUB + 1) * (MSUB + 1) + 1, NPT)); VS = np.zeros((MSUB + 1) * (MSUB + 1), np.uint8)
        o = np.empty(17); tb = np.empty(6); rvl = np.empty(maxc * NOUT)
        if True:
            counts[a] = process_tracklet(T[a], U[a], UD[a], EP[a], EV[a], SR[a], lrhos, rhodots, nb_, nras, starts, centers, cbases, cell_rad,
                                         vmin, vmax, qmax, step_fac, G, V, S, VS, o, tb, rc[a], rvl, maxc, st[a], mode, blo, min(bhi, nb_), keep)
            for q in range(counts[a] * NOUT): rv[a, q] = rvl[q]
    return counts, rc, rv, st

def default_scan(rho_min=0.03, rho_max=12.0, nl=40, vmax_kms=110.0, nr=56):
    return np.linspace(math.log(rho_min), math.log(rho_max), nl), np.linspace(-vmax_kms, vmax_kms, nr) / KMS
