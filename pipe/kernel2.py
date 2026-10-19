"""Fast hyperbolic-orbit radiant-hypothesis kernel.
For each tracklet (attributable u, udot at time t, observer heliocentric state E, Edot) and each sky cell
(radiant hypothesis), solve the two unknowns (range rho, range-rate rhodot) so the incoming asymptote of the
two-body hyperbola points exactly at the cell centre. Output the remaining time-invariant elements
(v_inf, impact vector b, perihelion time t_p) plus error estimates."""
import numpy as np, math
from numba import njit, prange

GM = 2.9591220828559115e-4   # AU^3/day^2
CLIGHT = 173.144632674       # AU/day
KMS = 1731.456829            # km/s per AU/day
NOUT = 12  # vinf_kms, bx, by, bz, tp, s_vinf, s_b, s_tp, rho, rhodot_kms, q, e

def make_grid(cell_deg):
    d = math.radians(cell_deg)
    nb_ = int(round(math.pi / d))
    nras = []; starts = [0]
    for k in range(nb_):
        z0 = -1 + 2 * k / nb_; z1 = -1 + 2 * (k + 1) / nb_
        zmid = 0.5 * (z0 + z1); cosd = math.sqrt(max(1 - zmid * zmid, 1e-12))
        nra = max(1, int(round(2 * math.pi * cosd / d)))
        nras.append(nra); starts.append(starts[-1] + nra)
    nras = np.array(nras, np.int64); starts = np.array(starts, np.int64)
    ncell = int(starts[-1]); centers = np.zeros((ncell, 3))
    for k in range(nb_):
        zmid = -1 + 2 * (k + 0.5) / nb_; cosd = math.sqrt(1 - zmid * zmid)
        for j in range(nras[k]):
            ra = 2 * math.pi * (j + 0.5) / nras[k]
            centers[starts[k] + j] = (cosd * math.cos(ra), cosd * math.sin(ra), zmid)
    return dict(nb=nb_, nras=nras, starts=starts, centers=centers, ncell=ncell, cell_rad=d)

@njit(cache=True, fastmath=True, error_model='numpy')
def cell_of(x, y, z, nb_, nras, starts):
    k = int((z + 1) * 0.5 * nb_)
    if k >= nb_: k = nb_ - 1
    if k < 0: k = 0
    ra = math.atan2(y, x)
    if ra < 0: ra += 6.283185307179586
    j = int(ra / 6.283185307179586 * nras[k])
    if j >= nras[k]: j = nras[k] - 1
    return starts[k] + j

@njit(cache=True, fastmath=True, error_model='numpy')
def tangent_basis(c, tb):
    if abs(c[2]) < 0.999:
        ex, ey, ez = -c[1], c[0], 0.0
    else:
        ex, ey, ez = 1.0, 0.0, 0.0
    n = math.sqrt(ex * ex + ey * ey + ez * ez); ex /= n; ey /= n; ez /= n
    tb[0] = ex; tb[1] = ey; tb[2] = ez
    tb[3] = c[1] * ez - c[2] * ey; tb[4] = c[2] * ex - c[0] * ez; tb[5] = c[0] * ey - c[1] * ex

@njit(cache=True, fastmath=True, error_model='numpy')
def radiant(t, u, udot, E, Edot, rho, rhodot, out):
    """out: vinf(AU/d), radx, rady, radz, e, q, rn, hx, hy, hz, ex, ey, ez, px, py, pz, s. returns 1 if unbound"""
    rx = E[0] + rho * u[0]; ry = E[1] + rho * u[1]; rz = E[2] + rho * u[2]
    vx = Edot[0] + rhodot * u[0] + rho * udot[0]
    vy = Edot[1] + rhodot * u[1] + rho * udot[1]
    vz = Edot[2] + rhodot * u[2] + rho * udot[2]
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    eps = 0.5 * (vx * vx + vy * vy + vz * vz) - GM / rn
    if eps <= 0.0: return 0
    hx = ry * vz - rz * vy; hy = rz * vx - rx * vz; hz = rx * vy - ry * vx
    ex = (vy * hz - vz * hy) / GM - rx / rn
    ey = (vz * hx - vx * hz) / GM - ry / rn
    ez = (vx * hy - vy * hx) / GM - rz / rn
    e2 = ex * ex + ey * ey + ez * ez
    if e2 <= 1.0: return 0
    e = math.sqrt(e2)
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    ex /= e; ey /= e; ez /= e
    hhx = hx / hn; hhy = hy / hn; hhz = hz / hn
    px = hhy * ez - hhz * ey; py = hhz * ex - hhx * ez; pz = hhx * ey - hhy * ex
    s = math.sqrt(1.0 - 1.0 / e2)
    ie = 1.0 / e
    out[0] = math.sqrt(2.0 * eps)
    out[1] = -(ex * ie + s * px); out[2] = -(ey * ie + s * py); out[3] = -(ez * ie + s * pz)
    out[4] = e; out[5] = GM * (e - 1.0) / (2.0 * eps); out[6] = rn
    out[7] = hx; out[8] = hy; out[9] = hz; out[10] = ex; out[11] = ey; out[12] = ez
    out[13] = px; out[14] = py; out[15] = pz; out[16] = s
    return 1

@njit(cache=True, fastmath=True, error_model='numpy')
def finish(t, u, E, rho, o, res):
    """b vector and perihelion time from radiant() output o -> res[0:3]=b, res[3]=tp"""
    vinf = o[0]; e = o[4]
    vinx = -o[1]; viny = -o[2]; vinz = -o[3]
    res[0] = (viny * o[9] - vinz * o[8]) / vinf
    res[1] = (vinz * o[7] - vinx * o[9]) / vinf
    res[2] = (vinx * o[8] - viny * o[7]) / vinf
    rx = E[0] + rho * u[0]; ry = E[1] + rho * u[1]; rz = E[2] + rho * u[2]
    rn = o[6]
    cosnu = (rx * o[10] + ry * o[11] + rz * o[12]) / rn
    sinnu = (rx * o[13] + ry * o[14] + rz * o[15]) / rn
    nu = math.atan2(sinnu, cosnu)
    eps = 0.5 * vinf * vinf
    a_abs = GM / (2 * eps)
    x = math.sqrt((e - 1) / (e + 1)) * math.tan(0.5 * nu)
    if abs(x) >= 1.0:
        res[3] = np.nan; return
    H = 2.0 * math.atanh(x)
    M = e * math.sinh(H) - H
    n = math.sqrt(GM / (a_abs * a_abs * a_abs))
    res[3] = (t - rho / CLIGHT) - M / n

@njit(cache=True, fastmath=True, error_model='numpy')
def resid(t, u, udot, E, Edot, lrho, rhodot, c, tb, o):
    ok = radiant(t, u, udot, E, Edot, math.exp(lrho), rhodot, o)
    if ok == 0 or (o[1] * c[0] + o[2] * c[1] + o[3] * c[2]) <= 0.0:
        return 0, 0.0, 0.0
    return 1, o[1] * tb[0] + o[2] * tb[1] + o[3] * tb[2], o[1] * tb[3] + o[2] * tb[4] + o[3] * tb[5]

@njit(cache=True, fastmath=True, error_model='numpy')
def newton(t, u, udot, E, Edot, lrho, rhodot, c, tb, o, tmp, J, lrho_min, lrho_max, maxit, tol):
    """2D Newton on (lrho, rhodot) with Jacobian refreshed only when needed. J: 4 floats (a11,a12,a21,a22) in/out.
    returns (ok, lrho, rhodot); on success o holds radiant() output at solution"""
    d1 = 1e-4; d2 = 2e-6
    have_J = J[4] > 0.5
    ok, f1, f2 = resid(t, u, udot, E, Edot, lrho, rhodot, c, tb, o)
    if ok == 0: return 0, lrho, rhodot
    for it in range(maxit):
        if abs(f1) < tol and abs(f2) < tol:
            return 1, lrho, rhodot
        if not have_J:
            ok1, g1, g2 = resid(t, u, udot, E, Edot, lrho + d1, rhodot, c, tb, tmp)
            ok2, k1, k2 = resid(t, u, udot, E, Edot, lrho, rhodot + d2, c, tb, tmp)
            if ok1 == 0 or ok2 == 0: return 0, lrho, rhodot
            J[0] = (g1 - f1) / d1; J[2] = (g2 - f2) / d1
            J[1] = (k1 - f1) / d2; J[3] = (k2 - f2) / d2
            J[4] = 1.0; have_J = True
        det = J[0] * J[3] - J[1] * J[2]
        if abs(det) < 1e-300: return 0, lrho, rhodot
        dx1 = -(J[3] * f1 - J[1] * f2) / det
        dx2 = -(-J[2] * f1 + J[0] * f2) / det
        if dx1 > 0.3: dx1 = 0.3
        if dx1 < -0.3: dx1 = -0.3
        lim = 15.0 / KMS
        if dx2 > lim: dx2 = lim
        if dx2 < -lim: dx2 = -lim
        nl = lrho + dx1; nr = rhodot + dx2
        if nl < lrho_min or nl > lrho_max: return 0, lrho, rhodot
        ok, n1, n2 = resid(t, u, udot, E, Edot, nl, nr, c, tb, o)
        if ok == 0: return 0, lrho, rhodot
        # if not improving, force Jacobian refresh next time
        if n1 * n1 + n2 * n2 > 0.5 * (f1 * f1 + f2 * f2):
            have_J = False; J[4] = 0.0
        lrho = nl; rhodot = nr; f1 = n1; f2 = n2
    if abs(f1) < tol and abs(f2) < tol:
        return 1, lrho, rhodot
    return 0, lrho, rhodot

@njit(cache=True, fastmath=True, error_model='numpy')
def process_tracklet(t, u, udot, E, Edot, sig_rate, lrhos, rhodots, nb_, nras, starts, centers, cell_rad,
                     vmin, vmax, qmax, best_d, best_i, best_j, res_cell, res_val, maxc):
    ncell = centers.shape[0]
    for k in range(ncell):
        best_d[k] = 9.0
    o = np.empty(17); tmp = np.empty(17); tb = np.empty(6); J = np.zeros(5); fin = np.empty(4); finp = np.empty(4)
    udp = np.empty(3); op = np.empty(17)
    nl = lrhos.shape[0]; nr = rhodots.shape[0]
    lrho_min = lrhos[0] - 0.7; lrho_max = lrhos[nl - 1] + 0.7
    vmin_ad = vmin / KMS; vmax_ad = vmax / KMS
    # ---- coarse scan: mark touched cells (+ 4 neighbours) with best starting point
    for i in range(nl):
        rho = math.exp(lrhos[i])
        for j in range(nr):
            if radiant(t, u, udot, E, Edot, rho, rhodots[j], o) == 0: continue
            if o[0] < vmin_ad or o[0] > vmax_ad or o[5] > qmax: continue
            k = cell_of(o[1], o[2], o[3], nb_, nras, starts)
            d = 1.0 - (o[1] * centers[k, 0] + o[2] * centers[k, 1] + o[3] * centers[k, 2])
            if d < best_d[k]:
                best_d[k] = d; best_i[k] = i; best_j[k] = j
            # neighbours (dilation): shift radiant by +-cell_rad along local east/north
            tangent_basis(o[1:4], tb)
            for q in range(4):
                sgn = 1.0 if (q & 1) == 0 else -1.0
                if q < 2:
                    nx = o[1] + sgn * cell_rad * tb[0]; ny = o[2] + sgn * cell_rad * tb[1]; nz = o[3] + sgn * cell_rad * tb[2]
                else:
                    nx = o[1] + sgn * cell_rad * tb[3]; ny = o[2] + sgn * cell_rad * tb[4]; nz = o[3] + sgn * cell_rad * tb[5]
                kn = cell_of(nx, ny, nz, nb_, nras, starts)
                if kn != k and best_d[kn] > 1.0:
                    best_d[kn] = 1.0 + d; best_i[kn] = i; best_j[kn] = j
    # ---- refine per touched cell
    n = 0
    un = math.sqrt(udot[0] ** 2 + udot[1] ** 2 + udot[2] ** 2) + 1e-30
    cx = u[1] * udot[2] - u[2] * udot[1]; cy = u[2] * udot[0] - u[0] * udot[2]; cz = u[0] * udot[1] - u[1] * udot[0]
    for k in range(ncell):
        if best_d[k] >= 9.0: continue
        c = centers[k]
        tangent_basis(c, tb)
        J[4] = 0.0
        ok, lr, rd = newton(t, u, udot, E, Edot, lrhos[best_i[k]], rhodots[best_j[k]], c, tb, o, tmp, J, lrho_min, lrho_max, 8, 3e-8)
        if ok == 0: continue
        if o[0] < vmin_ad or o[0] > vmax_ad or o[5] > qmax: continue
        rho = math.exp(lr)
        finish(t, u, E, rho, o, fin)
        if not (fin[3] == fin[3]): continue
        if n >= maxc: return n
        # error propagation, linearised: df = f(x; udot+d) ; dx = -J^-1 df ; elements at (x+dx, udot+d)
        sv = 0.0; sb = 0.0; st = 0.0
        if J[4] < 0.5:
            ok1, g1, g2 = resid(t, u, udot, E, Edot, lr + 1e-4, rd, c, tb, tmp)
            ok2, k1, k2 = resid(t, u, udot, E, Edot, lr, rd + 2e-6, c, tb, tmp)
            if ok1 == 0 or ok2 == 0: continue
            J[0] = g1 / 1e-4; J[2] = g2 / 1e-4; J[1] = k1 / 2e-6; J[3] = k2 / 2e-6; J[4] = 1.0
        det = J[0] * J[3] - J[1] * J[2]
        if abs(det) < 1e-300: continue
        for pdir in range(2):
            if pdir == 0:
                udp[0] = udot[0] * (1 + sig_rate / un); udp[1] = udot[1] * (1 + sig_rate / un); udp[2] = udot[2] * (1 + sig_rate / un)
            else:
                udp[0] = udot[0] + sig_rate * cx / un; udp[1] = udot[1] + sig_rate * cy / un; udp[2] = udot[2] + sig_rate * cz / un
            okp, f1, f2 = resid(t, u, udp, E, Edot, lr, rd, c, tb, tmp)
            if okp == 1:
                dx1 = -(J[3] * f1 - J[1] * f2) / det
                dx2 = -(-J[2] * f1 + J[0] * f2) / det
                okp = radiant(t, u, udp, E, Edot, math.exp(lr + dx1), rd + dx2, op)
            if okp == 1:
                finish(t, u, E, math.exp(lr + dx1), op, finp)
                sv += ((op[0] - o[0]) * KMS) ** 2
                sb += (finp[0] - fin[0]) ** 2 + (finp[1] - fin[1]) ** 2 + (finp[2] - fin[2]) ** 2
                st += (finp[3] - fin[3]) ** 2
            else:
                sv += 1e4; sb += 1e2; st += 1e6
        base = n * NOUT
        res_cell[n] = k
        res_val[base + 0] = o[0] * KMS; res_val[base + 1] = fin[0]; res_val[base + 2] = fin[1]; res_val[base + 3] = fin[2]
        res_val[base + 4] = fin[3]; res_val[base + 5] = math.sqrt(sv); res_val[base + 6] = math.sqrt(sb); res_val[base + 7] = math.sqrt(st)
        res_val[base + 8] = rho; res_val[base + 9] = rd * KMS; res_val[base + 10] = o[5]; res_val[base + 11] = o[4]
        n += 1
    return n

@njit(parallel=True, cache=True, error_model='numpy')
def process_chunk(T, U, UD, EP, EV, SIG, lrhos, rhodots, nb_, nras, starts, centers, cell_rad, vmin, vmax, qmax, maxc):
    n = T.shape[0]; ncell = centers.shape[0]
    res_cell = np.empty((n, maxc), np.int32)
    res_val = np.empty((n, maxc * NOUT), np.float64)
    counts = np.zeros(n, np.int64)
    for a in prange(n):
        best_d = np.empty(ncell); best_i = np.empty(ncell, np.int64); best_j = np.empty(ncell, np.int64)
        counts[a] = process_tracklet(T[a], U[a], UD[a], EP[a], EV[a], SIG[a], lrhos, rhodots, nb_, nras, starts, centers,
                                     cell_rad, vmin, vmax, qmax, best_d, best_i, best_j, res_cell[a], res_val[a], maxc)
    return counts, res_cell, res_val

def default_scan(rho_min=0.03, rho_max=12.0, nl=40, vmax_kms=110.0, nr=56):
    return np.log(np.geomspace(rho_min, rho_max, nl)), np.linspace(-vmax_kms, vmax_kms, nr) / KMS
