"""Joint heliocentric two-body fit of a candidate cluster (stage 3 screen before find_orb).
State x = (r, v) at epoch t0 (AU, AU/day). Observables per tracklet: unit direction u (2 eq, tangent plane) and
angular rate udot (2 eq). Gauss-Newton with finite-difference Jacobian. Returns rms position residual (arcsec)."""
import math, numpy as np
from numba import njit
from kernel2 import GM, CLIGHT

@njit(cache=True, fastmath=True, error_model='numpy')
def stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z); return (s - math.sin(s)) / (s * z), (1.0 - math.cos(s)) / z
    elif z < -1e-8:
        s = math.sqrt(-z); return (math.sinh(s) - s) / (s * (-z)), (math.cosh(s) - 1.0) / (-z)
    return 1.0 / 6.0, 0.5

@njit(cache=True, fastmath=True, error_model='numpy')
def kepler(r0, v0, dt, out):
    """universal-variable propagation (Vallado); out[0:3]=r, out[3:6]=v; returns 0 on failure"""
    rn = math.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    vr = (r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2]) / rn
    v2 = v0[0]**2 + v0[1]**2 + v0[2]**2
    alpha = 2.0 / rn - v2 / GM
    sg = math.sqrt(GM)
    if alpha > 1e-12:
        x = sg * dt * alpha
    elif alpha < -1e-12:
        a = 1.0 / alpha
        num = -2.0 * GM * alpha * dt
        den = vr * rn * (1.0 - rn * alpha) + (1.0 if dt >= 0 else -1.0) * math.sqrt(-GM * a) * (1.0 - rn * alpha)
        x = (1.0 if dt >= 0 else -1.0) * math.sqrt(-a) * math.log(num / den) if den != 0.0 and num / den > 0 else sg * dt / rn
    else:
        x = sg * dt / rn
    for it in range(80):
        z = alpha * x * x
        S, C = stumpff(z)
        F = (rn * vr / sg) * x * x * C + (1.0 - alpha * rn) * x**3 * S + rn * x - sg * dt
        dF = (rn * vr / sg) * x * (1.0 - z * S) + (1.0 - alpha * rn) * x * x * C + rn
        dx = F / dF
        x -= dx
        if abs(dx) < 1e-12 * max(1.0, abs(x)): break
    z = alpha * x * x
    S, C = stumpff(z)
    f = 1.0 - x * x / rn * C
    g = dt - x**3 / sg * S
    for i in range(3): out[i] = f * r0[i] + g * v0[i]
    r1 = math.sqrt(out[0]**2 + out[1]**2 + out[2]**2)
    fd = sg / (r1 * rn) * (alpha * x**3 * S - x)
    gd = 1.0 - x * x / r1 * C
    for i in range(3): out[3 + i] = fd * r0[i] + gd * v0[i]
    return 1 if (math.isfinite(r1) and abs(dx) < 1e-8 * max(1.0, abs(x))) else 0

@njit(cache=True, fastmath=True, error_model='numpy')
def residuals(x, t0, T, U, UD, OP, OV, res, tmp):
    """res: per tracklet [dpos_x, dpos_y (rad), drate_x, drate_y (rad/day)] in the tangent plane of U"""
    n = len(T)
    for k in range(n):
        dt = T[k] - t0
        # one light-time iteration
        ok = kepler(x[0:3], x[3:6], dt, tmp)
        if ok == 0: return 0
        px = tmp[0] - OP[k, 0]; py = tmp[1] - OP[k, 1]; pz = tmp[2] - OP[k, 2]
        rho = math.sqrt(px*px + py*py + pz*pz)
        ok = kepler(x[0:3], x[3:6], dt - rho / CLIGHT, tmp)
        if ok == 0: return 0
        px = tmp[0] - OP[k, 0]; py = tmp[1] - OP[k, 1]; pz = tmp[2] - OP[k, 2]
        rho = math.sqrt(px*px + py*py + pz*pz)
        ux = px / rho; uy = py / rho; uz = pz / rho
        vx = tmp[3] - OV[k, 0]; vy = tmp[4] - OV[k, 1]; vz = tmp[5] - OV[k, 2]
        rd = ux*vx + uy*vy + uz*vz
        udx = (vx - rd*ux) / rho; udy = (vy - rd*uy) / rho; udz = (vz - rd*uz) / rho
        # tangent basis at observed U[k]
        ax, ay, az = U[k, 0], U[k, 1], U[k, 2]
        if abs(az) < 0.9: ex, ey, ez = -ay, ax, 0.0
        else: ex, ey, ez = 0.0, -az, ay
        en = math.sqrt(ex*ex + ey*ey + ez*ez); ex /= en; ey /= en; ez /= en
        fx = ay*ez - az*ey; fy = az*ex - ax*ez; fz = ax*ey - ay*ex
        res[k, 0] = ux*ex + uy*ey + uz*ez
        res[k, 1] = ux*fx + uy*fy + uz*fz
        res[k, 2] = (udx - UD[k, 0])*ex + (udy - UD[k, 1])*ey + (udz - UD[k, 2])*ez
        res[k, 3] = (udx - UD[k, 0])*fx + (udy - UD[k, 1])*fy + (udz - UD[k, 2])*fz
    return 1

@njit(cache=True, fastmath=True, error_model='numpy')
def fit(x0, t0, T, U, UD, OP, OV, sig_pos, sig_rate, maxit=40):
    """Gauss-Newton on the 6-state. Weighted: positions by sig_pos (rad), rates by sig_rate (rad/day).
    Returns (ok, rms_pos_arcsec, chi2, x)"""
    n = len(T); x = x0.copy(); tmp = np.empty(6); res = np.empty((n, 4)); res2 = np.empty((n, 4))
    J = np.empty((4 * n, 6)); b = np.empty(4 * n); w = np.empty(4 * n)
    for k in range(n):
        w[4*k] = 1.0 / max(sig_pos[k], 1e-12); w[4*k+1] = w[4*k]; w[4*k+2] = 1.0 / max(sig_rate[k], 1e-12); w[4*k+3] = w[4*k+2]
    chi2 = 1e300; rms = 1e300; lam = 1e-3; c2n = 1e300
    for it in range(maxit):
        if residuals(x, t0, T, U, UD, OP, OV, res, tmp) == 0: return 0, 1e300, 1e300, x
        for k in range(n):
            for j in range(4): b[4*k+j] = -res[k, j] * w[4*k+j]
        for p in range(6):
            h = 1e-7 if p < 3 else 1e-8
            xs = x.copy(); xs[p] += h
            if residuals(xs, t0, T, U, UD, OP, OV, res2, tmp) == 0: return 0, 1e300, 1e300, x
            for k in range(n):
                for j in range(4): J[4*k+j, p] = (res2[k, j] - res[k, j]) / h * w[4*k+j]
        A = J.T @ J; g = J.T @ b
        c2 = 0.0; s = 0.0
        for k in range(n):
            c2 += (res[k,0]*w[4*k])**2 + (res[k,1]*w[4*k+1])**2 + (res[k,2]*w[4*k+2])**2 + (res[k,3]*w[4*k+3])**2
            s += res[k,0]**2 + res[k,1]**2
        rms = math.sqrt(s / n) * 206264.806; chi2 = c2
        accepted = False
        for tries in range(8):
            Ad = A.copy()
            for p in range(6): Ad[p, p] *= 1.0 + lam
            dx = np.linalg.solve(Ad, g)
            xt = x + dx
            if residuals(xt, t0, T, U, UD, OP, OV, res2, tmp) == 0: lam *= 10.0; continue
            c2n = 0.0
            for k in range(n):
                c2n += (res2[k,0]*w[4*k])**2 + (res2[k,1]*w[4*k+1])**2 + (res2[k,2]*w[4*k+2])**2 + (res2[k,3]*w[4*k+3])**2
            if c2n < c2:
                x = xt; lam = max(lam / 3.0, 1e-9); accepted = True; break
            lam *= 10.0
        if not accepted or (c2 - c2n) < 1e-6 * c2: 
            if accepted:
                s = 0.0
                for k in range(n): s += res2[k,0]**2 + res2[k,1]**2
                rms = math.sqrt(s / n) * 206264.806; chi2 = c2n
            break
    return 1, rms, chi2, x

def initial_state(tr, idx, lrho, rhodot):
    """heliocentric state of tracklet idx at its epoch from a kernel solution (rho, rhodot)"""
    rho = math.exp(lrho); u = tr['u'][idx]; ud = tr['udot'][idx]
    r = tr['obs_pos'][idx] + rho * u
    v = tr['obs_vel'][idx] + rhodot * u + rho * ud
    return np.r_[r, v], tr['t'][idx] - rho / CLIGHT

def fit_members(tr, mem, x0, t0):
    T = tr['t'][mem]; U = tr['u'][mem]; UD = tr['udot'][mem]; OP = tr['obs_pos'][mem]; OV = tr['obs_vel'][mem]
    sp = np.deg2rad(np.maximum(tr['rms_arcsec'][mem], 0.3) / 3600.0) / np.sqrt(tr['nobs'][mem])
    try:
        return fit(np.asarray(x0, float), float(t0), T, U, UD, OP, OV, sp, tr['sig_rate'][mem])
    except np.linalg.LinAlgError:
        return 0, np.inf, np.inf, np.asarray(x0, float)

def robust_fit(tr, mem, seeds, thr_arcsec=60.0, min_members=3):
    """seeds: list of (x0, t0). Fit from each seed, keep best; then iteratively drop members whose position residual
    exceeds max(3*rms, thr) and refit. Returns (members_kept, rms, chi2_per_dof, x, t0)."""
    mem = np.asarray(mem, int); best = None
    for x0, t0 in seeds:
        ok, rms, chi2, x = fit_members(tr, mem, x0, t0)
        if ok and (best is None or rms < best[1]): best = (x, rms, chi2, t0)
    if best is None: return mem[:0], np.inf, np.inf, None, None
    x, rms, chi2, t0 = best
    tmp = np.empty(6)
    while len(mem) >= min_members:
        res = np.empty((len(mem), 4))
        if residuals(x, t0, tr['t'][mem], tr['u'][mem], tr['udot'][mem], tr['obs_pos'][mem], tr['obs_vel'][mem], res, tmp) == 0: break
        pos = np.hypot(res[:, 0], res[:, 1]) * 206264.806
        bad = pos > max(3.0 * rms, thr_arcsec)
        if not bad.any(): break
        if (~bad).sum() < min_members: mem = mem[~bad]; break
        mem = mem[~bad]
        ok, rms, chi2, x = fit_members(tr, mem, x, t0)
        if not ok: break
    dof = max(1, 4 * len(mem) - 6)
    return mem, rms, chi2 / dof, x, t0
