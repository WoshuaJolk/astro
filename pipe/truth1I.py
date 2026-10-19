import numpy as np, math
GM = 2.9591220828559115e-4
def propagate(r0, v0, dt):
    """universal-variable two-body propagation (Vallado), vectorised over dt"""
    r0n = np.linalg.norm(r0); v0n2 = v0 @ v0; alpha = 2/r0n - v0n2/GM  # 1/a (negative for hyperbola)
    sq = math.sqrt(GM); dot0 = r0 @ v0
    out_r = np.empty((len(dt), 3)); out_v = np.empty((len(dt), 3))
    for k, t in enumerate(dt):
        if alpha < 0:
            a = 1/alpha
            chi = np.sign(t)*math.sqrt(-a)*math.log(-2*GM*alpha*t/(dot0 + np.sign(t)*math.sqrt(-GM*a)*(1 - r0n*alpha)))
        else:
            chi = sq*t*alpha
        for _ in range(60):
            psi = chi*chi*alpha
            if psi > 1e-6: c2 = (1-math.cos(math.sqrt(psi)))/psi; c3 = (math.sqrt(psi)-math.sin(math.sqrt(psi)))/math.sqrt(psi**3)
            elif psi < -1e-6: c2 = (1-math.cosh(math.sqrt(-psi)))/psi; c3 = (math.sinh(math.sqrt(-psi))-math.sqrt(-psi))/math.sqrt((-psi)**3)
            else: c2 = 0.5; c3 = 1/6
            rr = chi*chi*c2 + dot0/sq*chi*(1-psi*c3) + r0n*(1-psi*c2)
            chin = chi + (sq*t - chi**3*c3 - dot0/sq*chi*chi*c2 - r0n*chi*(1-psi*c3))/rr
            if abs(chin-chi) < 1e-12: chi = chin; break
            chi = chin
        f = 1 - chi*chi/r0n*c2; g = t - chi**3/sq*c3
        r = f*r0 + g*v0; rn = np.linalg.norm(r)
        gdot = 1 - chi*chi/rn*c2; fdot = sq/(rn*r0n)*chi*(psi*c3-1)
        out_r[k] = r; out_v[k] = fdot*r0 + gdot*v0
    return out_r, out_v
R0 = np.array([2.689471917784, 0.525911225199, 0.856663661637]); V0 = np.array([19.169061262528, -0.094135159932, 8.441254145208])*1e-3; T0 = 58120.0
def topo(t, E, Edot):
    r, v = propagate(R0, V0, t - T0)
    rho = r - E; rhon = np.linalg.norm(rho, axis=1); uh = rho/rhon[:, None]
    rhodot = np.einsum('ij,ij->i', v - Edot, uh)
    return rhon, rhodot, uh, r, v
