# find_bound (planned)
Unknown bound asteroids and comets in unlinked archives (ITF, unflagged DECam). Replace the (v∞, b, t_p)
parameterization in `kernel4.py` with Keplerian elements (a, e, i, Ω, ω, M) sampled on the same adaptive (ρ, ρ̇)
grid; the hasher/triangle/fit stages are unchanged (they only need time-invariant elements). The ITF search already
produced 569 bound by-products; this makes that the goal.
