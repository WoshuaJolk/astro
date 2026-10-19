# find_bound (planned)
Unknown bound asteroids and comets in unlinked archives (ITF, unflagged DECam). Replace the (v∞, b, t_p)
parameterization in `kernel4.py` with Keplerian elements (a, e, i, Ω, ω, M) sampled on the same adaptive (ρ, ρ̇)
grid; the hasher/triangle/fit stages are unchanged (they only need time-invariant elements). The ITF search already
produced 569 bound by-products; this makes that the goal.

## Performance
To be measured with the same protocol as `tools/find_iso/README.md` ("How to compare your pipeline to this one"): hardware, orbit-space coverage per tracklet, per-stage wall time and ms/tracklet on a named ITF window, losslessness verification, injection-recovery on known objects, false objects after the final fit, peak RSS.
