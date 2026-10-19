# find_neo (planned)
Earth-approaching objects and temporarily captured mini-moons: restrict the kernel to small topocentric distance ρ
and add Earth's gravity to the two-body propagator in `fit2.py`; link over hours–days instead of months.

## Performance
To be measured with the same protocol as `tools/find_iso/README.md` ("How to compare your pipeline to this one"): hardware, orbit-space coverage per tracklet, per-stage wall time and ms/tracklet on a named ITF window, losslessness verification, injection-recovery on known objects, false objects after the final fit, peak RSS.
