# find_comet (planned)
Distant inbound long-period and near-parabolic comets: same kernel and linker as `find_iso`, with the orbit family
widened to e in [0.9, 1.1] and q up to ~15 AU (the kernel already spans this region; the change is the scan range plus
a slower-motion tracklet filter), validated on known long-period comets with ITF precoveries (e.g. C/2017 K2-class
objects found years before perihelion). Deep DECam detections are where such comets sit unnoticed for years.

## Performance
To be measured with the same protocol as `tools/find_iso/README.md` ("How to compare your pipeline to this one"): hardware, orbit-space coverage per tracklet, per-stage wall time and ms/tracklet on a named ITF window, losslessness verification, injection-recovery on known objects, false objects after the final fit, peak RSS.
