# find_iso — all-sky archival search for interstellar objects (hyperbolic orbits)

Pipeline used to search the MPC Isolated Tracklet File (ITF) and unflagged NOIRLab Source Catalog (DECam) detections
for a fourth interstellar object. Design rule throughout: **no physical priors as hard cuts** (all-sky radiant grid,
full v_inf / q range, every tracklet, every root); every speed-up must be lossless and is verified by diffing outputs.

## Method (one tracklet in, candidate objects out)
1. `tools/find_iso/pipe/ingest.py` — parse 80-column observations, build same-night tracklets with topocentric observer state
   (position/velocity from the JPL ephemeris); `nsc_tracklets2.py` / `nsc_assemble.py` do the same for unflagged
   DECam/NSC detections (≥3 epochs ≥10 min apart, straight-line rms < 2", KD-tree pairing); `nsc_obs80.py` formats
   NSC detections as MPC 80-col lines for find_orb.
2. `tools/find_iso/pipe/kernel4.py` (+ `kernel2.py` helpers) — grid-native hyperbolic-orbit kernel. For each tracklet it evaluates the
   map (rho, rho_dot) -> (radiant, v_inf, impact vector b, perihelion time t_p, q, e) once on an adaptive grid and
   reads off the solution for every one of the 8148 2.5° radiant cells its sheet covers, with analytic derivatives
   and error propagation. ~0.13 ms/tracklet interpolation-only, ~0.4 ms with exact elements (8 cores).
3. `tools/find_iso/pipe/search3.py` (`run_kernel`, `neighbours`, triangle utilities) and `tools/find_iso/pipe/search6b.py` (`find_pairs`,
   `hash_cells`, `triangle_clusters`) — lossless element-space linker: each tracklet's solution sheet is sampled and
   bucket-hashed per radiant cell, exact 5-D tests on collisions (incl. neighbour-cell probing), dense buckets swept
   once, per-cell pair buffers retried at a 20x larger cap when they overflow (so dense sky is never truncated), then
   triangles (three tracklets agreeing on radiant and (v_inf, b, t_p)) define per-cell clusters.
4. `tools/find_iso/pipe/fit2.py` + `tools/find_iso/pipe/itf_fit2.py` — Stage-3 seed-and-grow two-body fitter: fit each large cluster, grow accepted
   orbits over the window's leftover tracklets with time-expanding rounds, dedupe, merge fragments only if the union
   fits one orbit (rms < 10", chi2/dof < 100, ≥3 tracklets). Validated on injected 1I (52/52), 2I (624/644),
   3I (~2070/2143), each recovered as one object with zero false objects.
5. `tools/find_iso/pipe/fo_screen2.py` / `tools/find_iso/pipe/fo_screen_nsc.py` — headless find_orb (full planetary perturbations) on every
   survivor, in process groups with timeouts; records e, a/q, obs used, rms, arc, stations.

`tools/find_iso/pipe/itf_full.py` drives steps 2–3 per 2-year window (1-year stride) in memory-bounded passes over radiant-cell
ranges (`ISO_PASS_TRK`), with per-pass checkpoints, optional per-pass triangle clustering (`ISO_STREAM=1`) and pass
ranges for worker splits (`ISO_PASS_RANGE=a:b`). `tools/find_iso/workers/bootstrap*.sh` set up a fresh 8-core VM and run a window
or pass range. `tools/find_iso/pipe/blind3.py`, `truth1I.py`, `fitclusters2.py` are the injection/validation scripts;
`mpcorb_shortarc.py` extracts single-opposition MPCORB objects for e-free refits; `aggregate.py` / `pull_workers.py`
collect candidate lists.

## Usage
```bash
pip install "numpy>=2,<3" "numba>=0.60" scipy jplephem
# data/: itf.txt (MPC ITF), de440s.bsp, ObsCodes.html; find_orb built at src/find_orb/fo
python3 tools/find_iso/pipe/ingest.py data/itf.txt data/itf_tracklets.npz
ISO_TRK=data/itf_tracklets.npz ISO_OUT=data/itf python3 tools/find_iso/pipe/itf_full.py --plan
ISO_TRK=... ISO_OUT=... python3 tools/find_iso/pipe/itf_full.py 38 39        # windows 38-39
ISO_TRK=... ISO_OUT=... python3 tools/find_iso/pipe/itf_fit2.py 38 39        # Stage-3 -> fit2_wNN.npy + CANDIDATE2 lines
ISO_TRK=... ISO_OUT=... python3 tools/find_iso/pipe/fo_screen_nsc.py         # find_orb on every candidate -> fo_screen.txt
```
Large inputs (millions of tracklets): `ISO_STREAM=1 ISO_PASS_TRK=10000`, split passes across machines with
`ISO_PASS_RANGE`, collect `clusters_wNN_pPP.pkl`, then run `itf_fit2.py`.

## Results so far
See `../../docs/STATUS.md` (full run log). ITF 1993–2026: 600 linked objects, 569 bound, 31 fit artifacts, no ISO.
DECam/NSC tier 1 (clean ≥4-detection tracklets + ITF 2017–2021): 156 objects, 149 bound, 7 artifacts, no ISO.
Tier 2 (3-detection DECam tracklets) in progress.

## Performance (measured; for comparison against other pipelines)

**Hardware / software.** One VM: 8 vCPU Intel Xeon Platinum 8559C (1 thread/core), 31 GB RAM, 32 GB swap, no GPU.
Python 3.10, numpy 2.2, numba 0.68 (all hot loops `@njit(parallel=True)`), scipy 1.15; find_orb (Bill Gray) built from source.
All numbers are wall-clock on this machine with all 8 cores; nothing below used more than one machine unless stated.

**Problem size.** Orbit space searched per tracklet: all 8148 radiant cells (2.5° HEALPix-like grid, full sky),
v∞ 8–80 km/s, q ≤ 6 AU, every (ρ, ρ̇) root — no priors on radiant, speed, or sky region. A tracklet typically has a
valid hyperbolic solution in ~20–25 % of cells, so ~1,500–3,300 solutions per tracklet are emitted and stored
(24 float32 values each: elements, derivatives, error bars).

**Kernel (`kernel4.py`).** 2.4 ms/tracklet at the start → 0.134 ms/tracklet interpolation-only, 0.42 ms/tracklet
with exact final elements + error propagation (what the search actually uses), benchmarked on 6k–66k ITF tracklets.
On a production window (340,660 tracklets, 1.04 billion solutions) the kernel stage took 2,359 s = 6.9 ms/tracklet,
or 2.3 µs per emitted solution — the difference to the micro-benchmark is memory traffic for writing 1 billion
solutions, not arithmetic. Per-pass fixed cost ≈ 0.19 ms × N tracklets (per-tracklet setup), so windows are split
into passes over cell ranges (`ISO_PASS_TRK`) and the total kernel time scales ≈ N × (number of passes).

**Linker (`search6b.py`).** Full ITF window 1 (1I validation): 66k tracklets, 150M solutions, 4 sky bands: kernel
143 s, pair hashing 620 s (original) → ~5x faster after the sparse radix-bucket rewrite; triangle clustering 6 s.
Production (NSC tier 1, three 2-year windows, 905k tracklets, 2.6 billion solutions, 460M candidate pairs → 39.9M
unique, 0 dropped): 85 passes, 33,100 s total ≈ 36.6 ms per tracklet end-to-end including kernel, or 12.7 µs per
emitted solution, 59 µs per raw candidate pair. Window-level: w00 340,660 trk / 157M raw pairs / 9,248 s;
w01 385,283 trk / 221M raw pairs / 12,996 s; w02 178,872 trk / 82M raw pairs / 1,614 s. Pair volume (not tracklet
count) drives time, and it is super-linear in density: the ecliptic/opposition bands dominate.
Lossless: per-cell pair buffers that overflow are re-searched at a 20x cap; the retry was verified to reproduce the
full pair set exactly (0 missing, 0 extra) on a 40k-tracklet test, and every production pass reports `overflow 0`.
Memory bound: a pass only materializes solutions for its own cells plus their hash-probe neighbours
(`run_kernel(keep_cells=…)`), verified identical pair sets; peak RSS 15–31 GB per pass at `ISO_PASS_TRK=10000` on
3.37M tracklets.

**Stage-3 fitter (`fit2.py`, `itf_fit2.py`).** Two-body seed-and-grow with analytic Jacobians, microseconds per
fit; a full window (tens of thousands of clusters, tens of candidate objects) fits in ~10–60 min single-machine.

**find_orb screen.** Full planetary + asteroid perturbations, headless, 8 parallel processes, 600 s timeout per
object: 600 ITF objects screened in ~1 h.

**Sensitivity / selectivity (injection–recovery, same settings for all, nothing tuned per object).**
1I/ʻOumuamua: 52/52 ITF tracklets recovered as one object, rms 0.64", χ²/dof 1.42, v∞ 26.4 km/s (true 26.3).
2I/Borisov: 624/644 tracklets, rms 1.03", v∞ 32.3 km/s (true 32.3). 3I/ATLAS: ~2,070/2,143 tracklets, rms 0.69",
v∞ 58.0 km/s. False objects on the full 66k-tracklet 1I window: 0 (3,900 random candidate clusters all rejected by
the joint orbit fit, χ²/dof > 100). Full ITF (2.56M tracklets, 9.1M observations, 1993–2026, 51 windows):
600 linked objects, 569 bound asteroids, 31 non-converging/degenerate fits, 0 hyperbolic survivors.

**How to compare your pipeline to this one.** Use the same inputs (public MPC ITF; de440s ephemeris), report
(a) machine (cores, RAM), (b) orbit-space coverage per tracklet (radiant cells × v∞ × q range, or equivalent),
(c) wall time per stage and per tracklet for one named ITF window (we recommend the 2-year window containing
2017-10-19 for 1I, 66k ITF tracklets), (d) whether any stage drops candidates under load (buffer caps, sampling,
priors) and how that was verified, (e) 1I/2I/3I recovery fractions and the number of false objects after the final
fit stage, (f) peak RSS. Our numbers for (c): kernel 143 s, pairs ~120 s, triangles 6 s, Stage-3 + find_orb ~15 min,
total ≈ 20 min for the 1I window end-to-end on 8 cores.
