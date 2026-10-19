# iso-search — all-sky archival search for interstellar objects

Pipeline used to search the MPC Isolated Tracklet File (ITF) and unflagged NOIRLab Source Catalog (DECam) detections
for a fourth interstellar object. Design rule throughout: **no physical priors as hard cuts** (all-sky radiant grid,
full v_inf / q range, every tracklet, every root); every speed-up must be lossless and is verified by diffing outputs.

## Method (one tracklet in, candidate objects out)
1. `pipe/ingest.py` — parse 80-column observations, build same-night tracklets with topocentric observer state
   (position/velocity from the JPL ephemeris); `nsc_tracklets2.py` / `nsc_assemble.py` do the same for unflagged
   DECam/NSC detections (≥3 epochs ≥10 min apart, straight-line rms < 2", KD-tree pairing); `nsc_obs80.py` formats
   NSC detections as MPC 80-col lines for find_orb.
2. `pipe/kernel4.py` (+ `kernel2.py` helpers) — grid-native hyperbolic-orbit kernel. For each tracklet it evaluates the
   map (rho, rho_dot) -> (radiant, v_inf, impact vector b, perihelion time t_p, q, e) once on an adaptive grid and
   reads off the solution for every one of the 8148 2.5° radiant cells its sheet covers, with analytic derivatives
   and error propagation. ~0.13 ms/tracklet interpolation-only, ~0.4 ms with exact elements (8 cores).
3. `pipe/search3.py` (`run_kernel`, `neighbours`, triangle utilities) and `pipe/search6b.py` (`find_pairs`,
   `hash_cells`, `triangle_clusters`) — lossless element-space linker: each tracklet's solution sheet is sampled and
   bucket-hashed per radiant cell, exact 5-D tests on collisions (incl. neighbour-cell probing), dense buckets swept
   once, per-cell pair buffers retried at a 20x larger cap when they overflow (so dense sky is never truncated), then
   triangles (three tracklets agreeing on radiant and (v_inf, b, t_p)) define per-cell clusters.
4. `pipe/fit2.py` + `pipe/itf_fit2.py` — Stage-3 seed-and-grow two-body fitter: fit each large cluster, grow accepted
   orbits over the window's leftover tracklets with time-expanding rounds, dedupe, merge fragments only if the union
   fits one orbit (rms < 10", chi2/dof < 100, ≥3 tracklets). Validated on injected 1I (52/52), 2I (624/644),
   3I (~2070/2143), each recovered as one object with zero false objects.
5. `pipe/fo_screen2.py` / `pipe/fo_screen_nsc.py` — headless find_orb (full planetary perturbations) on every
   survivor, in process groups with timeouts; records e, a/q, obs used, rms, arc, stations.

`pipe/itf_full.py` drives steps 2–3 per 2-year window (1-year stride) in memory-bounded passes over radiant-cell
ranges (`ISO_PASS_TRK`), with per-pass checkpoints, optional per-pass triangle clustering (`ISO_STREAM=1`) and pass
ranges for worker splits (`ISO_PASS_RANGE=a:b`). `workers/bootstrap*.sh` set up a fresh 8-core VM and run a window
or pass range. `pipe/blind3.py`, `truth1I.py`, `fitclusters2.py` are the injection/validation scripts;
`mpcorb_shortarc.py` extracts single-opposition MPCORB objects for e-free refits; `aggregate.py` / `pull_workers.py`
collect candidate lists.

## Usage
```bash
pip install "numpy>=2,<3" "numba>=0.60" scipy jplephem
# data/: itf.txt (MPC ITF), de440s.bsp, ObsCodes.html; find_orb built at src/find_orb/fo
python3 pipe/ingest.py data/itf.txt data/itf_tracklets.npz
ISO_TRK=data/itf_tracklets.npz ISO_OUT=data/itf python3 pipe/itf_full.py --plan
ISO_TRK=... ISO_OUT=... python3 pipe/itf_full.py 38 39        # windows 38-39
ISO_TRK=... ISO_OUT=... python3 pipe/itf_fit2.py 38 39        # Stage-3 -> fit2_wNN.npy + CANDIDATE2 lines
ISO_TRK=... ISO_OUT=... python3 pipe/fo_screen_nsc.py         # find_orb on every candidate -> fo_screen.txt
```
Large inputs (millions of tracklets): `ISO_STREAM=1 ISO_PASS_TRK=10000`, split passes across machines with
`ISO_PASS_RANGE`, collect `clusters_wNN_pPP.pkl`, then run `itf_fit2.py`.

## Results so far
See `docs/STATUS.md` (full run log). ITF 1993–2026: 600 linked objects, 569 bound, 31 fit artifacts, no ISO.
DECam/NSC tier 1 (clean ≥4-detection tracklets + ITF 2017–2021): 156 objects, 149 bound, 7 artifacts, no ISO.
Tier 2 (3-detection DECam tracklets) in progress.
