# ISO search — working status (kept current for continuity)

## Kernel (updated: EXACT finalization)
- kernel4 now: bilinear patch locates each cell's (rho,rhodot), then exact radiant() + 1 Newton correction on the exact map
  (|d| 0.006 cell), exact finish() for b,tp, rate-error sigma via 2 perturbed udot evals. NOUT=22:
  0 vinf km/s, 1-3 b, 4 tp, 5 q, 6 e, 7 lrho, 8 rhodot, 9-16 A(4x2: vinf,bx,by,bz vs cell tangent d1,d2), 17-21 sigma(vinf,bx,by,bz,tp)
  process_chunk(T,U,UD,EP,EV,SR(sig_rate),lrhos,rhodots,nb,nras,starts,centers,cbases,cell_rad,vmin,vmax,qmax,step_fac,maxc,mode)
  config: default_scan(nl=40,nr=56), step_fac=1.5. Speed 0.42 ms/tr 8 thr (2.9 single): ~1 ms of single-thread time is the
  4 radiant + 3 finish per solution (analytic sigma / skipping correction when |d|<0.01 are the next wins).
  vs kernel3 (cmp34.py 40 56 1.5 NAME): exact-cell hits 1I 41/52 (k3 34), 2I 621/644 (k3 622), 3I 794/2143 (k3 790);
  misses 0.7-1.9% of k3 cells, finds 3-10% more; elements agree to 0.1% (99th pct <2%). kernel4_interp_backup.py = older interp-only.
## Kernel (older notes)
- pipe/kernel4.py = current kernel (grid-native, bilinear patch inversion, no per-cell Newton). API:
  process_chunk(T,U,UD,EP,EV, lrhos, rhodots, nb,nras,starts,centers, cbases(=cell_bases(g)), cell_rad, vmin,vmax,qmax, step_fac, maxc, mode)
  -> counts, rc(cells int32), rv(float32 NOUT=16 per sol: vinf_kms,bx,by,bz,lrho,rhodot(AU/d),q,e,A[4x2]), stats
  mode=0 normal (other modes = profiling stubs). default_scan(nl=16,nr=24) + step_fac=1.5 is the benchmarked config.
  sigma (rate-error) and t_p are NOT computed in the kernel any more -> compute lazily in pair stage for candidate pairs.
- Speed (8 cores, 1I window tracklets): kernel3 2.4 ms/tr -> kernel4 0.134 ms/tr (18x). Single-thread 0.96 ms:
  evals 0.2 (2700 x 75ns), candidate-cell loop 0.3 (17k x 18ns), bilinear solve+A 0.45.
- Correctness vs kernel3 (pipe/cmp34.py NL NR STEP NAME): exact true-cell hits
  1I 47/52 (k3 34/52); 2I 609/644 (k3 622/644; 13.6% of k3 cells missing in k4 at base 16x24 -> checking 24x32/40x56);
  3I 782/2143 (k3 790/2143; both low -> true cell often neighbour, high sensitivity at v_inf 58).
  Elements agree: median 1% of across-cell variation, 99th pct 0.16-0.48.
- Floor analysis: ~1800 cells covered per tracklet at 2.5 deg; per-cell irreducible ~50ns => ~0.05-0.07 ms/tr on 8 cores
  after micro-opts. 0.01 ms needs either 5 deg cells (4x fewer cells; resolution not coverage) or ~8x more cores.

## Next
1. Compiled pair/cluster stage consuming kernel4 output (per cell: own + 8 neighbours translated with A; E-box overlap
   candidates; exact pair test with lazy sigma/tp via finite differences on radiant()/finish()).
2. Blind recovery 1I/2I/3I (inj_*.npz, first_line thresholds 1I 231195, 2I 651864, 3I 366346; true states in cmp34.py).
3. Then full ITF, MPCORB refit, NSC/DECam same-night tracklets (nsc/exposures_c4d.csv downloaded, 340953 exposures).

## Stage 2 (pairs) — pipe/search3.py, driver pipe/blind3.py NAME NPASS
- Memory: ~1900 sols/tracklet x 88 B => 66k tracklets = 11 GB -> OOM. Solution: stream by sky-band passes:
  kernel4.process_chunk(..., blo, bhi) clips the radiant-band loop, search3.search() runs npass passes (kernel on bands
  [B0-1,B1+1), pairs for cells in [B0,B1)). Memory / npass; eval cost repeated per pass (~0.2 of kernel time).
- pair_cell: per cell k gather own + neighbour solutions (translated by nb_off), vinf-interval sweep, "at least one narrow"
  rule (sigma_vinf<5 km/s & sigma_b<0.2 AU; info-based, not a physics cut), weighted LS for common delta (clamped to the
  intersection of both cells' delta boxes), chi2 (4 eq/2 dof) < 16, |dtp| < 10 d + 3 sigma. Pairs -> dedupe -> union-find.
- Logs: data/test/blind3_1I.log ; outputs pairs_NAME.npz

## Stage 2 v2 — pipe/search4.py (hashed sheet linking), drivers small4.py NAME NOTH / blind3.py NAME NPASS
- search3 sweep was hopeless: boxes (A*cell + 3 sigma) span ~35-50% of the population -> 1e9 cand pairs/cell.
- search4: each solution is a linear sheet E(delta)=E0+A.delta (A from kernel). Sample at NSUB=8 x 8 sub-cell centres,
  bucket (counting sort) on (sub-cell, vinf bin, tp bin) with 2x2 shifted grids (bin = 2*tol, TOL = 1.5 km/s, 0.05 AU, 1.5 d),
  check bx,by,bz within tol in-bucket, dedupe per tracklet pair, |dt|>=1 d, then exact 5-eq/2-dof weighted LS chi2<20.
  Only 'narrow' (informative) solutions are hashed: sigma_v<5 km/s, sigma_b<0.2 AU, sigma_tp<5 d (error-based, not physics).
- kernel4 now NOUT=24 (22,23 = dtp/ddelta), NPT=11 (tp on grid). Backups: kernel4_pre_tp.py, search4_pre_tp.py.
- Before tp hashing (1I, 52 ISO + 4096 random): all 52 ISO paired, 1117 ISO-ISO pairs, but 178k random pairs (2% of all!)
  -> everything in one cluster. tp hashing (selectivity ~1/300) is the fix being tested.
- tp hashing result (1I small, 52+4096): 67k unique pairs (ISO-ISO 1071, random 63k = 0.75% of all pairs). Pair-level
  selectivity is intrinsically ~1e-3 at these tolerances -> selectivity must come from triangles: triangle_clusters() in
  search4 (per cell, 3 edges with consistent delta*). Found 958 per-cell clusters with >=3 ISO (max 48/52) vs 1230 random
  (1038 of size 3). Random k-clusters scale ~N^k -> need ~10x tighter pairs: NSUB=16 (testing) and chi cut:
  chi ISO-ISO quantiles [0, 0.01, 0.26, 4.8] vs random [0.66, 4.75, 15.3, 19.5] -> chi<3 keeps 98% ISO, 37% random
  (sigma from rate errors is conservative). Merge per-cell clusters into objects only if they share >=2 members.
- NSUB=16 w/o tightening TOL was worse (76k random pairs, 4x slower) -> reverted. TOL/3 = [0.5 km/s, 0.017 AU, 0.5 d], NSUB=8:
  1I small: 8152 unique pairs (ISO-ISO 952, random 6888), per-cell random triangle clusters 5 (was 1230), merged objects:
  ISO object 49/52 members + 3 random size-3. 81 s. Full window extrapolation: ~1.8M random pairs, ~20k random triplets
  before chi<3 cut (~1k after). Next: 2I/3I small (same settings), then full-window blind3 for 1I/2I/3I.
- 2I small: 635/644 paired, main object 476/644 + 12 small 2I-only fragments, 13 random triplets. 3I small: 1901/2143 paired
  but Python triangle enumeration choked on 6.7M raw pairs (dense ISO graph) -> triangle_clusters(dense=50000) accepts
  a cell's edges outright when pairs/cell > 50k (far above random density); chi<3 applied before triangles. Rerun queued.
- Full 1I window (blind3 1I 4): pass 1 (bands 18-36) 46M sols, 4.3M raw pairs in 293 s, overflow 10351 (cap=20000/cell hit)
  and runs>max 128926 (hash runs > maxrun=256 truncated) -> LOSSY at full scale; must find what makes long runs
  (degenerate geometry?) and raise/handle caps before trusting full-window completeness.
- Full 1I window result: 1.84M unique pairs (ISO-ISO 951, 938 at chi<3), 51/52 ISO paired, 7284 per-cell triangle clusters,
  4090 merged objects; ISO object 54 members (48 ISO + 6 random). ~4000 random objects come from degenerate populations
  (diag_rand.py): zero-rate repeated stationary sources, opposition main-belt asteroids with near-identical motion, slow
  distant movers. Element agreement alone cannot kill these -> stage 3: pipe/fit2.py joint heliocentric 2-body fit
  (universal-variable Kepler, Gauss-Newton on 6-state, tangent-plane position+rate residuals, robust_fit with outlier drop).
  Pure 1I members: initial resid ~1800" -> 10.4" rms (max 57", Earth-perturbation level) -> threshold 60". Timings for
  full 1I window: kernel 143 s, hashing 620 s (ecliptic bands dominate; overflow/runs>max truncation = lossy), triangles 6 s,
  fits 30 s for 3836 objects. fitclusters.py NAME drives stage 3 from pairs_NAME.npz.
- Stage 3 works: fit2.fit now Levenberg-Marquardt (maxit 40), seeds from up to 12 per-cell clusters of the object
  (the biggest ISO per-cell cluster sat at a wrong radiant cell, vinf 34 vs 26.4 -> plain GN from it stalled at 16000").
  1I object: 48/48 ISO members kept, rms 0.48", chi2/dof 1.35 (two-body is enough over 70 d). Random objects: 3857,
  201 with rms<10" but only 2 with >=4 members. fitclusters.py 1I: 111 s for 3887 objects.
- search4 hasher: long runs now sorted by bx and swept (lossless, no maxrun truncation); cap 200k/cell. Backup search4_pre_runs.py.
- 3I small (2143 ISO + 4096 random): 1901/2143 paired; merged objects: 3I fragments 607, 590, 123 + 20 small 3I-only
  triplets; ~10 random per-cell triplets. Python triangle step 342 s (dense ISO graph) -> needs numba eventually.
  Stage-3 acceptance from 1I full window: rms<10" & chi2/dof<100 passes 1I (0.48", 1.35) and 0/3857 random objects.
- Running: run_full.sh -> blind3 2I 10 passes, fitclusters 2I, blind3 3I 8 passes, fitclusters 3I (logs data/test/blind3_*.log, fit_*.log).
  TODO after: fitclusters should merge fragments whose fitted orbits agree; hashing speed on ecliptic bands; numba triangles.
- search5.py (hash v3): radix-sorted sparse buckets, 1 grid + neighbour probing, touched-list table clears, degenerate-run
  set-hash skip, dedupe per SOLUTION pair (search4 deduped per tracklet pair -> order-dependent root losses).
  cmp45.py (52 ISO + 4096 bg, under CPU contention): search4 201 s / 8110 distinct pairs; search5 40 s / 15631 distinct,
  superset of search4 except 1 borderline pair (chi 19.3 of 20). Kernel multiplicity check: 94% of (tid,cell) single root,
  duplicates are distinct roots (not redundancy). blind3 now uses search5. 2I/3I full chain relaunched on search5.
- search5 dense-cell profile (dbg7.py, 64 equatorial cells of 1I window, 18.5k sols/cell): search4 231 ms/cell, 30k raw / 15.5k
  distinct pairs; search5 124 ms/cell, 140k raw / 52.9k distinct (neighbour probing is a wider prefilter than 4 shifted grids,
  extra pairs all pass exact chi<20 -> search4 was lossy vs its own exact test). emit() now keeps min-chi row per tracklet pair
  per cell (slot map) after testing every solution pair once -> raw volume /6.5 vs first search5.
- search5 final criterion: pair candidate iff within TOL in all 5 dims at some common sub-cell sample (explicit |dv|,|dtp| checks;
  same guarantee search4's shifted grids gave, but deterministic). dbg8 (2I bands 0-7, 41k sols/cell): s4 134/225/58 ms/cell vs
  s5 194/197/27; distinct pairs s5 ~1.1-1.3x s4 (all root pairs tested). Dedupe table resets at 2M touched (was: infinite loop when
  >4M solution pairs in a cell -> first 2I run hung in pass 0). Compiled triangle stage (search5.triangle_clusters/_tri_cell):
  identical clusters to search4 on 1I pairs, 4x faster; fitclusters/small4/blind3 now import search5.
- Hasher speedup #2 experiments (dense 2I cells, bands 14-21 kernel output cached in data/test/k2I_b14_21.npz):
  search6 = search5 made cache-local (per-cell float32 gather of ~550k solutions, no solution-pair table, 8MB slot map):
  identical pairs/chi, only ~10% faster -> random val access was NOT the bottleneck. search7 = + 2-D (by,bz) binning in runs:
  identical output but 2-4x slower (binary-search overhead) -> the bx-sweep comparisons were not the bottleneck either.
  Phase profile (search6p.hash_cell_p) in progress to find the real one. Full 2I under search5: pass3 501s/22.8M pairs,
  pass4 765s/23.9M pairs (ecliptic bands, pair volume explodes).
- Parallel prep started: NSC revisit groups (nsc_groups.py: 17,963 same-night revisit groups, 113,746 exposures, 23.8B meas),
  nsc_fetch.py pulling ndet<=2 measurements per exposure (~2.6k rows/exposure, 4.5 s/query, 4 workers -> ~80 h; raise workers);
  MPCORB single-opposition short arcs: 194,651 objects with arc<=60 d (data/mpcorb_shortarc.pkl); find_orb binary at src/find_orb/fo.
- 2I full run (search5) killed in the triangle stage: search5._tri_cell used insertion sort on adjacency lists (deg up to 1e4 in dense
  2I cells -> quadratic), and the dense-cell path still enumerated triangles. Fixed (argsort for long lists; components-only for
  >50k-edge cells). blind3 now saves pairs before the triangle stage. Relaunched run_full.sh: 2I (search6b, 10 passes) -> fit -> 3I -> fit.
- Full ITF parallelized over 7 child Devin sessions (windows 38-51, 2 each; bundle data/../iso_pipe.tgz = bundle/bootstrap.sh + pipe files;
  prompts in iso/prompts/). Sessions: 5a472e8025534a7e9db5fb87a29dae04 (38-39), a148737807f642a0976f158713cc32a5 (40-41),
  09b8bc3c200843189fa3f2e487b531f8 (42-43), 32cda85b5dd24ff9b56f9b70276090c4 (44-45), 909b43a009164fec8985ecf427894828 (46-47),
  98e551695b3a419e8980513924ea55e2 (48-49), 525359e1fbf8438589a7e87d3823f3c0 (50-51). Local box: itf_full.py 15 37 (small windows),
  itf_fit.py <w> for clustering/fit; results data/itf/. GitHub integration cannot create repos (403).
- 2026-10-18 resume after quota pause (all VMs restarted). Relaunched locally: run_full.sh (2I->fit->3I->fit), itf_full.py 15 37,
  nsc_fetch.py 4 20 (fetch3.log). 7 children were resumed by the user ("Continue") and re-messaged to re-run bootstrap; note children
  downloaded a fresh ITF (e.g. 2,560,423 tracklets vs 2,562,669 here) - window results refer to each worker's ITF snapshot.
- NSC TAP throughput dropped to ~30 exp/min on 2026-10-18 (server-side latency, 3+ min per 20-exposure batch); left running, resumable.
- Candidate protocol: itf_fit accepted objects -> extract_obs.py <out> <ids> -> find_orb joint fit (common designation; fo treats
  different desigs as different objects). Size-3 objects: position rms is meaningless (6 constraints = 6 params); only the rates
  discriminate, so treat 3-tracklet 'accepted' objects as weak unless chi2/dof is small AND find_orb gives a consistent e>1 orbit.
  First local candidate (w26: AKK052/VR1543/pis180, 2001-11..2002-03): pis180 is a 4.8-min arc flagged 'probably a star' by fo.
  find_orb joint fit of w26 candidate: bound a=2.59 e=0.38 using only 4/9 obs -> rejected (chance alignment).
  w27 4-tracklet candidate: find_orb bound a=3.45 e=0.50 (12/12 obs) -> rejected; short arcs of bound objects pass the 2-body hyperbolic screen, find_orb bound-vs-hyperbolic comparison is the discriminator.
- 2I full run under search6b: kernel 813s + pairs 2180s over 10 passes (search5 run: 3822s total; search4 est. >6000s); pair sets identical (7,816,350 unique; 631/644 ISO tracklets in ISO-ISO pairs).
- fitclusters.py 2I was OOM-killed (ran concurrently with itf_full window 29/30, ~15 GB val arrays). itf_full paused (windows 15-29
  done, 30-37 remain). Sequence: run_full.sh continues with 3I (blind3 -> fitclusters) alone; then rerun `fitclusters.py 2I`
  (pairs_2I.npz saved), then restart `nice python3 -u itf_full.py 15 37` and `itf_fit.py 29..37`. NSC TAP at ~4 exp/min (server-side).
- 3I full window (108,489 tracklets, 8 passes, search6b): 286.4M solutions, 33.4M raw pairs, kernel+pairs 1214 s; 3.18M unique pairs;
  1830/2143 injected 3I tracklets in ISO-ISO pairs; 91,687 per-cell triangle clusters (11 s with compiled stage), 6242 with >=3 ISO,
  max 320 ISO members in one cluster, 85,334 pure-random clusters -> fitclusters.py 3I running (fit_3I.log).
- 3I fitclusters result: 4494 objects; the main 3I object (592 members, 570 injected) did NOT converge in the two-body screen
  (rms 376", chi2/dof 1.4e6) -> only fragments passed (74 ISO members rms 4.5" chi2/dof 91; 13; 7; 6; 3). 1I converged fully (48/52),
  2I pending. Gap to fix in Stage 3: robust_fit seeding/convergence on long dense arcs (seed from the pair-level common solution of
  the densest cell, grow membership iteratively). Random objects: 356 with rms<10" (mostly 3-tracklet exact position fits), 4 with
  <60" & kept>=4. Accepted after merge: 6 (1 is 3I, 1 is a 3I triplet, 4 small random).
- Worker results arriving (data/itf/workers/w*.txt): w38 142.6k tracklets, 352M sols, 57.4M raw pairs, 5.62M unique, OVERFLOW 7.45M
  (per-cell pair cap 200k hit in dense cells -> completeness caveat for that window), total 1494 s; 13 accepted (all 3-tracklet, weak).
  w46 303.8k tracklets, 1.02B sols, 64M raw pairs, 9.1M unique, overflow 0, 4807 s; 31 accepted. Children hit OOM in itf_fit (31 GB,
  old seeding code) and added swap; bootstrap.sh cannot resume a completed window (grep rc=1 under pipefail) - children worked around it.
  itf_fit.py/fitclusters.py now seed from <=3000 members (bundle updated) - fixes the OOM for future runs.
- 2I fitclusters (seed-subset code): 5732 objects; main 2I merged object = 88,912 members (544 injected) -> fit fails; only a 7-member
  2I fragment passes (rms 0.83", chi2/dof 4.8, vinf 32 km/s). KEY VALIDATION FINDING: on dense windows (2I, 3I) the per-cell
  cluster MERGE (share >=2 members) chains main-belt clusters into one giant object and Stage-3 fit cannot recover the ISO from it;
  pairs (631/644, 1830/2143) and triangles (clusters with up to 213/320 ISO members) do recover it. Fix needed: merge only clusters
  whose (vinf,b,tp) agree, and/or fit the largest per-cell clusters directly before merging.
- fo_screen.py (data/cand/fo_screen.txt): ALL 46 accepted candidates so far (local w26-28 + worker w38, w46) get BOUND find_orb
  orbits (e 0.02-0.86, a 1-8.5 AU), many using only a subset of obs -> no ISO candidate survives stage 5 yet. All were 3-tracklet
  (one 4-tracklet) objects; the 2-body hyperbolic screen passes short arcs of bound objects by construction.
- Window 30 (MJD 53165-53895, 82k tracklets) pass 2: 76.9M pairs with 270.8M OVERFLOW (per-cell cap 200k) -> that window is
  incomplete; cap/dynamic output buffers are a needed fix before claiming completeness for dense windows (also w38: 7.45M overflow).
- HANDOFF (if context resets): pending = (a) 7 children (ids above) finish -> read structured_output via devin_session_interact get,
  append lines to data/itf/workers/w<tag>.txt, run pipe/aggregate.py and pipe/fo_screen.py; (b) itf_full 15 37 finishing locally ->
  itf_fit.py 29..37; (c) nsc_fetch.py 3 5 running (fetch4.log, ~14 exp/min; 110k exposures to go); (d) final report to Josh:
  validation (1I full recovery; 2I/3I pairs+triangles recover, merge+fit fragments only), ITF results (windows, accepted, find_orb
  all bound), caveats (overflow cap, 3-tracklet weakness), next steps (merge fix, cap fix, NSC tracklets, MPCORB refit).
- Window 30 final: 82k tracklets, 146M sols, 123M raw pairs, 6.87M unique, OVERFLOW 317.5M (pathological density, likely many same-object repeats); results for w30 are incomplete pending the cap fix.
- w29 candidates: both bound per find_orb (a=1.89 e=0.50 2/8 obs; C000014 5-trk a=2.93 e=0.24) -> rejected. Running total: 48 accepted -> 0 survive find_orb.
- Tools for aggregation: pipe/pull_workers.py <overflow content.txt> (worker structured_output -> data/itf/workers/w*.txt), pipe/aggregate.py (table), pipe/fo_screen.py (find_orb on all CANDIDATE lines -> data/cand/fo_screen.txt), itf_fit.py <w> for local windows (29 done; 30-37 pending as itf_full finishes).
- Workers done: 42-43 (w42 234k trk, 645M sols, 10.5M unique, overflow 26.5M, 3487 s, 23 accepted; w43 237k trk, 7.43M unique,
  overflow 123M, 4836 s, fit line missing -> asked worker), 50-51 (w50 308.5k trk, 1.06B sols, 8.9M unique, overflow 0, 4248 s,
  25 accepted; w51 189k trk, 5.09M unique, overflow 0, 2150 s, fit line missing -> asked worker). fo_screen over 59 candidates:
  49 bound, 0 hyperbolic, 10 unparsed (checking elem files). Local w30: 4 accepted; w31 fit rerun (first attempt no output).
- fo_raw.txt (data/cand): raw find_orb one-liners for all 59 candidate obs files. 53 bound (e<0.95 mostly). Non-bound-looking:
  ISO0006 q=0.27 e=0.915 (4/6 obs), ISO0007 q=0.007 e=0.992 (3/9), ISO0012 e=0.942 (6/11), ISO0016/0031/0033/0044 e=1.000 (fo
  parabolic fallback, 4-5 of 9-11 obs used), ISO0022 e=0.973, ISO0023 e=0.940, ISO0052 q=1.61 e=1.047 i=12 (6/9 obs, 2021 Jun30-Sep30,
  w46 list), ISO0056 no solution. None fits ALL its observations -> none is a credible ISO; ISO0052 is the only e>1 and is weak
  (3 obs rejected). Report all as rejected/weak, with ISO0052 flagged for a manual look (which tracklet is rejected?).
  ISO0052 = w46 candidate P11hCFF/P11hufo/P21lQKv (F51 Pan-STARRS, MJD 59395-59487): fo q=1.61 e=1.047 using 6/9 obs -> weak, flag for manual look.
- w50-51 worker lines complete in workers/w50-51.txt (w51 fit line present). Still missing: w43 fit line (asked worker 09b8bc3c); windows 39,40,41,44,45,47,48,49 still running on workers.
- Added 64 GB /swapfile locally (fit 31 first attempt died after 'loaded' line, likely OOM); fit 31 relaunched -> fit_small_31b.log. Asked workers 42-43 and 50-51 to attach fit_wNN.log files (chat text truncates candidate lists).
- Fit logs for w42,43,50,51 downloaded from worker attachments and merged into workers/w*.txt (complete candidate lists). Re-running fo_screen over all candidates -> data/cand/fo_screen.txt
- fit 31 OOM-killed again (dmesg: 3 python3 kills at ~16 GB RSS while itf_full holds 14 GB). Decision: fits no longer run alongside
  itf_full; fit_queue.sh waits for itf_full 15 37 to exit, then runs itf_fit 31..37 sequentially (logs fit_small_NN.log, FITQUEUE_DONE marker).
- fo_raw.txt now covers 140 candidates: 125 bound; notable non-bound: ISO0089 q=1.15 e=1.96 i=3 (only 4/9 obs used, 2024 Nov24-Dec22,
  weak), ISO0099 q=0.97 e=1.001 (9/9 obs but 4-day arc 2024 Sep 9-13 -> e unconstrained), ISO0108 q=5.56 e=0.996 (10 obs, 7-day arc),
  ISO0133 = ISO0052 dup (e=1.047, 6/9 obs). Several e=1.000 = find_orb parabolic fallback. No credible hyperbolic object yet.
- Full find_orb elements: ISO0099 (P11Ype6/P11Yr46/P21YQzV, F51 2024 Sep 9-13) e=1.0008 +/- 0.097, q=0.971, H=24.1, 9/9 obs, 4-day arc
  -> e unconstrained, consistent with a small bound NEO; ISO0089 (C0YQGY5/P122kiG/P222FBs) e=1.96 +/- 8.4 from 4/9 obs -> meaningless.
  Verdict so far: 0 credible ISO among 140 fast-screen candidates (local w15-32 + worker w38,42,43,46,50,51).
- aggregate.py table at this point: 24 windows with pairs (1.97M tracklet-windows), 22 fitted, 159 accepted fast-screen objects.
- Planned Stage-3 merge fix (not yet implemented; needs free CPU): replace "merge clusters sharing any member" with fit-gated
  agglomeration: (1) seed from the best-chi2 cluster, (2) add a neighbouring cluster only if >=2 shared members AND the joint
  two-body fit of the union keeps rms < thr and chi2/dof < thr, else keep as separate object; (3) after convergence, re-test
  every leftover tracklet of the window against the final orbit (attribution) so a dense true object (2I/3I) is rebuilt from
  fragments instead of chained with unrelated main-belt clusters. Validate on 2I/3I pairs_*.npz (already saved), then rerun
  itf_fit on all windows (pairs files kept, kernel need not rerun).
- w47 done (worker 909b43a0): 312027 trk, 1.066B sols, 9.60M unique, overflow 0, accepted 17; worker.log merged into workers/w46-47.txt. Heavy windows done: 38,42,43,46,47,50,51 (7/14). fo_screen run3 launched.
- w48-49 done (worker 98e55169): worker.log merged into workers/w48-49.txt. Heavy windows done: 38,42,43,46,47,48,49,50,51 (9/14). Remaining: 39,40,41,44,45.
- w38-39 done (worker 5a472e80): merged. Heavy done: 38,39,42,43,46,47,48,49,50,51 (10/14). Remaining 40,41,44,45.
- fo_raw over 190 candidates: 167 bound; new non-bound: ISO0170 q=0.75 e=2.34 i=5 (4/9 obs used, 2022 Oct-2023 Mar) -> weak like ISO0089; rest are repeats (e=1.000 parabolic fallbacks, ISO0052 dup). Still 0 credible ISO.

## Update 2026-10-18 (local windows complete)
- Local ITF windows 15-37: search + itf_fit complete for all. Accepted (fast screen): w31 9, w32 17, w33 11, w34 8, w35 1, w36 4, w37 6.
- w31 fit: 121.4M chi<3 pairs, 229k clusters, 2795 objects, fits 382 s, 9 accepted. w32: 57.1M, 153k, 2552 objects, 17 accepted.
- find_orb screen (fo_screen run5): 227 candidate records → 197 bound, 31 unparsed/parabolic-fallback, 0 credible e>1. The e>1 lines (ISO0207 e=2.34 4/9 obs; ISO0220 e=1.047 6/9 obs; several e=1.000 parabolic fallbacks with 4-5/9 obs) are duplicates of earlier weak cases from overlapping windows and reject ≥1/3 of their observations — not credible.
- Heavy workers: 12/14 windows done (38,39,42,43,46-51); 40-41 and 44-45 still running.
- NSC: see fetch4.log; still ~19 exp/min.

## Stage-3 v2 (fitclusters2.py, seed-and-grow) — 2026-10-18
Replaces shared-member chaining: fit each large per-cell cluster alone, grow accepted orbits by re-testing leftover tracklets (group, then whole window). Same thresholds (rms<10", chi2/dof<100, >=3 kept). No physics cuts.
- 1I: 52/52 tracklets in one object (was 48/52), rms 0.64", chi2/dof 1.42, vinf 26.4 km/s; 0 other objects.
- 3I: 2072/2143 in one object (was fragmented, max 74), rms 0.81", chi2/dof 0.83, vinf 58.0 km/s; 2 small 3I fragments (53, 5) + ~20 bound 1-2-night false objects (3-4 tracklets, dense-field degenerates).
- 2I (v2 grow, residual gate 3"): best object only 21/644 (7-day arc) — short-arc two-body orbit cannot be extrapolated across the 500-day window at 3". v3 grow = time-expanding window (one arc-span per round) with 30" gate + outlier rejection at max(3 rms, 2"); running on 2I, 3I, 1I (fit3_*.log).
- v4 grow (time-expanding, 30" gate, >=90% member survival per round): **2I 624/644 in one object**, rms 1.03", chi2/dof 7.2, vinf 32.3 km/s (true 32.3), full span 58725-59122; 11 bound 1-3-night false objects with >=4 tracklets (dense-field degenerates, killed by find_orb screen). v3 (strict survival) had split 3I into 1414+912 time-disjoint pieces -> added joint-fit fragment merge; 3I/1I v4 running.
- v4 + joint-fit fragment merge: **3I 2013/2143 in one object** (rms 0.69", chi2/dof 3.9, vinf 58.0), 0 false objects with >=4 tracklets; 1I 52/52 (v2 run) ; 2I 624/644. Stage-3 v2 validated on all three -> itf_fit2.py (same thresholds) now re-fitting all local windows (fit2_wNN.log, CANDIDATE2 lines) and being shipped to workers for windows 38-51.
- Worker 44-45: old itf_fit.py 44 OOM-killed (102 GB vm) even with 99 GB swap; will use itf_fit2.py instead.
- Stage-3 v2 on local windows 15-37 (fit2_wNN.log): 15-23 → 0 objects, 24-29 → 1-4 each, 33-37 → 17/13/8/9/8. find_orb screen of v2 candidates from windows 15-34 (fo_screen2.txt): 34 candidates (32 with 3 tracklets, one 5, one 10; arcs 5-167 d, median 29 d) → all 34 bound (e ≤ 0.985), 0 hyperbolic, 0 unparsed.
- Stage-3 v2 find_orb screen (fo_screen2.py run4, 2026-10-18 ~16:30 UTC): local windows 15-37 + worker windows 38,39,42,43,46-51 → 405 candidates (353 with 3 tracklets, 39 with 4, 13 with ≥5). 384 bound; 20 nominal e≥1, all 3-tracklet with find_orb using ≤7 of 9-11 obs (rejecting ≥1 tracklet) or parabolic fallback e=1.000 → mislinks; 1 timeout (ISO0107, w42, 3 trk F51+G96 178 d, vinf 42.6 km/s, re-running without limit). All ≥5-tracklet candidates bound with tight residuals. Zero credible hyperbolic survivors. Pending: worker v2 for windows 40,41,44,45; NSC pool.
- fo_screen2.py now caches prior results, runs 6 find_orb processes in parallel, 900 s timeout recorded as 'find_orb timeout' instead of crashing.
- NSC/DECam tracklet builder (pipe/nsc_tracklets.py): per fully-fetched same-night revisit group with >=3 exposures, pair ndet<=2 detections between exposures within 12 deg/day * dt (kernel limit), reject pairs consistent with stationary (<3 sigma + 0.3"), confirm with a 3rd exposure by linear prediction (0.7" + 3 sigma), require >=3 detections (MOPS-standard; bare pairs at DECam depth are noise: a 2-exposure test gave 1.6M pairs from 4.6k detections). Output nsc_tracklets.npz in the ITF schema (+obs_table) so itf_full.py/itf_fit2.py run unchanged via ISO_TRK/ISO_OUT env vars. Groups with only 2 exposures cannot yield a confirmable tracklet and are skipped (reported as a coverage gap, not a physics cut).
- ISO0107 (w42, 3 trk F51+G96, 178 d): find_orb did not converge in 30 min with no limit; 3-tracklet, recorded as unconstrained, not a lead.
- Stage-3 v2 find_orb screen, COMPLETE for all ITF windows (fo_screen2.py run7, 2026-10-18 ~19:30 UTC): local windows 15-37 + all 14 worker windows 38-51 -> 600 unique candidates (512 with 3 tracklets, 57 with 4, 31 with >=5). 569 bound; 31 nominal e>=1, all 3-tracklet except one 4-tracklet parabolic fallback (w42, 146 d, 6/12 obs used). Of the 31: 10 are e=1.000 parabolic fallbacks, 19 reject >=3 of 9-11 obs (one whole tracklet), 1 is a 2-day arc, 1 (ISO0477, w45: W84 14 det 2019-10-25 + F51 3 det 2019-09-25, 16/17 used) converges to q=0.018 AU e=1.007 — a degenerate sun-grazing solution for a 2-night 30 d arc, unconstrained. ISO0107 resolved: a=4.41 e=0.61 (bound, 6/10 obs). Zero credible hyperbolic survivors in the ITF. Worker fit2 logs for 40/41/44/45 reconstructed from their structured_output (data/itf/workers/fit2_w4x.log). fo_screen2.py: as_completed + process-group kill on 600 s timeout (ex.map was blocked by one fo that never exits).
- nsc_tracklets2.py: raerr/decerr are already arcsec (bug: *3600 made sigma~1000", confirmation accepted everything -> 26M tracklets/group). Also: pair exposure i with first exposure >=10 min later; confirmer must be >=10 min from both members (3 distinct epochs), else group is a coverage gap; per-group checkpoints in data/nsc/trk/g{night}_{ib}_{jb}.npz; chunk 500 first-exposure detections to bound memory (dense fields: 19k det/deg^2, 0.5 deg pairing radius).

## 2026-10-18 tier-1 NSC+ITF restart with lossless overflow retry
- First tier-1 run (NSC nobs>=4 + ITF 2017-2021, 519,532 tracklets, 22 passes/window) overflowed the per-cell pair buffer (cap 200k) in passes 6-8: 254k / 70M / 46M dropped candidate pairs — the ecliptic/opposition bands where DECam tracklets concentrate, i.e. not lossless exactly where it matters.
- Fix (search6b.find_pairs, backup search6b.py.bak_preretry): slot table sized dynamically from cap; any cell whose buffer overflows is re-hashed alone with retry_cap=4,000,000. Verified lossless on a 40k-tracklet ITF slice: cap=300+retry reproduces the cap=200k pair set exactly (328,866 pairs, 0 missing/extra, residual overflow 0); without retry cap=300 keeps only 91,919.
- w00 was killed at pass 9/22 and the whole tier-1 chain restarted 19:30 UTC (itf_full -> chain.sh: itf_fit2 0 1 2 -> fo_screen_nsc). Expect ~5-6 h to the NSC verdict.
- Ops lesson (again): `pkill -f "...chain.sh"` matched the calling shell's own command line and killed it; kill by PID.
- 20:29 UTC second restart: the serial per-cell retry left pass 7 single-threaded for >45 min (one thread busy, 7 idle). hash_cells now takes an explicit cell list, and overflowed cells are retried 8 at a time in parallel (retry_batch=8). Re-verified lossless on the same 40k ITF slice (328,866 pairs, 0 missing/extra); retry overhead 9 s vs 98 s serial. Backups: search6b.py.bak_preretry, search6b.py.bak_retry1.
- Third restart (pass 7 lossless: 24 cells retried, 22.79M pairs vs 19.86M+70M dropped, 1127 s): projected w00 pair volume (~150M rows) would have OOM'd the end-of-window `S3.dedupe(P_all.astype(float64))` (~230 B/row peak). itf_full now dedupes each pass in float32 (`dedupe_lean`, verified identical to search3.dedupe) and writes per-pass checkpoints `pairs_wNN_pPP.npy`, which a restart re-loads instead of recomputing. Backup: itf_full.py.bak_prechk.

## NSC tier-1 kernel/pair search (lossless run, 2026-10-19)
- window 00 MJD 58001-58731: 340660 tracklets, 1.04e9 solutions, 157.4M raw pairs -> 13.23M unique, overflow 0, 9248 s
- window 01 MJD 58366-59096: 385283 tracklets, 1.26e9 solutions, 220.6M raw pairs -> 19.16M unique, overflow 0, 12996 s
- all overflowed cells retried at cap 4M, residual overflow 0 in every pass; per-pass checkpoints in data/nsc/tier1/pairs_wNN_pPP.npy
- window 02 running (12 passes); chain.sh then runs itf_fit2.py 0 1 2 and fo_screen_nsc.py
- window 02 MJD 58731-59461: 178872 tracklets, 2.99e8 solutions, 82.1M raw pairs -> 7.46M unique, overflow 0, 1614 s
- 2026-10-19 ~04:00 UTC: disk filled at w02 pass 5 (64G swapfile + per-pass checkpoints). Swap shrunk to 32G, finished w00/w01 per-pass checkpoints removed (pairs_w00/01.npz verified loadable), w02 resumed from checkpoint p04; nsc_fetch restarted (22410 exposures done).
- search done 04:34 UTC; itf_fit2.py 0 1 2 running, then fo_screen_nsc.py
- 04:46 UTC itf_fit2 w00 crashed in merge stage: numba np.linalg.solve raised LinAlgError (singular normal matrix) inside fit2.fit; fixed in fit2.fit_members (LinAlgError -> failed fit). Rerunning itf_fit2.py 0 1 2 + fo_screen_nsc.py (chain2.log).
- w00 Stage-3: 86 objects (11 with an NSC tracklet); find_orb: 83 bound, 3 nominal e>=1 all 3-tracklet fits using 3-4/10-11 obs (NSC0029 e=1.000 3/11, NSC0042 e=1.633 4/10, NSC0085 e=1.000 3/10). All >=4-tracklet objects bound (23-trk NSC0001 a=3.16 e=0.12 148/150 obs; NSC0002 a=2.61 e=0.07 102/105; 20-trk NSC0003 a=2.22 e=0.54 82/94). No lead. (fo_w00.log / fo_screen.txt, superseded by chain2 full run)
- 05:31 UTC NSC tier-1 find_orb verdict (fo_screen.txt): 156 Stage-3 objects (w00 86, w01 55, w02 35; 19 involve an NSC tracklet) -> 149 bound, 7 nominal e>=1, none credible: NSC0029/0085/0107/0113 e=1.000-1.001 parabolic fallbacks (3/11, 3/10, 3/16, 8/10 obs; 0113 is a 2-night 30 d arc), NSC0042 e=1.63 4/10 obs, NSC0146 e=20.3 4/9 obs 7 d, NSC0110 q=0.30 e=1.19 7/10 obs = 2 nights (2019-07-06 + 2020-01-27, F51 only), degenerate two-night fit. All >=4-tracklet objects bound. Coverage caveat: tier 1 = NSC tracklets with >=4 detections (673k) + ITF 58001-59130 only; the 11.06M 3-detection NSC tracklets are unsearched; NSC pool = 22.4k/113.7k target exposures (W84, Sep 2018-Sep 2019).

## NSC tier 2 (2026-10-19)
- Loader quality filter (rms<2") passed only 91,116 of 673,347 nobs>=4 NSC tracklets (median rms 6.8"): tier 1 effectively searched 91k DECam tracklets + ITF. Defensible as lossless for real objects (DECam astrometry ~0.15").
- 3-detection NSC tracklets passing the loader: 8,252,052. 4.67M of them are from ONE night (MJD 58447) in the Magellanic-Cloud field (RA 6-87, dec -75..-65); that field has 12 nights total.
- Tier 2a = all NSC tracklets passing the loader outside that field (3-det + 4-det) + ITF 58001-59130 -> data/nsc/tier2a_tracklets.npz, chain in data/nsc/tier2a/ (full.log). Tier 2b = the field (data/nsc/tier2b_field_idx.npy), to be linked against its other nights separately.
- tier 2a relaunched 05:58 UTC with cell-based pass edges + ISO_STREAM=1 (per-pass triangle clusters, no window concat) + ISO_PASS_TRK=32000 (~99 passes/window); first launch had 126/198 empty passes because npass exceeded the 72 radiant bands
- 06:2x UTC tier 2a relaunch #3: search3.run_kernel now fills preallocated arrays chunk-by-chunk and permutes val in place (verified identical output on 6000 tracklets, 529,890 sols); ISO_PASS_TRK=24000 (~132 passes/window). Launch #2 (PASS_TRK=32000) reached 29.5 GB RSS in pass 0 and was killed.
- 06:45 UTC tier 2a fan-out: measured 30 min/pass at PASS_TRK=24000 (pass 0: 72.7M sols, 7.98M pairs, 0 overflow, 1803 s, RSS peak ~27.5 GB) -> 276 passes = ~5.5 days on one box. 7 worker sessions created (bundle iso_pipe2a.tgz + tier2a_tracklets_c.npz as attachments; bootstrap2a.sh; ISO_STREAM=1 ISO_PASS_TRK=24000 ISO_PASS_RANGE):
  c9f6570420fa4ea8a6ccc30dc78ce4ac w00 26:61 | af5ea8ee461d45f39e57688935372476 w00 61:96 | 51be7987ee2a4ab5a5064b59e308b499 w00 96:132 | 9270b957b6fb43fab641ff652b347f97 w01 0:34 | eac4cabb8f834abaa58b79add80ec4a4 w01 34:68 | db7fbc43173643a294d082b515231c08 w01 68:102 | 2f1c3214a0484807b47adb5bdc939cce w01 102:135
  Local box: w00 passes 0:26 (chain2.sh, switch after current pass finishes) then w02 all 9 passes. Deliverable per worker: clusters_w0W_pLO_HI.tgz (per-pass cluster pickles) -> unpack into data/nsc/tier2a/, then itf_fit2.py 0 1 2 (reads per-pass clusters when pairs_wNN.npz has 'stream'; the window npz must be written locally once all cluster files are present) + fo_screen_nsc.py.
- 07:0x UTC: PASS_TRK=24000 pass 1 reached 31.2 GB RSS / 0 avail -> killed; switched everything to ISO_PASS_TRK=16000 (w00 198 passes, w01 203, w02 14). Local: w00 0:40 then w02 (chain2.sh). Workers re-assigned: c9f6 w00 40:93 | af5e w00 93:146 | 51be w00 146:198 | 9270 w01 0:51 | eac4 w01 51:102 | db7f w01 102:152 | 2f1c w01 152:203.
- 07:40 UTC root cause of the memory wall: run_kernel materialized ALL solutions in the 3 radiant bands around a pass (B0-1..B1+1) for all 3.16M tracklets, independent of PASS_TRK (pass 0 at 24000 and 16000 both = 72,694,976 sols); mid-latitude bands (~290 cells) would need 60+ GB. Fix: run_kernel(keep_cells=mask) keeps only the pass's cells + their hasher neighbour cells (first-order only, hash_cell reads nb_ids[k,*] of k). Verified identical (pair,cell) sets on 40k-tracklet slices (cells 4000-4041: 2908 pairs both; 60-101: 1665 both), sols 2.4x fewer at mid-lat. PASS_TRK=10000 now (kernel setup ~0.19 ms x N per pass dominates time anyway). Pass plan: w00 316, w01 324, w02 22. Local w00 0:64 + w02; workers (64:148,148:232,232:316 | 0:81,81:162,162:243,243:324) to be re-messaged with the new iso_pipe2a.tgz.
- 07:41 UTC local pass 0 with keep_cells + PASS_TRK=10000: 19.49M sols (was 72.7M), 3.16M pairs, overflow 0, 1081 s, peak RSS ~15 GB (kernel phase 3 GB, hasher ~12 GB). 662 passes total / 8 boxes ~ 25 h.
- local w00 passes 1-3 (PASS_TRK=10000, keep_cells): 41.4M/44.7M/54.2M sols, 3.5M/4.1M/3.7M pairs, overflow 0, 1176/1341/1527 s, peak RSS ~29 GB on pass 3 (swap unused). ~20-25 min/pass.
- local w00 passes 4-5: 50.8M/55.0M sols, 4.5M/4.0M pairs, overflow 0, 1554/1727 s; pass 5 peaked 31.2 GB RSS + 1 GB swap (survived). Memory is at the ceiling for dense passes at PASS_TRK=10000; if a worker OOMs, fall back to PASS_TRK=8000 with the equivalent cell range (edge 64/316 = cell 1650 = edge 80/395).
- local w00 pass 6: 53.9M sols, 5.47M pairs, 12 cells retried at cap 4M, residual overflow 0, 2589 s (retry stage), peak ~31 GB + 1 GB swap. Passes 0-6 all lossless.
- local w00 pass 7: 51.5M sols, 4.80M pairs, 5 cells retried, residual overflow 0, 2392 s. Dense passes run ~40 min; local w00 0:64 ETA ~1.5 days at this rate (workers similar).
- local w00 pass 8: 56.2M sols, 4.20M pairs, overflow 0, 1779 s, peak 31.3 GB + 5 GB swap (survived). Passes 0-8 lossless.
