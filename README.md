# astro

Fast archival-search tools for moving objects, built around one engine: a grid-native orbit kernel that maps each
tracklet's sky motion onto a whole family of orbits in one evaluation, and a lossless element-space linker
(hash → pairs → triangles → joint orbit fit → find_orb). Each tool in `tools/` is the same engine with a different
orbit family.

| tool | target | status |
|---|---|---|
| [`tools/find_iso`](tools/find_iso) | interstellar objects (hyperbolic, v∞ 8–80 km/s, q < 6 AU), all-sky, no priors | complete; ITF + DECam/NSC searches running (see `docs/STATUS.md`) |
| [`tools/find_comet`](tools/find_comet) | distant inbound long-period / near-parabolic comets (e ≈ 1, large q) | planned (small change to the e/q range) |
| [`tools/find_bound`](tools/find_bound) | unknown bound asteroids/comets in unlinked archives | planned (Keplerian parameterization of the same kernel) |
| [`tools/find_neo`](tools/find_neo) | Earth-approaching objects, mini-moons (small ρ, Earth gravity) | planned |

`docs/STATUS.md` is the running log of the ISO search (methods, bugs found, every run's numbers).
