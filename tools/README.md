# tools
Each subfolder is a self-contained search tool sharing the kernel/linker design described in `find_iso/README.md`.
Common inputs: MPC 80-column observations (ITF, MPCORB short arcs) or NSC/DECam detections built into tracklets;
common outputs: linked objects with two-body fits, then find_orb orbits with full planetary perturbations.
