# AGENTS.md

## Project invariants
- Build a Python package named `nvdyn` with Python 3.11+, `src/` layout, and `pytest`.
- Use `qutip==5.2.3`.
- Keep the implementation typed, modular, documented, and scientifically honest.
- Default physical model:
  - one NV electron in the ground-state spin-1 manifold
  - N nearby `13C` nuclei, each spin-1/2
  - no direct `13C-13C` interactions anywhere
- Default electron-nuclear hyperfine model for each carbon:
  `H_hf^(k) = A_par^(k) * Sz * Iz_k + A_perp^(k) * Sx * Ix_k`
- Public MVP API for `C13Spec` exposes:
  - `A_par`
  - `A_perp`
  - optional `tensor` override
- If a full tensor is supplied, treat it as an advanced alternate mode, clearly separated from the default secular model and clearly documented.
- Perfect initialization into electron `m_s = 0` and perfect reset between shots by default.
- Experiment sequences and circuit operations must compile into the same `PulseSchedule` / `PulseIR`.
- The main circuit simulation path must be pulse-accurate, not a disconnected ideal-gate simulator.
- Use one consistent basis ordering and one consistent internal unit convention throughout the repo.
- Do not silently add `13C-13C` couplings, `theta/phi`, or mandatory reset physics.

## Subagent workflow
Use subagents only for bounded, parallelizable work. The main agent owns architecture, integration, conflict resolution, and final verification.

Recommended delegation:
1. `physics-core`
   - files: `constants.py`, `units.py`, `operators.py`, `device.py`, `hamiltonian.py`
   - responsibilities: basis conventions, operator construction, register model, default secular hyperfine model, optional full-tensor path
   - tests: dimensions, tensor products, default hyperfine assembly, no `13C-13C` terms

2. `controls-and-sequences`
   - files: `waveforms.py`, `channels.py`, `pulses.py`, `schedule.py`, `sequences.py`
   - responsibilities: MW/RF pulse objects, analytic and sampled waveforms, delays, frame changes, sequence macros
   - tests: waveform sampling, schedule assembly, CPMG/DDRF/PulsePol/SWAP structure

3. `solvers-and-readout`
   - files: `sim/exact.py`, `sim/lindblad.py`, `observables.py`, `readout.py`, `results.py`
   - responsibilities: QuTiP-based exact and Lindblad simulation, expectations, trajectories, perfect-init defaults, optional count model
   - tests: Rabi, Ramsey, dephasing, trajectory/results interfaces

4. `circuit-compiler`
   - files: `circuit.py`, `compiler.py`
   - responsibilities: circuit IR, pulse-accurate compilation to the same `PulseSchedule` used by sequences
   - tests: circuit-to-schedule lowering, `swap`, native entangling macro hook

5. `docs-and-examples`
   - files: `README.md`, `examples/`
   - responsibilities: runnable demos, usage docs, implementation notes aligned with actual code
   - tests: example smoke tests where reasonable

## Integration rules
- Subagents must not invent their own basis order, units, or public naming conventions.
- Each subagent should leave a short integration note describing files changed, assumptions made, and tests added.
- The main agent merges the work, resolves conflicts, runs the full test suite, and makes the final pass on API consistency and documentation accuracy.
## Performance and parallel execution
- Pin `qutip==5.2.3`.
- Build time-dependent Hamiltonians and time-dependent collapse operators with `QobjEvo`.
- Prefer list-based or coefficient-based `QobjEvo` construction for compiled pulse schedules; avoid function-returning-`Qobj` forms unless necessary.
- Use QuTiP's process-based parallel APIs from `qutip.solver.parallel`:
  - `serial_map`
  - `parallel_map`
  - `loky_pmap`
- Introduce a small `ParallelConfig` or equivalent with:
  - `map`: `"serial" | "parallel" | "loky"`
  - `num_cpus`: `int | None`
  - `timeout`: `float | None`
  - `fail_fast`: `bool`
- Deterministic backends (`sesolve`, `mesolve`, `SESolver`, `MESolver`) must parallelize only over independent jobs:
  - ODMR / Rabi / Ramsey / ENDOR frequency sweeps
  - scan points over pulse amplitude / phase / duration / tau
  - repeated experiment batches
  - batched circuits
  - calibration scans
- Do not claim or implement QuTiP-native multiprocessing over the internal ODE time steps of one deterministic trajectory.
- Use solver class interfaces (`SESolver`, `MESolver`, and where relevant `MCSolver`) inside workers when repeating the same physical model, so operator preparation can be reused.
- Avoid nested parallelism by default.
- If code uses solver `step()` with multiple solver instances in parallel, prefer `dop853` or `vern9` rather than the default `adams`.
- If a stochastic / Monte-Carlo backend is implemented, use QuTiP's native solver options such as `options={"map": "parallel", "num_cpus": ...}` or `options={"map": "loky", "num_cpus": ...}`.
- Do not assume parallel Monte-Carlo trajectories return in the same order as serial execution.
- Keep serial mode as the reference path; parallel modes must match serial results within numerical tolerances.

## Dependency policy for parallelism
- Core dependency: `qutip==5.2.3`
- Optional extra: `loky` for `loky_pmap`
- Do not use removed legacy APIs such as `parfor`.

## Subagent workflow update
6. `performance-and-parallel`
   - files: `parallel.py` or `dispatch.py`, `sim/*`, and sweep-related utilities
   - responsibilities:
     - `ParallelConfig`
     - serial / multiprocessing / loky dispatch
     - batch sweep execution
     - result collation
     - serial-vs-parallel parity tests
     - avoiding nested parallel execution by default