# nvdyn

`nvdyn` is a typed Python package for pulse-level simulation of NV-center spin dynamics with one ground-state NV electron spin-1 and `N` nearby `13C` spin-1/2 nuclei. The package uses one shared `PulseSchedule` for experiment sequences and circuit compilation, and the primary simulation path is pulse-accurate through the same laboratory-frame Hamiltonian.

## Physical model

The MVP models:

- one NV electron in the ground-state spin-1 manifold
- `N` nearby `13C` nuclei, each spin-1/2
- NV zero-field splitting
- electron Zeeman interaction
- nuclear Zeeman interaction
- electron-`13C` hyperfine interaction
- external MW and RF controls

The default Hamiltonian is

```text
H(t) = D * Sz^2
     + gamma_e * (B · S)
     + sum_k [ -gamma_c * (B · I_k) + A_par_k * Sz * Iz_k + A_perp_k * Sx * Ix_k ]
     + H_MW(t)
     + H_RF(t)
```

The default hyperfine interaction is exactly the secular lab model

```text
H_hf^(k) = A_par_k * Sz * Iz_k + A_perp_k * Sx * Ix_k
```

An advanced alternate path is available per carbon through an optional full `3x3` tensor override. When that tensor is supplied, the code uses the full bilinear interaction for that carbon only. The default secular path and the full-tensor path are kept separate in the API and docs.

## Explicit simplifications

- Direct `13C-13C` couplings are intentionally omitted everywhere.
- No host `14N` or `15N` nucleus is included in the MVP.
- No strain, electric-field, excited-state optical physics, or bath/CCE model is included.
- Optical initialization and reset are modeled phenomenologically rather than microscopically.
- Perfect electron initialization into `m_s = 0` and perfect reset between shots are the defaults.
- The logical electron qubit for controls and circuits is the `|m_s=0>` / `|m_s=-1>` subspace, while the static Hamiltonian still lives in the full spin-1 manifold.

## Conventions

- Public frequencies are in Hz; times are in seconds.
- Internally, Hamiltonians are assembled in angular-frequency units.
- Tensor-factor ordering is always `electron ⊗ carbon_0 ⊗ carbon_1 ⊗ ...`.
- Electron basis ordering is QuTiP's spin-1 ordering: `|m_s=+1>`, `|m_s=0>`, `|m_s=-1>`.
- Carbon basis ordering is `|m_I=+1/2>`, `|m_I=-1/2>`.

## What is implemented

- device specs: `NVElectronSpec`, `C13Spec`, `Register`
- pulse primitives: analytic and sampled waveforms, MW/RF pulses, delays, virtual frame changes, measurements
- one canonical `PulseSchedule`
- sequence builders: ODMR, electron Rabi, nuclear Rabi, Ramsey, Hahn echo, CPMG, DDRF, PulsePol, SWAP-style interaction macro
- circuit layer: `rx`, `ry`, `rz`, `delay`, `swap`, native electron-nuclear interaction hook
- exact unitary backend
- simple Lindblad backend with phenomenological `T1` / `T2`
- observable and synthetic-count readout helpers
- serial / multiprocessing / loky outer-loop scans through `ParallelConfig`

## What is left as extension points

- host nitrogen spin
- richer calibrated two-qubit gates beyond the default interaction-window hook
- explicit laser/reset physics
- more detailed noise models
- Monte-Carlo trajectories
- higher-performance compiled or reduced-model backends

## Installation

```bash
python -m pip install -e .[dev]
```

Optional parallel extra:

```bash
python -m pip install -e .[dev,parallel]
```

## Quick start

```python
from nvdyn import C13Spec, NVElectronSpec, Register, Simulator
from nvdyn.sequences import CPMG

register = Register(
    electron=NVElectronSpec(D=2.87e9, B=(0.0, 0.0, 0.04)),
    carbons=[
        C13Spec(name="c1", A_par=414e3, A_perp=95e3),
        C13Spec(name="c2", A_par=120e3, A_perp=40e3),
    ],
)

sequence = CPMG(
    target="electron",
    n=8,
    tau=8e-6,
    frequency=register.default_electron_transition_hz(),
)

simulator = Simulator(method="exact", dt=2e-9, perfect_init=True)
result = simulator.run(register, sequence, observables=["P_ms0"])
print(result.expectation("P_ms0")[-1])
```

## Examples

Runnable examples live in `src/nvdyn/examples`:

- `python -m nvdyn.examples.odmr_scan`
- `python -m nvdyn.examples.electron_rabi`
- `python -m nvdyn.examples.ramsey`
- `python -m nvdyn.examples.cpmg_with_c13`
- `python -m nvdyn.examples.pulsepol_demo`
- `python -m nvdyn.examples.circuit_swap`

The bundled examples use reduced frequencies compared to a physical room-temperature NV so they stay interactive in a laboratory-frame pulse simulation. The API itself accepts physical-scale inputs.

## Tests

```bash
pytest
```

The tests cover operator dimensions, default and tensor hyperfine assembly, the deliberate absence of `13C-13C` terms, sampled waveforms, CPMG schedule structure, perfect initialization, pulse-accurate circuit lowering, exact Rabi behavior, Ramsey/dephasing behavior, and serial/parallel parity.
