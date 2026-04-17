"""Simple Lindblad simulation backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qutip import QobjEvo, basis
from qutip.solver.mesolve import MESolver

from ..device import Register
from ..hamiltonian import CompiledSchedule, compile_schedule
from ..observables import resolve_observables
from ..operators import electron_qubit_operators, embed_single_operator
from ..results import Result, result_from_qutip
from ..schedule import PulseSchedule


@dataclass(frozen=True, slots=True)
class LindbladConfig:
    T1: float | None = None
    T2: float | None = None


def build_collapse_operators(register: Register, model: LindbladConfig) -> list[QobjEvo]:
    collapse_ops: list[QobjEvo] = []
    if model.T1 is not None and model.T1 > 0.0:
        jump_local = basis(3, 1) * basis(3, 2).dag()
        jump = embed_single_operator(register, 0, jump_local)
        collapse_ops.append(QobjEvo((1.0 / model.T1) ** 0.5 * jump))
    if model.T2 is not None and model.T2 > 0.0:
        dephasing = electron_qubit_operators(register)["z"]
        collapse_ops.append(QobjEvo((0.5 / model.T2) ** 0.5 * dephasing))
    return collapse_ops


@dataclass(slots=True)
class PreparedLindbladSimulation:
    register: Register
    schedule: PulseSchedule
    dt: float
    perfect_init: bool
    nuclear_state: str
    noise_model: LindbladConfig = field(default_factory=LindbladConfig)
    solver_options: dict[str, Any] = field(default_factory=dict)
    compiled: CompiledSchedule = field(init=False)
    _solver: MESolver = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compiled = compile_schedule(self.register, self.schedule, dt=self.dt)
        options = {
            "progress_bar": "",
            "store_final_state": True,
            "store_states": True,
            "method": "dop853",
            "max_step": self.dt,
        }
        options.update(self.solver_options)
        self._solver = MESolver(
            self.compiled.hamiltonian,
            c_ops=build_collapse_operators(self.register, self.noise_model),
            options=options,
        )

    def run(
        self,
        *,
        observables: list[str] | tuple[str, ...] | dict[str, Any] | None = None,
        initial_state: Any = None,
        shots: int | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Result:
        state = initial_state if initial_state is not None else self.register.default_state(
            perfect_init=self.perfect_init,
            nuclear_state=self.nuclear_state,
        )
        if state.isket:
            state = state.proj()
        e_ops = resolve_observables(self.register, observables)
        qresult = self._solver.run(state, self.compiled.tlist, e_ops=e_ops)
        return result_from_qutip(
            qutip_result=qresult,
            times=self.compiled.tlist,
            schedule=self.schedule,
            observables=e_ops,
            measurements=self.compiled.measurements,
            shots=shots,
            seed=seed,
            metadata=metadata,
        )
