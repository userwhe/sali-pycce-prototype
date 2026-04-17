"""Exact unitary simulation backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qutip.solver.mesolve import MESolver
from qutip.solver.sesolve import SESolver

from ..device import Register
from ..hamiltonian import CompiledSchedule, compile_schedule
from ..observables import resolve_observables
from ..results import Result, result_from_qutip
from ..schedule import PulseSchedule


@dataclass(slots=True)
class PreparedExactSimulation:
    register: Register
    schedule: PulseSchedule
    dt: float
    perfect_init: bool
    nuclear_state: str
    solver_options: dict[str, Any] = field(default_factory=dict)
    compiled: CompiledSchedule = field(init=False)
    _sesolver: SESolver = field(init=False, repr=False)
    _mesolver: MESolver = field(init=False, repr=False)
    _density_solver: MESolver = field(init=False, repr=False)

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
        self._sesolver = SESolver(self.compiled.hamiltonian, options=options)
        self._mesolver = MESolver(self.compiled.hamiltonian, c_ops=[], options=options)
        self._density_solver = self._mesolver

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
        e_ops = resolve_observables(self.register, observables)
        if state.isket:
            qresult = self._sesolver.run(state, self.compiled.tlist, e_ops=e_ops)
        else:
            qresult = self._mesolver.run(state, self.compiled.tlist, e_ops=e_ops)
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
