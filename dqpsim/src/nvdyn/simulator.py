"""High-level simulator API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from qutip import Qobj

from .circuit import Circuit
from .compiler import CompilerConfig, compile_circuit
from .device import Register
from .observables import resolve_observables
from .parallel import ParallelConfig, dispatch_jobs, normalize_parallel_config
from .results import BatchResult, Result, SweepResult
from .schedule import PulseSchedule
from .sequences import Sequence as ExperimentSequence
from .sim import LindbladConfig, PreparedExactSimulation, PreparedLindbladSimulation

ProgramLike = PulseSchedule | ExperimentSequence | Circuit


def _normalize_lindblad_config(config: LindbladConfig | dict[str, object] | None) -> LindbladConfig:
    if config is None:
        return LindbladConfig()
    if isinstance(config, LindbladConfig):
        return config
    electron_t1 = config.get("electron_t1", config.get("T1"))
    electron_t2 = config.get("electron_t2", config.get("T2"))
    return LindbladConfig(
        T1=None if electron_t1 is None else float(electron_t1),
        T2=None if electron_t2 is None else float(electron_t2),
    )


@dataclass(slots=True)
class PreparedSimulation:
    """Prepared simulator object that reuses compiled operators and solver state."""

    register: Register
    schedule: PulseSchedule
    backend: PreparedExactSimulation | PreparedLindbladSimulation
    observables: dict[str, Qobj]
    method: str

    def run(
        self,
        *,
        initial_state: Qobj | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Result:
        return self.backend.run(
            observables=self.observables,
            initial_state=initial_state,
            shots=shots,
            seed=seed,
            metadata={"method": self.method},
        )


@dataclass(slots=True, frozen=True)
class _SimulationJob:
    simulator: "Simulator"
    register: Register
    program: ProgramLike
    observables: tuple[str, ...] | None
    shots: int | None
    seed: int | None


def _run_simulation_job(job: _SimulationJob) -> Result:
    serial_simulator = replace(job.simulator, parallel=ParallelConfig(map="serial"))
    return serial_simulator.run(
        job.register,
        job.program,
        observables=list(job.observables) if job.observables is not None else None,
        shots=job.shots,
        seed=job.seed,
    )


@dataclass(slots=True)
class Simulator:
    """High-level simulator entry point."""

    method: str = "exact"
    dt: float = 2e-9
    perfect_init: bool = True
    perfect_reset: bool = True
    nuclear_state: str = "mixed"
    parallel: ParallelConfig | dict[str, object] | None = None
    lindblad: LindbladConfig | dict[str, object] | None = None
    solver_options: dict[str, object] = field(
        default_factory=lambda: {"store_states": True, "method": "dop853"}
    )
    compiler_config: CompilerConfig = field(default_factory=CompilerConfig)

    def __post_init__(self) -> None:
        self.parallel = normalize_parallel_config(self.parallel)
        self.lindblad = _normalize_lindblad_config(self.lindblad)
        if self.dt <= 0.0:
            raise ValueError("simulation dt must be positive")
        if self.method not in {"exact", "lindblad"}:
            raise ValueError("simulator method must be 'exact' or 'lindblad'")
        if self.nuclear_state not in {"mixed", "up"}:
            raise ValueError("nuclear_state must be 'mixed' or 'up'")

    def _to_schedule(self, register: Register, program: ProgramLike) -> PulseSchedule:
        if isinstance(program, PulseSchedule):
            return program.copy()
        if isinstance(program, ExperimentSequence):
            if hasattr(program, "to_schedule"):
                return program.to_schedule().copy()
            return program.compile()
        if isinstance(program, Circuit):
            return compile_circuit(register, program, self.compiler_config)
        raise TypeError(f"unsupported program type {type(program)!r}")

    def prepare(
        self,
        register: Register,
        program: ProgramLike,
        *,
        observables: Sequence[str] | None = None,
    ) -> PreparedSimulation:
        schedule = self._to_schedule(register, program)
        resolved_observables = resolve_observables(register, observables)
        if self.method == "exact":
            backend = PreparedExactSimulation(
                register=register,
                schedule=schedule,
                dt=self.dt,
                perfect_init=self.perfect_init,
                nuclear_state=self.nuclear_state,
                solver_options=dict(self.solver_options),
            )
        else:
            backend = PreparedLindbladSimulation(
                register=register,
                schedule=schedule,
                dt=self.dt,
                perfect_init=self.perfect_init,
                nuclear_state=self.nuclear_state,
                noise_model=self.lindblad,
                solver_options=dict(self.solver_options),
            )
        return PreparedSimulation(
            register=register,
            schedule=schedule,
            backend=backend,
            observables=resolved_observables,
            method=self.method,
        )

    def run(
        self,
        register: Register,
        program: ProgramLike,
        *,
        observables: Sequence[str] | None = None,
        initial_state: Qobj | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Result:
        prepared = self.prepare(register, program, observables=observables)
        return prepared.run(initial_state=initial_state, shots=shots, seed=seed)

    def run_batch(
        self,
        register: Register,
        programs: Sequence[ProgramLike],
        *,
        observables: Sequence[str] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> BatchResult:
        jobs = [
            _SimulationJob(
                simulator=self,
                register=register,
                program=program,
                observables=None if observables is None else tuple(observables),
                shots=shots,
                seed=None if seed is None else seed + index,
            )
            for index, program in enumerate(programs)
        ]
        results = dispatch_jobs(_run_simulation_job, jobs, self.parallel)
        return BatchResult(results=results, metadata={"parallel": self.parallel.map})

    def run_sweep(
        self,
        register: Register,
        values: Sequence[float],
        builder: Callable[[float], ProgramLike],
        *,
        parameter_name: str = "value",
        observables: Sequence[str] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> SweepResult:
        value_list = [float(value) for value in values]
        programs = [builder(value) for value in value_list]
        batch = self.run_batch(register, programs, observables=observables, shots=shots, seed=seed)
        return SweepResult(
            parameter_name=parameter_name,
            values=np.asarray(value_list, dtype=float),
            results=batch.results,
            metadata=batch.metadata,
        )

    def run_parameter_scan(
        self,
        register: Register,
        values: Sequence[float],
        builder: Callable[[float], ProgramLike],
        *,
        parameter_name: str = "value",
        observables: Sequence[str] | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> SweepResult:
        return self.run_sweep(
            register,
            values,
            builder,
            parameter_name=parameter_name,
            observables=observables,
            shots=shots,
            seed=seed,
        )
