"""Outer-loop parallel dispatch helpers."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, replace
from typing import Callable, Iterable, TypeVar

from qutip.solver.parallel import loky_pmap, parallel_map, serial_map

T = TypeVar("T")
R = TypeVar("R")

_PARALLEL_SENTINEL = "NVDYN_IN_PARALLEL"


@dataclass(slots=True, frozen=True)
class ParallelConfig:
    """Configuration for outer-loop batch parallelism."""

    map: str = "serial"
    num_cpus: int | None = None
    timeout: float | None = None
    fail_fast: bool = True


def normalize_parallel_config(config: ParallelConfig | dict[str, object] | None) -> ParallelConfig:
    if config is None:
        return ParallelConfig()
    if isinstance(config, ParallelConfig):
        return config
    return ParallelConfig(
        map=str(config.get("map", "serial")),
        num_cpus=None if config.get("num_cpus") is None else int(config["num_cpus"]),
        timeout=None if config.get("timeout") is None else float(config["timeout"]),
        fail_fast=bool(config.get("fail_fast", True)),
    )


def effective_parallel_config(config: ParallelConfig | dict[str, object] | None) -> ParallelConfig:
    """Disable nested or unavailable parallel execution automatically."""

    resolved = normalize_parallel_config(config)
    if os.environ.get(_PARALLEL_SENTINEL) == "1" and resolved.map != "serial":
        return replace(resolved, map="serial")
    if resolved.map != "serial" and resolved.num_cpus == 1:
        return replace(resolved, map="serial")
    if resolved.map == "loky" and importlib.util.find_spec("loky") is None:
        return replace(resolved, map="serial")
    return resolved


def _indexed_job_runner(payload: tuple[Callable[[T], R], int, T]) -> tuple[int, R]:
    job_fn, index, job = payload
    os.environ[_PARALLEL_SENTINEL] = "1"
    return (index, job_fn(job))


def dispatch_jobs(job_fn: Callable[[T], R], jobs: Iterable[T], config: ParallelConfig | dict[str, object] | None) -> list[R]:
    """Run jobs in serial, multiprocessing, or loky mode while preserving order."""

    indexed_jobs = [(job_fn, index, job) for index, job in enumerate(jobs)]
    resolved = effective_parallel_config(config)
    if resolved.map == "serial" or len(indexed_jobs) <= 1:
        pairs = serial_map(_indexed_job_runner, indexed_jobs)
    elif resolved.map == "parallel":
        pairs = parallel_map(
            _indexed_job_runner,
            indexed_jobs,
            num_cpus=resolved.num_cpus,
            timeout=resolved.timeout,
            fail_fast=resolved.fail_fast,
        )
    elif resolved.map == "loky":
        pairs = loky_pmap(
            _indexed_job_runner,
            indexed_jobs,
            num_cpus=resolved.num_cpus,
            timeout=resolved.timeout,
            fail_fast=resolved.fail_fast,
        )
    else:
        raise ValueError(f"unsupported parallel map mode {resolved.map!r}")
    ordered = sorted(pairs, key=lambda pair: pair[0])
    return [result for _, result in ordered]
