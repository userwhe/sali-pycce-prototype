from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path


@dataclass(slots=True)
class FieldConfig:
    name: str
    bz_tesla: float

    @classmethod
    def from_name(cls, field: str) -> "FieldConfig":
        if field == "high":
            return cls(name="high", bz_tesla=0.056)
        if field == "low":
            return cls(name="low", bz_tesla=0.0056)
        raise ValueError("field must be either 'high' or 'low'")


@dataclass(slots=True)
class PhysicsConfig:
    bz_tesla: float
    gamma_mhz_per_t: float = 10.705
    t2_us: float = 200.0
    measurements: int = 1000
    signal_points: int = 1000
    tau_32_min_us: float = 6.0
    tau_32_max_us: float = 50.0
    tau_256_min_us: float = 10.0
    tau_256_max_us: float = 40.0
    add_decoherence: bool = True
    add_shot_noise: bool = True


@dataclass(slots=True)
class DataConfig:
    train_samples: int
    val_samples: int
    test_samples: int
    min_nuclei: int = 1
    max_nuclei: int = 20
    az_min_khz: float = -100.0
    az_max_khz: float = 100.0
    aperp_min_khz: float = 2.0
    aperp_max_khz: float = 102.0
    norm_epsilon: float = 0.001
    seed: int = 12345

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples + self.test_samples


@dataclass(slots=True)
class ModelConfig:
    signal_length: int = 1000
    conv1_filters: int = 16
    conv2_filters: int = 32
    dense_height: int = 102
    dense_width: int = 52
    output_height: int = 204
    output_width: int = 104
    dropout: float = 0.2


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 250
    samples_per_epoch: int | None = None
    learning_rate: float = 0.001
    lr_reduction_factor: float = 0.7
    lr_plateau_patience: int = 5
    early_stopping_patience: int = 20
    min_delta: float = 0.0
    device: str = "auto"
    loss_type: str = "mse"
    positive_weight: float = 25.0
    border_penalty_weight: float = 0.0
    border_width: int = 2


@dataclass(slots=True)
class PostprocessConfig:
    threshold: float = 0.25
    min_area: int = 3
    local_max_min_distance_px: int = 3
    erosion_size: int = 1
    dilation_size: int = 1


@dataclass(slots=True)
class RunConfig:
    physics: PhysicsConfig
    data: DataConfig
    model: ModelConfig = dataclass_field(default_factory=ModelConfig)
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
    postprocess: PostprocessConfig = dataclass_field(default_factory=PostprocessConfig)
    output_dir: Path = Path("runs/practical")


def practical_config(field: str = "low") -> RunConfig:
    field_cfg = FieldConfig.from_name(field)
    return RunConfig(
        physics=PhysicsConfig(bz_tesla=field_cfg.bz_tesla),
        data=DataConfig(train_samples=2048, val_samples=512, test_samples=512),
        training=TrainingConfig(max_epochs=10),
        output_dir=Path(f"runs/practical-{field}"),
    )


def paper_config(field: str = "low") -> RunConfig:
    field_cfg = FieldConfig.from_name(field)
    return RunConfig(
        physics=PhysicsConfig(bz_tesla=field_cfg.bz_tesla),
        data=DataConfig(
            train_samples=2_520_000,
            val_samples=540_000,
            test_samples=540_000,
        ),
        training=TrainingConfig(),
        output_dir=Path(f"runs/paper-{field}"),
    )
