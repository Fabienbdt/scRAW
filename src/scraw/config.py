"""JSON-backed configuration objects for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    data_path: str = "data/baron_human_pancreas.h5ad"
    output_dir: str = "results/default_run"
    label_key: Optional[str] = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"


@dataclass
class PreprocessingConfig:
    min_genes_per_cell: int = 200
    max_genes_per_cell: Optional[int] = None
    min_cells_per_gene: int = 3
    target_sum: float = 20000.0
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat"
    scale_max_value: float = 10.0


@dataclass
class ModelConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [512, 256, 128])
    latent_dim: int = 192
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    epochs: int = 210
    warmup_epochs: int = 74
    batch_size: int = 192
    learning_rate: float = 0.00233670337683859
    masking_rate: float = 0.15
    masking_value: float = 0.0
    masked_recon_weight: float = 0.8
    masking_in_weighted_phase: bool = True
    gradient_clip: float = 5.0


@dataclass
class WeightingConfig:
    weight_exponent: float = 0.7
    cluster_density_alpha: float = 0.3
    density_knn_k: int = 15
    density_weight_exponent: float = 1.0
    density_weight_clip: float = 8.0
    dynamic_weight_momentum: float = 0.85
    dynamic_weight_update_interval: int = 10
    min_cell_weight: float = 0.45
    max_cell_weight: float = 8.0


@dataclass
class TripletConfig:
    enabled: bool = True
    weight: float = 0.2346243650039478
    start_epoch: int = 84
    margin: float = 0.4
    min_anchor_weight: float = 1.8
    max_anchors_per_batch: int = 64


@dataclass
class ClusteringConfig:
    pseudo_label_method: str = "leiden"
    pseudo_k: int = 0
    pseudo_k_min: int = 8
    pseudo_k_max: int = 30
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: int = 8
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_reassign_noise: bool = True


@dataclass
class BatchCorrectionConfig:
    enabled: bool = True
    key: str = "batch"
    adversarial_weight: float = 0.056150696336115635
    adversarial_lambda: float = 1.75
    start_epoch: int = 55
    ramp_epochs: int = 60
    mmd_weight: float = 0.0


@dataclass
class OutputConfig:
    save_figures: bool = True
    save_model: bool = True


@dataclass
class ScRAWConfig:
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    triplet: TripletConfig = field(default_factory=TripletConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    batch_correction: BatchCorrectionConfig = field(default_factory=BatchCorrectionConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScRAWConfig":
        """Create a config object by merging a partial dict with defaults."""
        payload = dict(payload or {})
        return cls(
            data=DataConfig(**{**asdict(DataConfig()), **payload.get("data", {})}),
            runtime=RuntimeConfig(**{**asdict(RuntimeConfig()), **payload.get("runtime", {})}),
            preprocessing=PreprocessingConfig(
                **{**asdict(PreprocessingConfig()), **payload.get("preprocessing", {})}
            ),
            model=ModelConfig(**{**asdict(ModelConfig()), **payload.get("model", {})}),
            training=TrainingConfig(**{**asdict(TrainingConfig()), **payload.get("training", {})}),
            weighting=WeightingConfig(**{**asdict(WeightingConfig()), **payload.get("weighting", {})}),
            triplet=TripletConfig(**{**asdict(TripletConfig()), **payload.get("triplet", {})}),
            clustering=ClusteringConfig(
                **{**asdict(ClusteringConfig()), **payload.get("clustering", {})}
            ),
            batch_correction=BatchCorrectionConfig(
                **{**asdict(BatchCorrectionConfig()), **payload.get("batch_correction", {})}
            ),
            outputs=OutputConfig(**{**asdict(OutputConfig()), **payload.get("outputs", {})}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dictionary suitable for JSON serialization."""
        return asdict(self)


def load_config(path: str | Path) -> ScRAWConfig:
    """Load one JSON config file from disk."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ScRAWConfig.from_dict(payload)


def save_config(config: ScRAWConfig, path: str | Path) -> None:
    """Save a config object as pretty JSON."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
