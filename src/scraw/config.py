"""JSON-backed configuration objects for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TypeVar


@dataclass
class DataConfig:
    data_path: str = "data/baron_human_pancreas.h5ad"
    output_dir: str = "results/default_run"
    label_key: Optional[str] = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    strict_repro: bool = True


@dataclass
class PreprocessingConfig:
    input_mode: str = "auto"
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
    latent_dim: int = 256
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    epochs: int = 120
    warmup_epochs: int = 55
    batch_size: int = 192
    learning_rate: float = 0.00164076083297036
    masking_rate: float = 0.1
    masking_value: float = 0.0
    masked_recon_weight: float = 0.8
    masking_in_weighted_phase: bool = True
    gradient_clip: float = 5.0


@dataclass
class WeightingConfig:
    weight_exponent: float = 0.2
    cluster_density_alpha: float = 0.3483603718613933
    weight_fusion_mode: str = "multiplicative"
    density_knn_k: int = 15
    density_weight_exponent: float = 1.0
    density_weight_clip: float = 3.0
    dynamic_weight_momentum: float = 0.6884621079434989
    dynamic_weight_update_interval: int = 20
    min_cell_weight: float = 0.3845423008053828
    max_cell_weight: float = 10.0


@dataclass
class TripletConfig:
    enabled: bool = True
    weight: float = 0.05007581780188212
    start_epoch: int = 60
    margin: float = 0.4
    min_anchor_weight: float = 1.2
    max_anchors_per_batch: int = 64


@dataclass
class ClusteringConfig:
    pseudo_label_method: str = "leiden"
    pseudo_k: int = 0
    pseudo_k_min: int = 8
    pseudo_k_max: int = 30
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: int = 6
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_reassign_noise: bool = False


@dataclass
class BatchCorrectionConfig:
    enabled: bool = True
    key: Optional[str] = "batch"
    adversarial_weight: float = 0.11763398875166495
    adversarial_lambda: float = 1.0
    start_epoch: int = 30
    ramp_epochs: int = 30
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
        """Create a validated config by merging a partial mapping with defaults."""
        if not isinstance(payload, Mapping):
            raise TypeError("The configuration root must be a JSON object.")

        payload = dict(payload)
        section_types = {
            "data": DataConfig,
            "runtime": RuntimeConfig,
            "preprocessing": PreprocessingConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "weighting": WeightingConfig,
            "triplet": TripletConfig,
            "clustering": ClusteringConfig,
            "batch_correction": BatchCorrectionConfig,
            "outputs": OutputConfig,
        }
        unknown_sections = sorted(set(payload) - set(section_types))
        if unknown_sections:
            raise ValueError(
                "Unknown configuration section(s): " + ", ".join(unknown_sections)
            )

        section_values = {
            name: _merge_section(section_type, payload.get(name, {}), name)
            for name, section_type in section_types.items()
        }
        config = cls(**section_values)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid or unsupported settings before a run starts."""
        if not str(self.data.data_path).strip():
            raise ValueError("data.data_path must not be empty.")
        if not str(self.data.output_dir).strip():
            raise ValueError("data.output_dir must not be empty.")
        if self.data.label_key is not None and not str(self.data.label_key).strip():
            raise ValueError("data.label_key must be null or a non-empty column name.")

        if not str(self.runtime.device).strip():
            raise ValueError("runtime.device must not be empty.")

        input_mode = str(self.preprocessing.input_mode).strip().lower()
        if input_mode not in {"auto", "raw", "preprocessed"}:
            raise ValueError(
                "preprocessing.input_mode must be one of: auto, raw, preprocessed."
            )
        _require_nonnegative(
            "preprocessing.min_genes_per_cell",
            self.preprocessing.min_genes_per_cell,
        )
        if self.preprocessing.max_genes_per_cell is not None:
            _require_nonnegative(
                "preprocessing.max_genes_per_cell",
                self.preprocessing.max_genes_per_cell,
            )
        _require_nonnegative(
            "preprocessing.min_cells_per_gene",
            self.preprocessing.min_cells_per_gene,
        )
        _require_positive("preprocessing.target_sum", self.preprocessing.target_sum)
        _require_nonnegative(
            "preprocessing.n_top_genes",
            self.preprocessing.n_top_genes,
        )
        _require_positive(
            "preprocessing.scale_max_value",
            self.preprocessing.scale_max_value,
        )

        hidden_layers = self.model.hidden_layers
        if isinstance(hidden_layers, str):
            tokens = [
                token.strip()
                for token in hidden_layers.split(",")
                if token.strip()
            ]
            if not tokens or any(not token.isdigit() or int(token) <= 0 for token in tokens):
                raise ValueError("model.hidden_layers must contain only positive integers.")
        elif not isinstance(hidden_layers, (list, tuple)) or not hidden_layers:
            raise ValueError(
                "model.hidden_layers must be a non-empty list of positive integers."
            )
        elif any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in hidden_layers
        ):
            raise ValueError("model.hidden_layers must contain only positive integers.")
        _require_positive("model.latent_dim", self.model.latent_dim)
        _require_range(
            "model.dropout",
            self.model.dropout,
            0.0,
            1.0,
            upper_inclusive=False,
        )

        _require_positive("training.epochs", self.training.epochs)
        _require_nonnegative("training.warmup_epochs", self.training.warmup_epochs)
        if int(self.training.warmup_epochs) > int(self.training.epochs):
            raise ValueError("training.warmup_epochs must not exceed training.epochs.")
        if int(self.training.batch_size) < 2:
            raise ValueError(
                "training.batch_size must be at least 2 because the model uses BatchNorm."
            )
        _require_positive("training.learning_rate", self.training.learning_rate)
        _require_range(
            "training.masking_rate",
            self.training.masking_rate,
            0.0,
            1.0,
            upper_inclusive=False,
        )
        _require_range(
            "training.masked_recon_weight",
            self.training.masked_recon_weight,
            0.0,
            1.0,
        )
        _require_positive("training.gradient_clip", self.training.gradient_clip)

        _require_nonnegative("weighting.weight_exponent", self.weighting.weight_exponent)
        _require_range(
            "weighting.cluster_density_alpha",
            self.weighting.cluster_density_alpha,
            0.0,
            1.0,
        )
        fusion_mode = str(self.weighting.weight_fusion_mode).strip().lower()
        if fusion_mode not in {"additive", "multiplicative"}:
            raise ValueError(
                "weighting.weight_fusion_mode must be 'additive' or 'multiplicative'."
            )
        _require_positive("weighting.density_knn_k", self.weighting.density_knn_k)
        _require_nonnegative(
            "weighting.density_weight_exponent",
            self.weighting.density_weight_exponent,
        )
        _require_positive(
            "weighting.density_weight_clip",
            self.weighting.density_weight_clip,
        )
        _require_range(
            "weighting.dynamic_weight_momentum",
            self.weighting.dynamic_weight_momentum,
            0.0,
            1.0,
        )
        _require_nonnegative(
            "weighting.dynamic_weight_update_interval",
            self.weighting.dynamic_weight_update_interval,
        )
        _require_positive("weighting.min_cell_weight", self.weighting.min_cell_weight)
        _require_positive("weighting.max_cell_weight", self.weighting.max_cell_weight)
        if float(self.weighting.max_cell_weight) < float(
            self.weighting.min_cell_weight
        ):
            raise ValueError("weighting.max_cell_weight must be >= weighting.min_cell_weight.")

        _require_nonnegative("triplet.weight", self.triplet.weight)
        _require_nonnegative("triplet.start_epoch", self.triplet.start_epoch)
        _require_nonnegative("triplet.margin", self.triplet.margin)
        _require_nonnegative("triplet.min_anchor_weight", self.triplet.min_anchor_weight)
        _require_nonnegative(
            "triplet.max_anchors_per_batch",
            self.triplet.max_anchors_per_batch,
        )

        pseudo_method = str(self.clustering.pseudo_label_method).strip().lower()
        if pseudo_method not in {"leiden", "kmeans"}:
            raise ValueError("clustering.pseudo_label_method must be 'leiden' or 'kmeans'.")
        pseudo_k = int(self.clustering.pseudo_k)
        if pseudo_k == 1 or pseudo_k < 0:
            raise ValueError("clustering.pseudo_k must be 0 (automatic) or at least 2.")
        if int(self.clustering.pseudo_k_min) < 2:
            raise ValueError("clustering.pseudo_k_min must be at least 2.")
        if int(self.clustering.pseudo_k_max) < int(self.clustering.pseudo_k_min):
            raise ValueError("clustering.pseudo_k_max must be >= clustering.pseudo_k_min.")
        if int(self.clustering.hdbscan_min_cluster_size) < 2:
            raise ValueError("clustering.hdbscan_min_cluster_size must be at least 2.")
        _require_positive(
            "clustering.hdbscan_min_samples",
            self.clustering.hdbscan_min_samples,
        )
        selection_method = str(
            self.clustering.hdbscan_cluster_selection_method
        ).strip().lower()
        if selection_method not in {"eom", "leaf"}:
            raise ValueError(
                "clustering.hdbscan_cluster_selection_method must be 'eom' or 'leaf'."
            )

        if self.batch_correction.key is not None and not str(
            self.batch_correction.key
        ).strip():
            raise ValueError("batch_correction.key must be null or a non-empty column name.")
        _require_nonnegative(
            "batch_correction.adversarial_weight",
            self.batch_correction.adversarial_weight,
        )
        _require_nonnegative(
            "batch_correction.adversarial_lambda",
            self.batch_correction.adversarial_lambda,
        )
        _require_nonnegative(
            "batch_correction.start_epoch",
            self.batch_correction.start_epoch,
        )
        _require_nonnegative(
            "batch_correction.ramp_epochs",
            self.batch_correction.ramp_epochs,
        )
        if float(self.batch_correction.mmd_weight) != 0.0:
            raise NotImplementedError(
                "batch_correction.mmd_weight is not implemented; set it to 0.0."
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
    config.validate()
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )


SectionT = TypeVar("SectionT")


def _merge_section(
    section_type: type[SectionT],
    override: Any,
    section_name: str,
) -> SectionT:
    """Merge one validated section mapping into its dataclass defaults."""
    if not isinstance(override, Mapping):
        raise TypeError(f"Configuration section '{section_name}' must be a JSON object.")
    allowed_fields = {item.name for item in fields(section_type)}
    unknown_fields = sorted(set(override) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"Unknown field(s) in '{section_name}': " + ", ".join(unknown_fields)
        )
    return section_type(**{**asdict(section_type()), **dict(override)})


def _require_nonnegative(name: str, value: Any) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be >= 0.")


def _require_positive(name: str, value: Any) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be > 0.")


def _require_range(
    name: str,
    value: Any,
    lower: float,
    upper: float,
    *,
    upper_inclusive: bool = True,
) -> None:
    numeric = float(value)
    valid_upper = numeric <= upper if upper_inclusive else numeric < upper
    if not math.isfinite(numeric) or numeric < lower or not valid_upper:
        upper_operator = "<=" if upper_inclusive else "<"
        raise ValueError(f"{name} must satisfy {lower} <= value {upper_operator} {upper}.")
