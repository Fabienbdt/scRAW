"""
Configuration module for scDeepCluster GUI.
Centralized configuration management with support for hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
from pathlib import Path


class ParamType(Enum):
    """Types of hyperparameters."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    RANGE = "range"


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter."""
    name: str
    display_name: str
    param_type: ParamType
    default: Any
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    category: str = "General"
    advanced: bool = False
    tuning_guide: str = ""  # New field for helpful tuning advice

    def validate(self, value: Any) -> bool:
        """Validate a value against this parameter's constraints."""
        if self.param_type == ParamType.INTEGER:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        elif self.param_type == ParamType.CHOICE:
            if self.choices and value not in self.choices:
                return False
        elif self.param_type == ParamType.MULTI_CHOICE:
            if self.choices and not all(v in self.choices for v in value):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'param_type': self.param_type.value,
            'default': self.default,
            'description': self.description,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step': self.step,
            'choices': self.choices,
            'category': self.category,
            'advanced': self.advanced,
            'tuning_guide': self.tuning_guide
        }


@dataclass
class Config:
    """Main configuration class for the application."""

    # Application settings
    app_name: str = "scDeepCluster Analysis Suite"
    version: str = "1.0.0"

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    models_dir: Path = field(default_factory=lambda: Path("models"))

    # Default preprocessing parameters
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'n_top_genes': 2000,
        'min_genes_per_cell': 100,
        'max_genes_per_cell': 10000,
        'min_cells_per_gene': 3,
        'target_sum': 20000,
        'scale_max_value': 10.0,
        'hvg_flavor': 'seurat'
    })

    # Device settings
    device: str = "auto"

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.models_dir = Path(self.models_dir)

    def ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        path = Path(path)
        config_dict = {
            'app_name': self.app_name,
            'version': self.version,
            'data_dir': str(self.data_dir),
            'results_dir': str(self.results_dir),
            'models_dir': str(self.models_dir),
            'preprocessing': self.preprocessing,
            'device': self.device,
            'seed': self.seed
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    DEFAULT_TRAIN_RATIO: float = 0.7
    DEFAULT_VAL_RATIO: float = 0.1
    DEFAULT_TEST_RATIO: float = 0.2
    
    MIN_TRAIN_RATIO: float = 0.5
    MAX_TRAIN_RATIO: float = 0.9
    
    MIN_VAL_RATIO: float = 0.0
    MAX_VAL_RATIO: float = 0.3
    
    MIN_TEST_RATIO: float = 0.05
    MAX_TEST_RATIO: float = 0.5
    
    MIN_TEST_CELLS: int = 50  # Minimum cells for reliable evaluation


# Default preprocessing hyperparameters
PREPROCESSING_PARAMS = [
    HyperparameterConfig(
        name='n_top_genes',
        display_name='Number of HVGs',
        param_type=ParamType.INTEGER,
        default=2000,
        description='Number of highly variable genes to select',
        min_value=100,
        max_value=10000,
        step=100,
        category='Preprocessing'
    ),
    HyperparameterConfig(
        name='min_genes_per_cell',
        display_name='Min Genes per Cell',
        param_type=ParamType.INTEGER,
        default=100,
        description='Minimum number of genes expressed in a cell',
        min_value=0,
        max_value=1000,
        step=50,
        category='Preprocessing'
    ),
    HyperparameterConfig(
        name='max_genes_per_cell',
        display_name='Max Genes per Cell',
        param_type=ParamType.INTEGER,
        default=10000,
        description='Maximum number of genes expressed in a cell',
        min_value=1000,
        max_value=50000,
        step=1000,
        category='Preprocessing'
    ),
    HyperparameterConfig(
        name='min_cells_per_gene',
        display_name='Min Cells per Gene',
        param_type=ParamType.INTEGER,
        default=3,
        description='Minimum number of cells expressing a gene',
        min_value=1,
        max_value=100,
        step=1,
        category='Preprocessing'
    ),
    HyperparameterConfig(
        name='target_sum',
        display_name='Normalization Target',
        param_type=ParamType.FLOAT,
        default=20000.0,
        description='Target sum for normalization',
        min_value=1000,
        max_value=100000,
        step=1000,
        category='Preprocessing'
    ),
    HyperparameterConfig(
        name='scale_max_value',
        display_name='Scale Max Value',
        param_type=ParamType.FLOAT,
        default=10.0,
        description='Maximum value for scaling (clipping)',
        min_value=1.0,
        max_value=50.0,
        step=1.0,
        category='Preprocessing',
        advanced=True
    ),
    HyperparameterConfig(
        name='hvg_flavor',
        display_name='HVG Selection Method',
        param_type=ParamType.CHOICE,
        default='seurat',
        description='Method for selecting highly variable genes',
        choices=['seurat', 'seurat_v3', 'cell_ranger'],
        category='Preprocessing'
    )
]


# Default clustering hyperparameters
CLUSTERING_PARAMS = [
    HyperparameterConfig(
        name='n_clusters',
        display_name='Number of Clusters',
        param_type=ParamType.INTEGER,
        default=0,
        description='Number of clusters (0 = auto-detect)',
        min_value=0,
        max_value=100,
        step=1,
        category='Clustering'
    ),
    HyperparameterConfig(
        name='cluster_method',
        display_name='Auto-detection Method',
        param_type=ParamType.CHOICE,
        default='silhouette',
        description='Method for automatic cluster number detection',
        choices=['silhouette', 'louvain'],
        category='Clustering'
    )
]


# Training hyperparameters
TRAINING_PARAMS = [
    HyperparameterConfig(
        name='batch_size',
        display_name='Batch Size',
        param_type=ParamType.INTEGER,
        default=256,
        description='Training batch size',
        min_value=16,
        max_value=1024,
        step=16,
        category='Training'
    ),
    HyperparameterConfig(
        name='pretrain_epochs',
        display_name='Pretraining Epochs',
        param_type=ParamType.INTEGER,
        default=50,
        description='Number of autoencoder pretraining epochs',
        max_value=500,
        step=10,
        category='Training'
    ),
    HyperparameterConfig(
        name='maxiter',
        display_name='Max Iterations',
        param_type=ParamType.INTEGER,
        default=2000,
        description='Maximum clustering iterations',
        min_value=100,
        max_value=10000,
        step=100,
        category='Training'
    ),
    HyperparameterConfig(
        name='pretrain_lr',
        display_name='Pretraining Learning Rate',
        param_type=ParamType.FLOAT,
        default=0.001,
        description='Learning rate for pretraining',
        min_value=0.0001,
        max_value=0.1,
        step=0.0001,
        category='Training'
    ),
    HyperparameterConfig(
        name='lr',
        display_name='Clustering Learning Rate',
        param_type=ParamType.FLOAT,
        default=1.0,
        description='Learning rate for clustering phase',
        min_value=0.01,
        max_value=10.0,
        step=0.1,
        category='Training'
    ),
    HyperparameterConfig(
        name='tol',
        display_name='Convergence Tolerance',
        param_type=ParamType.FLOAT,
        default=0.001,
        description='Convergence tolerance (label change percentage)',
        min_value=0.0001,
        max_value=0.1,
        step=0.0001,
        category='Training',
        advanced=True
    )
]


# Network architecture hyperparameters
NETWORK_PARAMS = [
    HyperparameterConfig(
        name='z_dim',
        display_name='Latent Dimension',
        param_type=ParamType.INTEGER,
        default=32,
        description='Dimension of the latent space',
        min_value=8,
        max_value=256,
        step=8,
        category='Network'
    ),
    HyperparameterConfig(
        name='encoder_layers',
        display_name='Encoder Layers',
        param_type=ParamType.STRING,
        default='256,64',
        description='Encoder hidden layer sizes (comma-separated)',
        category='Network',
        advanced=True
    ),
    HyperparameterConfig(
        name='decoder_layers',
        display_name='Decoder Layers',
        param_type=ParamType.STRING,
        default='64,256',
        description='Decoder hidden layer sizes (comma-separated)',
        category='Network',
        advanced=True
    ),
    HyperparameterConfig(
        name='activation',
        display_name='Activation Function',
        param_type=ParamType.CHOICE,
        default='relu',
        description='Activation function for hidden layers',
        choices=['relu', 'sigmoid', 'elu', 'leaky_relu'],
        category='Network'
    ),
    HyperparameterConfig(
        name='sigma',
        display_name='Noise Sigma',
        param_type=ParamType.FLOAT,
        default=1.0,
        description='Standard deviation for denoising noise',
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        category='Network'
    ),
    HyperparameterConfig(
        name='gamma',
        display_name='Clustering Loss Weight',
        param_type=ParamType.FLOAT,
        default=1.0,
        description='Weight for clustering loss',
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        category='Network',
        advanced=True
    ),
    HyperparameterConfig(
        name='alpha',
        display_name='Student-t DOF',
        param_type=ParamType.FLOAT,
        default=1.0,
        description="Degrees of freedom for Student's t-distribution",
        min_value=0.1,
        max_value=10.0,
        step=0.1,
        category='Network',
        advanced=True
    )
]


# PCA-specific parameters
PCA_PARAMS = [
    HyperparameterConfig(
        name='n_pca_components',
        display_name='PCA Components',
        param_type=ParamType.INTEGER,
        default=8,
        description='Number of principal components',
        min_value=2,
        max_value=100,
        step=1,
        category='PCA'
    )
]


# Leiden-specific parameters
LEIDEN_PARAMS = [
    HyperparameterConfig(
        name='leiden_resolution',
        display_name='Leiden Resolution',
        param_type=ParamType.FLOAT,
        default=0.8,
        description='Resolution parameter for Leiden clustering',
        min_value=0.1,
        max_value=3.0,
        step=0.1,
        category='Leiden'
    ),
    HyperparameterConfig(
        name='leiden_neighbors',
        display_name='KNN Neighbors',
        param_type=ParamType.INTEGER,
        default=15,
        description='Number of neighbors for KNN graph',
        min_value=5,
        max_value=100,
        step=5,
        category='Leiden'
    )
]


def get_all_hyperparameters() -> List[HyperparameterConfig]:
    """Get all available hyperparameters."""
    return (PREPROCESSING_PARAMS + CLUSTERING_PARAMS + TRAINING_PARAMS +
            NETWORK_PARAMS + PCA_PARAMS + LEIDEN_PARAMS)


def get_hyperparameters_by_category() -> Dict[str, List[HyperparameterConfig]]:
    """Get hyperparameters organized by category."""
    all_params = get_all_hyperparameters()
    by_category = {}
    for param in all_params:
        if param.category not in by_category:
            by_category[param.category] = []
        by_category[param.category].append(param)
    return by_category
