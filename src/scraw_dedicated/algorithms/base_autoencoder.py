"""Base autoencoder utilities shared by scRAW modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..core.algorithm_registry import AlgorithmInfo, BaseAlgorithm
from ..core.config import HyperparameterConfig, ParamType


class _GradientReversalFunction(torch.autograd.Function):
    """Identity in forward pass, sign flip in backward pass."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        """Réalise l'opération `forward` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            ctx: Paramètre d'entrée `ctx` utilisé dans cette étape du pipeline.
            x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
            lambda_: Paramètre d'entrée `lambda_` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        ctx.lambda_ = float(lambda_.item())
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Réalise l'opération `backward` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            ctx: Paramètre d'entrée `ctx` utilisé dans cette étape du pipeline.
            grad_output: Paramètre d'entrée `grad_output` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Réalise l'opération `gradient reversal` du module `base_autoencoder`.
    
    Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
    
    Args:
        x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
        lambda_: Paramètre d'entrée `lambda_` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    lam = torch.tensor(float(lambda_), dtype=torch.float32, device=x.device)
    return _GradientReversalFunction.apply(x, lam)


@dataclass
class NetworkShape:
    input_dim: int
    hidden_layers: List[int]
    z_dim: int


class MLPAutoencoder(nn.Module):
    """Simple MLP autoencoder used by scRAW."""

    def __init__(self, shape: NetworkShape, dropout: float = 0.1) -> None:
        """Helper interne: init.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            shape: Paramètre d'entrée `shape` utilisé dans cette étape du pipeline.
            dropout: Paramètre d'entrée `dropout` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        super().__init__()
        enc: List[nn.Module] = []
        prev = shape.input_dim
        for h in shape.hidden_layers:
            enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        enc.append(nn.Linear(prev, shape.z_dim))

        dec: List[nn.Module] = []
        prev = shape.z_dim
        for h in reversed(shape.hidden_layers):
            dec.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        dec.append(nn.Linear(prev, shape.input_dim))

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Réalise l'opération `forward` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def parse_hidden_layers(raw: Any) -> List[int]:
    """Réalise l'opération `parse hidden layers` du module `base_autoencoder`.
    
    Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
    
    Args:
        raw: Paramètre d'entrée `raw` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    if isinstance(raw, (list, tuple)):
        vals = [int(x) for x in raw if int(x) > 0]
        return vals or [512, 256, 128]
    if raw is None:
        return [512, 256, 128]
    text = str(raw).strip()
    if not text:
        return [512, 256, 128]

    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
            if v > 0:
                out.append(v)
        except Exception:
            continue
    return out or [512, 256, 128]


class BaseAutoencoderAlgorithm(BaseAlgorithm):
    """Minimal shared base for autoencoder-style algorithms."""

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        """Réalise l'opération `get info` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return AlgorithmInfo(
            name="enhanced_autoencoder",
            display_name="Enhanced AE (base)",
            description="Internal base class used by scRAW.",
            category="deep_learning",
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes="Expects preprocessed dense matrix in adata.X",
            has_internal_preprocessing=False,
            recommended_data="preprocessed",
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        """Réalise l'opération `get hyperparameters` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return [
            HyperparameterConfig(
                name="hidden_layers",
                display_name="Hidden Layers",
                param_type=ParamType.STRING,
                default="512,256,128",
                description="Comma-separated hidden layer sizes.",
                category="Network",
            ),
            HyperparameterConfig(
                name="z_dim",
                display_name="Latent Dimension",
                param_type=ParamType.INTEGER,
                default=128,
                min_value=2,
                max_value=512,
                description="Latent embedding size.",
                category="Network",
            ),
            HyperparameterConfig(
                name="dropout",
                display_name="Dropout",
                param_type=ParamType.FLOAT,
                default=0.1,
                min_value=0.0,
                max_value=0.8,
                description="Dropout used in encoder/decoder.",
                category="Network",
            ),
            HyperparameterConfig(
                name="epochs",
                display_name="Epochs",
                param_type=ParamType.INTEGER,
                default=120,
                min_value=1,
                max_value=2000,
                description="Total training epochs.",
                category="Training",
            ),
            HyperparameterConfig(
                name="warmup_epochs",
                display_name="Warm-up Epochs",
                param_type=ParamType.INTEGER,
                default=30,
                min_value=0,
                max_value=1000,
                description="Number of warm-up epochs without weighted loss.",
                category="Training",
            ),
            HyperparameterConfig(
                name="batch_size",
                display_name="Batch Size",
                param_type=ParamType.INTEGER,
                default=256,
                min_value=8,
                max_value=8192,
                description="Mini-batch size.",
                category="Training",
            ),
            HyperparameterConfig(
                name="lr",
                display_name="Learning Rate",
                param_type=ParamType.FLOAT,
                default=1e-3,
                min_value=1e-6,
                max_value=1.0,
                description="Optimizer learning rate.",
                category="Training",
            ),
        ]

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Helper interne: init.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            params: Paramètre d'entrée `params` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        super().__init__(params=params)
        self.model: Optional[MLPAutoencoder] = None

    def _as_numpy_matrix(self, data: Any) -> np.ndarray:
        """Helper interne: as numpy matrix.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        X = data.X if hasattr(data, "X") else data
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input matrix must be 2D with non-zero shape.")
        return X

    def _build_model(self, input_dim: int) -> MLPAutoencoder:
        """Helper interne: build model.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            input_dim: Paramètre d'entrée `input_dim` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        shape = NetworkShape(
            input_dim=int(input_dim),
            hidden_layers=parse_hidden_layers(self.params.get("hidden_layers", "512,256,128")),
            z_dim=int(self.params.get("z_dim", 128)),
        )
        model = MLPAutoencoder(shape=shape, dropout=float(self.params.get("dropout", 0.1)))
        self.model = model
        return model

    def _encode_numpy(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Helper interne: encode numpy.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            X: Paramètre d'entrée `X` utilisé dans cette étape du pipeline.
            batch_size: Paramètre d'entrée `batch_size` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        device = torch.device(self.get_device())
        self.model.to(device)
        self.model.eval()

        parts: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
                zb = self.model.encoder(xb)
                parts.append(zb.detach().cpu().numpy())
        if parts:
            return np.concatenate(parts, axis=0)
        return np.zeros((0, int(self.params.get("z_dim", 128))), dtype=np.float32)

    def fit(self, data: Any, labels: Optional[Any] = None) -> "BaseAutoencoderAlgorithm":
        """Entraîne le modèle sur les données fournies.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
            labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        raise NotImplementedError("Use ScRAWAlgorithm for training.")

    def predict(self, data: Any = None) -> Any:
        """Retourne les clusters prédits après entraînement.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        if not self._fitted:
            raise RuntimeError("Algorithm not fitted.")
        return self._labels

    def encode(self, data: Any) -> np.ndarray:
        """Projette les données dans l'espace latent appris par l'encodeur.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        X = self._as_numpy_matrix(data)
        return self._encode_numpy(X)

    def get_num_parameters(self) -> Optional[int]:
        """Réalise l'opération `get num parameters` du module `base_autoencoder`.
        
        Cette docstring est rédigée pour faciliter la lecture du code, même pour un débutant.
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        if self.model is None:
            return None
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))


class EnhancedAutoencoderAlgorithm(BaseAutoencoderAlgorithm):
    """Backward-compatible alias kept for existing imports."""

    pass
