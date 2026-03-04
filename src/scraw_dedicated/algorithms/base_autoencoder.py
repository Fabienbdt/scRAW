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
        """Forward pass of gradient reversal.

        The tensor is returned unchanged, while `lambda_` is cached in the
        autograd context to scale and invert gradients during backward.
        """
        ctx.lambda_ = float(lambda_.item())
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass of gradient reversal.

        The gradient is multiplied by `-lambda_` so the upstream branch learns
        representations that confuse the adversarial head.
        """
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to `x` with scale `lambda_`."""
    # On passe explicitement lambda sur le bon device pour éviter les mismatch CPU/GPU.
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
        """Build a symmetric MLP autoencoder from the requested architecture.

        Args:
            shape: Network dimensions (input, hidden stack, latent size).
            dropout: Dropout applied after each hidden block.
        """
        super().__init__()
        # Construction de l'encodeur: réduction progressive vers l'espace latent.
        enc: List[nn.Module] = []
        prev = shape.input_dim
        for h in shape.hidden_layers:
            enc.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(), nn.Dropout(dropout)])
            prev = h
        enc.append(nn.Linear(prev, shape.z_dim))

        # Construction du décodeur: symétrique de l'encodeur pour reconstruire l'entrée.
        dec: List[nn.Module] = []
        prev = shape.z_dim
        for h in reversed(shape.hidden_layers):
            dec.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(), nn.Dropout(dropout)])
            prev = h
        dec.append(nn.Linear(prev, shape.input_dim))

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode an input batch.

        Returns:
            Tuple `(z, recon)` where `z` is the latent embedding and `recon`
            is the reconstruction in feature space.
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def parse_hidden_layers(raw: Any) -> List[int]:
    """Parse hidden-layer configuration from list/tuple or comma string.

    Invalid/empty values are ignored and the default `[512, 256, 128]` is
    returned when no valid positive layer size is provided.
    """
    # Accepte soit une liste Python, soit une chaîne "512,256,128".
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
        """Return static metadata for the base autoencoder algorithm."""
        return AlgorithmInfo(
            name="base_autoencoder",
            display_name="Base AE",
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
        """Return shared network/training hyperparameters for descendants."""
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
        """Initialize the base algorithm container and model placeholder."""
        super().__init__(params=params)
        self.model: Optional[MLPAutoencoder] = None

    def _as_numpy_matrix(self, data: Any) -> np.ndarray:
        """Convert supported input containers to a dense `float32` 2D matrix."""
        # Supporte AnnData (data.X) ou matrice numpy directe.
        X = data.X if hasattr(data, "X") else data
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input matrix must be 2D with non-zero shape.")
        return X

    def _build_model(self, input_dim: int) -> MLPAutoencoder:
        """Instantiate and store the MLP autoencoder from current parameters."""
        # Lit les hyperparamètres réseau puis instancie l'autoencodeur MLP.
        shape = NetworkShape(
            input_dim=int(input_dim),
            hidden_layers=parse_hidden_layers(self.params.get("hidden_layers", "512,256,128")),
            z_dim=int(self.params.get("z_dim", 128)),
        )
        model = MLPAutoencoder(shape=shape, dropout=float(self.params.get("dropout", 0.1)))
        self.model = model
        return model

    def _encode_numpy(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Encode a full matrix into latent space using mini-batches."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        device = torch.device(self.get_device())
        self.model.to(device)
        was_training = bool(self.model.training)
        self.model.eval()

        # Encodage par mini-batch pour limiter la mémoire sur grands jeux de données.
        parts: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=device)
                zb = self.model.encoder(xb)
                parts.append(zb.detach().cpu().numpy())
        out = (
            np.concatenate(parts, axis=0)
            if parts
            else np.zeros((0, int(self.params.get("z_dim", 128))), dtype=np.float32)
        )

        # Preserve caller context: _encode_numpy is used during training for
        # pseudo-label/weight refresh and must not permanently switch to eval mode.
        if was_training:
            self.model.train()
        return out

    def fit(self, data: Any, labels: Optional[Any] = None) -> "BaseAutoencoderAlgorithm":
        """Abstract training entrypoint implemented by concrete algorithms."""
        raise NotImplementedError("Use ScRAWAlgorithm for training.")

    def predict(self, data: Any = None) -> Any:
        """Return predicted labels once the algorithm has been fitted."""
        if not self._fitted:
            raise RuntimeError("Algorithm not fitted.")
        return self._labels

    def encode(self, data: Any) -> np.ndarray:
        """Project input data into the latent space of the trained encoder."""
        X = self._as_numpy_matrix(data)
        return self._encode_numpy(X)

    def get_num_parameters(self) -> Optional[int]:
        """Return the number of trainable parameters, or `None` if unbuilt."""
        if self.model is None:
            return None
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
