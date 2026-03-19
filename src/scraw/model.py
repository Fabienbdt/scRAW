"""Model definition and device helpers for the scRAW pipeline."""

from __future__ import annotations

from typing import Iterable, List
import logging

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


logger = logging.getLogger(__name__)


class _GradientReversalFunction(torch.autograd.Function):
    """Identity in forward pass, sign flip in backward pass."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        ctx.lambda_ = float(lambda_.item())
        return x.clone()

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to the latent representation."""
    lam = torch.tensor(float(lambda_), dtype=torch.float32, device=x.device)
    return _GradientReversalFunction.apply(x, lam)


def parse_hidden_layers(raw: Iterable[int] | str | None) -> List[int]:
    """Parse hidden-layer sizes from a list/tuple or a comma-separated string."""
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw if int(v) > 0]
        return values or [512, 256, 128]
    if raw is None:
        return [512, 256, 128]

    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return out or [512, 256, 128]


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto|cpu|cuda|mps` into a concrete torch device."""
    requested = str(requested or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


class MLPAutoencoder(nn.Module):
    """Simple symmetric MLP autoencoder used by scRAW."""

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        hidden_layers = parse_hidden_layers(config.hidden_layers)

        encoder_layers: List[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            encoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, int(config.latent_dim)))

        decoder_layers: List[nn.Module] = []
        previous_dim = int(config.latent_dim)
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(float(config.dropout)),
                ]
            )
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, int(input_dim)))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode one batch."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def encode_in_batches(
    model: MLPAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode a full matrix, preferring a single pass for exact reproducibility."""
    model.to(device)
    was_training = bool(model.training)
    model.eval()

    try:
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device)
            emb = model.encoder(xb).detach().cpu().numpy().astype(np.float32, copy=False)
        return emb
    except RuntimeError as exc:
        message = str(exc).lower()
        oom_markers = ("out of memory", "cuda error: out of memory")
        if not any(marker in message for marker in oom_markers):
            raise

        if device.type == "cuda":
            torch.cuda.empty_cache()

        parts: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = torch.tensor(
                    np.asarray(X[start : start + batch_size], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                zb = model.encoder(xb)
                parts.append(zb.detach().cpu().numpy())

        if not parts:
            return np.zeros((0, model.encoder[-1].out_features), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    finally:
        if was_training:
            model.train()
