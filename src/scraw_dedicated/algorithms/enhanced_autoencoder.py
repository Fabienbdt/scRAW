"""Compatibility wrapper around the new base_autoencoder module."""

from .base_autoencoder import (
    BaseAutoencoderAlgorithm,
    EnhancedAutoencoderAlgorithm,
    MLPAutoencoder,
    NetworkShape,
    gradient_reversal,
    parse_hidden_layers,
)

__all__ = [
    "BaseAutoencoderAlgorithm",
    "EnhancedAutoencoderAlgorithm",
    "MLPAutoencoder",
    "NetworkShape",
    "gradient_reversal",
    "parse_hidden_layers",
]
