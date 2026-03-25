from __future__ import annotations

from pathlib import Path

import torch

from scraw.config import ScRAWConfig
from scraw.model import MLPAutoencoder
from scraw.pipeline import _load_checkpoint_model


def test_load_checkpoint_model_restores_state_dict(tmp_path: Path) -> None:
    config = ScRAWConfig()
    config.runtime.device = "cpu"
    input_dim = 12
    checkpoint_path = tmp_path / "autoencoder.pt"

    source_model = MLPAutoencoder(input_dim=input_dim, config=config.model)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.125)
    torch.save(source_model.state_dict(), checkpoint_path)

    loaded_model = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        config=config,
        device=torch.device("cpu"),
    )

    assert loaded_model.training is False
    for expected_param, actual_param in zip(source_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(expected_param, actual_param)
