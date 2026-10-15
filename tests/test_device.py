"""Étape 0 : get_device() et passage d'un tenseur sur le device détecté."""

import pytest
import torch

from common import device as device_module
from common.device import get_device


def test_tensor_runs_on_detected_device():
    dev = get_device()
    x = torch.arange(6, dtype=torch.float32, device=dev).reshape(2, 3)
    y = (x @ x.T).cpu()
    assert torch.equal(y, torch.tensor([[5.0, 14.0], [14.0, 50.0]]))
    assert x.device.type == dev.type


def test_cpu_can_be_forced():
    assert get_device("cpu").type == "cpu"


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv(device_module.ENV_VAR, "cpu")
    assert get_device("cuda").type == "cpu"


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        get_device("tpu")


# Les branches CUDA/MPS ne sont pas exécutables sur toutes les machines : on simule
# la disponibilité pour vérifier l'ordre de priorité cuda > mps > cpu.
@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.delenv(device_module.ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_device("auto").type == expected


def test_explicit_unavailable_device_raises(monkeypatch):
    monkeypatch.delenv(device_module.ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        get_device("cuda")
