"""Store model weights without torch.compile's wrapper prefix."""
import os
import tempfile
import torch


def atomic_save(state, destination):
    destination = os.fspath(destination)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp",
                                     dir=os.path.dirname(os.path.abspath(destination)))
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def unwrap_model(model):
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def clean_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key in cleaned:
            raise ValueError(f"Duplicate checkpoint key after unwrapping: {key}")
        cleaned[key] = value
    return cleaned
