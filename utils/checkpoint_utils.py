"""Configuration-path helpers required by the patched LLaVA runtime.

Model loading for the current scheme lives in mvpg_run.llava_inference.
These small accessors accept the existing attribute-style OmegaConf objects
without importing the legacy training stack or loading model frameworks.
"""
from pathlib import Path


def get_model_checkpoint(checkpoint_config) -> Path:
    return Path(checkpoint_config.base_model)


def has_image_model_checkpoint(checkpoint_config) -> bool:
    return checkpoint_config.image_model is not None


def get_image_model_checkpoint(checkpoint_config):
    if not has_image_model_checkpoint(checkpoint_config):
        return None
    return checkpoint_config.image_model
