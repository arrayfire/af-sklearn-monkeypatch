__all__ = ["Patcher"]

import importlib
from pathlib import Path
from typing import Any, Optional

from .. import patches_info

# TODO: remove
temporary_storage = {}

class Patcher:
    @staticmethod
    def patch(module_name: str) -> None:
        patch_config = patches_info[module_name]
        _apply_patch(patch_config["module"], patch_config["name"], patch_config["module_patch"])

    @staticmethod
    def rollback(module_name: str) -> None:
        patch_config = patches_info[module_name]
        _apply_patch(patch_config["module"], patch_config["name"], None)

    @staticmethod
    def patch_all():
        for p in patches_info:
            Patcher.patch(p)

    @staticmethod
    def rollback_all():
        for p in patches_info:
            if p in temporary_storage:
                Patcher.rollback(p)


def _load_module(module_path: str) -> Any:
    return importlib.import_module(module_path)


def _load_instance(module_path: str, instance_name: str) -> Any:
    return getattr(_load_module(module_path), instance_name)


def _apply_patch(target_module: str, target_instance_name: str, module_patch: Optional[str]) -> None:
    loaded_target_module = _load_module(target_module)

    if module_patch is not None:
        temporary_storage[target_instance_name] = getattr(loaded_target_module, target_instance_name)
        loaded_patch_instance = _load_instance(module_patch, target_instance_name)
    if module_patch is None:
        loaded_patch_instance = temporary_storage[target_instance_name]
        del temporary_storage[target_instance_name]

    setattr(loaded_target_module, target_instance_name, loaded_patch_instance)
