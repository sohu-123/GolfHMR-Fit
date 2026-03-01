"""Syntax-only tests for training_step and config helpers."""

# These tests are syntax-only and do not require torch to be installed.
import importlib


def test_imports():
    importlib.import_module("sam_3d_body.models.meta_arch.base_model")
    importlib.import_module("sam_3d_body.utils.hand_constraints")


if __name__ == "__main__":
    test_imports()
    print("imports OK")
