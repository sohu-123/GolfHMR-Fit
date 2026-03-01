"""Default hand constraint indices and helper utilities."""

from typing import Dict

# Default joint indices based on MHR ordering observed in code
DEFAULT_HAND_IDXS: Dict[str, int] = {
    # body joint indices (used for forearm/wrist in body outputs)
    "l_elbow": 76,
    "r_elbow": 40,
    "l_wrist": 77,
    "r_wrist": 41,
    # hand output wrist/hand indices (from hand decoder outputs)
    "l_hand": 78,
    "r_hand": 42,
}

# Default lambda (weights) for the hand constraint losses
DEFAULT_HAND_LAMBDAS: Dict[str, float] = {
    "lambda_hand_rot": 0.1,
    "lambda_hand_axis": 0.5,
    "lambda_hand_wrist": 0.5,
}


def get_hand_idxs_from_cfg(cfg) -> Dict[str, int]:
    """Read hand idx overrides from cfg if present; otherwise return defaults."""
    if cfg is None:
        return DEFAULT_HAND_IDXS
    training_cfg = getattr(cfg, "TRAIN", None)
    if training_cfg is None:
        return DEFAULT_HAND_IDXS

    idxs = DEFAULT_HAND_IDXS.copy()
    overrides = getattr(training_cfg, "HAND_CONSTRAINT_IDXS", None)
    if overrides is None:
        return idxs
    for k, v in overrides.items():
        if k in idxs:
            idxs[k] = int(v)
    return idxs


def get_hand_lambdas_from_cfg(cfg) -> Dict[str, float]:
    if cfg is None:
        return DEFAULT_HAND_LAMBDAS
    training_cfg = getattr(cfg, "TRAIN", None)
    if training_cfg is None:
        return DEFAULT_HAND_LAMBDAS

    lambdas = DEFAULT_HAND_LAMBDAS.copy()
    overrides = getattr(training_cfg, "HAND_CONSTRAINT_LAMBDAS", None)
    if overrides is None:
        return lambdas
    for k, v in overrides.items():
        if k in lambdas:
            lambdas[k] = float(v)
    return lambdas
