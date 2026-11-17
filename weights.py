import json
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_FIT_WEIGHTS = CONFIG_DIR / "fit_weights.json"
DEFAULT_MAPPING_WEIGHTS = CONFIG_DIR / "mapping_weights.json"

SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf",
    "right_eye_smplhf", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
    "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2",
    "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2",
    "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2",
    "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
    "right_thumb3", "nose", "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", "right_thumb",
    "right_index", "right_middle", "right_ring", "right_pinky",
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)

_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _fingers(side):
    joints = [f"{side}_{f}{k}" for f in _FINGERS for k in (1, 2, 3)]
    return joints + [f"{side}_{f}" for f in _FINGERS]


GROUPS = {
    "wrists": ["left_wrist", "right_wrist"],
    "left_fingers": _fingers("left"),
    "right_fingers": _fingers("right"),
    "fingers": _fingers("left") + _fingers("right"),
}


def _read(path, default):
    path = Path(path) if path else default
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _joint_index(key):
    if key in SMPLX_JOINT_NAMES:
        return SMPLX_JOINT_NAMES.index(key)
    if str(key).isdigit() and int(key) < NUM_SMPLX_JOINTS:
        return int(key)
    raise KeyError(f"Unknown SMPL-X joint {key!r}. Use a name from SMPLX_JOINT_NAMES or an index 0-{NUM_SMPLX_JOINTS - 1}.")


def _check_keys(cfg, allowed, path):
    unknown = [k for k in cfg if k not in allowed and not k.startswith("_")]
    if unknown:
        raise KeyError(f"{path}: unknown key(s) {unknown}; valid: {sorted(allowed)}")


def load_fit_weights(path=None):
    cfg, path = _read(path, DEFAULT_FIT_WEIGHTS)
    _check_keys(cfg, {"default", "groups", "joints"}, path)
    w = np.full(NUM_SMPLX_JOINTS, float(cfg.get("default", 1.0)), dtype=np.float32)
    for group, value in cfg.get("groups", {}).items():
        if group not in GROUPS:
            raise KeyError(f"{path}: unknown group {group!r}; valid: {sorted(GROUPS)}")
        for name in GROUPS[group]:
            w[_joint_index(name)] = value
    for key, value in cfg.get("joints", {}).items():
        w[_joint_index(key)] = value
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError(f"{path}: weights must be finite and not negative.")
    return w


def load_priority(path, names):
    cfg, path = _read(path, DEFAULT_MAPPING_WEIGHTS)
    _check_keys(cfg, {"priority"}, path)
    priority = []
    for item in cfg.get("priority", []):
        if isinstance(item, str):
            if item not in names:
                raise KeyError(f"{path}: unknown keypoint {item!r}; valid: {sorted(names)} or a combined index.")
            item = names[item]
        priority.append(int(item))
    return priority
