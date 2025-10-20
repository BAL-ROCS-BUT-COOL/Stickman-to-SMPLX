import numpy as np

joint_mapping = {
    0: -1,   # nose
    1: 12,   # neck
    2: 0,   # pelvis
    3: 59, # left ear
    4: 58, # right ear
    5: 16,   # left shoulder
    6: 17,   # right shoulder
    7: 18,   # left elbow
    8: 19,   # right elbow
    9: 1,   # Left hip
    10: 2,   # Right hip
    11: 4,   # left knee
    12: 5, # right knee
    13: 7, # left ankle
    14: 8, # right ankle
    15: 20, # Left wrist 20
    16: 37, # left thumb1
    17: 38, # left thumb2
    18: 39, # Left thumb3
    19: 66, # left thumb4
    20: 25, # Left index1
    21: 26, # left index2
    22: 27, # Left index3
    23: 67, # left index4
    24: 28, # left middle1
    25: 29, # Left middle2
    26: 30, # left middle3
    27: 68, # left middle4
    28: 34, # left ring1
    29: 35, # left ring2
    30: 36, # left ring3
    31: 69, # left ring4
    32: 31, # left pinky1
    33: 32, # left pinky2
    34: 33, # left pinky3
    35: 70, # left pinky4
    36: 21, # right wrist 21
    37: 52, # right thumb1
    38: 53, # right thumb2
    39: 54, # right thumb3
    40: 71, # right thumb4
    41: 40, # right index1
    42: 41, # right index2
    43: 42, # right index3
    44: 72, # right index4
    45: 43, # right middle1
    46: 44, # right middle2
    47: 45, # right middle3
    48: 73, # right middle4
    49: 49, # right ring1
    50: 50, # right ring2
    51: 51, # right ring3
    52: 74, # right ring4
    53: 46, # right pinky1
    54: 47, # right pinky2
    55: 48, # right pinky3
    56: 75, # right pinky4
}

def apply_joint_mapping(joints: np.ndarray, mapping: dict, fill_value=np.nan):
    """
    Map (76,3) -> (57,3) or (T,76,3) -> (T,57,3) using 'mapping'.
    Returns (joints_mapped, valid_mask).
    """
    if joints.ndim == 2:
        joints = joints[None, ...]  # (1,76,3)
        squeeze_back = True
    elif joints.ndim == 3:
        squeeze_back = False
    else:
        raise ValueError("joints must be (76,3) or (T,76,3)")

    T = joints.shape[0]
    out_len = max(mapping.keys()) + 1  # 57
    out = np.full((T, out_len, 3), fill_value, dtype=float)
    mask = np.zeros((out_len,), dtype=bool)

    for tgt, src in mapping.items():
        if src is not None and src >= 0:
            out[:, tgt, :] = joints[:, src, :]
            mask[tgt] = True

    if squeeze_back:
        out = out[0]
    return out, mask