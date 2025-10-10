import numpy as np
import argparse
import os

# ─── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Combine body and hand 3D joints into SMPL-X format")
parser.add_argument("--body", type=str, required=True, help="Path to body .npz file")
parser.add_argument("--hand", type=str, required=True, help="Path to hand .npz file")
parser.add_argument("--output", type=str, default="smplx_joints.npy", help="Output .npy filename")
args = parser.parse_args()

# ─── Load Data ────────────────────────────────────────────────────────────────
if not os.path.exists(args.body):
    raise FileNotFoundError(f"Body file not found: {args.body}")
if not os.path.exists(args.hand):
    raise FileNotFoundError(f"Hand file not found: {args.hand}")


body = np.load(args.body, allow_pickle=True)['poses_3d']
hand = np.load(args.hand, allow_pickle=True)['poses_3d']

# ─── Constants ────────────────────────────────────────────────────────────────
HIDDEN = {9, 10}
MIDPOINTS = {
    1: (5, 6),    # Midpoint between body[5] and body[6]
    2: (11, 12),  # Midpoint between body[11] and body[12]
}

# ─── Helper Function: Get point by index ──────────────────────────────────────
def get_joint_point(idx, body_frame, hand_frame):
    num_body = len(body_frame)
    num_hand = len(hand_frame)
    
    if idx in MIDPOINTS:
        i1, i2 = MIDPOINTS[idx]
        return (body_frame[i1] + body_frame[i2]) / 2
    elif 0 <= idx < num_body:
        return body_frame[idx]
    elif num_body <= idx < num_body + num_hand:
        return hand_frame[idx - num_body]
    else:
        raise IndexError(f"Invalid index {idx} for combined joints.")

# ─── Main Processing: Compute all visible joint points ────────────────────────
total_kps = len(body[0]) + len(hand[0])
visible_indices = [i for i in range(total_kps) if i not in HIDDEN]

joints_cha1 = []
for b_frame, h_frame in zip(body, hand):
    frame_points = np.array([get_joint_point(i, b_frame, h_frame) for i in visible_indices])
    joints_cha1.append(frame_points)

joints_cha1 = np.stack(joints_cha1)
joints_cha1.shape

def permute_axes(joints):
    """
    Permutes axes as follows: x→z, y→x, z→y
    joints: (N, 55, 3)
    """
    # Split into components
    x = joints[..., 0]
    y = joints[..., 1]
    z = joints[..., 2]

    # Reorder: [z, x, y]
    new_joints = np.stack([y, z, x], axis=-1)

    rot_mat = np.array([
        [-1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]
    ])


    return new_joints @ rot_mat.T

def center_joints_at_pelvis(joints, pelvis_index=2, offset=(0.01, 0.01, 0.01)):
    """
    Center the body at the pelvis with a slight offset
    Also make sure that the body moves relatively to the original movement and is not fixed at center for every frame. 
    This would cause Jitter
    """
    assert joints.ndim == 3 and joints.shape[-1] == 3, "Expected (T, J, 3)"
    pelvis0 = joints[0, pelvis_index]            # (3,)
    offset_vec = np.asarray(offset, dtype=joints.dtype)  # (3,)

    # Broadcast: subtract pelvis0 from all joints, then add the offset
    centered = joints - pelvis0.reshape(1, 1, 3) + offset_vec.reshape(1, 1, 3)
    return centered

# Apply to your dataset
joints_cha1_transformed = permute_axes(joints_cha1)
joints_cha1_transformed = center_joints_at_pelvis(joints_cha1_transformed)



# ------------------------------------------- Mapping ---------------------------------------------------------------------

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
    15: -1, # Left wrist 20
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
    36: -1, # right wrist 21
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
def reorder_joints(joints_cha1, joint_mapping, log_unmapped=False):

    # Validate input
    if joints_cha1.ndim != 3 or joints_cha1.shape[2] != 3:
        raise ValueError(f"Expected joints_cha1 shape (n_frames, n_joints, 3), got {joints_cha1.shape}")
    
    n_frames, n_joints_input, _ = joints_cha1.shape
    n_joints_output = 76  # SMPL-X has 75 joints
    
    print(f"Input shape: {joints_cha1.shape}")
    print(f"Expected input joints: {len(joint_mapping)}, got: {n_joints_input}")
    
    # Initialize output array with zeros
    smplx_joints = np.zeros((n_frames, n_joints_output, 3))
    
    # Track unmapped joints
    unmapped_joints = []
    mapped_count = 0
    
    # Apply mapping
    for src_idx, dst_idx in joint_mapping.items():
        if src_idx >= n_joints_input:
            print(f"Warning: Source index {src_idx} exceeds input size {n_joints_input}")
            continue
            
        if dst_idx == -1:
            unmapped_joints.append(src_idx)
            continue
            
        if dst_idx >= n_joints_output:
            print(f"Warning: Destination index {dst_idx} exceeds output size {n_joints_output}")
            continue
            
        # Copy the joint data
        smplx_joints[:, dst_idx, :] = joints_cha1[:, src_idx, :]
        mapped_count += 1
    
    if log_unmapped:
        print(f"Mapped {mapped_count} joints")
        print(f"Unmapped source joints: {unmapped_joints}")
        
        # Check which output joints remain zero
        zero_joints = []
        for i in range(n_joints_output):
            if np.allclose(smplx_joints[0, i, :], 0):
                zero_joints.append(i)
        print(f"Output joints that remain zero: {zero_joints}")

    return smplx_joints

smplx_joints = reorder_joints(joints_cha1_transformed, joint_mapping, log_unmapped=True)

print(f"Reordered joints shape: {smplx_joints.shape}")
np.save(args.output, smplx_joints)
print(f"Saved joints as {args.output}")
 