import numpy as np
import argparse
import os
from joints import joint_mapping

# ─── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Combine body and hand 3D joints into SMPL-X format")
parser.add_argument("--body", type=str, required=True, help="Path to body .npz file")
parser.add_argument("--hand", type=str, required=True, help="Path to hand .npz file")
parser.add_argument("--output", type=str, default="data/smplx_joints.npy", help="Output .npy filename")
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
 