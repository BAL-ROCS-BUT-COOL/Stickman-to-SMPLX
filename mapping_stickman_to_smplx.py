import numpy as np
import argparse
import os
from joints import joint_mapping

# ─── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Combine body and hand 3D joints into SMPL-X format")
parser.add_argument("--body", type=str, required=True, help="Path to body .npz file")
parser.add_argument("--hand", type=str, required=True, help="Path to hand .npz file")
parser.add_argument("--output-dir", type=str, default="data", help="Output directory name")
parser.add_argument("--output", type=str, default="smplx_joints.npy", help="Output .npy filename")
parser.add_argument("--permute_axes", action=argparse.BooleanOptionalAction, default=True, help="toggle axis permutation.")
parser.add_argument("--center-at-pelvis", action=argparse.BooleanOptionalAction, default=True, help="toggle centering at pelvis.")
parser.add_argument("--transform_output", type=str, default="joint_transforms.npy", help="Output .npy file for the manual transfomations.")
parser.add_argument("--apply-priority", action=argparse.BooleanOptionalAction, default=True, help="toggle the PRIORITY step (see PRIORITY in the script).")
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

# ─── Joint priority ───────────────────────────────────────────────────────────
# PRIORITY lists joint indexes, most important first. A listed joint keeps its
# position exactly as it is in the pose / hand file; every joint that is NOT
# listed has the lowest priority and is the one that gives way (see
# prioritize_joints() below). Leave the list empty to get the old behaviour.
#
# Indexes refer to the COMBINED keypoint list, i.e. the same index space as
# HIDDEN, MIDPOINTS and get_joint_point():
#
#     0 .. N_BODY-1                  pose file  (COCO-17; 1 = neck, 2 = pelvis
#                                    are the virtual MIDPOINTS joints)
#     N_BODY .. N_BODY+N_HAND-1      hand file  (left hand first, then right)
#
# Careful: body[9] / body[10] are the wrist ends of the pose-data arms. They are
# in HIDDEN and are NOT used. The wrists we want are the ones from the HAND file.
N_BODY = body.shape[1]                         # 17
N_HAND = hand.shape[1]                         # 42
HAND_JOINTS = N_HAND // 2                      # 21 per hand
LEFT_HAND_WRIST = N_BODY                       # hand[0]   -> combined 17
RIGHT_HAND_WRIST = N_BODY + HAND_JOINTS        # hand[21]  -> combined 38

PRIORITY = [
    LEFT_HAND_WRIST,
    RIGHT_HAND_WRIST,
]

# Rigid "bones" (combined indexes) whose length is kept constant over the whole
# sequence (reference length = median over all frames). COCO-17 body, but the
# wrists are the hand-file wrists.
_NECK, _PELVIS = 1, 2
_L_SHO, _R_SHO, _L_ELB, _R_ELB = 5, 6, 7, 8
_L_HIP, _R_HIP, _L_KNEE, _R_KNEE, _L_ANK, _R_ANK = 11, 12, 13, 14, 15, 16
BONES = [
    # (no shoulder-shoulder / hip-hip bones on purpose: together with the two
    #  midpoint bones they would form flat collinear triangles that converge slowly)
    (_NECK, _L_SHO), (_NECK, _R_SHO),                              # shoulder girdle
    (_L_SHO, _L_ELB), (_L_ELB, LEFT_HAND_WRIST),                   # left arm
    (_R_SHO, _R_ELB), (_R_ELB, RIGHT_HAND_WRIST),                  # right arm
    (_PELVIS, _L_HIP), (_PELVIS, _R_HIP),                          # pelvis
    (_L_HIP, _L_KNEE), (_L_KNEE, _L_ANK),                          # left leg
    (_R_HIP, _R_KNEE), (_R_KNEE, _R_ANK),                          # right leg
    (_PELVIS, _NECK),                                              # spine
]

# Joints that just hang off a parent and are translated with it when the parent
# is moved (so hands and head keep their own shape instead of being deformed).
FOLLOWERS = {
    LEFT_HAND_WRIST: list(range(LEFT_HAND_WRIST + 1, LEFT_HAND_WRIST + HAND_JOINTS)),
    RIGHT_HAND_WRIST: list(range(RIGHT_HAND_WRIST + 1, RIGHT_HAND_WRIST + HAND_JOINTS)),
    _NECK: [0, 3, 4],   # nose, left ear, right ear
}

PRIORITY_MAX_ITERS = 500
PRIORITY_TOL = 1e-6      # relative to the mean bone length

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

def center_joints_at_pelvis(joints, pelvis_index=2, offset=(0.00, 0.00, 0.00)):
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
if args.permute_axes:
    joints_cha1_transformed = permute_axes(joints_cha1)
    # Recover the exact 3x3 linear map permute_axes() applies.
    basis = np.eye(3).reshape(3, 1, 3)
    permute_matrix = permute_axes(basis)[:, 0, :].T
else:
    joints_cha1_transformed = joints_cha1.copy()
    permute_matrix = np.eye(3)

pelvis_index = 2
if args.center_at_pelvis:
    pelvis0 = joints_cha1_transformed[0, pelvis_index].copy()
	"""
	remove center_offset and figure out why it even existed. We do not want magic variables.
	"""
    center_offset = np.array((0.01, 0.01, 0.01))
    joints_cha1_transformed = center_joints_at_pelvis(joints_cha1_transformed, pelvis_index=pelvis_index, offset=center_offset)
else:
    pelvis0 = np.zeros(3)
    center_offset = np.zeros(3)

# Save the per-frame 4x4 affine transform actually applied (permute_matrix,
# then subtract pelvis0, then add center_offset), as a homogeneous matrix
# with per frame data.
# (The PRIORITY step below is applied afterwards and is not part of this record.
# Prioritized joints are left untouched by it, so for them this transform still
# maps the original data onto the output exactly.)
n_frames = joints_cha1.shape[0]
frame_transform = np.eye(4)
frame_transform[:3, :3] = permute_matrix
frame_transform[:3, 3] = -pelvis0 + center_offset
joint_transforms = np.tile(frame_transform, (n_frames, 1, 1))
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, args.transform_output), joint_transforms)
print(f"Saved per-frame transform record ({joint_transforms.shape}) -> {args.transform_output}")


# ------------------------------------------- Priority ---------------------------------------------------------------------

def prioritize_joints(joints, priority, visible_indices, bones, followers,
                      max_iters=PRIORITY_MAX_ITERS, tol=PRIORITY_TOL):
    """
    Make the skeleton consistent (constant bone lengths, like SMPL-X) while the
    joints in `priority` keep their position.

    joints          : (T, J, 3), columns ordered like `visible_indices`
                      (combined indexes without HIDDEN).
    priority        : combined joint indexes, most important first.
    visible_indices : combined index of every column of `joints`.
    bones           : (a, b) pairs of combined indexes that must keep a fixed length.
    followers       : {parent: [children]}; children are translated with their
                      parent unless they are listed in `priority` themselves.

    For every bone the endpoint with the higher priority stays where it is and the
    other one takes the whole correction. Two endpoints of equal priority (which
    can only be two unlisted joints) share the correction half and half. A listed
    joint therefore only moves if it sits on a bone together with a joint that is
    listed EARLIER, and the bone length cannot be met otherwise.

    Returns (adjusted_joints, info).
    """
    col = {raw: c for c, raw in enumerate(visible_indices)}

    # --- validate the priority list ---
    seen = set()
    for raw in priority:
        if raw in HIDDEN:
            raise ValueError(
                f"Priority index {raw} is a hidden pose-data wrist (HIDDEN) and is never used. "
                f"Use the hand-file wrists instead: {LEFT_HAND_WRIST} (left) / {RIGHT_HAND_WRIST} (right).")
        if raw not in col:
            raise IndexError(f"Priority index {raw} is not a valid combined joint index (0..{total_kps - 1}).")
        if raw in seen:
            raise ValueError(f"Priority index {raw} is listed twice.")
        seen.add(raw)

    n_joints = joints.shape[1]
    n_listed = len(priority)
    rank = np.full(n_joints, n_listed, dtype=int)      # unlisted = lowest priority
    for r, raw in enumerate(priority):
        rank[col[raw]] = r

    bone_cols = []
    for a, b in bones:
        for raw in (a, b):
            if raw not in col:
                raise IndexError(f"Bone endpoint {raw} is not a visible joint.")
        bone_cols.append((col[a], col[b]))

    # --- rigid bone lengths: median over the sequence ---
    lengths = np.array([np.nanmedian(np.linalg.norm(joints[:, b] - joints[:, a], axis=-1))
                        for a, b in bone_cols])
    tol_abs = tol * float(np.mean(lengths))

    # --- projection (position based, Gauss-Seidel) ---
    out = joints.copy()
    worst = np.inf
    iters_used = 0
    for it in range(max_iters):
        worst = 0.0
        for (a, b), L in zip(bone_cols, lengths):
            d = out[:, b] - out[:, a]
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            err = dist - L
            worst = max(worst, float(np.nanmax(np.abs(err))))
            corr = np.nan_to_num(err * d / np.maximum(dist, 1e-12))   # moving a by +corr or b by -corr shortens the bone by err
            if rank[a] == rank[b]:
                wa, wb = 0.5, 0.5
            elif rank[a] < rank[b]:      # a is more important -> only b moves
                wa, wb = 0.0, 1.0
            else:                        # b is more important -> only a moves
                wa, wb = 1.0, 0.0
            out[:, a] += wa * corr
            out[:, b] -= wb * corr
        iters_used = it + 1
        if worst < tol_abs:
            break

    # --- followers: hands / head travel with their parent, shape is preserved ---
    delta = out - joints
    for parent, children in followers.items():
        shift = np.nan_to_num(delta[:, col[parent]])             # (T, 3)
        for child in children:
            c = col[child]
            if rank[c] < n_listed:                               # listed -> keeps its position
                continue
            out[:, c] += shift

    moved = np.linalg.norm(np.nan_to_num(out - joints), axis=-1)  # (T, J)
    info = {
        "iterations": iters_used,
        "worst_bone_error": worst,
        "bone_lengths": dict(zip(bones, lengths)),
        "priority_moved": {raw: float(moved[:, col[raw]].max()) for raw in priority},
        "other_moved_max": float(np.max(moved[:, rank == n_listed])) if np.any(rank == n_listed) else 0.0,
    }
    return out, info


if args.apply_priority and len(PRIORITY) > 0:
    joints_cha1_transformed, prio_info = prioritize_joints(
        joints_cha1_transformed, PRIORITY, visible_indices, BONES, FOLLOWERS)
    print(f"Priority order (combined indexes): {PRIORITY}")
    print(f"Priority solver: {prio_info['iterations']} iterations, "
          f"worst remaining bone length error {prio_info['worst_bone_error']:.2e}")
    for raw, dist in prio_info["priority_moved"].items():
        flag = "" if dist < 1e-9 else "   <-- moved: conflicts with a joint of higher priority"
        print(f"  joint {raw}: max displacement {dist:.2e}{flag}")
    print(f"  largest displacement of a non-priority joint: {prio_info['other_moved_max']:.4f}")
else:
    print("Priority step skipped.")


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
np.save(os.path.join(args.output_dir, args.output), smplx_joints)
print(f"Saved joints as {args.output}")
