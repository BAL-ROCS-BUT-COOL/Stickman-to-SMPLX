import numpy as np
import argparse
import os
from joints import joint_mapping
from weights import load_priority

parser = argparse.ArgumentParser(description="Combine body and hand 3D joints into SMPL-X format")
parser.add_argument("--body", type=str, required=True, help="Path to body .npz file")
parser.add_argument("--hand", type=str, required=True, help="Path to hand .npz file")
parser.add_argument("--output-dir", type=str, default="data", help="Output directory name")
parser.add_argument("--output", type=str, default="smplx_joints.npy", help="Output .npy filename")
parser.add_argument("--permute_axes", action=argparse.BooleanOptionalAction, default=True, help="toggle axis permutation.")
parser.add_argument("--center-at-pelvis", action=argparse.BooleanOptionalAction, default=True, help="toggle centering at pelvis.")
parser.add_argument("--transform_output", type=str, default="joint_transforms.npy", help="Output .npy file for the manual transfomations.")
parser.add_argument("--apply-priority", action=argparse.BooleanOptionalAction, default=True, help="toggle the PRIORITY step (see config/mapping_weights.json).")
parser.add_argument("--weights", type=str, default=None, help="priority config JSON (default: config/mapping_weights.json)")
args = parser.parse_args()

if not os.path.exists(args.body):
    raise FileNotFoundError(f"Body file not found: {args.body}")
if not os.path.exists(args.hand):
    raise FileNotFoundError(f"Hand file not found: {args.hand}")


body = np.load(args.body, allow_pickle=True)['poses_3d']
hand = np.load(args.hand, allow_pickle=True)['poses_3d']

HIDDEN = {9, 10}
MIDPOINTS = {
    1: (5, 6),
    2: (11, 12),
}

N_BODY = body.shape[1]
N_HAND = hand.shape[1]
HAND_JOINTS = N_HAND // 2
LEFT_HAND_WRIST = N_BODY
RIGHT_HAND_WRIST = N_BODY + HAND_JOINTS

KEYPOINT_NAMES = {
    "nose": 0, "neck": 1, "pelvis": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
    "left_hand_wrist": LEFT_HAND_WRIST, "right_hand_wrist": RIGHT_HAND_WRIST,
}
PRIORITY = load_priority(args.weights, KEYPOINT_NAMES)

_NECK, _PELVIS = 1, 2
_L_SHO, _R_SHO, _L_ELB, _R_ELB = 5, 6, 7, 8
_L_HIP, _R_HIP, _L_KNEE, _R_KNEE, _L_ANK, _R_ANK = 11, 12, 13, 14, 15, 16
BONES = [
    (_NECK, _L_SHO), (_NECK, _R_SHO),
    (_L_SHO, _L_ELB), (_L_ELB, LEFT_HAND_WRIST),
    (_R_SHO, _R_ELB), (_R_ELB, RIGHT_HAND_WRIST),
    (_PELVIS, _L_HIP), (_PELVIS, _R_HIP),
    (_L_HIP, _L_KNEE), (_L_KNEE, _L_ANK),
    (_R_HIP, _R_KNEE), (_R_KNEE, _R_ANK),
    (_PELVIS, _NECK),
]

FOLLOWERS = {
    LEFT_HAND_WRIST: list(range(LEFT_HAND_WRIST + 1, LEFT_HAND_WRIST + HAND_JOINTS)),
    RIGHT_HAND_WRIST: list(range(RIGHT_HAND_WRIST + 1, RIGHT_HAND_WRIST + HAND_JOINTS)),
    _NECK: [0, 3, 4],
}

PRIORITY_MAX_ITERS = 500
PRIORITY_TOL = 1e-6

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

total_kps = len(body[0]) + len(hand[0])
visible_indices = [i for i in range(total_kps) if i not in HIDDEN]

joints_cha1 = []
for b_frame, h_frame in zip(body, hand):
    frame_points = np.array([get_joint_point(i, b_frame, h_frame) for i in visible_indices])
    joints_cha1.append(frame_points)

joints_cha1 = np.stack(joints_cha1)
joints_cha1.shape

def permute_axes(joints):
    """Permutes axes as follows: x→z, y→x, z→y"""
    x = joints[..., 0]
    y = joints[..., 1]
    z = joints[..., 2]

    new_joints = np.stack([y, z, x], axis=-1)

    rot_mat = np.array([
        [-1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]
    ])


    return new_joints @ rot_mat.T

def center_joints_at_pelvis(joints, pelvis_index=2, offset=(0.00, 0.00, 0.00)):
    """Center the body at the pelvis with a slight offset"""
    assert joints.ndim == 3 and joints.shape[-1] == 3, "Expected (T, J, 3)"
    pelvis0 = joints[0, pelvis_index]
    offset_vec = np.asarray(offset, dtype=joints.dtype)

    centered = joints - pelvis0.reshape(1, 1, 3) + offset_vec.reshape(1, 1, 3)
    return centered

if args.permute_axes:
    joints_cha1_transformed = permute_axes(joints_cha1)
    basis = np.eye(3).reshape(3, 1, 3)
    permute_matrix = permute_axes(basis)[:, 0, :].T
else:
    joints_cha1_transformed = joints_cha1.copy()
    permute_matrix = np.eye(3)

pelvis_index = 2
if args.center_at_pelvis:
    pelvis0 = joints_cha1_transformed[0, pelvis_index].copy()
    # TODO: remove center_offset (magic variable, origin unclear)
    center_offset = np.array((0.01, 0.01, 0.01))
    joints_cha1_transformed = center_joints_at_pelvis(joints_cha1_transformed, pelvis_index=pelvis_index, offset=center_offset)
else:
    pelvis0 = np.zeros(3)
    center_offset = np.zeros(3)

n_frames = joints_cha1.shape[0]
frame_transform = np.eye(4)
frame_transform[:3, :3] = permute_matrix
frame_transform[:3, 3] = -pelvis0 + center_offset
joint_transforms = np.tile(frame_transform, (n_frames, 1, 1))
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, args.transform_output), joint_transforms)
print(f"Saved per-frame transform record ({joint_transforms.shape}) -> {args.transform_output}")


def prioritize_joints(joints, priority, visible_indices, bones, followers,
                      max_iters=PRIORITY_MAX_ITERS, tol=PRIORITY_TOL):
    """Make the skeleton consistent (constant bone lengths, like SMPL-X) while the"""
    col = {raw: c for c, raw in enumerate(visible_indices)}

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
    rank = np.full(n_joints, n_listed, dtype=int)
    for r, raw in enumerate(priority):
        rank[col[raw]] = r

    bone_cols = []
    for a, b in bones:
        for raw in (a, b):
            if raw not in col:
                raise IndexError(f"Bone endpoint {raw} is not a visible joint.")
        bone_cols.append((col[a], col[b]))

    lengths = np.array([np.nanmedian(np.linalg.norm(joints[:, b] - joints[:, a], axis=-1))
                        for a, b in bone_cols])
    tol_abs = tol * float(np.mean(lengths))

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
            corr = np.nan_to_num(err * d / np.maximum(dist, 1e-12))
            if rank[a] == rank[b]:
                wa, wb = 0.5, 0.5
            elif rank[a] < rank[b]:
                wa, wb = 0.0, 1.0
            else:
                wa, wb = 1.0, 0.0
            out[:, a] += wa * corr
            out[:, b] -= wb * corr
        iters_used = it + 1
        if worst < tol_abs:
            break

    delta = out - joints
    for parent, children in followers.items():
        shift = np.nan_to_num(delta[:, col[parent]])
        for child in children:
            c = col[child]
            if rank[c] < n_listed:
                continue
            out[:, c] += shift

    moved = np.linalg.norm(np.nan_to_num(out - joints), axis=-1)
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


def reorder_joints(joints_cha1, joint_mapping, log_unmapped=False):

    if joints_cha1.ndim != 3 or joints_cha1.shape[2] != 3:
        raise ValueError(f"Expected joints_cha1 shape (n_frames, n_joints, 3), got {joints_cha1.shape}")

    n_frames, n_joints_input, _ = joints_cha1.shape
    n_joints_output = 76

    print(f"Input shape: {joints_cha1.shape}")
    print(f"Expected input joints: {len(joint_mapping)}, got: {n_joints_input}")

    smplx_joints = np.zeros((n_frames, n_joints_output, 3))

    unmapped_joints = []
    mapped_count = 0

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

        smplx_joints[:, dst_idx, :] = joints_cha1[:, src_idx, :]
        mapped_count += 1

    if log_unmapped:
        print(f"Mapped {mapped_count} joints")
        print(f"Unmapped source joints: {unmapped_joints}")

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
