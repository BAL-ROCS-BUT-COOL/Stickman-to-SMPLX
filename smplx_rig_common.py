"""
smplx_rig_common.py

Shared, numpy-only rig module. It is the single source of truth for everything the SMPL-X fitters and the Blender
importer must agree on exactly:

  * the bone structure   re-rooted hierarchy, detached joints, rest orientations, bone lengths
  * the A-pose rest      rest joints / rest mesh, built the same way as the armature in Blender
  * SMPL-X kinematics    pose parameters -> world rotations / positions -> per-joint skin matrices
  * pose-bone solving    skin matrices -> Blender matrix_basis (location / rotation / scale) per bone
  * body chains          named groups of joints (torso, left_arm, right_index, ...) used to choose which parts are "raw"
  * the CHANNEL FILE     the one pose format the importer reads, written by the fitters (see save_channels)

No bpy, no scipy, no torch: it is imported by command-line fitters and loaded inside Blender
(importlib, from script_dir) alike.

Typical use
    meta = np.load("armature_rig_metadata.npz")
    rig = build_rig(meta)                                    # rest structure, identical for every script
    pose = load_pose_params("armature_pose_params_smoothed.npz")
    channels = channels_from_params(rig, pose)               # plain SMPL-X fit -> bone channels
    save_channels("channels.npz", rig, channels, meta=dict(source="plain fit"))
"""

import json
import time
from math import radians

import numpy as np

_t0 = time.time()


def _log(msg):
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


# ── RIG STRUCTURE (one definition for the fitters AND the importer) ───────────
NUM_JOINTS = 55
leaf_bone_length = 0.03      # display length of leaf bones (jaw, eyes, feet)
ROOT_BONE_IDX = 9            # the hierarchy is re-rooted here (spine3)
DETACHED_JOINTS = {12, 13, 14}   # neck and both collars are independent roots
LEFT_HIP_IDX = 1
RIGHT_HIP_IDX = 2
LEFT_COLLAR_IDX = 13
RIGHT_COLLAR_IDX = 14
NON_TRUNK_CHILDREN = {LEFT_COLLAR_IDX, RIGHT_COLLAR_IDX}
KEEP_SMPLX_HIERARCHY = True  # False: every bone is a root (the skinning result is identical)

# A-pose rest: the armature's rest pose (bones + mesh) is an A pose instead of the SMPL-X T pose.
APOSE_REST = True
ARM_DOWN_DEG = 45.0
ELBOW_BEND_DEG = 0.0

SIDES = ["left", "right"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]    # order inside a hand block of the pose data
# SMPL-X joint numbers (see smplx/joint_names.py)
SMPLX_BODY = {
    "left":  {"sho": 16, "elb": 18, "wrist": 20, "hip": 1, "knee": 4, "ank": 7, "foot": 10, "collar": 13, "hand": 25},
    "right": {"sho": 17, "elb": 19, "wrist": 21, "hip": 2, "knee": 5, "ank": 8, "foot": 11, "collar": 14, "hand": 40},
}
SMPLX_FINGER_OFFSET = {"index": 0, "middle": 3, "pinky": 6, "ring": 9, "thumb": 12}   # inside a 15-joint hand

# ── BODY CHAINS ───────────────────────────────────────────────────────────────
# The unit of "which part comes from where": a fitter's RAW chains, the importer's bone tags and bone collections.
CHAINS = {
    "torso": [0, 3, 6, 9, 12, 13, 14, 15, 22, 23, 24],     # pelvis, spine, neck, collars, head, jaw, eyes
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "left_arm": [16, 18],                                   # shoulder, elbow  (= upper arm, forearm)
    "right_arm": [17, 19],
    "left_wrist": [20],                                     # the palm bones
    "right_wrist": [21],
}
for _side in SIDES:
    for _finger, _off in SMPLX_FINGER_OFFSET.items():
        _first = SMPLX_BODY[_side]["hand"] + _off
        CHAINS[f"{_side}_{_finger}"] = [_first, _first + 1, _first + 2]

_FINGER_CHAINS = [f"{s}_{f}" for s in SIDES for f in SMPLX_FINGER_OFFSET]
# Groups are shortcuts for several chains (these four are the "parts" the previous scripts called USE_SMPLX_FOR).
GROUPS = {
    "legs": ["left_leg", "right_leg"],
    "arms": ["left_arm", "right_arm"],
    "wrists": ["left_wrist", "right_wrist"],
    "fingers": _FINGER_CHAINS,
    "left_hand": ["left_wrist"] + [f"left_{f}" for f in SMPLX_FINGER_OFFSET],
    "right_hand": ["right_wrist"] + [f"right_{f}" for f in SMPLX_FINGER_OFFSET],
    "hands": ["left_wrist", "right_wrist"] + _FINGER_CHAINS,
    "all": list(CHAINS),
}


def resolve_chains(spec):
    """Chain / group names (a list, a set or a comma-separated string) -> sorted list of SMPL-X joint numbers."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    joints = set()
    for name in spec:
        if name in CHAINS:
            joints.update(CHAINS[name])
        elif name in GROUPS:
            for c in GROUPS[name]:
                joints.update(CHAINS[c])
        else:
            raise ValueError(f"Unknown chain / group {name!r}. Chains: {sorted(CHAINS)}. Groups: {sorted(GROUPS)}.")
    return sorted(joints)


def chain_of_joint(j):
    """Name of the chain a joint belongs to (None if it is in none)."""
    for name, joints in CHAINS.items():
        if j in joints:
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MOVED VERBATIM from inject_animation.py (rig structure, kinematics, pose-bone solving)
# ══════════════════════════════════════════════════════════════════════════════
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _swing(a, b):
    """Minimal rotation matrix taking direction a onto direction b (batched over leading dims)."""
    a = _unit(a)
    b = _unit(b)
    c = np.asarray(np.sum(a * b, axis=-1))
    axis = np.cross(a, b)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    R = np.eye(3) + K + (K @ K) / np.maximum(1.0 + c, 1e-9)[..., None, None]
    anti = c < -0.999999
    if np.any(anti):   # 180 degrees: turn about any axis perpendicular to a
        helper = np.where((np.abs(a[..., 0]) < 0.9)[..., None], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        n = _unit(np.cross(a, helper))
        R180 = 2.0 * n[..., :, None] * n[..., None, :] - np.eye(3)
        R = np.where(anti[..., None, None], R180, R)
    return R


def build_rest_rotation(y_dir, lateral_ref):
    """Rotation matrix with local Y along y_dir and X from lateral_ref projected orthogonal to it."""
    y_dir = y_dir / np.linalg.norm(y_dir)
    ref = lateral_ref / np.linalg.norm(lateral_ref)
    if abs(np.dot(ref, y_dir)) > 0.99:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fallback, y_dir)) > 0.99:
            fallback = np.array([1.0, 0.0, 0.0])
        x_dir = np.cross(fallback, y_dir)
        used_fallback = True
    else:
        x_dir = ref - y_dir * np.dot(ref, y_dir)  # project ref orthogonal to y_dir
        used_fallback = False
    x_dir = x_dir / np.linalg.norm(x_dir)
    z_dir = np.cross(x_dir, y_dir)
    return np.column_stack([x_dir, y_dir, z_dir]), used_fallback


def compute_descendant_counts(children):
    counts = {}

    def count(j):
        if j in counts:
            return counts[j]
        total = sum(1 + count(c) for c in children[j])
        counts[j] = total
        return total

    for j in children:
        count(j)
    return counts


def compute_child_map(parents):
    children = {j: [] for j in range(len(parents))}
    for j, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(j)
    for j in children:
        children[j].sort()
    return children


def reroot_parents(parents, new_root):
    """Returns (new_parents, path) with new_root re-designated as the hierarchy's root by
    reversing the parent chain between it and the current root; other branches are unchanged."""
    new_parents = parents.copy()
    path = [new_root]
    cur = new_root
    while parents[cur] != -1:
        cur = int(parents[cur])
        path.append(cur)
    for i in range(len(path) - 1):
        new_parents[path[i + 1]] = path[i]
    new_parents[new_root] = -1
    return new_parents, path


def compute_rest_orientations(parents, joint_rest_positions, lateral_ref):
    """Per-joint rest rotation (local Y toward the primary child, tie-broken by lowest index)
    and bone length. Identical to smplx_armature_import.py."""
    num_joints = len(parents)
    children = compute_child_map(parents)
    descendant_counts = compute_descendant_counts(children)
    rest = [None] * num_joints
    length = [leaf_bone_length] * num_joints

    order = []
    visited = [False] * num_joints

    def visit(j):
        if visited[j]:
            return
        visited[j] = True
        order.append(j)
        for c in children[j]:
            visit(c)

    for r in [j for j in range(num_joints) if parents[j] < 0]:
        visit(r)

    for j in order:
        kids = children[j]
        if kids:
            eligible = [c for c in kids if c not in NON_TRUNK_CHILDREN]
            if not eligible:
                eligible = kids
            primary = max(eligible, key=lambda c: (descendant_counts[c], -c))
            if primary in (LEFT_HIP_IDX, RIGHT_HIP_IDX) and LEFT_HIP_IDX in kids and RIGHT_HIP_IDX in kids:
                midpoint = (joint_rest_positions[LEFT_HIP_IDX] + joint_rest_positions[RIGHT_HIP_IDX]) / 2.0
                direction = midpoint - joint_rest_positions[j]
            else:
                direction = joint_rest_positions[primary] - joint_rest_positions[j]
            dist = np.linalg.norm(direction)
            if dist < 1e-8:
                direction = (rest[parents[j]] @ np.array([0, 1.0, 0])) if parents[j] >= 0 else np.array([0, 1.0, 0])
                dist = leaf_bone_length
            rest[j], _ = build_rest_rotation(direction, lateral_ref)
            length[j] = max(dist, 1e-4)
        else:
            rest[j] = rest[parents[j]] if parents[j] >= 0 else np.eye(3)
            length[j] = leaf_bone_length
    return rest, length


def build_smplx_hierarchy(parents_in):
    """The parents smplx_armature_import.py gives its bones: re-rooted at ROOT_BONE_IDX, DETACHED_JOINTS cut loose."""
    parents = np.array(parents_in, dtype=int).copy()
    if parents[ROOT_BONE_IDX] == -1:
        path = [ROOT_BONE_IDX]
    else:
        parents, path = reroot_parents(parents, ROOT_BONE_IDX)
    for dj in DETACHED_JOINTS:
        if dj in path:
            raise ValueError(f"DETACHED_JOINTS contains {dj}, already on reroot path {path}.")
        parents[dj] = -1
    return parents


def parents_first_order(parents):
    order, done = [], set()
    while len(order) < len(parents):
        progressed = False
        for j in range(len(parents)):
            if j not in done and (parents[j] < 0 or int(parents[j]) in done):
                order.append(j)
                done.add(j)
                progressed = True
        if not progressed:
            raise ValueError("Cycle in bone hierarchy.")
    return order


def solve_joint_poses(A, t, rest_R, rest_J, parents):
    """Pose-bone location / rotation / scale (Blender 'matrix_basis') that realise the skin matrices A, t.

    Blender, for a bone with parent p (bones use inherit_scale = 'NONE'; a root behaves like a parent that is the
    identity), with W the bone's pose matrix [RW diag(scale) | head] and rel = rest(p)^-1 rest(j):
        W_j.linear = RW_p rel.linear  basis.linear        (the parent's scale does NOT reach the child's axes)
        W_j.head   = W_p.head + W_p.linear (rel.head + rel.linear basis.loc)     (it DOES reach the child's offset)
    basis = T(loc) R diag(scale). The head of every bone is exact. Where the wanted linear part is not a rotation
    times an axis-aligned scale (torso, palm) the closest such one is used (polar decomposition, diagonal of the
    stretch part). Returns loc (F,J,3), rot (F,J,3,3), scale (F,J,3), RW (F,J,3,3: scale-free world rotation),
    Wt (F,J,3: world head).
    """
    n_frames, nj = A.shape[:2]
    loc = np.zeros((n_frames, nj, 3))
    rot = np.tile(np.eye(3), (n_frames, nj, 1, 1))
    scl = np.ones((n_frames, nj, 3))
    RW = np.zeros((n_frames, nj, 3, 3))
    Wt = np.zeros((n_frames, nj, 3))
    I3 = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
    ONE = np.ones((n_frames, 3))
    for j in parents_first_order(parents):
        Rj, Jj = rest_R[j], rest_J[j]
        Wlin = A[:, j] @ Rj
        Wtr = A[:, j] @ Jj + t[:, j]
        p = int(parents[j])
        if p < 0:
            R_rel, rel_t = Rj, Jj
            PR, PS, PT = I3, ONE, np.zeros((n_frames, 3))
        else:
            R_rel = rest_R[p].T @ Rj
            rel_t = rest_R[p].T @ (Jj - rest_J[p])
            PR, PS, PT = RW[:, p], scl[:, p], Wt[:, p]
        PRt = np.swapaxes(PR, -1, -2)
        B = R_rel.T @ (PRt @ Wlin)
        d = np.einsum("fab,fb->fa", PRt, Wtr - PT) / PS      # diag(1/scale_p) RW_p^T (head - parent head)
        loc_j = (d - rel_t) @ R_rel                           # R_rel^T (d - rel_t)
        U, _, Vt = np.linalg.svd(B)
        Rm = U @ Vt
        neg = np.linalg.det(Rm) < 0
        if neg.any():
            U[neg, :, 2] *= -1
            Rm = U @ Vt
        P = np.swapaxes(Rm, -1, -2) @ B
        loc[:, j], rot[:, j] = loc_j, Rm
        scl[:, j] = np.stack([P[:, 0, 0], P[:, 1, 1], P[:, 2, 2]], axis=-1)
        RW[:, j] = PR @ R_rel @ Rm
        Wt[:, j] = PT + np.einsum("fab,fb->fa", PR, PS * (rel_t + loc_j @ R_rel.T))
    return loc, rot, scl, RW, Wt


def realized_affines(rot_world, scl, Wt, rest_R, rest_J):
    """The skin matrices the pose bones really produce (rotation x axis-aligned scale), same layout as joint_affines()."""
    A = np.zeros(rot_world.shape)
    t = np.zeros(Wt.shape)
    for j in range(rot_world.shape[1]):
        A[:, j] = (rot_world[:, j] * scl[:, j][:, None, :]) @ rest_R[j].T
        t[:, j] = Wt[:, j] - A[:, j] @ rest_J[j]
    return A, t


def skin_joint_affines(A, t, W, verts, frames=None):
    """Linear blend skinning in numpy with per-joint skin matrices: (len(frames), V, 3)."""
    frames = np.arange(A.shape[0]) if frames is None else np.asarray(frames)
    out = np.zeros((len(frames),) + verts.shape)
    for j in range(A.shape[1]):
        idx = np.where(W[:, j] > 0)[0]
        if len(idx) == 0:
            continue
        y = np.einsum("fab,vb->fva", A[frames, j], verts[idx]) + t[frames, j][:, None, :]
        out[:, idx] += W[idx, j][None, :, None] * y
    return out


def axis_angle_to_matrix_np(v):
    """v: (3,) axis-angle, magnitude = angle in radians. Returns 3x3 rotation matrix."""
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3)
    axis = v / theta
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def apose_local_rotation(j):
    """SMPL-X local rotation of joint j in the A pose (identity except shoulders / elbows); same as the import."""
    arm = radians(ARM_DOWN_DEG)
    bend = radians(ELBOW_BEND_DEG)
    if j == SMPLX_BODY["left"]["sho"]:
        return axis_angle_to_matrix_np(np.array([0.0, 0.0, -arm]))
    if j == SMPLX_BODY["right"]["sho"]:
        return axis_angle_to_matrix_np(np.array([0.0, 0.0, arm]))
    if j == SMPLX_BODY["left"]["elb"]:
        return axis_angle_to_matrix_np(np.array([0.0, -bend, 0.0]))
    if j == SMPLX_BODY["right"]["elb"]:
        return axis_angle_to_matrix_np(np.array([0.0, bend, 0.0]))
    return np.eye(3)


def make_4x4(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def inv_rigid(M):
    Mi = np.eye(4)
    Mi[:3, :3] = M[:3, :3].T
    Mi[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return Mi


def apose_rest(parents_orig, J, rest_R_T, verts, W):
    """The T-pose rig posed once into the A pose and kept as the rest: returns (rest 4x4 per joint, joint positions,
    rest rotations, mesh vertices). Same maths as smplx_armature_import.py (SMPL-X hierarchy FK + linear blend skinning)."""
    nj = len(parents_orig)
    rest_T = [make_4x4(rest_R_T[j], J[j]) for j in range(nj)]
    world = {}

    def get(j):
        if j in world:
            return world[j]
        p = int(parents_orig[j])
        basis = make_4x4(rest_R_T[j].T @ apose_local_rotation(j) @ rest_R_T[j], np.zeros(3))
        world[j] = rest_T[j] @ basis if p < 0 else get(p) @ (inv_rigid(rest_T[p]) @ rest_T[j]) @ basis
        return world[j]

    M = [get(j) for j in range(nj)]
    new_v = np.zeros(verts.shape)
    for j in range(nj):
        S = M[j] @ inv_rigid(rest_T[j])
        new_v += W[:, j:j + 1] * (verts @ S[:3, :3].T + S[:3, 3])
    return M, np.array([m[:3, 3] for m in M]), [m[:3, :3] for m in M], new_v


def _polar_rotation(M):
    """Rotation part (proper, det +1) of (..., 3, 3) matrices: the closest rotation to M."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    if neg.any():
        U[neg, :, 2] *= -1
        R = U @ Vt
    return R


def _quat_wxyz(R):
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions (w, x, y, z) with w >= 0."""
    w = np.sqrt(np.maximum(0.0, 1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])) / 2.0
    x = np.sqrt(np.maximum(0.0, 1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2.0
    y = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2.0
    z = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2.0
    x = np.copysign(x, R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(y, R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(z, R[..., 1, 0] - R[..., 0, 1])
    return np.stack([w, x, y, z], axis=-1)


def axis_angle_to_matrix_batch(v):
    """v: (..., 3) axis-angle (magnitude = angle in radians) -> (..., 3, 3) rotation matrices."""
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.maximum(theta, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    th = theta[..., None]
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def smplx_local_rotations(pose, rig, nj):
    """(F,J,3,3) SMPL-X local rotations from the pose parameters, laid out like smplx_armature_import.py:
    joint 0 = global_orient, 1..21 = body_pose, 22..24 (jaw, eyes) = identity, 25..39 / 40..54 = hand_pose + hand_mean."""
    n_frames = pose["global_orient"].shape[0]
    aa = np.zeros((n_frames, nj, 3))
    aa[:, 0] = pose["global_orient"]
    aa[:, 1:22] = pose["body_pose"]
    aa[:, 25:40] = pose["left_hand_pose"] + np.asarray(rig["left_hand_mean"], dtype=float).reshape(15, 3)
    aa[:, 40:55] = pose["right_hand_pose"] + np.asarray(rig["right_hand_mean"], dtype=float).reshape(15, 3)
    return axis_angle_to_matrix_batch(aa)


def smplx_pose_world(parents_orig, J_T, rest_R_T, local_rot):
    """World rotation (F,J,3,3) and position (F,J,3) of every joint: forward kinematics down the ORIGINAL SMPL-X hierarchy
    from the T-pose rest, with the same maths as smplx_armature_import.py / apose_rest() (basis = R_rest^T R_local R_rest)."""
    n_frames, nj = local_rot.shape[:2]
    Rw = np.zeros((n_frames, nj, 3, 3))
    Tw = np.zeros((n_frames, nj, 3))
    for j in parents_first_order(parents_orig):
        basis = rest_R_T[j].T @ local_rot[:, j] @ rest_R_T[j]
        p = int(parents_orig[j])
        if p < 0:
            Rw[:, j] = rest_R_T[j] @ basis
            Tw[:, j] = J_T[j]
        else:
            Rw[:, j] = Rw[:, p] @ (rest_R_T[p].T @ rest_R_T[j]) @ basis
            Tw[:, j] = Tw[:, p] + Rw[:, p] @ (rest_R_T[p].T @ (J_T[j] - J_T[p]))
    return Rw, Tw


# ══════════════════════════════════════════════════════════════════════════════
#  NEW IN THE REFACTOR: rig object, plain-fit skin matrices, channel file
# ══════════════════════════════════════════════════════════════════════════════
class Rig:
    """Rest structure of the armature, computed once from armature_rig_metadata.npz.

    J, rest_R, verts : the rest the Blender bones / mesh are built at (A pose when APOSE_REST, else T pose)
    J_T, rest_R_T    : the T pose the SMPL-X pose parameters are defined against
    parents_orig     : the true SMPL-X hierarchy;  parents_smplx / parents_bl : the Blender bone hierarchy
    """


def build_rig(meta, apose=None, keep_hierarchy=None):
    """Rest structure from the rig metadata (an npz or dict with parents, lbs_weights, v_template, joint_rest_positions,
    left_hand_mean, right_hand_mean and optionally faces). Same maths the old importers ran inline."""
    apose = APOSE_REST if apose is None else apose
    keep_hierarchy = KEEP_SMPLX_HIERARCHY if keep_hierarchy is None else keep_hierarchy
    r = Rig()
    J = np.asarray(meta["joint_rest_positions"], dtype=float)
    W = np.asarray(meta["lbs_weights"], dtype=float)
    verts = np.asarray(meta["v_template"], dtype=float)
    r.parents_orig = np.asarray(meta["parents"]).astype(int)
    r.nj = int(r.parents_orig.shape[0])
    if r.nj != NUM_JOINTS or J.shape[0] < NUM_JOINTS or W.shape[1] < NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} SMPL-X joints in the rig metadata, got parents {r.parents_orig.shape}, "
                         f"joint_rest_positions {J.shape}, lbs_weights {W.shape}.")
    J, r.W = J[:r.nj], W[:, :r.nj]
    r.faces = np.asarray(meta["faces"]).astype(int) if "faces" in getattr(meta, "files", meta) else None
    r.left_hand_mean = np.asarray(meta["left_hand_mean"], dtype=float).reshape(45)
    r.right_hand_mean = np.asarray(meta["right_hand_mean"], dtype=float).reshape(45)
    r.parents_smplx = build_smplx_hierarchy(r.parents_orig)
    lateral_ref = J[RIGHT_HIP_IDX] - J[LEFT_HIP_IDX]
    rest_R, bone_len = compute_rest_orientations(r.parents_smplx, J, lateral_ref)
    r.J_T, r.rest_R_T = J.copy(), np.asarray(rest_R)
    r.v_template = verts
    if apose:
        _, J, rest_R, verts = apose_rest(r.parents_orig, J, r.rest_R_T, verts, r.W)
    r.J, r.rest_R, r.verts = np.asarray(J), np.asarray(rest_R), verts
    r.bone_len = list(bone_len)
    r.parents_bl = r.parents_smplx if keep_hierarchy else np.full(r.nj, -1)
    r.apose = bool(apose)
    r.bone_names = [f"joint_{j:02d}" for j in range(r.nj)]
    return r


def load_pose_params(path):
    """The SMPL-X fit's parameter file (armature_pose_params[_smoothed].npz) -> dict of float arrays."""
    P = np.load(path)
    return dict(
        global_orient=np.asarray(P["global_orient"], dtype=float).reshape(-1, 3),
        body_pose=np.asarray(P["body_pose"], dtype=float).reshape(-1, 21, 3),
        left_hand_pose=np.asarray(P["left_hand_pose"], dtype=float).reshape(-1, 15, 3),
        right_hand_pose=np.asarray(P["right_hand_pose"], dtype=float).reshape(-1, 15, 3),
        transl=np.asarray(P["transl"], dtype=float).reshape(-1, 3),
    )


def smplx_skin_matrices(rig, pose):
    """Skin matrices (x -> A x + t, written against the bones' rest) of all joints from SMPL-X pose parameters, plus the
    joint positions. A (F,J,3,3), t (F,J,3), pos (F,J,3)."""
    local = smplx_local_rotations(pose, dict(left_hand_mean=rig.left_hand_mean, right_hand_mean=rig.right_hand_mean), rig.nj)
    Rw, pos = smplx_pose_world(rig.parents_orig, rig.J_T, rig.rest_R_T, local)
    A = np.einsum("fjab,jcb->fjac", Rw, rig.rest_R)       # A = R_world R_rest^T
    t = pos - np.einsum("fjab,jb->fja", A, rig.J)         # so that the joint head is the SMPL-X joint position
    return A, t, pos


# ── channel solving ───────────────────────────────────────────────────────────
def matrix_to_quat(R):
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions (w, x, y, z), w >= 0. Numerically stable for every
    rotation (the largest component is computed first); _quat_wxyz() above loses ~1e-8 near 180 degrees / small components."""
    c = np.stack([1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2],
                  1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2]], axis=-1)
    k = c.argmax(axis=-1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(c, k[..., None], axis=-1)[..., 0], 1e-300))      # 4 * the largest component
    d21, d02, d10 = R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]
    s01, s02, s12 = R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1]
    cand = [np.stack([s / 4, d21 / s, d02 / s, d10 / s], -1), np.stack([d21 / s, s / 4, s01 / s, s02 / s], -1),
            np.stack([d02 / s, s01 / s, s / 4, s12 / s], -1), np.stack([d10 / s, s02 / s, s12 / s, s / 4], -1)]
    q = np.where((k == 0)[..., None], cand[0], np.where((k == 1)[..., None], cand[1], np.where((k == 2)[..., None], cand[2], cand[3])))
    q = np.where(q[..., :1] < 0, -q, q)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)



SOURCE_SMPLX, SOURCE_RAW, SOURCE_ADJUSTED = 0, 1, 2
SOURCE_NAMES = {SOURCE_SMPLX: "smplx", SOURCE_RAW: "raw", SOURCE_ADJUSTED: "adjusted"}
CHANNEL_FORMAT_VERSION = 1


def solve_channels(rig, A, t, transl, source=None):
    """Skin matrices -> the channel dict (Blender pose-bone location / rotation / scale for every bone).
    quat is (w, x, y, z), w >= 0, NOT made continuous over time (the importer does that, like it always did)."""
    loc, rot, scl, RW, Wt = solve_joint_poses(A, t, rig.rest_R, rig.J, rig.parents_bl)
    n_frames = A.shape[0]
    return dict(
        loc=loc, quat=matrix_to_quat(rot), scale=scl, transl=np.asarray(transl, dtype=float).reshape(n_frames, 3),
        source=np.zeros(rig.nj, dtype=np.int8) if source is None else np.asarray(source, dtype=np.int8),
        heads=Wt, world_rot=RW)


def channels_from_params(rig, pose):
    """A plain SMPL-X fit (rotations only, fixed bone lengths) -> channels. Joints whose Blender parent is not their
    SMPL-X parent (re-rooted spine chain, detached neck / collars) get the location the reversed edge needs."""
    A, t, _ = smplx_skin_matrices(rig, pose)
    return solve_channels(rig, A, t, pose["transl"])


def quat_to_matrix(q):
    """(..., 4) quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    return R


def channels_to_pose(rig, loc, quat, scale):
    """What the channels produce in Blender (inherit_scale = None): world head positions (F,J,3) and scale-free world
    rotations (F,J,3,3). The inverse of solve_joint_poses; used to check a channel file against the intended pose."""
    rot = quat_to_matrix(quat)
    n_frames = rot.shape[0]
    RW = np.zeros((n_frames, rig.nj, 3, 3))
    Wt = np.zeros((n_frames, rig.nj, 3))
    I3 = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
    for j in parents_first_order(rig.parents_bl):
        Rj, Jj = rig.rest_R[j], rig.J[j]
        p = int(rig.parents_bl[j])
        if p < 0:
            R_rel, rel_t, PR, PS, PT = Rj, Jj, I3, np.ones((n_frames, 3)), np.zeros((n_frames, 3))
        else:
            R_rel = rig.rest_R[p].T @ Rj
            rel_t = rig.rest_R[p].T @ (Jj - rig.J[p])
            PR, PS, PT = RW[:, p], scale[:, p], Wt[:, p]
        RW[:, j] = PR @ R_rel @ rot[:, j]
        Wt[:, j] = PT + np.einsum("fab,fb->fa", PR, PS * (rel_t + loc[:, j] @ R_rel.T))
    return Wt, RW


# ── channel file ──────────────────────────────────────────────────────────────
def save_channels(path, rig, ch, meta=None):
    """Writes the CHANNEL FILE, the one pose format the importer reads. Keys:
        format_version  1
        loc    (F,J,3)  pose-bone location  (Blender matrix_basis, in the parent's bone space)
        quat   (F,J,4)  pose-bone rotation, (w, x, y, z)
        scale  (F,J,3)  pose-bone scale (all ones for a plain SMPL-X fit)
        transl (F,3)    root translation the armature OBJECT carries (SMPL-X 'transl')
        source (J,)     0 = SMPL-X fit, 1 = raw (keypoint-driven), 2 = SMPL-X bone adjusted to meet a raw one
        parents_bl (J,) Blender bone parents;  rest_J (J,3), rest_R (J,3,3): the rest the channels were solved against
        meta_json       settings and fit residuals that produced the file (free text for humans)
    """
    np.savez_compressed(
        path, format_version=np.int32(CHANNEL_FORMAT_VERSION), loc=ch["loc"], quat=ch["quat"], scale=ch["scale"],
        transl=ch["transl"], source=ch["source"].astype(np.int8), parents_bl=rig.parents_bl.astype(np.int32),
        rest_J=rig.J, rest_R=rig.rest_R, meta_json=np.array(json.dumps(meta or {}, default=str)))


def is_channel_file(npz):
    return all(k in npz.files for k in ("loc", "quat", "scale", "transl", "source"))


def load_channels(path, rig=None):
    """Reads a channel file; with `rig`, also checks that the rest it was solved against equals this rig's rest."""
    z = np.load(path, allow_pickle=False)
    if not is_channel_file(z):
        raise ValueError(f"{path} is not a channel file (keys: {z.files}).")
    if int(z["format_version"]) != CHANNEL_FORMAT_VERSION:
        raise ValueError(f"{path}: channel format {int(z['format_version'])}, this module reads {CHANNEL_FORMAT_VERSION}.")
    ch = dict(loc=z["loc"], quat=z["quat"], scale=z["scale"], transl=z["transl"], source=z["source"].astype(int),
              parents_bl=z["parents_bl"], rest_J=z["rest_J"], rest_R=z["rest_R"], meta=json.loads(str(z["meta_json"])))
    if rig is not None:
        if not np.array_equal(ch["parents_bl"], rig.parents_bl):
            raise ValueError(f"{path}: bone hierarchy differs from the rig's (was the file made with other rest settings?).")
        dJ = float(np.abs(ch["rest_J"] - rig.J).max())
        dR = float(np.abs(ch["rest_R"] - rig.rest_R).max())
        if dJ > 1e-6 or dR > 1e-6:
            raise ValueError(f"{path}: rest pose differs from the rig's (joints {dJ:.2e} m, axes {dR:.2e}); the channels were solved "
                             f"against another rest (APOSE_REST / ARM_DOWN_DEG / rig metadata).")
    return ch
