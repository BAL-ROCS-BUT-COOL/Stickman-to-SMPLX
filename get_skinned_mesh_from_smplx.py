import torch
import smplx
import numpy as np
import argparse
import os
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as ScipyRotation

finger_indices = [
        25, 26, 27, 67,  # left index
        28, 29, 30, 68,  # left middle
        31, 32, 33, 70,  # left pinky
        34, 35, 36, 69, # left ring
        37, 38, 39, 66, # left thumb
        40, 41, 42, 72, # right index
        43, 44, 45, 73, # right middle
        46, 47, 48, 75,  # right pinky
        49, 50, 51, 74, # right ring
        52, 53, 54, 71, # right thumb
        20, 21, # both wrists
        # 18,19 #include elbows
    ]

# --- Optimiser iteration counts (per frame) ---
# Every frame is fitted in three steps: Stage 1 (Adam, body only), Stage 2 (Adam,
# body + hands) and a final L-BFGS polish. These are the iteration counts.
#
# FIT_ITER_STAGE1 / FIT_ITER_STAGE2: for every frame that is warm-started from the
# previous frame's result (all frames except the first). The logged loss of a
# typical frame is already close to its final value after ~100 (Stage 1) and
# ~100 (Stage 2) iterations, so lowering these saves a lot of time. Defaults are
# the original values. Can be overridden per run with --iter-stage1 / --iter-stage2
# (the launcher passes them when FIT_ITER_STAGE1 / FIT_ITER_STAGE2 are set).
FIT_ITER_STAGE1 = 100
FIT_ITER_STAGE2 = 100
# The very first frame starts from zeros (no previous frame), so it keeps its own,
# full counts: (stage 1, stage 2).
FIT_ITER_FIRST_FRAME = (200, 400)
# Max iterations of the final L-BFGS polish.
FIT_LBFGS_ITER = 50
# The first frame is fitted from zeros, and a pose the stickman data leaves ambiguous (leg twist, an upside-down body) can settle
# in a different solution than the one the rest of the clip then follows, so the first frames jump in rotation although the joint
# positions barely move. After the normal pass, this many frames from the start are fitted AGAIN, backwards, each warm-started from
# the frame after it (which already sits in the clip's solution). 0 = off. Costs this many extra frames of fit time.
FIT_BACKFILL_FRAMES = 30

# Discourages implausible rotation magnitudes in weakly-constrained joints
# (e.g. the neck). 1e-6 is too weak to have any real effect -- this needs
# to actually compete with position loss in genuinely flat directions.
# Starting guess, not a derived optimum; tune by inspecting the result.
POSE_REG_WEIGHT = 1e-4

# Fingers have dense position constraints (finger_indices are 5x
# upweighted) and don't share the neck's near-zero-gradient degeneracy --
# a much smaller weight here avoids pulling legitimate finger curl back
# toward rest just because POSE_REG_WEIGHT above needed to be strong.
HAND_POSE_REG_WEIGHT = 1e-6

# joint_mapping (joints.py) gives us position targets for the shoulders,
# elbows, wrists and neck, but none for spine1/spine2/spine3 or the
# collars. Forward kinematics still lets shoulder/neck position error
# flow back into these joints' rotation gradients -- but rotation *about
# a spine joint's own long (vertical) axis* barely moves any downstream
# joint position at all. That's a near-zero-gradient direction: Adam and
# LBFGS can wander arbitrarily far along it for free, and because every
# frame is solved from scratch (see fit_pose_to_partial_joints' `init`
# arg) each frame can wander to a *different* twist value -- which is
# what shows up as the spine appearing to snap through a huge, random
# twist from one frame to the next.
#
# Local (0-based, within the 21-joint body_pose vector) indices of the
# three spine joints in the standard SMPL/SMPL-X kinematic tree:
# spine1 = body joint 3, spine2 = body joint 6, spine3 = body joint 9.
SPINE_LOCAL_INDICES = [2, 5, 8]

# All SMPL-X joint-local rotation frames coincide with the world frame at
# the rest (zero) pose, so "twist" (rotation about the spine's own long
# axis) is approximately rotation about world/local +Y for a roughly
# upright figure. This is an approximation -- it degrades for very bent
# or heavily-rotated poses -- but it's more than good enough for
# suppressing the runaway-twist artifact.
SPINE_TWIST_AXIS = np.array([0.0, 1.0, 0.0])

# Set True to remove spine twist entirely (the "cheat" fix): after every
# optimizer step, the twist component of spine1/spine2/spine3 is
# projected out, leaving natural forward/back and side bending intact.
# Set False to fall back to the plain regularization-only behavior.
#
# Currently OFF: testing whether per-frame warm-starting (see `init` in
# fit_pose_to_partial_joints) is enough on its own to stop the runaway
# twist before reaching for the harder lock. Flip back to True any time.
LOCK_SPINE_TWIST = False


# ─── Per-bone temporal smoothing settings ────────────────────────────────────
# Each bone can be smoothed independently. A setting is one of:
#
#   OFF         no smoothing, the raw fitted values are kept
#   STANDARD    zero-phase Butterworth low-pass at the --cutoff_hz command-line
#               value (default 3 Hz): the "standard amount" of smoothing
#   hz(x)       zero-phase Butterworth low-pass at a custom cutoff of x Hz
#               (lower = smoother; e.g. hz(1.5) is stronger than STANDARD)
#   frames(n)   light moving average over +-n neighbouring frames
#               (frames(1) = average of previous, current and next frame).
#               Removes single-frame jitter but keeps fast motion and detail.
#
# Edit the group constants below to change whole body regions at once, or
# override single bones in BONE_SMOOTHING further down.

OFF = ("off", None)
STANDARD = ("standard", None)


def hz(cutoff_hz):
    return ("hz", float(cutoff_hz))


def frames(n):
    return ("frames", int(n))


# --- Group settings (defaults) ---
HIP_SMOOTHING = STANDARD      # pelvis (global orientation) + left/right hip joints
LEG_SMOOTHING = STANDARD      # knees, ankles, feet
SPINE_SMOOTHING = STANDARD    # spine1, spine2, spine3
HEAD_SMOOTHING = STANDARD     # neck + head
ARM_SMOOTHING = STANDARD #frames(1)     # collars, shoulders, elbows, wrists
HAND_SMOOTHING = STANDARD #frames(1)    # all finger joints
TRANSL_SMOOTHING = STANDARD   # root translation (not a bone; moves the whole body)

# --- Per-bone settings for body_pose / global_orient ---
# Every key must be present (a typo or missing bone raises an error).
BONE_SMOOTHING = {
    "pelvis":         HIP_SMOOTHING,     # = global_orient
    "left_hip":       HIP_SMOOTHING,
    "right_hip":      HIP_SMOOTHING,
    "left_knee":      LEG_SMOOTHING,
    "right_knee":     LEG_SMOOTHING,
    "left_ankle":     LEG_SMOOTHING,
    "right_ankle":    LEG_SMOOTHING,
    "left_foot":      LEG_SMOOTHING,
    "right_foot":     LEG_SMOOTHING,
    "spine1":         SPINE_SMOOTHING,
    "spine2":         SPINE_SMOOTHING,
    "spine3":         SPINE_SMOOTHING,
    "neck":           HEAD_SMOOTHING,
    "head":           HEAD_SMOOTHING,
    "left_collar":    ARM_SMOOTHING,
    "right_collar":   ARM_SMOOTHING,
    "left_shoulder":  ARM_SMOOTHING,
    "right_shoulder": ARM_SMOOTHING,
    "left_elbow":     ARM_SMOOTHING,
    "right_elbow":    ARM_SMOOTHING,
    "left_wrist":     ARM_SMOOTHING,
    "right_wrist":    ARM_SMOOTHING,
}

# --- Per-finger settings (each covers all 3 joints of that finger) ---
FINGER_SMOOTHING = {
    "left_index":   HAND_SMOOTHING,
    "left_middle":  HAND_SMOOTHING,
    "left_pinky":   HAND_SMOOTHING,
    "left_ring":    HAND_SMOOTHING,
    "left_thumb":   HAND_SMOOTHING,
    "right_index":  HAND_SMOOTHING,
    "right_middle": HAND_SMOOTHING,
    "right_pinky":  HAND_SMOOTHING,
    "right_ring":   HAND_SMOOTHING,
    "right_thumb":  HAND_SMOOTHING,
}

# Order of the 21 joints inside body_pose (SMPL-X joints 1..21; joint 0, the
# pelvis, is global_orient) and of the 5 fingers inside each hand pose vector.
BODY_POSE_BONES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
FINGER_ORDER = ["index", "middle", "pinky", "ring", "thumb"]


def remove_twist_(body_pose, joint_local_indices, twist_axis=SPINE_TWIST_AXIS):
    """In-place: zero the rotation-about-twist_axis ("twist") component of
    the given joints in body_pose (shape (B, 63)), leaving the
    perpendicular ("swing", i.e. bend) component unchanged.

    Uses the standard swing-twist quaternion decomposition: given a
    rotation's quaternion (v, w), the twist around unit axis t is
    normalize(proj_t(v), w); the swing is rotation * twist^-1. Zeroing
    the twist and recomposing is equivalent to just returning the swing.

    Runs as a numpy/scipy round-trip under torch.no_grad(), exactly like
    wrap_axis_angle_ -- this is a post-step projection, never call it
    inside the loss/forward pass.
    """
    axis = twist_axis / np.linalg.norm(twist_axis)
    with torch.no_grad():
        flat = body_pose.view(-1, 21, 3)
        arr = flat.detach().cpu().numpy().copy()
        for b in range(arr.shape[0]):
            for j in joint_local_indices:
                rot = ScipyRotation.from_rotvec(arr[b, j])
                quat = rot.as_quat()  # (x, y, z, w)
                v, w = quat[:3], quat[3]
                twist_v = axis * np.dot(v, axis)
                twist_quat = np.array([*twist_v, w])
                norm = np.linalg.norm(twist_quat)
                twist_quat = twist_quat / norm if norm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
                swing = rot * ScipyRotation.from_quat(twist_quat).inv()
                arr[b, j] = swing.as_rotvec()
        flat.copy_(torch.as_tensor(arr, dtype=body_pose.dtype, device=body_pose.device).view(flat.shape))


def lock_spine_twist_(body_pose):
    """Convenience wrapper: applies remove_twist_ to the spine joints only,
    gated on the LOCK_SPINE_TWIST flag so it can be disabled in one place.
    """
    if LOCK_SPINE_TWIST:
        remove_twist_(body_pose, SPINE_LOCAL_INDICES)


def wrap_axis_angle_np(arr):
    """Numpy equivalent of wrap_axis_angle_, for post-processing already-
    detached arrays (e.g. after temporal smoothing, which can reintroduce
    magnitudes slightly over pi near antipodal blends). Same guarantee:
    output represents the identical rotation, just with any winding removed.
    """
    flat = arr.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=-1, keepdims=True)
    safe_theta = np.clip(theta, 1e-8, None)
    axis = flat / safe_theta
    theta_mod = np.mod(theta, 2 * np.pi)
    over_pi = theta_mod > np.pi
    theta_short = np.where(over_pi, 2 * np.pi - theta_mod, theta_mod)
    axis_short = np.where(over_pi, -axis, axis)
    return (axis_short * theta_short).reshape(arr.shape)


def unwrap_axis_angle_np(arr):
    """arr: (F, 3) axis-angle of ONE joint over time -> the same rotations as one continuous curve.
    Every rotation has two axis-angle forms (angle a about n == angle 2*pi-a about -n). A joint near 180 degrees, like a pelvis
    that has to hang upside down, flips between the two from frame to frame, and low-pass filtering such a jumping curve
    blends the two forms into a rotation that is nowhere near either of them. The quaternion sign is made continuous over
    time first, so the angle may grow past pi here; wrap_axis_angle_np brings it back to <= pi after the smoothing."""
    if arr.shape[0] < 2:
        return arr
    quat = ScipyRotation.from_rotvec(arr).as_quat()                # (F, 4) x y z w, w >= 0
    for t in range(1, quat.shape[0]):
        if np.dot(quat[t], quat[t - 1]) < 0.0:
            quat[t] = -quat[t]
    vec, w = quat[:, :3], quat[:, 3]
    norm = np.linalg.norm(vec, axis=1)
    angle = 2.0 * np.arctan2(norm, w)                              # 0 .. 2*pi, continuous because quat is
    scale = np.where(norm > 1e-12, angle / np.maximum(norm, 1e-12), 2.0)
    return vec * scale[:, None]


def lowpass_sos(cutoff_hz, fs, order=3):
    nyq = fs * 0.5
    wn = cutoff_hz / nyq
    return butter(order, wn, btype='low', output='sos')


def smooth_time(data, cutoff_hz, fps, order=3):
    """data: (F, N). Zero-phase low-pass filter along the frame axis."""
    sos = lowpass_sos(cutoff_hz, fps, order)
    return sosfiltfilt(sos, data, axis=0)


def smooth_with_setting(data, setting, standard_cutoff_hz, fps):
    """data: (F, N). Applies one smoothing setting (OFF / STANDARD / hz(x) /
    frames(n), see top of file) along the frame axis."""
    kind, value = setting
    if kind == "off":
        return data
    if kind == "standard":
        cutoff = standard_cutoff_hz
    elif kind == "hz":
        cutoff = value
    elif kind == "frames":
        if value <= 0:
            return data
        return uniform_filter1d(data, size=2 * value + 1, axis=0, mode="nearest")
    else:
        raise ValueError(f"Unknown smoothing setting: {setting!r}")
    if cutoff <= 0 or cutoff >= fps * 0.5:
        return data  # cutoff at/above Nyquist: nothing to filter out
    return smooth_time(data, cutoff, fps)


def smooth_pose_per_bone(stacked, standard_cutoff_hz, fps):
    """Smooths every bone with its own setting from BONE_SMOOTHING /
    FINGER_SMOOTHING / TRANSL_SMOOTHING. Returns a dict with the same keys
    as `stacked` (rotations wrapped back to <= pi)."""
    expected = set(BODY_POSE_BONES) | {"pelvis"}
    if set(BONE_SMOOTHING) != expected:
        raise KeyError(f"BONE_SMOOTHING keys mismatch. Missing: {sorted(expected - set(BONE_SMOOTHING))}, "
                       f"unknown: {sorted(set(BONE_SMOOTHING) - expected)}")
    expected_fingers = {f"{s}_{f}" for s in ("left", "right") for f in FINGER_ORDER}
    if set(FINGER_SMOOTHING) != expected_fingers:
        raise KeyError(f"FINGER_SMOOTHING keys mismatch. Missing: {sorted(expected_fingers - set(FINGER_SMOOTHING))}, "
                       f"unknown: {sorted(set(FINGER_SMOOTHING) - expected_fingers)}")

    def run(data, setting):
        return smooth_with_setting(data, setting, standard_cutoff_hz, fps)

    def run_rot(data, setting):
        """Rotation curves (F, 3): made continuous first (see unwrap_axis_angle_np), then smoothed."""
        return run(unwrap_axis_angle_np(data), setting)

    out = {}
    out["global_orient"] = wrap_axis_angle_np(run_rot(stacked["global_orient"], BONE_SMOOTHING["pelvis"]))

    body = stacked["body_pose"].reshape(-1, 21, 3)
    body_out = np.empty_like(body)
    for j, name in enumerate(BODY_POSE_BONES):
        body_out[:, j] = run_rot(body[:, j], BONE_SMOOTHING[name])
    out["body_pose"] = wrap_axis_angle_np(body_out.reshape(-1, 63))

    for side in ("left", "right"):
        hand = stacked[f"{side}_hand_pose"].reshape(-1, 15, 3)
        hand_out = np.empty_like(hand)
        for j in range(15):
            finger = f"{side}_{FINGER_ORDER[j // 3]}"
            hand_out[:, j] = run_rot(hand[:, j], FINGER_SMOOTHING[finger])
        out[f"{side}_hand_pose"] = wrap_axis_angle_np(hand_out.reshape(-1, 45))

    out["transl"] = run(stacked["transl"], TRANSL_SMOOTHING)
    out["betas"] = stacked["betas"]
    return out


def wrap_axis_angle_(param):
    """In-place: rewrites each 3-vector in param (axis-angle, magnitude =
    rotation angle in radians) to its shortest equivalent representation
    (magnitude <= pi). R(theta) == R(theta + 2*pi*k) exactly, so this
    changes nothing about the represented rotation -- it only removes
    hidden extra winding accumulated during optimization. Call after each
    optimizer step, never inside the loss/forward pass.
    """
    with torch.no_grad():
        flat = param.view(-1, 3)
        theta = flat.norm(dim=-1, keepdim=True)
        safe_theta = theta.clamp_min(1e-8)
        axis = flat / safe_theta
        theta_mod = torch.remainder(theta, 2 * torch.pi)
        over_pi = theta_mod > torch.pi
        theta_short = torch.where(over_pi, 2 * torch.pi - theta_mod, theta_mod)
        axis_short = torch.where(over_pi, -axis, axis)
        flat.copy_((axis_short * theta_short).view(-1, 3))


def fit_pose_to_partial_joints(partial_joints_np, model, device, missing_threshold=1e-6, init=None):
    """Fits SMPL-X pose parameters (not baked vertices) to partial_joints_np
    using the same Stage1/Stage2/LBFGS procedure as
    get_mesh_from_3dpoints.py, including axis-angle wrapping. model is a
    shared, already-created smplx model instance (not recreated per frame).

    init: optional dict (e.g. the previous frame's returned result) used to
    warm-start global_orient/body_pose/left_hand_pose/right_hand_pose/transl
    instead of zero-initializing. Every frame is otherwise solved completely
    independently, which -- combined with the near-zero-gradient spine-twist
    direction described above SPINE_LOCAL_INDICES -- lets that twist wander
    to an unrelated value each frame. Warm-starting from a converged
    neighboring frame keeps the optimizer in the same basin instead, which
    removes most of the remaining frame-to-frame flicker even in joints
    LOCK_SPINE_TWIST doesn't touch. Pass None (default) for zero-init, e.g.
    on the first frame of a sequence.

    Returns a dict of detached numpy arrays: global_orient, body_pose,
    left_hand_pose, right_hand_pose, transl, valid_mask.
    """
    partial_joints = torch.tensor(partial_joints_np, dtype=torch.float32, device=device)
    joint_norms = torch.norm(partial_joints, dim=1)
    valid_mask = joint_norms > missing_threshold
    print(f"Valid joints: {valid_mask.sum().item()}/{len(valid_mask)}")

    def init_param(key, shape):
        if init is not None and key in init:
            val = torch.as_tensor(init[key], dtype=torch.float32, device=device).reshape(shape)
        else:
            val = torch.zeros(shape, device=device)
        return val.clone().detach().requires_grad_(True)

    global_orient = init_param("global_orient", (1, 3))
    body_pose = init_param("body_pose", (1, 63))
    betas = torch.zeros((1, 10), device=device, requires_grad=False)
    left_hand_pose = init_param("left_hand_pose", (1, 45))
    right_hand_pose = init_param("right_hand_pose", (1, 45))
    transl = init_param("transl", (1, 3))

    weights = torch.ones(len(valid_mask), device=device)
    for idx in finger_indices:
        if 0 <= idx < len(weights):
            weights[idx] = 5.0

    def weighted_mse_loss(pred_joints, target_joints, valid_mask, weights):
        diff = (pred_joints - target_joints) ** 2
        weighted = diff * weights.unsqueeze(1)
        valid_rows = weighted[valid_mask]
        if valid_rows.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return valid_rows.mean()

    # ---------------- Stage 1: Fit body (no hands) ----------------
    optimizer_stage1 = torch.optim.Adam([global_orient, body_pose, transl], lr=0.02)
    n_iter_stage1, n_iter_stage2 = FIT_ITER_FIRST_FRAME if init is None else (FIT_ITER_STAGE1, FIT_ITER_STAGE2)
    for i in range(n_iter_stage1):
        optimizer_stage1.zero_grad()
        output = model(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl)
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        loss.backward()
        optimizer_stage1.step()
        wrap_axis_angle_(global_orient)
        wrap_axis_angle_(body_pose)
        lock_spine_twist_(body_pose)
        if (i + 1) % 50 == 0 or i == 0 or i == n_iter_stage1 - 1:
            print(f"[Stage1] Iter {i+1}/{n_iter_stage1}  loss={loss.item():.8f}")

    # ---------------- Stage 2: Fit hands (Adam to get close) ----------------
    params_stage2 = [global_orient, body_pose, transl, left_hand_pose, right_hand_pose]
    optimizer_stage2 = torch.optim.Adam(params_stage2, lr=0.01)
    for i in range(n_iter_stage2):
        optimizer_stage2.zero_grad()
        output = model(
            global_orient=global_orient, body_pose=body_pose, betas=betas,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl,
        )
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        reg = POSE_REG_WEIGHT * (global_orient.pow(2).sum() + body_pose.pow(2).sum()) \
            + HAND_POSE_REG_WEIGHT * (left_hand_pose.pow(2).sum() + right_hand_pose.pow(2).sum())
        (loss + reg).backward()
        optimizer_stage2.step()
        wrap_axis_angle_(global_orient)
        wrap_axis_angle_(body_pose)
        lock_spine_twist_(body_pose)
        wrap_axis_angle_(left_hand_pose)
        wrap_axis_angle_(right_hand_pose)
        if (i + 1) % 100 == 0 or i == 0 or i == n_iter_stage2 - 1:
            print(f"[Stage2] Iter {i+1}/{n_iter_stage2}  loss={loss.item():.8f}")

    # ---------------- Final refinement: L-BFGS ----------------
    params_refine = [global_orient, body_pose, transl, left_hand_pose, right_hand_pose]
    for p in params_refine:
        p.requires_grad = True
    optimizer_refine = torch.optim.LBFGS(params_refine, max_iter=FIT_LBFGS_ITER, line_search_fn="strong_wolfe", lr=1.0)

    def closure():
        optimizer_refine.zero_grad()
        output = model(
            global_orient=global_orient, body_pose=body_pose, betas=betas,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl,
        )
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        reg = POSE_REG_WEIGHT * (global_orient.pow(2).sum() + body_pose.pow(2).sum()) \
            + HAND_POSE_REG_WEIGHT * (left_hand_pose.pow(2).sum() + right_hand_pose.pow(2).sum())
        (loss + reg).backward()
        return loss + reg

    try:
        optimizer_refine.step(closure)
    except Exception as e:
        print("LBFGS failed or terminated early:", e)

    # Applied once here, not inside closure() -- see get_mesh_from_3dpoints.py for why.
    wrap_axis_angle_(global_orient)
    wrap_axis_angle_(body_pose)
    lock_spine_twist_(body_pose)
    wrap_axis_angle_(left_hand_pose)
    wrap_axis_angle_(right_hand_pose)

    return {
        "global_orient": global_orient.detach().cpu().numpy()[0],
        "body_pose": body_pose.detach().cpu().numpy()[0],
        "left_hand_pose": left_hand_pose.detach().cpu().numpy()[0],
        "right_hand_pose": right_hand_pose.detach().cpu().numpy()[0],
        "transl": transl.detach().cpu().numpy()[0],
        "valid_mask": valid_mask.cpu().numpy(),
    }


def extract_rig_metadata(model):
    """One-time (not per-frame) extraction of everything needed to build a
    real skinned armature instead of baked shape keys."""
    v_template = model.v_template.detach().cpu().numpy()
    J_regressor = model.J_regressor.detach().cpu().numpy()
    return {
        "parents": model.parents.detach().cpu().numpy(),          # (num_joints,), root = -1
        "lbs_weights": model.lbs_weights.detach().cpu().numpy(),  # (num_vertices, num_joints)
        "v_template": v_template,                                  # (num_vertices, 3), rest pose
        "joint_rest_positions": J_regressor @ v_template,          # (num_joints, 3), rest pose
        "faces": model.faces,                                      # (num_faces, 3)
        # SMPLX.forward() does full_pose += self.pose_mean internally, and
        # the hand portion of pose_mean is a real, nonzero "relaxed curl"
        # baseline (flat_hand_mean defaults to False) -- fitted
        # left_hand_pose/right_hand_pose values are corrections *relative*
        # to this baseline, not absolute rotations. Anything applying these
        # poses outside SMPL-X's own forward() (e.g. driving Blender bones
        # directly) must add this back in, or hands collapse toward "no
        # curl" instead of their actual fitted shape.
        "left_hand_mean": model.left_hand_mean.detach().cpu().numpy(),    # (45,)
        "right_hand_mean": model.right_hand_mean.detach().cpu().numpy(),  # (45,)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SMPL-X pose parameters (not baked meshes) from partial joints")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory name")
    parser.add_argument("--joints", type=str, default="smplx_joints.npy")
    parser.add_argument("--model", type=str, default="models")
    parser.add_argument("--out-poses", type=str, default="armature_pose_params.npz")
    parser.add_argument("--out-rig", type=str, default="armature_rig_metadata.npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cutoff-hz", type=float, default=3.0, help="Cutoff (Hz) of the STANDARD low-pass smoothing amount. Per-bone settings are in BONE_SMOOTHING etc. at the top of the script.")
    parser.add_argument("--backfill-frames", type=int, default=None, help=f"Re-fit this many frames at the start of the clip backwards, from the settled solution (default {FIT_BACKFILL_FRAMES}, FIT_BACKFILL_FRAMES at the top of the script; 0 = off)")
    parser.add_argument("--iter-stage1", type=int, default=None, help=f"Stage 1 (Adam) iterations per warm-started frame (default {FIT_ITER_STAGE1}, FIT_ITER_STAGE1 at the top of the script)")
    parser.add_argument("--iter-stage2", type=int, default=None, help=f"Stage 2 (Adam) iterations per warm-started frame (default {FIT_ITER_STAGE2}, FIT_ITER_STAGE2 at the top of the script)")
    args = parser.parse_args()
    if args.iter_stage1 is not None:
        FIT_ITER_STAGE1 = args.iter_stage1
    if args.iter_stage2 is not None:
        FIT_ITER_STAGE2 = args.iter_stage2
    if args.backfill_frames is not None:
        FIT_BACKFILL_FRAMES = args.backfill_frames
    if FIT_BACKFILL_FRAMES < 0:
        raise ValueError(f"--backfill-frames must be 0 or more, got {FIT_BACKFILL_FRAMES}.")
    if FIT_ITER_STAGE1 < 1 or FIT_ITER_STAGE2 < 1:
        raise ValueError(f"Iteration counts must be at least 1 (stage 1: {FIT_ITER_STAGE1}, stage 2: {FIT_ITER_STAGE2}).")
    print(f"Fit iterations: first frame {FIT_ITER_FIRST_FRAME[0]} + {FIT_ITER_FIRST_FRAME[1]}, other frames {FIT_ITER_STAGE1} + {FIT_ITER_STAGE2}, L-BFGS up to {FIT_LBFGS_ITER}")

    args.joints = os.path.join(args.output_dir, args.joints)
    args.out_poses = os.path.join(args.output_dir, args.out_poses)
    args.out_rig = os.path.join(args.output_dir, args.out_rig)

    if not os.path.exists(args.joints):
        raise FileNotFoundError(f"Joints file not found: {args.joints}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")

    partial_joints = np.load(args.joints)
    print(f"Loaded joints from {args.joints}, shape = {partial_joints.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(
        model_path=args.model, model_type="smplx", gender="male",
        use_pca=False, num_pca_comps=12,
        create_global_orient=True, create_body_pose=True, create_betas=True,
        create_left_hand_pose=True, create_right_hand_pose=True,
        use_face_contour=False,
    ).to(device)

    rig_metadata = extract_rig_metadata(model)
    np.savez(args.out_rig, **rig_metadata)
    print(f"Saved rig metadata -> {args.out_rig}")
    for k, v in rig_metadata.items():
        print(f"  {k}: {np.asarray(v).shape}")

    keys = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]
    all_params = {k: [] for k in keys}
    prev_result = None  # warm-start: see fit_pose_to_partial_joints' `init` arg
    fitted = []
    for i in range(len(partial_joints)):
        print(f"\n=== Processing frame {i+1}/{len(partial_joints)} ===")
        result = fit_pose_to_partial_joints(partial_joints[i], model, device, init=prev_result)
        for k in keys:
            all_params[k].append(result[k])
        prev_result = result
        fitted.append(result)

    # Backfill: see FIT_BACKFILL_FRAMES. Frame K is the settled solution; frames K-1 ... 0 are refitted from it, one after the other.
    n_back = min(FIT_BACKFILL_FRAMES, len(partial_joints) - 1)
    if n_back > 0:
        print(f"\n=== Backfill: re-fitting frames {n_back} down to 1 backwards from the settled solution of frame {n_back + 1} ===")
        prev_result = fitted[n_back]
        for i in range(n_back - 1, -1, -1):
            print(f"\n=== Backfill frame {i+1}/{len(partial_joints)} ===")
            result = fit_pose_to_partial_joints(partial_joints[i], model, device, init=prev_result)
            for k in keys:
                all_params[k][i] = result[k]
            prev_result = result

    stacked = {k: np.stack(v) for k, v in all_params.items()}
    stacked["betas"] = np.zeros(10, dtype=np.float32)  # fixed/unoptimized, see get_mesh_from_3dpoints.py
    np.savez(args.out_poses, **stacked)
    print(f"\nSaved {len(partial_joints)} frames of pose parameters -> {args.out_poses}")
    for k, v in stacked.items():
        print(f"  {k}: {v.shape}")

    # ─── Smooth Outputs with Low-pass Filter ─────────────────────────────────
    # translation smooths directly like any position signal. Rotation
    # components get the same linear filter for consistency with
    # get_mesh_from_3dpoints.py's convention, then a wrap-safety pass,
    # since linearly blending two axis-angle vectors near antipodal
    # orientations isn't a fully correct rotation average (true rotation
    # smoothing needs quaternion SLERP) -- this is an approximation, good
    # for modest frame-to-frame rotation change, not a rigorous fix.
    #
    # How much each bone is smoothed is configured per bone (or per body
    # region) via the constants near the top of this file.
    smoothed = smooth_pose_per_bone(stacked, args.cutoff_hz, args.fps)

    folder, fname = os.path.split(args.out_poses)
    name, ext = os.path.splitext(fname)
    smoothed_path = os.path.join(folder, f"{name}_smoothed{ext}")
    np.savez(smoothed_path, **smoothed)
    print(f"Saved smoothed pose parameters -> {smoothed_path}")
