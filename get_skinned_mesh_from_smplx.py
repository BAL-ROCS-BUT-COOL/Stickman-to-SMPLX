import torch
import smplx
import numpy as np
import argparse
import os
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as ScipyRotation

from weights import load_fit_weights

JOINT_WEIGHTS = None

FIT_ITER_STAGE1 = 100
FIT_ITER_STAGE2 = 100
FIT_ITER_FIRST_FRAME = (200, 400)
FIT_LBFGS_ITER = 50
FIT_BACKFILL_FRAMES = 30
POSE_REG_WEIGHT = 1e-4
HAND_POSE_REG_WEIGHT = 1e-6
SPINE_LOCAL_INDICES = [2, 5, 8]
SPINE_TWIST_AXIS = np.array([0.0, 1.0, 0.0])
LOCK_SPINE_TWIST = False


OFF = ("off", None)
STANDARD = ("standard", None)


def hz(cutoff_hz):
    return ("hz", float(cutoff_hz))


def frames(n):
    return ("frames", int(n))


HIP_SMOOTHING = STANDARD
LEG_SMOOTHING = STANDARD
SPINE_SMOOTHING = STANDARD
HEAD_SMOOTHING = STANDARD
ARM_SMOOTHING = STANDARD
HAND_SMOOTHING = STANDARD
TRANSL_SMOOTHING = STANDARD
BONE_SMOOTHING = {
    "pelvis":         HIP_SMOOTHING,
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

BODY_POSE_BONES = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
FINGER_ORDER = ["index", "middle", "pinky", "ring", "thumb"]


def remove_twist_(body_pose, joint_local_indices, twist_axis=SPINE_TWIST_AXIS):
    """In-place: zero the rotation-about-twist_axis ("twist") component of"""
    axis = twist_axis / np.linalg.norm(twist_axis)
    with torch.no_grad():
        flat = body_pose.view(-1, 21, 3)
        arr = flat.detach().cpu().numpy().copy()
        for b in range(arr.shape[0]):
            for j in joint_local_indices:
                rot = ScipyRotation.from_rotvec(arr[b, j])
                quat = rot.as_quat()
                v, w = quat[:3], quat[3]
                twist_v = axis * np.dot(v, axis)
                twist_quat = np.array([*twist_v, w])
                norm = np.linalg.norm(twist_quat)
                twist_quat = twist_quat / norm if norm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
                swing = rot * ScipyRotation.from_quat(twist_quat).inv()
                arr[b, j] = swing.as_rotvec()
        flat.copy_(torch.as_tensor(arr, dtype=body_pose.dtype, device=body_pose.device).view(flat.shape))


def lock_spine_twist_(body_pose):
    """Convenience wrapper: applies remove_twist_ to the spine joints only,"""
    if LOCK_SPINE_TWIST:
        remove_twist_(body_pose, SPINE_LOCAL_INDICES)


def wrap_axis_angle_np(arr):
    """Numpy equivalent of wrap_axis_angle_, for post-processing already-"""
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
    """arr: (F, 3) axis-angle of ONE joint over time -> the same rotations as one continuous curve."""
    if arr.shape[0] < 2:
        return arr
    quat = ScipyRotation.from_rotvec(arr).as_quat()
    for t in range(1, quat.shape[0]):
        if np.dot(quat[t], quat[t - 1]) < 0.0:
            quat[t] = -quat[t]
    vec, w = quat[:, :3], quat[:, 3]
    norm = np.linalg.norm(vec, axis=1)
    angle = 2.0 * np.arctan2(norm, w)
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
    """data: (F, N). Applies one smoothing setting (OFF / STANDARD / hz(x) /"""
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
        return data
    return smooth_time(data, cutoff, fps)


def smooth_pose_per_bone(stacked, standard_cutoff_hz, fps):
    """Smooths every bone with its own setting from BONE_SMOOTHING /"""
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
    """In-place: rewrites each 3-vector in param (axis-angle, magnitude ="""
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
    """Fits SMPL-X pose parameters (not baked vertices) to partial_joints_np"""
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

    joint_weights = JOINT_WEIGHTS if JOINT_WEIGHTS is not None else load_fit_weights()
    weights = torch.as_tensor(joint_weights, dtype=torch.float32, device=device)[: len(valid_mask)]

    def weighted_mse_loss(pred_joints, target_joints, valid_mask, weights):
        diff = (pred_joints - target_joints) ** 2
        weighted = diff * weights.unsqueeze(1)
        valid_rows = weighted[valid_mask]
        if valid_rows.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return valid_rows.mean()

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
    """One-time (not per-frame) extraction of everything needed to build a"""
    v_template = model.v_template.detach().cpu().numpy()
    J_regressor = model.J_regressor.detach().cpu().numpy()
    return {
        "parents": model.parents.detach().cpu().numpy(),
        "lbs_weights": model.lbs_weights.detach().cpu().numpy(),
        "v_template": v_template,
        "joint_rest_positions": J_regressor @ v_template,
        "faces": model.faces,
        "left_hand_mean": model.left_hand_mean.detach().cpu().numpy(),
        "right_hand_mean": model.right_hand_mean.detach().cpu().numpy(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SMPL-X pose parameters (not baked meshes) from partial joints")
    parser.add_argument("--output-dir", type=str, default="data", help="Base directory: smplx_joints.npy is read from here, the results go to <output-dir>/<out-subdir>/")
    parser.add_argument("--out-subdir", type=str, default="skinned_smplx", help="Folder inside --output-dir that receives the pose parameters and the rig metadata (default: skinned_smplx)")
    parser.add_argument("--joints", type=str, default="smplx_joints.npy")
    parser.add_argument("--model", type=str, default="models")
    parser.add_argument("--weights", type=str, default=None, help="Joint weight config JSON (default: config/fit_weights.json)")
    parser.add_argument("--out-poses", type=str, default="armature_pose_params.npz")
    parser.add_argument("--out-rig", type=str, default="armature_rig_metadata.npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cutoff-hz", type=float, default=3.0, help="Cutoff (Hz) of the STANDARD low-pass smoothing amount. Per-bone settings are in BONE_SMOOTHING etc. at the top of the script.")
    parser.add_argument("--backfill-frames", type=int, default=None, help=f"Re-fit this many frames at the start of the clip backwards, from the settled solution (default {FIT_BACKFILL_FRAMES}, FIT_BACKFILL_FRAMES at the top of the script; 0 = off)")
    parser.add_argument("--iter-stage1", type=int, default=None, help=f"Stage 1 (Adam) iterations per warm-started frame (default {FIT_ITER_STAGE1}, FIT_ITER_STAGE1 at the top of the script)")
    parser.add_argument("--iter-stage2", type=int, default=None, help=f"Stage 2 (Adam) iterations per warm-started frame (default {FIT_ITER_STAGE2}, FIT_ITER_STAGE2 at the top of the script)")
    args = parser.parse_args()
    JOINT_WEIGHTS = load_fit_weights(args.weights)
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
    out_dir = os.path.join(args.output_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    args.out_poses = os.path.join(out_dir, args.out_poses)
    args.out_rig = os.path.join(out_dir, args.out_rig)

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
    prev_result = None
    fitted = []
    for i in range(len(partial_joints)):
        print(f"\n=== Processing frame {i+1}/{len(partial_joints)} ===")
        result = fit_pose_to_partial_joints(partial_joints[i], model, device, init=prev_result)
        for k in keys:
            all_params[k].append(result[k])
        prev_result = result
        fitted.append(result)

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
    stacked["betas"] = np.zeros(10, dtype=np.float32)
    np.savez(args.out_poses, **stacked)
    print(f"\nSaved {len(partial_joints)} frames of pose parameters -> {args.out_poses}")
    for k, v in stacked.items():
        print(f"  {k}: {v.shape}")

    smoothed = smooth_pose_per_bone(stacked, args.cutoff_hz, args.fps)

    folder, fname = os.path.split(args.out_poses)
    name, ext = os.path.splitext(fname)
    smoothed_path = os.path.join(folder, f"{name}_smoothed{ext}")
    np.savez(smoothed_path, **smoothed)
    print(f"Saved smoothed pose parameters -> {smoothed_path}")
