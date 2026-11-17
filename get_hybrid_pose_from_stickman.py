"""
get_hybrid_pose_from_stickman.py          (fitter B: SMPL-X + RAW chains)

Command-line fitter, numpy only (no Blender, no torch). It writes the CHANNEL FILE that inject_animation.py reads.

Fitter A (get_skinned_mesh_from_smplx.py) is the plain SMPL-X fit: rotations only, fixed bone lengths. This script
starts from that fit and replaces the chains you name with RAW chains: bones driven directly by the (triangulated)
keypoints, so every bone head lands on its keypoint -- typically the hands. Everything else stays exactly the fit.

    python get_hybrid_pose_from_stickman.py --pose-dir <Pose_set> [--raw hands] [--raw-mode exact|rotation]

  --raw         chains / groups (comma separated) that come from the pose data. Chains: torso, left_leg, right_leg,
                left_arm, right_arm, left_wrist, right_wrist, <side>_<finger> (left_index, right_thumb, ...).
                Groups: legs, arms, wrists, fingers, left_hand, right_hand, hands, all.   Default: hands.
                (The old USE_SMPLX_FOR = {"torso", "legs", "arms"} is exactly --raw hands.)
                --raw all needs no SMPL-X fit at all (a pure keypoint-driven rig).
  --raw-mode    exact     every raw bone head sits on its keypoint (location + scale channels; what the old hybrid made)
                rotation  raw bones only ROTATE and keep their template length; only the root of each raw region (the
                          wrist) sits on its keypoint, the rest follows rigidly. No location / scale channels to fight
                          when editing by hand; the price is that finger heads no longer hit noisy keypoints.
  --config      JSON file overriding any setting below (SCALE_MODE, SNAP_SCALE, LOCK_LENGTH_BONES, ...)

Reads (all in --pose-dir):   body_poses_3d.npz, hand_poses_3d.npz, joint_transforms.npy, armature_rig_metadata.npz,
                             and the SMPL-X fit named by import_config.json's "pose_params" (or --pose-params)
Writes:                      hybrid_channels.npz (or --out); add  "pose_channels": "hybrid_channels.npz"  to
                             import_config.json and inject_animation.py picks it up.

Every function between the settings and "NEW" is moved verbatim from inject_animation.py (the numpy half of the old
hybrid importer); the solver itself is unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import smplx_rig_common as rc
from smplx_rig_common import (CHAINS, SIDES, FINGERS, SMPLX_BODY, SMPLX_FINGER_OFFSET, NUM_JOINTS,
                              SOURCE_SMPLX, SOURCE_RAW, SOURCE_ADJUSTED, _log, _unit, _swing, _polar_rotation, _quat_wxyz,
                              realized_affines, skin_joint_affines, resolve_chains)

# ── SETTINGS (each can be overridden with --config <json>) ────────────────────
# Which side is which. Defaults follow joints.py: body[5/7/11/13/15] and the FIRST hand block are the LEFT side.
SWAP_BODY_SIDES = False
SWAP_HAND_SIDES = False

# Chains that come from the pose data instead of the SMPL-X fit (names: see the top of this file).
RAW_CHAINS = ["hands"]
# "exact" | "rotation": see the top of this file.
RAW_MODE = "exact"

# How the bones are scaled to the subject (bone HEADS land exactly on the keypoints in every mode):
#   "none" no scaling (bones only rotate and move) | "static" one constant uniform scale per bone (clip median)
#   "uniform" uniform scale following the measured length every frame | "along_bone" per-frame stretch along the bone
SCALE_MODE = "none"

# SEAM SNAP: where an SMPL-X bone ends at a raw joint (forearm -> raw wrist) the SMPL-X bone is aimed at that joint's
# head and scaled so its end lands on it. SNAP_SCALE "along_bone" | "uniform" | "none".
SNAP_SEAMS = True
SNAP_SCALE = "along_bone"
SNAP_MAX_STRETCH = 0.25
# LOCKED BONE LENGTH: the listed SMPL-X seam bones (18 / 19 = forearms) keep one constant length for the clip; the
# difference is absorbed by their parent (upper arm). LOCK_LENGTH_TO "mean" | "median" | "template".
LOCK_LENGTH_BONES = {18, 19}
LOCK_LENGTH_TO = "mean"
LOCK_PARENT_MAX_STRETCH = 0.25
LOCK_OVERFLOW_MAX_STRETCH = 0.25
# FOREARM / UPPER ARM ROLL: turn the bone about its own axis towards the roll of the next bone down (percent, 0 = off).
FOREARM_ROLL_TO_WRIST_PERCENT = 80.0
FOREARM_ROLL_BONES = {18: 20, 19: 21}
UPPERARM_ROLL_TO_FOREARM_PERCENT = 0.0
UPPERARM_ROLL_BONES = {16: 18, 17: 19}
ROLL_UNWRAP_OVER_TIME = True

RIGID_FEET = True                 # ankles and feet follow the shin as a rigid body anchored at the ankle keypoint
FINGERTIP_BONE_TO_KEYPOINT = True # fingertip leaf bones are as long as the tip itself (display only)
weight_threshold = 1e-3
CHEST_STUB = 0.08                 # driver bookkeeping only (the chest driver has no tail keypoint)
HEAD_STUB = 0.16                  # driver bookkeeping only (the head driver has no tail keypoint)
FALLBACK_TIP_LENGTH = 0.03        # fingertip length if the template has no vertex to measure it from

# ── DATA LAYOUT (same as mapping_stickman_to_smplx.py / animated_stickman.py) ─
BODY_LEN = 17          # COCO-17; body[9] / body[10] (pose-data wrists) are NOT used, the hand-file wrists are
HAND_LEN = 21          # per hand: wrist, then thumb / index / middle / ring / pinky x 4 (the 4th is the tip)
BODY_IDX = {   # COCO indexes of the LEFT / RIGHT limb keypoints
    "left":  {"sho": 5, "elb": 7, "hip": 11, "knee": 13, "ank": 15},
    "right": {"sho": 6, "elb": 8, "hip": 12, "knee": 14, "ank": 16},
}

_SETTING_KEYS = [
    "SWAP_BODY_SIDES", "SWAP_HAND_SIDES", "RAW_CHAINS", "RAW_MODE", "SCALE_MODE", "SNAP_SEAMS", "SNAP_SCALE", "SNAP_MAX_STRETCH",
    "LOCK_LENGTH_BONES", "LOCK_LENGTH_TO", "LOCK_PARENT_MAX_STRETCH", "LOCK_OVERFLOW_MAX_STRETCH",
    "FOREARM_ROLL_TO_WRIST_PERCENT", "FOREARM_ROLL_BONES", "UPPERARM_ROLL_TO_FOREARM_PERCENT", "UPPERARM_ROLL_BONES",
    "ROLL_UNWRAP_OVER_TIME", "RIGID_FEET", "FINGERTIP_BONE_TO_KEYPOINT", "weight_threshold", "CHEST_STUB", "HEAD_STUB",
    "FALLBACK_TIP_LENGTH"]


def apply_settings(overrides):
    """Overrides module settings from a dict (e.g. a --config JSON). Unknown names raise; JSON lists / string keys are
    turned back into the sets / int-keyed dicts the code expects."""
    g = globals()
    for k, v in overrides.items():
        if k not in _SETTING_KEYS:
            raise KeyError(f"Unknown setting {k!r}. Valid: {_SETTING_KEYS}")
        if k == "LOCK_LENGTH_BONES":
            v = {int(x) for x in v}
        elif k in ("FOREARM_ROLL_BONES", "UPPERARM_ROLL_BONES"):
            v = {int(a): int(b) for a, b in dict(v).items()}
        g[k] = v


def settings_snapshot():
    g = globals()
    return {k: (sorted(g[k]) if isinstance(g[k], set) else g[k]) for k in _SETTING_KEYS}


# ══════════════════════════════════════════════════════════════════════════════
#  MOVED VERBATIM from inject_animation.py (the numpy solver)
# ══════════════════════════════════════════════════════════════════════════════
def _vec(kp, pair):
    """Vector from keypoint pair[0] to keypoint pair[1]."""
    return kp[pair[1]] - kp[pair[0]]


def _dist(kp, pair):
    return np.linalg.norm(_vec(kp, pair), axis=-1)


def _frame2(u, v):
    """Rotation matrices (columns X, Y, Z) with Y along u and X taken from v orthogonalised to Y."""
    y = _unit(u)
    x = v - np.sum(v * y, axis=-1, keepdims=True) * y
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    bad = nx < 1e-6
    if np.any(bad):
        alt = np.where((np.abs(y[..., 0]) < 0.9)[..., None], np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        x2 = alt - np.sum(alt * y, axis=-1, keepdims=True) * y
        x = np.where(bad, x2, x)
    x = _unit(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def fill_missing(arr, label):
    """Forward/backward-fills frames where a keypoint is NaN/inf (arr: (F, K, 3))."""
    bad = ~np.isfinite(arr).all(axis=-1)
    if not bad.any():
        return arr
    arr = arr.copy()
    n_frames = arr.shape[0]
    for k in np.where(bad.any(axis=0))[0]:
        good = ~bad[:, k]
        if not good.any():
            raise ValueError(f"{label} keypoint {k} is missing in every frame.")
        idx = np.where(good, np.arange(n_frames), -1)
        idx = np.maximum.accumulate(idx)
        first = np.argmax(good)
        idx[idx < 0] = first
        arr[:, k] = arr[idx, k]
    print(f"[get_hybrid_pose_from_stickman] WARNING: {int(bad.sum())} missing {label} keypoint value(s) filled from neighbouring frames.")
    return arr


def pose_keypoints(body, hand):
    """Named keypoints per frame from the pose files. body: (F,17,3), hand: (F,42,3) -> {name: (F,3)}."""
    kp = {
        "pelvis": (body[:, 11] + body[:, 12]) / 2.0,
        "neck": (body[:, 5] + body[:, 6]) / 2.0,
    }
    for s in SIDES:
        body_side = ({"left": "right", "right": "left"} if SWAP_BODY_SIDES else {"left": "left", "right": "right"})[s]
        hand_side = ({"left": "right", "right": "left"} if SWAP_HAND_SIDES else {"left": "left", "right": "right"})[s]
        for part, idx in BODY_IDX[body_side].items():
            kp[f"{s}_{part}"] = body[:, idx]
        block = 0 if hand_side == "left" else 1          # first hand block = left (joints.py)
        base = HAND_LEN * block
        kp[f"{s}_wrist"] = hand[:, base]
        for f, finger in enumerate(FINGERS):
            for k in range(1, 5):
                kp[f"{s}_{finger}{k}"] = hand[:, base + 1 + 4 * f + (k - 1)]
    return kp


def rest_keypoints(J, verts, W):
    """The same named keypoints, taken from the SMPL-X rest pose. Each value is shape (1,3)."""
    def one(p):
        return np.asarray(p, dtype=float).reshape(1, 3)

    kp = {
        "pelvis": one((J[SMPLX_BODY["left"]["hip"]] + J[SMPLX_BODY["right"]["hip"]]) / 2.0),
        #"pelvis": one(J[0]),
        "neck": one((J[SMPLX_BODY["left"]["sho"]] + J[SMPLX_BODY["right"]["sho"]]) / 2.0),
    }
    for s in SIDES:
        sj = SMPLX_BODY[s]
        for part in ("sho", "elb", "wrist", "hip", "knee", "ank"):
            kp[f"{s}_{part}"] = one(J[sj[part]])
        for finger in FINGERS:
            first = sj["hand"] + SMPLX_FINGER_OFFSET[finger]
            for k in range(1, 4):
                kp[f"{s}_{finger}{k}"] = one(J[first + k - 1])
            # tip: furthest point of the distal bone's skin along the finger direction
            j3, j2 = first + 2, first + 1
            d = J[j3] - J[j2]
            d = d / np.linalg.norm(d)
            cand = np.where(W[:, j3] > 0.5)[0]
            reach = float(((verts[cand] - J[j3]) @ d).max()) if len(cand) else 0.0
            if reach < 1e-3:
                reach = FALLBACK_TIP_LENGTH
            kp[f"{s}_{finger}4"] = one(J[j3] + d * reach)
    return kp


def build_specs():
    """Bone table. Parents come before their children. Names use the SMPL-X side."""
    specs = [
        dict(name="pelvis", kind="abs", head="pelvis", tail="neck",
             u=("pelvis", "neck"), v=("right_hip", "left_hip"),
             scale="box", length=("pelvis", "neck"), width=("right_hip", "left_hip")),
        dict(name="chest", kind="abs", head="neck", tail=None, stub=CHEST_STUB,
             u=("pelvis", "neck"), v=("right_sho", "left_sho"),
             scale="box", length=("pelvis", "neck"), width=("right_sho", "left_sho")),
        dict(name="head", kind="abs", head="neck", tail=None, stub=HEAD_STUB,
             u=("pelvis", "neck"), v=("right_sho", "left_sho"), scale="rigid"),
    ]
    for s in SIDES:
        specs += [
            dict(name=f"upperarm_{s}", kind="chain", parent="chest", head=f"{s}_sho", tail=f"{s}_elb", scale="stretch"),
            dict(name=f"forearm_{s}", kind="chain", parent=f"upperarm_{s}", head=f"{s}_elb", tail=f"{s}_wrist", scale="stretch"),
            dict(name=f"hand_{s}", kind="abs", head=f"{s}_wrist", tail=f"{s}_middle1",
                 u=(f"{s}_wrist", f"{s}_middle1"), v=(f"{s}_pinky1", f"{s}_index1"),
                 scale="box", length=(f"{s}_wrist", f"{s}_middle1"), width=(f"{s}_pinky1", f"{s}_index1")),
            dict(name=f"thigh_{s}", kind="chain", parent="pelvis", head=f"{s}_hip", tail=f"{s}_knee", scale="stretch"),
            dict(name=f"shin_{s}", kind="chain", parent=f"thigh_{s}", head=f"{s}_knee", tail=f"{s}_ank", scale="stretch"),
        ]
        for finger in FINGERS:
            for k in (1, 2, 3):
                specs.append(dict(
                    name=f"{finger}{k}_{s}", kind="chain",
                    parent=f"hand_{s}" if k == 1 else f"{finger}{k - 1}_{s}",
                    head=f"{s}_{finger}{k}", tail=f"{s}_{finger}{k + 1}", scale="stretch"))
    return specs


def build_joint_weights():
    """SMPL-X joint -> [(bone, share)]. Shares of one joint sum to 1."""
    m = {
        0: [("pelvis", 1.0)],
        3: [("pelvis", 2.0 / 3), ("chest", 1.0 / 3)],      # spine1
        6: [("pelvis", 1.0 / 3), ("chest", 2.0 / 3)],      # spine2
        9: [("chest", 1.0)],                               # spine3
        12: [("head", 1.0)], 15: [("head", 1.0)],          # neck, head
        22: [("head", 1.0)], 23: [("head", 1.0)], 24: [("head", 1.0)],   # jaw, eyes
    }
    for s in SIDES:
        sj = SMPLX_BODY[s]
        m[sj["collar"]] = [("chest", 1.0)]
        m[sj["sho"]] = [(f"upperarm_{s}", 1.0)]
        m[sj["elb"]] = [(f"forearm_{s}", 1.0)]
        m[sj["wrist"]] = [(f"hand_{s}", 1.0)]
        m[sj["hip"]] = [(f"thigh_{s}", 1.0)]
        m[sj["knee"]] = [(f"shin_{s}", 1.0)]
        m[sj["ank"]] = [(f"shin_{s}", 1.0)]                # no foot keypoints: foot follows the shin
        m[sj["foot"]] = [(f"shin_{s}", 1.0)]
        for finger in FINGERS:
            for k in (1, 2, 3):
                m[sj["hand"] + SMPLX_FINGER_OFFSET[finger] + k - 1] = [(f"{finger}{k}_{s}", 1.0)]
    return m


def solve_bones(specs, kp_rest, kp_pose):
    """Per bone: rest head / frame / length and, per frame, posed head / frame / scale."""
    n_frames = next(iter(kp_pose.values())).shape[0]
    out = {}
    for sp in specs:
        head, tail = sp["head"], sp["tail"]
        if sp["kind"] == "abs":
            if tail is not None:
                assert sp["u"] == (head, tail), f"{sp['name']}: u must run head->tail so the tail is hit exactly"
            R_rest = _frame2(_vec(kp_rest, sp["u"]), _vec(kp_rest, sp["v"]))[0]
            R_pose = _frame2(_vec(kp_pose, sp["u"]), _vec(kp_pose, sp["v"]))
        else:
            par = out[sp["parent"]]
            u_rest = _vec(kp_rest, (head, tail))[0]
            u_pose = _vec(kp_pose, (head, tail))
            R_rest = _swing(par["R_rest"][:, 1], u_rest) @ par["R_rest"]
            carried = par["R_pose"] @ par["R_rest"].T @ R_rest          # child's rest frame carried by the parent
            R_pose = _swing(carried[..., :, 1], u_pose) @ carried
        length = float(np.linalg.norm(_vec(kp_rest, (head, tail))[0])) if tail is not None else sp["stub"]

        kind = sp["scale"]
        if kind == "rigid":
            sy = sx = np.ones(n_frames)
        elif kind == "stretch":
            sy = _dist(kp_pose, (head, tail)) / _dist(kp_rest, (head, tail))
            sx = sy
        elif kind == "box":     # torso / palm: length, width and (unobservable) depth
            sy = _dist(kp_pose, sp["length"]) / _dist(kp_rest, sp["length"])
            sx = _dist(kp_pose, sp["width"]) / _dist(kp_rest, sp["width"])
        else:
            raise ValueError(kind)
        g = np.sqrt(sx * sy)                       # one uniform factor per frame (= sy for a plain limb / finger bone)
        if kind == "rigid" or SCALE_MODE == "none":
            scale = np.ones((n_frames, 3))
        elif SCALE_MODE == "static":
            scale = np.tile(np.full(3, float(np.median(g))), (n_frames, 1))
        elif SCALE_MODE == "uniform":
            scale = np.stack([g, g, g], axis=-1)
        elif SCALE_MODE == "along_bone":
            scale = np.stack([np.ones_like(sy), sy, np.ones_like(sy)], axis=-1) if kind == "stretch" \
                else np.stack([sx, sy, 0.5 * (sx + sy)], axis=-1)
        else:
            raise ValueError(f"SCALE_MODE must be 'none', 'static', 'uniform' or 'along_bone', got {SCALE_MODE!r}")

        out[sp["name"]] = dict(
            head_rest=kp_rest[head][0], R_rest=R_rest, length=length,
            head_pose=kp_pose[head], R_pose=R_pose, scale=scale, len_ratio=sy)
    return out


def joint_keypoint_names():
    """SMPL-X joint -> name of the keypoint that sits on that joint (only joints that have one)."""
    m = {}
    for s in SIDES:
        sj = SMPLX_BODY[s]
        for part in ("sho", "elb", "wrist", "hip", "knee", "ank"):
            m[sj[part]] = f"{s}_{part}"
        for finger in FINGERS:
            first = sj["hand"] + SMPLX_FINGER_OFFSET[finger]
            for k in (1, 2, 3):
                m[first + k - 1] = f"{s}_{finger}{k}"
    return m


def fingertip_joints():
    """SMPL-X joint of each fingertip leaf bone -> name of its tip keypoint."""
    m = {}
    for s in SIDES:
        for finger in FINGERS:
            first = SMPLX_BODY[s]["hand"] + SMPLX_FINGER_OFFSET[finger]
            m[first + 2] = f"{s}_{finger}4"
    return m


def driver_affines(results):
    """Per driver: the rest -> posed map  x -> A x + t,  A = R_pose diag(scale) R_rest^T.  {name: (A (F,3,3), t (F,3))}"""
    out = {}
    for name, r in results.items():
        A = (r["R_pose"] * r["scale"][:, None, :]) @ r["R_rest"].T
        t = r["head_pose"] - A @ r["head_rest"]
        out[name] = (A, t)
    return out


def joint_affines(drivers, joint_weights, num_joints):
    """Skin matrix of every SMPL-X joint: the share-weighted sum of its drivers' maps.  A (F,J,3,3), t (F,J,3)."""
    n_frames = next(iter(drivers.values()))[0].shape[0]
    A = np.zeros((n_frames, num_joints, 3, 3))
    t = np.zeros((n_frames, num_joints, 3))
    for j, shares in joint_weights.items():
        for bone, share in shares:
            A[:, j] += share * drivers[bone][0]
            t[:, j] += share * drivers[bone][1]
    return A, t


def regularize_joint_affines(A, t, J, mode):
    """For every scale mode except 'along_bone': a joint's skin matrix becomes a rotation times ONE uniform scale
    (1 for 'none'; the mean stretch for 'static' / 'uniform'). Only the blended joints (spine1 / spine2: a mix of the
    pelvis and chest maps) are changed by this, their mix would otherwise shrink along the bending axis and show up as a
    non-uniform bone scale. The posed joint position stays where it was. Modifies A, t in place."""
    if mode == "along_bone":
        return
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    if neg.any():
        U[neg, :, 2] *= -1
        R = U @ Vt
    scale = S.mean(axis=-1) if mode in ("static", "uniform") else np.ones(S.shape[:-1])
    head = np.einsum("fjab,jb->fja", A, J) + t
    A_new = scale[..., None, None] * R
    t_new = head - np.einsum("fjab,jb->fja", A_new, J)
    A[:] = A_new
    t[:] = t_new


def report_length_ratios(results, threshold=0.05, top=8):
    """Logs the bones whose measured length differs from the SMPL-X template's by more than `threshold` (median over the clip)."""
    rows = []
    for name, r in results.items():
        ratio = float(np.median(r["len_ratio"]))
        if abs(ratio - 1.0) > threshold:
            rows.append((abs(ratio - 1.0), name, ratio))
    rows.sort(reverse=True)
    if not rows:
        _log(f"Bone lengths: all within {threshold * 100:.0f}% of the SMPL-X template.")
        return
    _log(f"Bone lengths vs the SMPL-X template (data / template, median over the clip; {len(rows)} bone(s) off by more than "
         f"{threshold * 100:.0f}%, largest first): " + ", ".join(f"{n} {r:.2f}" for _, n, r in rows[:top]))


def pin_ankles(A, t, J, anchors):
    """Ankle and foot joints keep their skin matrix's linear part; the translation is set so the ankle lands on its
    keypoint. Needed when bones are not stretched: the shin's own map then ends short of / beyond the ankle keypoint."""
    for s in SIDES:
        ank, foot = SMPLX_BODY[s]["ank"], SMPLX_BODY[s]["foot"]
        for j in (ank, foot):
            t[:, j] = anchors[s] - np.einsum("fab,b->fa", A[:, j], J[ank])


def make_feet_rigid(A, t, J, anchors=None):
    """Ankle and foot joints: rotation of the shin only, pinned to the ankle keypoint (modifies A, t in place).
    anchors: {side: (F,3)} ankle keypoints; without them the shin's own posed ankle position is used."""
    for s in SIDES:
        ank, foot = SMPLX_BODY[s]["ank"], SMPLX_BODY[s]["foot"]
        U, _, Vt = np.linalg.svd(A[:, ank])
        R = U @ Vt
        neg = np.linalg.det(R) < 0
        if neg.any():
            U[neg, :, 2] *= -1
            R = U @ Vt
        anchor = anchors[s] if anchors is not None else A[:, ank] @ J[ank] + t[:, ank]   # the ankle keypoint
        for j in (ank, foot):
            A[:, j] = R
            t[:, j] = anchor - R @ J[ank]


def find_seams(parents_bl, J, rest_R, smplx_joints):
    """[(p, c)]: SMPL-X-sourced joint p whose bone points at (ends at) the pose-driven joint c, i.e. c is p's child and
    the bone's own Y axis runs along p -> c."""
    smplx = set(smplx_joints)
    seams = []
    for c in range(len(parents_bl)):
        p = int(parents_bl[c])
        if p < 0 or p not in smplx or c in smplx:
            continue
        d = np.asarray(J[c]) - np.asarray(J[p])
        n = float(np.linalg.norm(d))
        if n > 1e-6 and float(np.asarray(rest_R[p])[:, 1] @ (d / n)) > 0.99:
            seams.append((p, c))
    return seams


def _aim_scale(A_b, t_b, j_head, j_tail, target, mode, factor):
    """Swings bone b about its own head (rest position j_head) so its tail (rest position j_tail) points at `target`, then
    scales it by `factor` (F,) along the bone ("along_bone") or on all axes ("uniform"). Returns the new A, t; the head stays."""
    head = np.einsum("fab,b->fa", A_b, j_head) + t_b
    tail = np.einsum("fab,b->fa", A_b, j_tail) + t_b
    v_tgt = target - head
    A_new = _swing(tail - head, v_tgt) @ A_b
    if mode == "along_bone":
        y = _unit(v_tgt)
        A_new = A_new + (factor - 1.0)[:, None, None] * (y[:, :, None] * np.einsum("fa,fab->fb", y, A_new)[:, None, :])
    elif mode == "uniform":
        A_new = factor[:, None, None] * A_new
    return A_new, head - np.einsum("fab,b->fa", A_new, j_head)


def _snap_locked(A, t, J, p, c, q, mode, lock_to, parent_max, overflow_max=0.0):
    """Seam bone p (SMPL-X) -> pose-driven joint c, with p's length LOCKED and its parent bone q absorbing the difference.
    1. p's length L is one constant for the clip (mean / median of |wrist - SMPL-X elbow|, or the template length).
    2. Per frame the elbow is moved to  wrist - L * unit(wrist - SMPL-X elbow)  (p keeps its direction, its end hits c).
    3. q is aimed at that elbow from its unchanged head and stretched along itself (clamped to parent_max) to reach it.
    4. p's head is put on where q's tail really ended up, p is aimed at c and scaled by the constant factor L / template. In
       frames where q hit its limit the remainder overflows into p: its factor is changed just enough to reach c, within
       overflow_max of the locked length.
    Modifies A, t of joints p and q in place; returns a report row."""
    lock_mode = "uniform" if mode == "uniform" else "along_bone"
    head_p = np.einsum("fab,b->fa", A[:, p], J[p]) + t[:, p]
    tail_p = np.einsum("fab,b->fa", A[:, p], J[c]) + t[:, p]
    target = np.einsum("fab,b->fa", A[:, c], J[c]) + t[:, c]
    d = target - head_p
    n_tgt = np.linalg.norm(d, axis=-1)
    n_cur = np.linalg.norm(tail_p - head_p, axis=-1)           # constant (a rotation) for an SMPL-X bone
    if lock_to == "template":
        L = float(np.mean(n_cur))
    elif lock_to == "median":
        L = float(np.median(n_tgt))
    elif lock_to == "mean":
        L = float(np.mean(n_tgt))
    else:
        raise ValueError(f"LOCK_LENGTH_TO must be 'mean', 'median' or 'template', got {lock_to!r}")
    factor_p = np.full(len(n_cur), L) / np.maximum(n_cur, 1e-9)   # one constant per bone (n_cur does not vary)

    # (2) where the elbow has to be for the locked forearm to end on the wrist
    elbow_want = target - _unit(d) * L
    # (3) the parent absorbs it
    head_q = np.einsum("fab,b->fa", A[:, q], J[q]) + t[:, q]
    tail_q = np.einsum("fab,b->fa", A[:, q], J[p]) + t[:, q]
    raw_q = np.linalg.norm(elbow_want - head_q, axis=-1) / np.maximum(np.linalg.norm(tail_q - head_q, axis=-1), 1e-9)
    s_q = np.clip(raw_q, 1.0 - parent_max, 1.0 + parent_max)
    A[:, q], t[:, q] = _aim_scale(A[:, q], t[:, q], J[q], J[p], elbow_want, lock_mode, s_q)
    # (4) the locked bone, from where the parent's tail actually landed
    elbow_act = np.einsum("fab,b->fa", A[:, q], J[p]) + t[:, q]
    A_p = A[:, p]
    t_p = elbow_act - np.einsum("fab,b->fa", A_p, J[p])           # head moved onto the parent's tail
    need = np.linalg.norm(target - elbow_act, axis=-1) / np.maximum(n_cur, 1e-9)   # factor that would end exactly on c
    factor = np.clip(need, factor_p * (1.0 - overflow_max), factor_p * (1.0 + overflow_max)) if overflow_max > 0 else factor_p
    factor = np.where(np.abs(need - factor_p) < 1e-9, factor_p, factor)            # unclamped frames: exactly the locked factor
    A[:, p], t[:, p] = _aim_scale(A_p, t_p, J[p], J[c], target, lock_mode, factor)
    gap_after = np.linalg.norm(target - (np.einsum("fab,b->fa", A[:, p], J[c]) + t[:, p]), axis=-1)
    return dict(kind="locked", p=p, c=c, q=q, gap_before=np.linalg.norm(target - tail_p, axis=-1), gap_after=gap_after,
                scale=factor, length=L, overflow=int(np.sum(np.abs(factor - factor_p) > 1e-9)), template_length=float(np.mean(n_cur)), parent_scale=s_q,
                clamped=int(np.sum(np.abs(s_q - raw_q) > 1e-9)),
                elbow_shift=np.linalg.norm(elbow_act - head_p, axis=-1))


def snap_seams(A, t, J, seams, mode, max_stretch, parents=None, rest_R=None, smplx_joints=(),
               lock_bones=(), lock_to="mean", lock_parent_max=0.25, lock_overflow_max=0.0):
    """Aims each seam bone p at the head of its pose-driven child c (swing about p's head) and scales it per `mode`, so
    its end lands on that head. Modifies A, t of joint p in place. Returns one report row per seam.
    Seam bones listed in `lock_bones` are handled by _snap_locked() instead: constant length, parent bone absorbs the rest."""
    if mode not in ("none", "along_bone", "uniform"):
        raise ValueError(f"SNAP_SCALE must be 'none', 'along_bone' or 'uniform', got {mode!r}")
    smplx = set(smplx_joints)
    seam_bones = {p for p, _ in seams}
    for j in sorted(set(lock_bones) - seam_bones):
        _log(f"WARNING: LOCK_LENGTH_BONES contains joint_{j:02d}, which is not a seam bone (SMPL-X bone ending at a pose-driven "
             f"joint) with the current RAW_CHAINS; it is not locked (and not stretched either).")
    rows = []
    for p, c in seams:
        if p in lock_bones:
            q = int(parents[p]) if parents is not None else -1
            ok = q >= 0 and q in smplx and q not in lock_bones and q not in seam_bones
            if ok:
                dq = np.asarray(J[p]) - np.asarray(J[q])
                nq = float(np.linalg.norm(dq))
                ok = nq > 1e-6 and float(np.asarray(rest_R[q])[:, 1] @ (dq / nq)) > 0.99
            if not ok:
                _log(f"WARNING: joint_{p:02d} is locked but its parent joint_{q:02d} is not an SMPL-X bone ending at it; "
                     f"falling back to the plain seam snap for this bone.")
            else:
                rows.append(_snap_locked(A, t, J, p, c, q, mode, lock_to, lock_parent_max, lock_overflow_max))
                continue
        head = np.einsum("fab,b->fa", A[:, p], J[p]) + t[:, p]
        tail = np.einsum("fab,b->fa", A[:, p], J[c]) + t[:, p]
        target = np.einsum("fab,b->fa", A[:, c], J[c]) + t[:, c]
        v_cur, v_tgt = tail - head, target - head
        n_cur = np.linalg.norm(v_cur, axis=-1)
        n_tgt = np.linalg.norm(v_tgt, axis=-1)
        A_new = _swing(v_cur, v_tgt) @ A[:, p]
        raw = n_tgt / np.maximum(n_cur, 1e-9)
        s = np.clip(raw, 1.0 - max_stretch, 1.0 + max_stretch) if mode != "none" else np.ones_like(raw)
        if mode == "along_bone":
            y = _unit(v_tgt)
            A_new = A_new + (s - 1.0)[:, None, None] * (y[:, :, None] * np.einsum("fa,fab->fb", y, A_new)[:, None, :])
        elif mode == "uniform":
            A_new = s[:, None, None] * A_new
        A[:, p] = A_new
        t[:, p] = head - np.einsum("fab,b->fa", A_new, J[p])
        gap_after = np.linalg.norm(target - (np.einsum("fab,b->fa", A_new, J[c]) + t[:, p]), axis=-1)
        rows.append(dict(kind="snap", p=p, c=c, gap_before=np.linalg.norm(target - tail, axis=-1), gap_after=gap_after, scale=s,
                         clamped=int(np.sum(np.abs(s - raw) > 1e-9)) if mode != "none" else 0))
    return rows


def report_snap(m):
    """Logs what the seam snap did, per seam (gap = distance between the SMPL-X bone's end and the pose-driven joint)."""
    if not m["snap"]:
        return
    for r in m["snap"]:
        gb, ga, sc = r["gap_before"] * 1000, r["gap_after"] * 1000, r["scale"]
        if r["kind"] == "locked":
            ps = r["parent_scale"]
            _log(f"Seam LOCK joint_{r['p']:02d} -> joint_{r['c']:02d}: length locked to {r['length'] * 1000:.1f} mm "
                 f"(template {r['template_length'] * 1000:.1f} mm, factor {np.median(sc):.3f}); parent joint_{r['q']:02d} absorbs it: "
                 f"scale mean {ps.mean():.3f}, range {ps.min():.3f}-{ps.max():.3f}"
                 + (f", limited by LOCK_PARENT_MAX_STRETCH in {r['clamped']} of {len(ps)} frames" if r["clamped"] else "")
                 + (f"; overflow into the locked bone in {r['overflow']} frames (factor {sc.min():.3f}-{sc.max():.3f})" if r["overflow"] else "")
                 + f"; elbow moved {r['elbow_shift'].mean() * 1000:.1f}/{r['elbow_shift'].max() * 1000:.1f} mm (mean/max) from the fit; "
                 f"gap at joint_{r['c']:02d}: {gb.mean():.1f}/{gb.max():.1f} -> {ga.mean():.1f}/{ga.max():.1f} mm")
            continue
        _log(f"Seam snap joint_{r['p']:02d} -> joint_{r['c']:02d}: gap {gb.mean():.1f}/{gb.max():.1f} mm (mean/max) -> "
             f"{ga.mean():.1f}/{ga.max():.1f} mm after; {SNAP_SCALE} scale mean {sc.mean():.3f}, range {sc.min():.3f}-{sc.max():.3f}"
             + (f"; limited by SNAP_MAX_STRETCH in {r['clamped']} of {len(sc)} frames" if r["clamped"] else ""))


def _axis_rotation(axis, angle):
    """Rodrigues: unit axes (F, 3) and angles (F,) -> rotation matrices (F, 3, 3)."""
    K = np.zeros(axis.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def fix_bone_roll(A, t, J, bones, percent, label="forearm"):
    """Turns every bone p about its own axis (its head -> the head of its child c) by `percent` % of the roll the child has relative to it.
    A (F,J,3,3), t (F,J,3) are the skin matrices (rest -> posed); both ends of the bone stay where they are.
    For bone p with child c (forearm -> wrist, upper arm -> forearm):  D = R_c R_p^T  is the child's motion relative to the bone in the world frame
    (identity at rest, whatever the rest orientations of the two bones are); its twist about the bone's posed axis a is  theta = 2 atan2(q_xyz . a, q_w)
    (swing-twist split of D), and the bone is rotated about a by  percent/100 * theta.  A pure bend of the child has no twist, so it is left alone.
    A stretch of the bone along a (seam snap / locked length) commutes with that rotation. Modifies A, t in place, returns report rows.
    Call it child-first (forearm, then upper arm) so that the upper arm aligns with the forearm's already corrected roll."""
    k = float(percent) / 100.0
    rows = []
    for p, c in bones.items():
        head = np.einsum("fab,b->fa", A[:, p], J[p]) + t[:, p]
        tail = np.einsum("fab,b->fa", A[:, p], J[c]) + t[:, p]
        axis = _unit(tail - head)
        D = _polar_rotation(A[:, c]) @ np.swapaxes(_polar_rotation(A[:, p]), -1, -2)
        q = _quat_wxyz(D)
        theta = 2.0 * np.arctan2(np.sum(q[:, 1:] * axis, axis=-1), q[:, 0])        # [-pi, pi] (modulo 360 deg), per frame
        if ROLL_UNWRAP_OVER_TIME and len(theta) > 1:
            theta = np.unwrap(theta)                                               # continuous over time: no 360 deg jump at +-180 deg
        phi = k * theta
        A_new = _axis_rotation(axis, phi) @ A[:, p]
        A[:, p], t[:, p] = A_new, head - np.einsum("fab,b->fa", A_new, J[p])      # the head (elbow) stays exactly where it was
        rows.append(dict(p=p, c=c, twist=theta, applied=phi, label=label, percent=float(percent)))
    return rows


def report_roll(m):
    for r in m.get("roll") or []:
        tw, ap = np.degrees(np.abs(r["twist"])), np.degrees(np.abs(r["applied"]))
        lab = r.get("label", "forearm")
        child = "wrist" if lab == "forearm" else "forearm"
        _log(f"{lab.capitalize()} roll joint_{r['p']:02d} -> {child} joint_{r['c']:02d}: {child} twist relative to the {lab} {tw.mean():.1f} deg mean / "
             f"{tw.max():.1f} deg max; {lab} turned {r.get('percent', 100.0):g} % of it ({ap.mean():.1f} / {ap.max():.1f} deg mean/max)."
             + (f" NOTE: the twist winds past 360 deg ({tw.max() / 360.0:.2f} turns)." if tw.max() > 360.0 else ""))


def transform_keypoints(kp, M):
    """Applies per-frame 4x4 transforms M (F,4,4) to every keypoint {name: (F,3)}."""
    R, tt = M[:, :3, :3], M[:, :3, 3]
    return {k: np.einsum("fij,fj->fi", R, v) + tt for k, v in kp.items()}


def diagnose_motion(m, max_frames=24):
    """Self-checks. Returns (worst head error [m], worst mesh deviation from exact skinning [m])."""
    names = joint_keypoint_names()
    head_err = 0.0
    for j, key in names.items():
        if j in m["smplx_joints"]:          # taken from the SMPL-X fit: see report_smplx_agreement()
            continue
        head_err = max(head_err, float(np.abs(m["Wt"][:, j] - m["kp_local"][key]).max()))
    frames = np.unique(np.linspace(0, m["n_frames"] - 1, min(max_frames, m["n_frames"])).astype(int))
    W = np.where(m["W"] > weight_threshold, m["W"], 0.0)
    A_real, t_real = realized_affines(m["RW"], m["scale"], m["Wt"], m["rest_R"], m["J"])
    ideal = skin_joint_affines(m["A"], m["t"], W, m["verts"], frames)
    real = skin_joint_affines(A_real, t_real, W, m["verts"], frames)
    return head_err, float(np.abs(ideal - real).max())


def report_smplx_agreement(m):
    """Logs how far the SMPL-X fit's joints are from the pose-data keypoints (mean / max over the clip, millimetres)."""
    if m["smplx_pos"] is None:
        return
    names = joint_keypoint_names()
    parts = [("hip", 1, 2), ("knee", 4, 5), ("ankle", 7, 8), ("shoulder", 16, 17), ("elbow", 18, 19), ("wrist", 20, 21)]
    rows = []
    for label, jl, jr in parts:
        d = np.concatenate([np.linalg.norm(m["smplx_pos"][:, j] - m["kp_local"][names[j]], axis=-1) for j in (jl, jr)])
        src = "SMPL-X" if jl in m["smplx_joints"] else "pose data"
        rows.append(f"{label} [{src}] {d.mean() * 1000:.0f}/{d.max() * 1000:.0f}")
    fingers = [j for j in names if j >= 25]
    df = np.concatenate([np.linalg.norm(m["smplx_pos"][:, j] - m["kp_local"][names[j]], axis=-1) for j in fingers])
    rows.append(f"finger joints {df.mean() * 1000:.0f}/{df.max() * 1000:.0f}")
    _log("SMPL-X fit vs pose data, joint distance in mm (mean/max; the joints marked [pose data] are NOT used from the fit): "
         + "; ".join(rows))


# ══════════════════════════════════════════════════════════════════════════════
#  NEW IN THE REFACTOR: raw-chain selection, rotation-only raw mode, channel output, command line
# ══════════════════════════════════════════════════════════════════════════════
def rigidify_raw(A, t, J, parents_bl, raw_set):
    """RAW_MODE 'rotation'. Every raw joint keeps only the ROTATION of its keypoint-driven skin matrix (no scale). The root of
    a raw region (a raw joint whose Blender parent is not raw: the wrist) keeps its head on its keypoint; every other raw joint
    is carried rigidly by its raw parent, so it has no location or scale channel at all. Modifies A, t in place."""
    for j in rc.parents_first_order(parents_bl):
        if j not in raw_set:
            continue
        p = int(parents_bl[j])
        if p >= 0 and p in raw_set:
            head = np.einsum("fab,b->fa", A[:, p], J[j]) + t[:, p]          # where the (already rigid) parent carries it
        else:
            head = np.einsum("fab,b->fa", A[:, j], J[j]) + t[:, j]          # region root: stays on its keypoint
        R = _polar_rotation(A[:, j])
        A[:, j] = R
        t[:, j] = head - np.einsum("fab,b->fa", R, J[j])


def compute_hybrid_motion(kp_data, rig, joint_transforms=None, smplx_pose=None, raw_joints=None):
    """Everything the channel file needs, in numpy only. (inject_animation.py's compute_rig_motion, with the SMPL-X / raw
    split given as a joint set instead of USE_SMPLX_FOR parts, and the rig structure coming from smplx_rig_common.)

    kp_data     {name: (F,3)} keypoints in the data frame (pose_keypoints())
    rig         smplx_rig_common.build_rig(...)
    joint_transforms (F,4,4) data frame -> SMPL-X fitted frame (mapping step), or None
    smplx_pose  the plain SMPL-X fit (load_pose_params), or None for a pure keypoint rig (then every joint is raw)
    raw_joints  SMPL-X joints that come from the keypoints; all other joints come from smplx_pose
    """
    J, verts, W_all, nj = rig.J, rig.verts, rig.W, rig.nj
    n_frames = next(iter(kp_data.values())).shape[0]
    if joint_transforms is None:
        joint_transforms = np.tile(np.eye(4), (n_frames, 1, 1))
    kp_fit = transform_keypoints(kp_data, joint_transforms)
    raw = set(range(nj)) if smplx_pose is None else set(raw_joints)
    smplx_joints = sorted(set(range(nj)) - raw) if smplx_pose is not None else []
    if smplx_pose is not None and smplx_pose["transl"].shape[0] != n_frames:
        raise ValueError(f"Frame count mismatch: pose data has {n_frames} frames, the SMPL-X fit {smplx_pose['transl'].shape[0]}.")

    kp_rest = rest_keypoints(J, verts, W_all)
    # root translation: the fit's own when part of the body comes from it (the armature object carries it), else from the data
    transl = smplx_pose["transl"] if smplx_pose is not None else kp_fit["pelvis"] - kp_rest["pelvis"][0]
    kp_local = {k: v - transl for k, v in kp_fit.items()}

    # keypoint-driven drivers -> per-joint skin matrices
    specs = build_specs()
    joint_weights = build_joint_weights()
    missing = sorted(set(range(nj)) - set(joint_weights))
    if missing:
        raise RuntimeError(f"SMPL-X joints without a driver assignment: {missing}")
    results = solve_bones(specs, kp_rest, kp_local)
    drivers = driver_affines(results)
    A, t = joint_affines(drivers, joint_weights, nj)
    regularize_joint_affines(A, t, J, SCALE_MODE)
    ankle_kp = {s: kp_local[f"{s}_ank"] for s in SIDES}
    if RIGID_FEET:
        make_feet_rigid(A, t, J, ankle_kp)
    elif SCALE_MODE != "along_bone":
        pin_ankles(A, t, J, ankle_kp)
    if RAW_MODE == "rotation":
        rigidify_raw(A, t, J, rig.parents_bl, raw)
    elif RAW_MODE != "exact":
        raise ValueError(f"RAW_MODE must be 'exact' or 'rotation', got {RAW_MODE!r}")

    # hybrid: every joint that is not raw takes its skin matrix from the plain SMPL-X fit
    smplx_pos = None
    if smplx_pose is not None:
        A_s, t_s, smplx_pos = rc.smplx_skin_matrices(rig, smplx_pose)
        for j in smplx_joints:
            A[:, j], t[:, j] = A_s[:, j], t_s[:, j]
    snap_rows = []
    if smplx_pose is not None and SNAP_SEAMS and smplx_joints:
        snap_rows = snap_seams(A, t, J, find_seams(rig.parents_smplx, J, rig.rest_R, smplx_joints), SNAP_SCALE, SNAP_MAX_STRETCH,
                               parents=rig.parents_smplx, rest_R=rig.rest_R, smplx_joints=smplx_joints,
                               lock_bones=set(LOCK_LENGTH_BONES), lock_to=LOCK_LENGTH_TO,
                               lock_parent_max=LOCK_PARENT_MAX_STRETCH, lock_overflow_max=LOCK_OVERFLOW_MAX_STRETCH)

    # Roll alignment only touches a bone that is next to a raw one: forearm -> wrist when either is raw, upper arm -> forearm
    # when either is raw or the forearm was rolled. Two plain SMPL-X bones next to each other stay exactly the fit.
    roll_rows = []
    fore = {p: c for p, c in FOREARM_ROLL_BONES.items() if p in raw or c in raw}
    if FOREARM_ROLL_TO_WRIST_PERCENT and fore:
        roll_rows += fix_bone_roll(A, t, J, fore, FOREARM_ROLL_TO_WRIST_PERCENT, label="forearm")
    up = {p: c for p, c in UPPERARM_ROLL_BONES.items() if p in raw or c in raw or c in fore}
    if UPPERARM_ROLL_TO_FOREARM_PERCENT and up:      # after the forearm, so it follows the forearm's corrected roll
        roll_rows += fix_bone_roll(A, t, J, up, UPPERARM_ROLL_TO_FOREARM_PERCENT, label="upper arm")

    bone_len = list(rig.bone_len)
    if FINGERTIP_BONE_TO_KEYPOINT:
        for j, key in fingertip_joints().items():
            bone_len[j] = max(float(np.linalg.norm(kp_rest[key][0] - J[j])), 1e-4)

    # provenance per bone: 1 raw, 2 SMPL-X bone adjusted to meet a raw one (snap / lock / roll), 0 plain SMPL-X
    source = np.full(nj, SOURCE_SMPLX, dtype=np.int8)
    for j in raw:
        source[j] = SOURCE_RAW
    for r in snap_rows:
        for key in ("p", "q"):
            if key in r and r[key] not in raw:
                source[r[key]] = SOURCE_ADJUSTED
    for r in roll_rows:
        if r["p"] not in raw:
            source[r["p"]] = SOURCE_ADJUSTED

    ch = rc.solve_channels(rig, A, t, transl, source)
    return dict(J=J, W=W_all, verts=verts, nj=nj, n_frames=n_frames, specs=specs, results=results, kp_rest=kp_rest,
                kp_fit=kp_fit, kp_local=kp_local, transl=transl, A=A, t=t, smplx_joints=set(smplx_joints), raw_joints=raw,
                smplx_pos=smplx_pos, snap=snap_rows, roll=roll_rows, parents_smplx=rig.parents_smplx, parents_bl=rig.parents_bl,
                rest_R=rig.rest_R, bone_len=bone_len, loc=ch["loc"], rot=None, scale=ch["scale"], RW=ch["world_rot"],
                Wt=ch["heads"], channels=ch, source=source)


def chain_residuals(m):
    """Per chain: distance (mm, mean / max over the clip) between the FINAL bone heads and the raw keypoints. ~0 for raw
    chains in 'exact' mode; for SMPL-X chains it is how far the plain fit sits from the data."""
    names = joint_keypoint_names()
    out = {}
    for chain, joints in CHAINS.items():
        d = [np.linalg.norm(m["Wt"][:, j] - m["kp_local"][names[j]], axis=-1) for j in joints if j in names]
        if not d:
            continue
        d = np.concatenate(d) * 1000.0
        src = sorted({rc.SOURCE_NAMES[int(m["source"][j])] for j in joints})
        out[chain] = dict(source="/".join(src), mean_mm=float(d.mean()), max_mm=float(d.max()))
    return out


def report_chain_residuals(res):
    _log("Final bone heads vs raw keypoints, per chain (mm mean / max):")
    for chain, r in res.items():
        _log(f"  {chain:<13} [{r['source']:<12}] {r['mean_mm']:7.1f} / {r['max_mm']:7.1f}")


# ── command line ──────────────────────────────────────────────────────────────
def load_keypoints(pose_dir):
    body = np.load(pose_dir / "body_poses_3d.npz")["poses_3d"].astype(float)
    hand = np.load(pose_dir / "hand_poses_3d.npz")["poses_3d"].astype(float)
    if body.shape[1] != BODY_LEN or hand.shape[1] != 2 * HAND_LEN:
        raise ValueError(f"Expected body (F,{BODY_LEN},3) and hand (F,{2 * HAND_LEN},3), got {body.shape} and {hand.shape}.")
    if body.shape[0] != hand.shape[0]:
        raise ValueError(f"Frame count mismatch: body {body.shape[0]} vs hand {hand.shape[0]}.")
    return pose_keypoints(fill_missing(body, "body"), fill_missing(hand, "hand")), body.shape[0]


def main(argv=None):
    global RAW_CHAINS, RAW_MODE, SCALE_MODE, SNAP_SEAMS, SWAP_BODY_SIDES, SWAP_HAND_SIDES
    ap = argparse.ArgumentParser(description="SMPL-X fit + RAW chains -> channel file")
    ap.add_argument("--pose-dir", required=True, type=Path)
    ap.add_argument("--rig", type=Path, default=None, help="armature_rig_metadata.npz (default: <pose-dir>/armature_rig_metadata.npz)")
    ap.add_argument("--pose-params", default=None, help="plain SMPL-X fit file name in pose-dir (default: import_config.json 'pose_params')")
    ap.add_argument("--out", default="hybrid_channels.npz", help="output file name (in pose-dir unless absolute)")
    ap.add_argument("--raw", default=None, help="chains / groups taken from the pose data, comma separated (default: hands)")
    ap.add_argument("--raw-mode", choices=["exact", "rotation"], default=None)
    ap.add_argument("--scale-mode", choices=["none", "static", "uniform", "along_bone"], default=None)
    ap.add_argument("--no-snap", action="store_true", help="no seam snap / lock / roll alignment of the SMPL-X bones")
    ap.add_argument("--no-joint-transform", action="store_true")
    ap.add_argument("--joint-transform", default="joint_transforms.npy")
    ap.add_argument("--swap-body-sides", action="store_true")
    ap.add_argument("--swap-hand-sides", action="store_true")
    ap.add_argument("--config", type=Path, default=None, help="JSON with setting overrides")
    args = ap.parse_args(argv)

    if args.config:
        apply_settings(json.loads(args.config.read_text()))
    if args.raw is not None:
        RAW_CHAINS = args.raw
    if args.raw_mode:
        RAW_MODE = args.raw_mode
    if args.scale_mode:
        SCALE_MODE = args.scale_mode
    if args.no_snap:
        SNAP_SEAMS = False
    SWAP_BODY_SIDES |= args.swap_body_sides
    SWAP_HAND_SIDES |= args.swap_hand_sides

    pose_dir = args.pose_dir
    rig_path = args.rig or pose_dir / "armature_rig_metadata.npz"
    rig = rc.build_rig(np.load(rig_path))
    raw_joints = resolve_chains(RAW_CHAINS)
    pure = set(raw_joints) == set(range(NUM_JOINTS))
    _log(f"Raw chains {RAW_CHAINS!r}: {len(raw_joints)} of {NUM_JOINTS} joints come from the pose data ({RAW_MODE} mode)"
         + ("; no SMPL-X fit needed." if pure else "."))

    kp_pose, n_frames = load_keypoints(pose_dir)
    if args.no_joint_transform:
        joint_transforms = np.tile(np.eye(4), (n_frames, 1, 1))
        _log("Joint-transform disabled -- working in the data frame.")
    else:
        jt_path = pose_dir / args.joint_transform
        joint_transforms = np.load(jt_path).astype(float)
        if joint_transforms.shape != (n_frames, 4, 4):
            raise ValueError(f"{jt_path.name} has shape {joint_transforms.shape}, expected {(n_frames, 4, 4)}.")

    smplx_pose, pp_name = None, None
    if not pure:
        pp_name = args.pose_params
        if pp_name is None:
            cfg_path = pose_dir / "import_config.json"
            pp_name = json.loads(cfg_path.read_text()).get("pose_params") if cfg_path.exists() else None
        if not pp_name:
            raise KeyError("The SMPL-X fit's file is needed for the non-raw chains: give --pose-params, add 'pose_params' to "
                           "import_config.json, or use --raw all for a pure keypoint rig.")
        pp_path = Path(pp_name) if Path(pp_name).is_absolute() else pose_dir / pp_name
        if "skin_affine_A" in np.load(pp_path).files:
            raise ValueError(f"{pp_path.name} carries skin_affine_A (a keypoint-solved pose file), not a plain SMPL-X fit.")
        _log(f"Loading the SMPL-X fit from {pp_path} ...")
        smplx_pose = rc.load_pose_params(pp_path)

    _log(f"Solving {NUM_JOINTS} joint poses for {n_frames} frames ...")
    m = compute_hybrid_motion(kp_pose, rig, joint_transforms, smplx_pose, raw_joints)
    head_err, mesh_dev = diagnose_motion(m)
    _log(f"Self-check: worst distance between a raw bone head and its keypoint = {head_err:.2e} m"
         + ("" if RAW_MODE == "exact" else "  (rotation mode: only region roots are expected to be exact)"))
    report_length_ratios(m["results"])
    report_smplx_agreement(m)
    report_snap(m)
    report_roll(m)
    res = chain_residuals(m)
    report_chain_residuals(res)

    out = Path(args.out) if Path(args.out).is_absolute() else pose_dir / args.out
    meta = dict(tool="get_hybrid_pose_from_stickman.py", raw_chains=RAW_CHAINS, raw_joints=raw_joints, settings=settings_snapshot(),
                pose_params=pp_name, residuals_mm=res,
                snap=[dict(kind=r["kind"], p=r["p"], c=r["c"], gap_before_mm_mean=float(r["gap_before"].mean() * 1000),
                           gap_after_mm_mean=float(r["gap_after"].mean() * 1000)) for r in m["snap"]])
    rc.save_channels(out, rig, m["channels"], meta)
    _log(f"Saved {out}.  To use it, add  \"pose_channels\": \"{out.name}\"  to {pose_dir / 'import_config.json'}.")
    return m


if __name__ == "__main__":
    main()
