"""
preview_compare.py        (one browser page with every stage side by side; no Blender needed)

    python preview_compare.py [--pose-dir data] [--model <smplx model folder>] [--no-open]

Makes <pose-dir>/preview_compare.html (one self-contained file) and opens it. Layout, two columns:

    pose                 SMPL-X joints            (smplx_joints.npy)
    SMPL-X mesh          skinned mesh + armature  (meshed_smplx/, skinned_smplx/)
    hybrid               overlay of everything, with a switch per layer

Playback is real time: space = play / stop, left / right arrow = one frame back / forward (shift = 10), Home / End = first / last.
Drag = rotate, right drag or shift drag = pan, wheel = zoom, double click = reset the view.
A panel whose data is missing says what to run to make it; the others still work.

  pose        body_poses_3d.npz + hand_poses_3d.npz, moved into the SMPL-X frame with joint_transforms.npy
  joints      smplx_joints.npy
  mesh        meshed_smplx/smoothed_all_meshes.npy (else all_meshes.npy); faces from the rig file or --model
  fit         skinned_smplx/armature_rig_metadata.npz + armature_pose_params_smoothed.npz (else armature_pose_params.npz)
  hybrid      skinned_smplx/armature_rig_metadata.npz + pose_smplx_hybrid/hybrid_channels.npz (--hybrid)

The fit and hybrid are skinned in the browser from the rig weights and the per-frame bone matrices (the same maths the
Blender importer uses, smplx_rig_common.py); the baked meshes are stored compressed. numpy is the only requirement.
"""

import argparse
import base64
import json
import sys
import webbrowser
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smplx_rig_common as rc

TEMPLATE = HERE / "assets" / "compare_viewer.html"
ALL_PANELS = ["pose", "joints", "mesh", "fit", "hybrid", "overlay"]

# COCO-17 body keypoints; body[9] / body[10] (wrists) are not used, the hand files carry the wrists
BODY_KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
BODY_EDGES = [(5, 6), (5, 7), (6, 8), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
HAND_LEN = 21
TIP_EDGES = [(39, 66), (27, 67), (30, 68), (36, 69), (33, 70), (54, 71), (42, 72), (45, 73), (51, 74), (48, 75)]


def _enc(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _first(base, names):
    for n in names:
        if (base / n).exists():
            return base / n
    return None


# ── loaders: each returns (dict of numpy arrays, None) or (None, reason) ─────────────────────────────────────
def load_pose(P):
    b, h = P / "body_poses_3d.npz", P / "hand_poses_3d.npz"
    if not (b.exists() and h.exists()):
        return None, "body_poses_3d.npz / hand_poses_3d.npz not found"
    body = np.load(b)["poses_3d"].astype(float)
    hand = np.load(h)["poses_3d"].astype(float)
    n = min(len(body), len(hand))
    body, hand = body[:n], hand[:n]
    jt = P / "joint_transforms.npy"
    if jt.exists():
        M = np.load(jt).astype(float)[:n]
        move = lambda x: np.einsum("fij,fnj->fni", M[:, :3, :3], x) + M[:, None, :3, 3]
        body, hand = move(body), move(hand)
        note = None
    else:
        note = "joint_transforms.npy missing: shown in the raw data frame"
    pts = [body[:, i] for i in BODY_KEEP]
    pts += [(body[:, 5] + body[:, 6]) / 2, (body[:, 11] + body[:, 12]) / 2]
    k = {i: n_ for n_, i in enumerate(BODY_KEEP)}
    neck, pelvis = len(BODY_KEEP), len(BODY_KEEP) + 1
    hb = len(pts)
    pts += [hand[:, i] for i in range(2 * HAND_LEN)]
    edges = [(k[a], k[b_]) for a, b_ in BODY_EDGES] + [(neck, k[0]), (neck, pelvis)]
    d_same = np.nanmean(np.linalg.norm(hand[:, 0] - body[:, 7], axis=-1))
    d_swap = np.nanmean(np.linalg.norm(hand[:, 0] - body[:, 8], axis=-1))
    left_block, right_block = (hb, hb + HAND_LEN) if d_same <= d_swap else (hb + HAND_LEN, hb)
    edges += [(k[7], left_block), (k[8], right_block)]
    for base in (hb, hb + HAND_LEN):
        for f in range(5):
            s = base + 1 + 4 * f
            edges += [(base, s), (s, s + 1), (s + 1, s + 2), (s + 2, s + 3)]
    return dict(pts=np.stack(pts, axis=1), edges=edges, note=note), None


def smplx_parents(rig_meta):
    if rig_meta is not None:
        return np.asarray(rig_meta["parents"]).astype(int)
    p = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15] + [0] * 30
    for base, wrist in ((25, 20), (40, 21)):
        for f in range(5):
            j = base + 3 * f
            p[j], p[j + 1], p[j + 2] = wrist, j, j + 1
    return np.array(p)


def load_joints(P, parents):
    f = P / "smplx_joints.npy"
    if not f.exists():
        return None, "smplx_joints.npy not found (run mapping_stickman_to_smplx.py)"
    J = np.load(f).astype(float)
    if J.ndim != 3 or J.shape[1] < 55:
        return None, f"smplx_joints.npy has shape {J.shape}, expected (frames, 55+, 3)"
    J = J.copy()
    J[:, np.all(J == 0, axis=(0, 2))] = np.nan             # joints the mapping left empty are not drawn
    edges = [(j, int(parents[j])) for j in range(1, 55) if parents[j] >= 0]
    if J.shape[1] >= 76:
        edges += TIP_EDGES
    return dict(pts=J, edges=edges), None


def load_mesh(P):
    M = P / "meshed_smplx"
    f = _first(M, ["smoothed_all_meshes.npy", "all_meshes.npy"]) if M.is_dir() else None
    if f is None:
        return None, "meshed_smplx/smoothed_all_meshes.npy not found (run_pipeline.py --meshes --model <smplx folder>)"
    V = np.load(f).astype(np.float32)
    fj = _first(M, ["smoothed_all_joints.npy", "all_joints.npy"])
    J = np.load(fj).astype(np.float32) if fj is not None else None
    return dict(verts=V, joints=J, file=f.name), None


def load_rig(P, rig_path):
    f = rig_path or (P / "skinned_smplx" / "armature_rig_metadata.npz")
    if not Path(f).exists():
        return None, None, "skinned_smplx/armature_rig_metadata.npz not found (run get_skinned_mesh_from_smplx.py)"
    meta = np.load(f)
    try:
        return rc.build_rig(meta), meta, None
    except Exception as e:
        return None, meta, f"rig file unusable: {e}"


def influences(rig, k_max=8):
    W = rig.W
    K = int(min(k_max, (W > 1e-6).sum(1).max()))
    idx = np.argsort(-W, axis=1)[:, :K]
    w = np.take_along_axis(W, idx, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return idx, w


def load_fit(P, rig, pose_params):
    f = pose_params or _first(P / "skinned_smplx", ["armature_pose_params_smoothed.npz", "armature_pose_params.npz"])
    if f is None or not Path(f).exists():
        return None, "skinned_smplx/armature_pose_params_smoothed.npz not found (run get_skinned_mesh_from_smplx.py --model <smplx folder>)"
    pose = rc.load_pose_params(f)
    A, t, pos = rc.smplx_skin_matrices(rig, pose)
    tr = pose["transl"][:, None, :]
    return dict(A=A, t=t + tr, pos=pos + tr, parents=rig.parents_orig, src=None, file=Path(f).name), None


def load_hybrid(P, rig, hybrid):
    f = Path(hybrid) if hybrid else P / "pose_smplx_hybrid" / "hybrid_channels.npz"
    if not f.is_absolute() and not f.exists():
        f = P / f
    if not f.exists():
        return None, "pose_smplx_hybrid/hybrid_channels.npz not found (run get_hybrid_pose_from_stickman.py or run_pipeline.py)"
    try:
        ch = rc.load_channels(f, rig)
    except Exception as e:
        return None, f"hybrid file does not match the rig: {e}"
    Wt, RW = rc.channels_to_pose(rig, ch["loc"], ch["quat"], ch["scale"])
    A, t = rc.realized_affines(RW, ch["scale"], Wt, rig.rest_R, rig.J)
    tr = ch["transl"][:, None, :]
    return dict(A=A, t=t + tr, pos=Wt + tr, parents=rig.parents_orig, src=np.asarray(ch["source"]).astype(int), file=f.name), None


def faces_from_model(model_dir):
    try:
        import smplx
        return np.asarray(smplx.create(str(model_dir), model_type="smplx").faces).astype(int)
    except Exception:
        return None


# ── camera hints: which way is up / forward in this data ────────────────────────────────────────────────────
def view_hints(layers):
    J = layers["joints"]["pts"] if layers.get("joints") else None
    if J is not None:
        up = np.nanmean(J[:, 12] - J[:, 0], axis=0)
        lat = np.nanmean(J[:, 2] - J[:, 1], axis=0)
        cloud = J[:, :55]
    elif layers.get("pose"):
        K = layers["pose"]["pts"]
        up = np.nanmean(K[:, len(BODY_KEEP)] - K[:, len(BODY_KEEP) + 1], axis=0)
        lat = np.nanmean(K[:, BODY_KEEP.index(12)] - K[:, BODY_KEEP.index(11)], axis=0)
        cloud = K
    elif layers.get("fit"):
        pos = layers["fit"]["pos"]
        up = np.nanmean(pos[:, 12] - pos[:, 0], axis=0)
        lat = np.nanmean(pos[:, 2] - pos[:, 1], axis=0)
        cloud = pos
    else:
        up, lat, cloud = np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.zeros((1, 1, 3))
    up = up / max(np.linalg.norm(up), 1e-9)
    axis = np.argmax(np.abs(up))
    if abs(up[axis]) > 0.7:                                  # keep a clean axis when the subject is roughly upright
        up = np.eye(3)[axis] * np.sign(up[axis])
    lat = lat - up * (lat @ up)
    lat = lat / max(np.linalg.norm(lat), 1e-9)
    facing = np.cross(up, lat)                               # SMPL-X: up = +Y, right hip - left hip = -X  ->  facing +Z
    extra = [layers[k]["pos"].reshape(-1, 3) for k in ("fit", "hybrid") if layers.get(k)]
    if layers.get("mesh") is not None and layers["mesh"]["joints"] is not None:
        extra.append(layers["mesh"]["joints"].reshape(-1, 3))
    pts = np.concatenate([cloud.reshape(-1, 3)] + extra)
    pts = pts[np.isfinite(pts).all(1)]
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2) * 1.15
    floor = float(np.min(pts @ up)) if len(pts) else 0.0
    return dict(up=up.tolist(), facing=facing.tolist(), center=center.tolist(), radius=max(radius, 0.3), floor=floor)


def pack_mesh(V):
    """(F,V,3) float -> deflate of byte-shuffled int16 time deltas (smooth motion compresses to a fraction)."""
    lo = V.reshape(-1, 3).min(0)
    step = max(1e-4, float((V.reshape(-1, 3).max(0) - lo).max()) / 60000)
    q = np.rint((V - lo) / step).astype(np.int32)
    q = q.reshape(len(V), -1)
    d = np.diff(q, axis=0, prepend=np.zeros((1, q.shape[1]), dtype=np.int32)).astype("<i2")
    raw = d.view("<u2")
    shuffled = np.concatenate([(raw & 0xFF).astype(np.uint8).ravel(), (raw >> 8).astype(np.uint8).ravel()])
    return dict(b64=base64.b64encode(zlib.compress(shuffled.tobytes(), 9)).decode("ascii"), origin=lo.tolist(), step=step)


def build(P, model=None, rig_path=None, pose_params=None, hybrid=None, fps=30.0, stride=1, panels=None):
    reasons, layers = {}, {}
    layers["pose"], reasons["pose"] = load_pose(P)
    rig, meta, why_rig = load_rig(P, rig_path)
    parents = smplx_parents(meta)
    layers["joints"], reasons["joints"] = load_joints(P, parents)
    layers["mesh"], reasons["mesh"] = load_mesh(P)
    if rig is not None:
        layers["fit"], reasons["fit"] = load_fit(P, rig, pose_params)
        layers["hybrid"], reasons["hybrid"] = load_hybrid(P, rig, hybrid)
    else:
        layers["fit"], reasons["fit"] = None, why_rig
        layers["hybrid"], reasons["hybrid"] = None, why_rig

    faces = None
    if rig is not None and rig.faces is not None:
        faces = rig.faces
    elif model is not None:
        faces = faces_from_model(model)
    if layers["mesh"] is not None and faces is None:
        layers["mesh"], reasons["mesh"] = None, "meshed_smplx found but there are no faces: need skinned_smplx/armature_rig_metadata.npz or --model <smplx folder>"

    if "overlay" not in (panels or ALL_PANELS):               # only embed what the chosen panels show
        for k in layers:
            if k not in (panels or ALL_PANELS) and layers[k] is not None:
                layers[k], reasons[k] = None, "not in --panels"

    counts = {k: (len(v["pts"]) if "pts" in v else len(v["verts"]) if "verts" in v else len(v["A"])) for k, v in layers.items() if v}
    if not counts:
        sys.exit(f"Nothing to show in {P}: " + "; ".join(f"{k}: {r}" for k, r in reasons.items() if r))
    n = min(counts.values())
    if len(set(counts.values())) > 1:
        print("Frame counts differ (" + ", ".join(f"{k} {c}" for k, c in counts.items()) + f"), showing the first {n}.")
    sl = slice(0, n, stride)
    for v in layers.values():
        if v:
            for key in ("pts", "verts", "joints", "A", "t", "pos"):
                if v.get(key) is not None:
                    v[key] = v[key][sl]
    n = len(range(0, n, stride))

    out = dict(N=n, fps=float(fps) / stride, panels=panels or ALL_PANELS, reasons={k: r for k, r in reasons.items() if layers.get(k) is None}, layers={})
    out.update(view_hints(layers))
    L = layers
    if L["pose"]:
        out["layers"]["pose"] = dict(n=int(L["pose"]["pts"].shape[1]), pts=_enc(L["pose"]["pts"], "<f4"), edges=L["pose"]["edges"], note=L["pose"]["note"])
    if L["joints"]:
        out["layers"]["joints"] = dict(n=int(L["joints"]["pts"].shape[1]), pts=_enc(L["joints"]["pts"], "<f4"), edges=L["joints"]["edges"])
    if L["mesh"]:
        out["layers"]["mesh"] = dict(V=int(L["mesh"]["verts"].shape[1]), file=L["mesh"]["file"], **pack_mesh(L["mesh"]["verts"]))
        if L["mesh"]["joints"] is not None:
            out["layers"]["mesh"]["joints"] = dict(n=int(L["mesh"]["joints"].shape[1]), pts=_enc(L["mesh"]["joints"], "<f4"))
    if faces is not None:
        out["faces"] = _enc(faces, "<u4")
    if rig is not None and (L["fit"] or L["hybrid"]):
        idx, w = influences(rig)
        out["rig"] = dict(V=int(len(rig.verts)), K=int(idx.shape[1]), nj=int(rig.nj), rest=_enc(rig.verts, "<f4"), idx=_enc(idx, "u1"), w=_enc(w, "<f4"))
    for name in ("fit", "hybrid"):
        v = L[name]
        if v:
            A12 = np.concatenate([v["A"].reshape(n, rig.nj, 9), v["t"]], axis=-1)
            bone_edges = [(j, int(v["parents"][j])) for j in range(rig.nj) if v["parents"][j] >= 0]
            out["layers"][name] = dict(A=_enc(A12, "<f4"), pos=_enc(v["pos"], "<f4"), n=int(rig.nj), edges=bone_edges, file=v["file"],
                                       src=None if v["src"] is None else [int(v["src"][j]) for j, _ in bone_edges])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", type=Path, default=Path("data"), help="pose folder (default: data)")
    ap.add_argument("--out", type=Path, default=None, help="html file to write (default: <pose-dir>/preview_compare.html)")
    ap.add_argument("--model", type=Path, default=None, help="SMPL-X model folder; only used to get the mesh faces when the rig file is missing")
    ap.add_argument("--rig", type=Path, default=None, help="armature_rig_metadata.npz (default: <pose-dir>/skinned_smplx/armature_rig_metadata.npz)")
    ap.add_argument("--pose-params", type=Path, default=None, help="SMPL-X fit file (default: skinned_smplx/armature_pose_params_smoothed.npz, else armature_pose_params.npz)")
    ap.add_argument("--hybrid", default=None, help="hybrid channel file, relative to pose-dir (default: pose_smplx_hybrid/hybrid_channels.npz)")
    ap.add_argument("--fps", type=float, default=30.0, help="playback speed at 1x (default: 30)")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame (smaller file for long clips)")
    ap.add_argument("--panels", default=",".join(ALL_PANELS), help="comma separated subset of: " + ", ".join(ALL_PANELS))
    ap.add_argument("--no-open", action="store_true", help="only write the file")
    a = ap.parse_args(argv)

    P = a.pose_dir.resolve()
    if not P.is_dir():
        sys.exit(f"--pose-dir is not a folder: {P}")
    panels = [p.strip() for p in a.panels.split(",") if p.strip()]
    bad = [p for p in panels if p not in ALL_PANELS]
    if bad:
        sys.exit(f"Unknown panel(s) {bad}; choose from {ALL_PANELS}")
    if not TEMPLATE.exists():
        sys.exit(f"Missing viewer template: {TEMPLATE}")

    data = build(P, a.model, a.rig, a.pose_params, a.hybrid, a.fps, max(1, a.stride), panels)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    out = (a.out or P / "preview_compare.html").resolve()
    out.write_text(html, encoding="utf-8")
    shown = [k for k in ("pose", "joints", "mesh", "fit", "hybrid") if k in data["layers"]]
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {data['N']} frames; layers: {', '.join(shown)})")
    for k, r in data["reasons"].items():
        if k not in panels and "overlay" not in panels:
            continue
        print(f"  {k}: not available - {r}")
    if not a.no_open:
        if not webbrowser.open(out.as_uri()):
            print("Could not open a browser; open the file by hand.")
    return out


if __name__ == "__main__":
    main()
