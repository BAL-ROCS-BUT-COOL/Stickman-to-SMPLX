"""
run_pipeline.py        (wrapper: mapping -> [meshes] -> fitter A -> fitter B, each only when needed, then the preview)

Runs the existing scripts one after the other as separate processes; it contains no pipeline logic of its own.

    python run_pipeline.py [--pose-dir <Pose_set>] --model <smplx model folder> [--raw hands] [--meshes]

  --pose-dir    the pose folder (default: data, relative to where you run the command)

  1. mapping_stickman_to_smplx.py        runs if smplx_joints.npy or joint_transforms.npy is missing (or --remap)
                                         writes into <pose-dir>
  (optional, with --meshes)
     get_mesh_from_smplx.py              runs if meshed_smplx/smoothed_all_meshes.npy or meshed_smplx/smoothed_all_joints.npy is missing
                                         (or --remesh); writes meshed_smplx/ (needs torch, smplx and --model; slow)

  2. get_skinned_mesh_from_smplx.py      runs if skinned_smplx/armature_pose_params_smoothed.npz or skinned_smplx/armature_rig_metadata.npz
                                         is missing (or --refit; with --raw all only the rig file is needed)
                                         (needs torch, smplx and --model; slow)
  3. get_hybrid_pose_from_stickman.py    always runs (seconds); writes pose_smplx_hybrid/hybrid_channels.npz

  4. preview                             opens <pose-dir>/preview_compare.html in the browser (preview_compare.py): pose, SMPL-X joints,
                                         SMPL-X mesh, skinned mesh + armature, hybrid and an overlay of all of them, with play / stop and
                                         frame-by-frame stepping. Missing data only blanks its panel. --no-compare opens the single
                                         previews instead (visualize_joints.py, visualize_smplx.py, skinned + hybrid page); --no-preview opens nothing.

Reads body_poses_3d.npz and hand_poses_3d.npz from --pose-dir; every output goes below --pose-dir:
the mapping files directly in it, the fit in skinned_smplx/, the hybrid in pose_smplx_hybrid/.
meshed_smplx/ is only made with --meshes.
Step 2 is skipped when its output already exists, so re-running with other --raw settings only repeats step 3.
Anything a step needs beyond the options below can be passed as one quoted string: --map-args, --fit-args, --hybrid-args.
    e.g.  --fit-args "--fps 25 --cutoff-hz 4"   --hybrid-args "--swap-hand-sides --no-snap"
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(step, script, args, required=True):
    cmd = [sys.executable, str(HERE / script)] + [str(a) for a in args]
    print(f"\n=== {step}: {script} ===\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        if not required:
            print(f"{step} failed (exit code {result.returncode}); the pipeline results are not affected.")
            return False
        sys.exit(f"\n{step} failed (exit code {result.returncode}); stopping.")
    return True


def preview(a, P):
    """Step 4. Never stops the pipeline: the results are already written."""
    if a.no_preview:
        return
    hybrid = P / a.out
    common = ["--pose-dir", P, "--hybrid", hybrid] + ([] if a.model is None else ["--model", a.model]) + shlex.split(a.preview_args)
    if not a.no_compare:
        run("preview", "preview_compare.py", common, required=False)
        return
    run("preview: pose / joints", "visualize_joints.py", ["--pose-dir", P, "--frame", a.preview_frame], required=False)
    if (P / "meshed_smplx" / "smoothed_all_meshes.npy").exists():
        run("preview: SMPL-X mesh", "visualize_smplx.py",
            ["--pose-dir", P, "--frame", a.preview_frame] + ([] if a.model is None else ["--model", a.model]), required=False)
    else:
        print("preview: SMPL-X mesh skipped, meshed_smplx/ does not exist (run with --meshes)")
    run("preview: skinned mesh + armature, hybrid", "preview_compare.py", common + ["--panels", "fit,hybrid", "--out", P / "preview_skinned.html"], required=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", type=Path, default=Path("data"), help="pose folder holding the inputs; every output goes below it (default: data)")
    ap.add_argument("--model", type=Path, help="SMPL-X model folder (needed only when fitter A has to run)")
    ap.add_argument("--raw", default=None, help="chains / groups taken from the pose data (fitter B default: hands)")
    ap.add_argument("--raw-mode", choices=["exact", "rotation"], default=None)
    ap.add_argument("--scale-mode", choices=["none", "static", "uniform", "along_bone"], default=None)
    ap.add_argument("--config", type=Path, default=None, help="settings JSON for fitter B")
    ap.add_argument("--out", default="pose_smplx_hybrid/hybrid_channels.npz", help="channel file (relative to pose-dir)")
    ap.add_argument("--meshes", action="store_true", help="also make the baked meshes (get_mesh_from_smplx.py -> meshed_smplx/); needs --model")
    ap.add_argument("--no-preview", action="store_true", help="do not open any preview at the end")
    ap.add_argument("--no-compare", action="store_true", help="open the single previews instead of the one compare page")
    ap.add_argument("--preview-frame", type=int, default=0, help="frame shown by the single previews with --no-compare (default: 0)")
    ap.add_argument("--preview-args", default="", help="extra arguments for preview_compare.py, one quoted string, e.g. \"--fps 25 --stride 2\"")
    ap.add_argument("--remap", action="store_true", help="run the mapping step even if its outputs exist")
    ap.add_argument("--remesh", action="store_true", help="with --meshes: run the mesh step even if its outputs exist")
    ap.add_argument("--refit", action="store_true", help="run fitter A even if its outputs exist")
    ap.add_argument("--map-args", default="", help="extra arguments for the mapping script, one quoted string")
    ap.add_argument("--mesh-args", default="", help="extra arguments for get_mesh_from_smplx.py, one quoted string")
    ap.add_argument("--fit-args", default="", help="extra arguments for fitter A, one quoted string")
    ap.add_argument("--hybrid-args", default="", help="extra arguments for fitter B, one quoted string")
    a = ap.parse_args(argv)

    P = a.pose_dir.resolve()
    if not P.is_dir():
        sys.exit(f"--pose-dir is not a folder: {P}")
    for name in ("body_poses_3d.npz", "hand_poses_3d.npz"):
        if not (P / name).exists():
            sys.exit(f"Missing input: {P / name}")

    total = 4 if a.meshes else 3
    step = 0

    # 1. mapping
    step += 1
    mapped = (P / "smplx_joints.npy").exists() and (P / "joint_transforms.npy").exists()
    if a.remap or not mapped:
        run(f"{step}/{total} mapping", "mapping_stickman_to_smplx.py",
            ["--body", P / "body_poses_3d.npz", "--hand", P / "hand_poses_3d.npz", "--output-dir", P] + shlex.split(a.map_args))
    else:
        print(f"{step}/{total} mapping: smplx_joints.npy and joint_transforms.npy exist, skipped (--remap to redo)")

    # (optional) baked meshes
    if a.meshes:
        step += 1
        M = P / "meshed_smplx"
        meshed = (M / "smoothed_all_meshes.npy").exists() and (M / "smoothed_all_joints.npy").exists()
        if a.remesh or not meshed:
            if a.model is None:
                sys.exit("The mesh step has to run (its outputs are missing or --remesh was given) but --model was not set.")
            run(f"{step}/{total} meshes", "get_mesh_from_smplx.py",
                ["--joints", P / "smplx_joints.npy", "--model", a.model, "--out_meshes", M / "all_meshes.npy", "--out_joints", M / "all_joints.npy"]
                + shlex.split(a.mesh_args))
        else:
            print(f"{step}/{total} meshes: meshed_smplx/smoothed_all_meshes.npy and smoothed_all_joints.npy exist, skipped (--remesh to redo)")

    # 2. fitter A
    step += 1
    pure = a.raw is not None and {s.strip() for s in a.raw.split(",")} == {"all"}      # --raw all needs the rig file only
    S = P / "skinned_smplx"
    fitted = (S / "armature_rig_metadata.npz").exists() and (pure or (S / "armature_pose_params_smoothed.npz").exists())
    if a.refit or not fitted:
        if a.model is None:
            sys.exit("Fitter A has to run (its outputs are missing or --refit was given) but --model was not set.")
        run(f"{step}/{total} fitter A", "get_skinned_mesh_from_smplx.py", ["--output-dir", P, "--model", a.model] + shlex.split(a.fit_args))
    else:
        print(f"{step}/{total} fitter A: skinned_smplx/armature_pose_params_smoothed.npz and armature_rig_metadata.npz exist, skipped (--refit to redo)")

    # 3. fitter B
    step += 1
    args = ["--pose-dir", P, "--pose-params", "skinned_smplx/armature_pose_params_smoothed.npz", "--out", a.out]
    for flag, value in (("--raw", a.raw), ("--raw-mode", a.raw_mode), ("--scale-mode", a.scale_mode), ("--config", a.config)):
        if value is not None:
            args += [flag, value]
    run(f"{step}/{total} fitter B", "get_hybrid_pose_from_stickman.py", args + shlex.split(a.hybrid_args))

    preview(a, P)

    print(f"\nDone. Add  \"pose_channels\": \"{Path(a.out).as_posix()}\"  to {P / 'import_config.json'} to use the hybrid in Blender.")


if __name__ == "__main__":
    main()
