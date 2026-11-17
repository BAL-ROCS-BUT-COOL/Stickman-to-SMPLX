"""
run_pipeline.py        (wrapper: mapping -> fitter A -> fitter B, each only when needed)

Runs the existing scripts one after the other as separate processes; it contains no pipeline logic of its own.

    python run_pipeline.py --pose-dir <Pose_set> --model <smplx model folder> [--raw hands]

  1. mapping_stickman_to_smplx.py        runs if smplx_joints.npy or joint_transforms.npy is missing (or --remap)
  2. get_skinned_mesh_from_smplx.py      runs if armature_pose_params_smoothed.npz or armature_rig_metadata.npz is missing (or --refit;
                                         with --raw all only the rig file is needed)
                                         (needs torch, smplx and --model; slow)
  3. get_hybrid_pose_from_stickman.py    always runs (seconds); writes hybrid_channels.npz

Reads body_poses_3d.npz and hand_poses_3d.npz from --pose-dir; every output goes to --pose-dir.
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


def run(step, script, args):
    cmd = [sys.executable, str(HERE / script)] + [str(a) for a in args]
    print(f"\n=== {step}: {script} ===\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"\n{step} failed (exit code {result.returncode}); stopping.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", required=True, type=Path)
    ap.add_argument("--model", type=Path, help="SMPL-X model folder (needed only when fitter A has to run)")
    ap.add_argument("--raw", default=None, help="chains / groups taken from the pose data (fitter B default: hands)")
    ap.add_argument("--raw-mode", choices=["exact", "rotation"], default=None)
    ap.add_argument("--scale-mode", choices=["none", "static", "uniform", "along_bone"], default=None)
    ap.add_argument("--config", type=Path, default=None, help="settings JSON for fitter B")
    ap.add_argument("--out", default="hybrid_channels.npz", help="channel file name (in pose-dir)")
    ap.add_argument("--remap", action="store_true", help="run the mapping step even if its outputs exist")
    ap.add_argument("--refit", action="store_true", help="run fitter A even if its outputs exist")
    ap.add_argument("--map-args", default="", help="extra arguments for the mapping script, one quoted string")
    ap.add_argument("--fit-args", default="", help="extra arguments for fitter A, one quoted string")
    ap.add_argument("--hybrid-args", default="", help="extra arguments for fitter B, one quoted string")
    a = ap.parse_args(argv)

    P = a.pose_dir.resolve()
    if not P.is_dir():
        sys.exit(f"--pose-dir is not a folder: {P}")
    for name in ("body_poses_3d.npz", "hand_poses_3d.npz"):
        if not (P / name).exists():
            sys.exit(f"Missing input: {P / name}")

    # 1. mapping
    mapped = (P / "smplx_joints.npy").exists() and (P / "joint_transforms.npy").exists()
    if a.remap or not mapped:
        run("1/3 mapping", "mapping_stickman_to_smplx.py",
            ["--body", P / "body_poses_3d.npz", "--hand", P / "hand_poses_3d.npz", "--output-dir", P] + shlex.split(a.map_args))
    else:
        print("1/3 mapping: smplx_joints.npy and joint_transforms.npy exist, skipped (--remap to redo)")

    # 2. fitter A
    pure = a.raw is not None and {s.strip() for s in a.raw.split(",")} == {"all"}      # --raw all needs the rig file only
    fitted = (P / "armature_rig_metadata.npz").exists() and (pure or (P / "armature_pose_params_smoothed.npz").exists())
    if a.refit or not fitted:
        if a.model is None:
            sys.exit("Fitter A has to run (its outputs are missing or --refit was given) but --model was not set.")
        run("2/3 fitter A", "get_skinned_mesh_from_smplx.py", ["--output-dir", P, "--model", a.model] + shlex.split(a.fit_args))
    else:
        print("2/3 fitter A: armature_pose_params_smoothed.npz and armature_rig_metadata.npz exist, skipped (--refit to redo)")

    # 3. fitter B
    args = ["--pose-dir", P, "--pose-params", "armature_pose_params_smoothed.npz", "--out", a.out]
    for flag, value in (("--raw", a.raw), ("--raw-mode", a.raw_mode), ("--scale-mode", a.scale_mode), ("--config", a.config)):
        if value is not None:
            args += [flag, value]
    run("3/3 fitter B", "get_hybrid_pose_from_stickman.py", args + shlex.split(a.hybrid_args))

    print(f"\nDone. Add  \"pose_channels\": \"{a.out}\"  to {P / 'import_config.json'} to use the hybrid in Blender.")


if __name__ == "__main__":
    main()
