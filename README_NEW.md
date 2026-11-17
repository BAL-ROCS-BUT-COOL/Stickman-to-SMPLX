# Stickman -> SMPL-X -> Blender pipeline

Turns triangulated 3D keypoints (COCO-17 body + 2 x 21 hand keypoints) into an animation on an SMPL-X armature in Blender.
Two kinds of animation can be made, and both end up as ordinary per-bone channels, so every bone chain stays editable by hand
without touching the others:

| Variant | Made by | What it is |
|---|---|---|
| **Plain SMPL-X** | `get_skinned_mesh_from_smplx.py` (fitter A) | An SMPL-X fit: rotations only, fixed bone lengths, template body (betas = 0). |
| **SMPL-X x raw hybrid** | `get_hybrid_pose_from_stickman.py` (fitter B) | The plain fit, with the chains you choose (default: the hands) replaced by raw bones driven straight from the keypoints. |

Why both: the raw keypoints carry offsets and biases, and a blind fit can make them more accurate or completely wrong. Keeping the
two as separate files lets you pick, per chain, which one to trust.

Per-script documentation is in [`docs/`](docs/); the paths every script reads and writes are in
[`docs/path_map.json`](docs/path_map.json).

---

## 1. Pipeline

```
 body_poses_3d.npz  (F,17,3)          hand_poses_3d.npz  (F,42,3)
              \                          /
               v                        v
     +-------------------------------------------+
     | mapping_stickman_to_smplx.py              |   uses joints.py (index map)
     | permute axes, centre at pelvis,           |
     | PRIORITY bone-length projection           |
     +-------------------------------------------+
              |                         |
   smplx_joints.npy (F,76,3)    joint_transforms.npy (F,4,4)
              |                         |
              v                         |
     +-------------------------------+  |
     | get_skinned_mesh_from_smplx.py|  |        needs torch + smplx + SMPL-X model files
     | FITTER A: plain SMPL-X fit    |  |
     +-------------------------------+  |
        |                  |            |
 armature_rig_metadata.npz |            |
        |        armature_pose_params(_smoothed).npz
        |                  |            |
        |                  v            |
        |   +-----------------------------------------+
        +-->| get_hybrid_pose_from_stickman.py        |<-- body/hand npz, joint_transforms.npy
        |   | FITTER B: SMPL-X + raw chains (numpy)   |
        |   +-----------------------------------------+
        |                  |
        |         hybrid_channels.npz   (channel file)
        |                  |
        v                  v
     +-----------------------------------------------+
     | inject_animation.py          (runs in Blender)|<-- import_config.json, joint_transforms.npy,
     | bakes channels onto the EXISTING armature     |    alignment_config.json (smplx_camera_alignment.py)
     +-----------------------------------------------+
                          |
                   animated armature in the .blend
```

`smplx_rig_common.py` is the numpy-only module that fitter B and the importer both load; it defines the rig (hierarchy, A-pose rest,
bone chains) and the channel-file format, so what fitter B solves is exactly what the importer bakes.

`animated_stickman.py` is a separate Blender script that draws the raw keypoints as a stickman (spheres and bones) for visual
comparison; it does not take part in the data flow above.

## 2. Files

| File | Runs in | Role | Doc |
|---|---|---|---|
| `mapping_stickman_to_smplx.py` | Python | Keypoints -> SMPL-X joint layout; writes `smplx_joints.npy`, `joint_transforms.npy` | [doc](docs/mapping_stickman_to_smplx.md) |
| `joints.py` | Python | Index table: keypoint index -> SMPL-X joint index | [doc](docs/joints.md) |
| `get_skinned_mesh_from_smplx.py` | Python + torch | Fitter A, plain SMPL-X fit; writes the pose parameters and the rig metadata | [doc](docs/get_skinned_mesh_from_smplx.md) |
| `get_hybrid_pose_from_stickman.py` | Python (numpy) | Fitter B, plain fit + raw chains -> channel file | [doc](docs/get_hybrid_pose_from_stickman.md) |
| `smplx_rig_common.py` | Python (numpy) | Shared rig maths, chain definitions, channel-file I/O | [doc](docs/smplx_rig_common.md) |
| `inject_animation.py` | Blender | Bakes a channel file (or a plain fit) onto the existing armature | [doc](docs/inject_animation.md) |
| `smplx_camera_alignment.py` | Blender | Camera-calibration world -> Blender world transform | [doc](docs/smplx_camera_alignment.md) |
| `smplx_joint_transform.py` | Python | Applies the recorded per-frame transform to mesh vertices (not used by the current scripts) | [doc](docs/smplx_joint_transform.md) |
| `animated_stickman.py` | Blender | Draws the raw keypoints as a stickman | [doc](docs/animated_stickman.md) |
| `run_pipeline.py` | Python | Wrapper: runs mapping, fitter A and fitter B in order, skipping steps whose outputs exist | [doc](docs/run_pipeline.md) |
| `tests/` | Python | Numpy tests, fake-Blender importer test, synthetic rig | [doc](docs/tests.md) |

Removed from the project: `import_armature_hybrid.py`, `DEPRECATED_smplx_armature_import.py`, `smplx_camera_bias.py`.

Put all the scripts in one folder (your `util` folder). `mapping_stickman_to_smplx.py` imports `joints.py`;
`inject_animation.py` loads `smplx_rig_common.py` and `smplx_camera_alignment.py` from its `script_dir` setting; fitter B imports
`smplx_rig_common.py`. Keep only one copy of `smplx_rig_common.py`.

## 3. Folder convention

Everything for one clip lives in one **pose folder** (`pose_dir`), inside the batch renderer's dataset tree:

```
<dataset_dir>/
  <camera folder>/
    alignment_config.json              camera alignment settings (smplx_camera_alignment.py)
    <Color_set>/
      <Pose_set>/                      <- pose_dir: everything below
        body_poses_3d.npz              input   (F,17,3) poses_3d  (+ poses_2d, unused)
        hand_poses_3d.npz              input   (F,42,3) poses_3d  (left hand first)
        import_config.json             you write it: armature name, which pose file to use
        smplx_joints.npy               made by mapping
        joint_transforms.npy           made by mapping
        armature_rig_metadata.npz      made by fitter A (same for every clip: male model, betas = 0)
        armature_pose_params.npz       made by fitter A
        armature_pose_params_smoothed.npz   made by fitter A (what you normally use)
        hybrid_channels.npz            made by fitter B
```

`docs/path_map.json` lists, for each script, what it reads, what it writes, and how each path is set.

## 4. Running it

Replace `<P>` with the pose folder.

```
# 1. keypoints -> SMPL-X layout (also writes joint_transforms.npy)
python mapping_stickman_to_smplx.py --body <P>/body_poses_3d.npz --hand <P>/hand_poses_3d.npz --output-dir <P>

# 2. plain SMPL-X fit (needs torch, smplx, the SMPL-X model folder; slow, GPU helps)
python get_skinned_mesh_from_smplx.py --output-dir <P> --model <smplx model folder>

# 3. optional: plain fit + raw chains
python get_hybrid_pose_from_stickman.py --pose-dir <P> --pose-params armature_pose_params_smoothed.npz --raw hands

# 4. in Blender, on a .blend that already contains the armature
blender -b scene.blend --python inject_animation.py -- --dataset-dir <dataset> --pose-dir <P>
```

Or all of steps 1-3 in one go (each step is skipped when its outputs already exist; see [docs/run_pipeline.md](docs/run_pipeline.md)):

```
python run_pipeline.py --pose-dir <P> --model <smplx model folder> --raw hands
```

`<P>/import_config.json`:

```json
{
  "armature_name": "SMPLX_Armature",
  "pose_params":   "armature_pose_params_smoothed.npz",
  "pose_channels": "hybrid_channels.npz"
}
```

`armature_name` is required. If `pose_channels` is present the importer uses it; otherwise it converts the plain fit named by
`pose_params`. Remove `pose_channels` (or point it elsewhere) to switch variants without re-running anything.

Which variant to use is chosen by one flag at step 3. `--raw hands` is what the old hybrid importer did. Chains and groups are
listed in [docs/get_hybrid_pose_from_stickman.md](docs/get_hybrid_pose_from_stickman.md); `--raw all` needs no plain fit at all.

## 5. Things to know

- **The armature must already exist in the .blend.** The importer animates an existing rig (bones `joint_00 ... joint_54`, A-pose
  rest, hierarchy re-rooted at joint 9) and refuses to run if the rest does not match. The old scripts that created the armature
  and mesh are not part of this project, so use the rig you already have. `armature_rig_metadata.npz` is made by fitter A and does
  not depend on the clip.
- **Not yet run in real Blender.** The importer is tested against a fake `bpy`; the Blender 4.0 bone-collection option is untested.
- **Template proportions do not match the data.** On the real 300-frame clip the thighs and shins measure 1.3-1.4 times the
  template and the hips are about 0.25 m apart against 0.115 m in the template. The plain fit therefore misses the knees by up
  to 43 cm and the hips by 8 cm on average; the arms fit to a few centimetres. Betas are fixed at zero.
- **Raw data glitches carry through.** In the real clip every left-hand keypoint jumps 60-70 mm at frame 75 (0-based; 76 in a
  1-based viewer) and is back by the next frame, which shows up as 50-80 degree finger flips on frames 75 and 76. It is in the input, not the solver.
- **Left / right.** Body indices 5, 7, 11, 13, 15 and the first hand block are the left side. If your data differs use
  `--swap-body-sides` / `--swap-hand-sides` on fitter B.
- **Units** are metres throughout.

## 6. Tests

```
python tests/test_pipeline.py                  # 10 tests; set OLD_INJECT_ANIMATION=<old inject_animation.py> to include the regression test
python tests/test_importer_fake_blender.py     # 3 tests
```

See [docs/tests.md](docs/tests.md).
