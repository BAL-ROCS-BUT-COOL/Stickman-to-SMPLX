# Stickman to SMPL-X
Turns triangulated 3D keypoints of a person (COCO-17 body + 2 x 21 hand keypoints, "stickman") into SMPL-X data. 
The results can then be brought into Blender with [smplx-to-blender](https://github.com/BAL-ROCS-BUT-COOL/SMPLX-to-Blender) repository.

This repo uses Python 3.11.9.

<p align="center">
  <img src="assets/both_plots_demo.png"
       alt="Stickman 3D visualization" width="700">
</p>

**Figure 1.** Stickman 3D visualization.

## 1. What it produces
The keypoints are mapped onto the SMPL-X joint layout once. From there three kinds of result can be made, each written into its own folder:

| Folder | Made by | What it is |
|---|---|---|
| `meshed_smplx/` | `get_mesh_from_smplx.py` | Baked SMPL-X **meshes** and joints per frame, with a low-pass-filtered copy (`smoothed_*`). One vertex array per frame. |
| `skinned_smplx/` | `get_skinned_mesh_from_smplx.py` (fitter A) | A plain SMPL-X **fit**: pose parameters (rotations only, fixed bone lengths, template body with betas = 0) plus the rig metadata that describes the armature. |
| `pose_smplx_hybrid/` | `get_hybrid_pose_from_stickman.py` (fitter B) | The plain fit with the chains you choose (default: the hands) replaced by raw bones driven straight from the keypoints. A **channel file** with ordinary per-bone channels, so every bone chain stays editable by hand without touching the others. |

Why both a fit and a hybrid: SMPLX fitting can make bad tracking plausible, but also make accurate tracking worse.
When using SMPLX to display accurate tracking, the hybrid step lets you keep the parts that are good and only replace the parts that are bad.

## 2. Setup

### Environment
Create a new Python virtual environment (recommended, to keep the dependencies isolated) and install the requirements.

**Windows (PowerShell or CMD):**

```
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### SMPL-X models
Download the SMPL-X models from the [project website](https://smpl-x.is.tue.mpg.de/) and put the `models` folder in the same directory as the scripts (or pass its location with `--model`).

### Shared module
`get_hybrid_pose_from_stickman.py` imports `smplx_rig_common.py`, which sits in the same folder as the script. The script looks for it there first, so nothing has to be copied or put on `PYTHONPATH`.
The smplx-to-blender repository uses the same file for its importer, so that what the hybrid step solves is exactly what the importer bakes. Keep only one copy of it.

## 3. Folder convention
Everything for one clip lives in one **pose folder**, shown here for the example data in `data/`. The inputs and the shared mapping files sit directly in it, and each result type gets its own subfolder:

```
data/
  body_poses_3d.npz                       input   (F,17,3) poses_3d  (+ poses_2d, unused)
  hand_poses_3d.npz                       input   (F,42,3) poses_3d  (left hand first)
  smplx_joints.npy                        made by mapping, read by everything below
  joint_transforms.npy                    made by mapping, read by the hybrid step
  import_config.json                      you write it (Blender side, see BLENDER_INTEGRATION.md)

  meshed_smplx/                           get_mesh_from_smplx.py
    all_meshes.npy
    all_joints.npy
    smoothed_all_meshes.npy               low-pass filtered, use this one for renders
    smoothed_all_joints.npy

  skinned_smplx/                          get_skinned_mesh_from_smplx.py
    armature_rig_metadata.npz             same for every clip
    armature_pose_params.npz
    armature_pose_params_smoothed.npz     what you normally use

  pose_smplx_hybrid/                      get_hybrid_pose_from_stickman.py
    hybrid_channels.npz
```

The mapping files stay outside the three folders because all three steps read them; they are inputs to the steps, not results of one of them. Every script takes the pose folder as `--output-dir` / `--pose-dir` and creates its subfolder when it is missing.

## 4. Pipeline

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
       +------+---------------+         |
       |                      |         |
       v                      v         |
 +----------------------+  +-------------------------------+
 | get_mesh_from_smplx  |  | get_skinned_mesh_from_smplx   |   needs torch + smplx + model files
 | -> meshed_smplx/     |  | FITTER A -> skinned_smplx/    |
 +----------------------+  +-------------------------------+
                              |                  |
                armature_rig_metadata.npz   armature_pose_params(_smoothed).npz
                              |                  |
                              v                  v
                     +-----------------------------------------+
                     | get_hybrid_pose_from_stickman.py        |<-- body/hand npz, joint_transforms.npy
                     | FITTER B -> pose_smplx_hybrid/ (numpy)  |
                     +-----------------------------------------+
                                       |
                              hybrid_channels.npz   (channel file)
```

## 5. Running it
All scripts use the `data/` folder by default. Put your `body_poses_3d.npz` and `hand_poses_3d.npz` there, or see the note at the end of this section to use another folder.

### 1. Map stickman data to SMPL-X
This script maps your custom 3D body and hand keypoints into the **SMPL-X joint format**. Run it with your input body and hand joint files:

```
python mapping_stickman_to_smplx.py --body BODY_FILE --hand HAND_FILE [--output-dir OUTPUT_DIR]
```

This writes `smplx_joints.npy` and `joint_transforms.npy` into the output folder (default `data/`).

Now visualize the data using the **`visualize_joints.py`** file:

```
python visualize_joints.py --frame FRAME_NUMBER
```

- Make sure that the body stands along the positive Y-axis.
- The pelvis should be close to the origin (0,0,0).
- The person should be facing the positive Z-direction.
- The left shoulder should be in the positive X-direction and the right shoulder in the negative X-direction (exactly like the right person in [Figure 1](#figure-1-stickman-3d-visualization)).

If this is not the case, change the **`rot_mat`** inside of **`mapping_stickman_to_smplx.py`** to adjust this and try again. Otherwise the SMPL-X mesh will not look good.

### 2. Get the SMPL-X mesh
Run the script to approximate the best fitting SMPL-X mesh to the given joints:

```
python get_mesh_from_smplx.py
```

This outputs 4 files into `data/meshed_smplx/`:
- **`all_meshes.npy`**
- **`all_joints.npy`**
- **`smoothed_all_meshes.npy`** (applied low-pass filter)
- **`smoothed_all_joints.npy`** (applied low-pass filter)

I recommend using the smoothed files for renders.

### 3. Visualize the mesh
After generating the meshes and joints, you can visualize any frame with:

```
python visualize_smplx.py --frame FRAME_NUMBER
```

This will open an interactive 3D plot showing the SMPL-X body mesh along with joint markers and axes.

### 4. Get the SMPL-X fit (for the armature)
Run the script to fit SMPL-X pose parameters to the given joints. This is slow, a GPU helps:

```
python get_skinned_mesh_from_smplx.py
```

This outputs 3 files into `data/skinned_smplx/`:
- **`armature_rig_metadata.npz`** (the same for every clip)
- **`armature_pose_params.npz`**
- **`armature_pose_params_smoothed.npz`** (the one you normally use)

### 5. Get the hybrid pose (optional)
Run the script to replace chosen parts of the fit with the raw keypoints (default: the hands):

```
python get_hybrid_pose_from_stickman.py --pose-params skinned_smplx/armature_pose_params_smoothed.npz --raw hands
```

This outputs **`hybrid_channels.npz`** into `data/pose_smplx_hybrid/`.

- `--raw` chooses what comes from the keypoints instead of the fit: `hands` (default), `arms`, `legs`, `wrists`, `fingers` or `all`, separated by commas. Single chains are also possible, e.g. `left_arm` or `right_index`.
- `--raw-mode exact` puts every raw bone on its keypoint (default). `--raw-mode rotation` lets raw bones only rotate and keep their template length.
- Left and right swapped? Add `--swap-body-sides` and/or `--swap-hand-sides`.
- Run it with `--help` to see all options.

### Everything at once
Steps 1, 4 and 5 can be run with one command. Steps that already have their output are skipped:

```
python run_pipeline.py --model models --raw hands
```

Step 2 (the meshes) is not part of this command, run it on its own.

### Using another folder
By default every script reads and writes `data/`. To use another folder:

- `mapping_stickman_to_smplx.py` and `get_skinned_mesh_from_smplx.py`: `--output-dir FOLDER`
- `get_hybrid_pose_from_stickman.py` and `run_pipeline.py`: `--pose-dir FOLDER`
- `get_mesh_from_smplx.py`: `--joints FOLDER/smplx_joints.npy --out_meshes FOLDER/meshed_smplx/all_meshes.npy --out_joints FOLDER/meshed_smplx/all_joints.npy`
- `visualize_joints.py` and `visualize_smplx.py` always read from `data/`.

If the SMPL-X models are not in a `models` folder next to the scripts, add `--model MODEL_FOLDER` to the scripts in steps 2 and 4. `run_pipeline.py` always needs `--model` (it has no default).

## 6. Files

| File | Needs | Role |
|---|---|---|
| `mapping_stickman_to_smplx.py` | numpy | Keypoints -> SMPL-X joint layout; writes `smplx_joints.npy` and `joint_transforms.npy` |
| `joints.py` | numpy | Index table: keypoint index -> SMPL-X joint index |
| `get_mesh_from_smplx.py` | torch, smplx | Baked meshes and joints, plus smoothed copies -> `meshed_smplx/` |
| `get_skinned_mesh_from_smplx.py` | torch, smplx | Fitter A, plain SMPL-X fit; pose parameters and rig metadata -> `skinned_smplx/` |
| `get_hybrid_pose_from_stickman.py` | numpy, `smplx_rig_common.py` | Fitter B, plain fit + raw chains -> `pose_smplx_hybrid/` |
| `smplx_rig_common.py` | numpy | Shared rig module: bone hierarchy, A-pose rest, SMPL-X skinning, chain definitions and the channel-file format. Imported by `get_hybrid_pose_from_stickman.py` and by the Blender importer |
| `run_pipeline.py` | | Wrapper: mapping, fitter A and fitter B in order, skipping steps whose outputs exist |
| `visualize_joints.py` | plotly | Plots the mapped joints of one frame |
| `visualize_smplx.py` | plotly, smplx | Plots the smoothed mesh and joints of one frame |

## 7. Things to know

- **Template proportions do not match the data.** On a real 300-frame clip the thighs and shins measured 1.3-1.4 times the template, and the hips were about 0.25 m apart against 0.115 m in the template. The plain fit therefore misses the knees by up to 43 cm and the hips by 8 cm on average; the arms fit to a few centimetres. Betas are fixed at zero.
- **Raw data glitches carry through.** In that clip every left-hand keypoint jumps 60-70 mm at frame 75 (0-based; 76 in a 1-based viewer) and is back by the next frame, which shows up as 50-80 degree finger flips on frames 75 and 76. It is in the input, not in the solver.
- **Left / right.** Body indices 5, 7, 11, 13, 15 and the first hand block are the left side. If your data differs, use `--swap-body-sides` / `--swap-hand-sides` on the hybrid step.
- **Units** are metres throughout.
- **Smoothing.** The low-pass filters use 30 fps by default. For the meshes this is set at the top of the smoothing block in `get_mesh_from_smplx.py` (3 Hz for meshes, 4 Hz for joints); for the fit use `--fps` and `--cutoff-hz`.