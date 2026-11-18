Example data you can use that works with the scripts: 300 frames of one clip.

Inputs: `body_poses_3d.npz`, `hand_poses_3d.npz`.
Made by `mapping_stickman_to_smplx.py`: `smplx_joints.npy`, `joint_transforms.npy`.
Rig description: `skinned_smplx/armature_rig_metadata.npz` (the same for every clip).
`import_config.json`: names and files for the Blender side.

Not included, because they need the SMPL-X model files (see the main README): the rest of `skinned_smplx/`, `pose_smplx_hybrid/` and `meshed_smplx/`.
Make the first two with `python run_pipeline.py --model models --raw hands`, or all three with the added `--meshes`.
