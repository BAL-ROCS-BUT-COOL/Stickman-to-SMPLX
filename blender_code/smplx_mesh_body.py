import numpy as np
import bpy
from pathlib import Path

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
folder = 'output'                               # subfolder in output_3d
translate = (1, -0.5, 0.06)               # final offset after grounding
subsurf_levels = 2                            # 0 to disable
object_name = "SMPLX_Body_Animated"           # name in the outliner

# File layout (relative to the .blend location or working dir if unsaved)
blend_path = bpy.data.filepath
base_dir = (Path(blend_path).parent if blend_path else Path.cwd())
mesh_file  = base_dir / folder / "all_meshes.npy"     # shape: (F, V, 3)
faces_file = base_dir / folder / "smplx_faces.npy"             # shape: (F_faces, 3)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
mesh_data = np.load(mesh_file)      # (num_frames, num_vertices, 3)
faces = np.load(faces_file)         # (num_faces, 3), dtype int
num_frames, num_vertices, _ = mesh_data.shape
print(f"Loaded {num_frames} frames, {num_vertices} vertices, {len(faces)} faces")

# ── TRANSFORM (swap axes, rotate around Z, ground to Z=0, translate) ─────────
def transform_mesh_data(arr, translate=(0.0, 0.0, 0.0)):
    out = arr.copy()

    # swap Y/Z like your original (x, z, y)
    out = np.stack([out[:, :, 0], out[:, :, 2], out[:, :, 1]], axis=-1)

    # 90° rotation variant from your original (flip x/y signs)
    Rz = np.array([[0, -1, 0],
                   [-1,  0, 0],
                   [0,   0, 1]], dtype=np.float32)
    out = out @ Rz.T

    # ground to Z=0 (global min over all frames/verts)
    out[:, :, 2] -= out[:, :, 2].min()

    # final translation
    out += np.array(translate, dtype=np.float32)
    return out

mesh_data = transform_mesh_data(mesh_data, translate=translate)

# ── CLEANUP OLD OBJECT/MESH ───────────────────────────────────────────────────
if object_name in bpy.data.objects:
    obj = bpy.data.objects[object_name]
    # unlink and remove mesh datablock if unique
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh, do_unlink=True)

if object_name in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes[object_name], do_unlink=True)

# ── CREATE FULL-BODY MESH FROM FRAME 0 ────────────────────────────────────────
mesh_datablock = bpy.data.meshes.new(object_name)
obj = bpy.data.objects.new(object_name, mesh_datablock)
bpy.context.collection.objects.link(obj)

verts_f0 = [tuple(v) for v in mesh_data[0]]
faces_all = faces.astype(int).tolist()  # ensure plain ints
mesh_datablock.from_pydata(verts_f0, [], faces_all)
mesh_datablock.update()

# Smooth shading
for poly in mesh_datablock.polygons:
    poly.use_smooth = True

# Optional subdivision
if subsurf_levels > 0:
    mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    mod.levels = subsurf_levels
    mod.render_levels = subsurf_levels

# ── SIMPLE MATERIAL (one for whole body) ──────────────────────────────────────
mat = bpy.data.materials.get("SMPLX_Mat") or bpy.data.materials.new("SMPLX_Mat")
mat.use_nodes = True
# set a soft gray base (Principled BSDF)
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.62, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.1
    bsdf.inputs["Roughness"].default_value = 0.45

if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)

# ── SHAPE KEYS (one per frame) ────────────────────────────────────────────────
obj.shape_key_add(name="Basis")
for f in range(num_frames):
    sk = obj.shape_key_add(name=f"Frame_{f:04d}")
    frame_verts = mesh_data[f]
    # assign vertex positions
    for vi in range(num_vertices):
        obj.data.shape_keys.key_blocks[f"Frame_{f:04d}"].data[vi].co = tuple(frame_verts[vi])

print("Shape keys created.")

# ── ANIMATE: each frame key toggles its corresponding shape key ───────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = num_frames
scene.frame_current = 1

# Zero all keys initially
for kb in obj.data.shape_keys.key_blocks:
    kb.value = 0.0

for f in range(num_frames):
    scene.frame_set(f + 1)
    # set all to 0 and keyframe
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
        kb.keyframe_insert(data_path="value")
    # turn on the one we want and keyframe
    key_name = f"Frame_{f:04d}"
    kb = obj.data.shape_keys.key_blocks.get(key_name)
    if kb:
        kb.value = 1.0
        kb.keyframe_insert(data_path="value")

print("Full-body animation ready.")