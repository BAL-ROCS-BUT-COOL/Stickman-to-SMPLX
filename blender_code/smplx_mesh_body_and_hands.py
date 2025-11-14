import numpy as np
import bpy
from pathlib import Path
from collections import deque

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
folder = 'test'                               # subfolder in output_3d
translate = (1.0, -0.5, 0.06)                 # final offset after grounding
subsurf_levels = 2                            # 0..3 typically
object_name = "SMPLX_BodyHands_Animated"      # name in the outliner

# Hand region detection
vertex_radius = 0.06
neighborhood_hops = 2
min_vertices_for_hand = 50
require_all_vertices_in_face = False  # True = stricter, gives crisper wrist cutoff

# SMPL-X joints (for seeds)
left_wrist_idx = 20
right_wrist_idx = 21
fingertip_indices = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

# ── FILES ─────────────────────────────────────────────────────────────────────
blend_path = bpy.data.filepath
base_dir = (Path(blend_path).parent if blend_path else Path.cwd()) / "output_3d"
mesh_file   = base_dir / folder / "smoothed_all_meshes.npy"     # (F, V, 3)
faces_file  = base_dir / "smplx_faces.npy"             # (F_faces, 3)
joints_file = base_dir / folder / "smoothed_all_joints.npy"   # (F, n_joints, 3)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
mesh_data = np.load(mesh_file)      # (num_frames, num_vertices, 3)
faces = np.load(faces_file).astype(int)  # (num_faces, 3)
all_joints = np.load(joints_file)   # (num_frames, n_joints, 3)
num_frames, num_vertices, _ = mesh_data.shape
num_faces = faces.shape[0]
print(f"Loaded {num_frames} frames, {num_vertices} vertices, {num_faces} faces")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def build_vertex_adjacency(nv, f):
    adj = [[] for _ in range(nv)]
    for tri in f:
        a, b, c = map(int, tri)
        adj[a].extend([b, c]); adj[b].extend([a, c]); adj[c].extend([a, b])
    return [list(set(neis)) for neis in adj]

def expand_vertex_set(seed_set, adjacency, hops=3):
    visited = set(seed_set)
    q = deque((v, 0) for v in seed_set)
    while q:
        v, d = q.popleft()
        if d >= hops:
            continue
        for nei in adjacency[v]:
            if nei not in visited:
                visited.add(nei)
                q.append((nei, d+1))
    return visited

def find_hand_vertices(vertices_frame, joints_frame):
    seeds = set()
    cand = []
    if left_wrist_idx is not None: cand.append(joints_frame[left_wrist_idx])
    if right_wrist_idx is not None: cand.append(joints_frame[right_wrist_idx])
    for fi in fingertip_indices:
        if fi < joints_frame.shape[0]: cand.append(joints_frame[fi])
    cand = np.array(cand)

    # joint-proximity seeds
    for jp in cand:
        d = np.linalg.norm(vertices_frame - jp.reshape(1, 3), axis=1)
        near = np.where(d <= vertex_radius)[0]
        seeds.update(near.tolist())

    # fallback if too few
    if len(seeds) < min_vertices_for_hand:
        xs = vertices_frame[:, 0]; ys = vertices_frame[:, 1]
        pct = 0.05
        left_cut = np.percentile(xs, pct*100)
        right_cut = np.percentile(xs, 100 - pct*100)
        left_candidates  = np.where(xs <= left_cut)[0]
        right_candidates = np.where(xs >= right_cut)[0]
        center = vertices_frame.mean(axis=0)
        distc = np.linalg.norm(vertices_frame - center.reshape(1, 3), axis=1)
        far_candidates = np.where(distc >= np.percentile(distc, 95))[0]
        seeds.update(left_candidates.tolist())
        seeds.update(right_candidates.tolist())
        seeds.update(far_candidates.tolist())

    adj = build_vertex_adjacency(vertices_frame.shape[0], faces)
    return expand_vertex_set(seeds, adj, neighborhood_hops)

def transform_mesh_data(arr, translate=(0.0, 0.0, 0.0)):
    out = arr.copy()
    # swap (x, y, z) -> (x, z, y)
    out = np.stack([out[:, :, 0], out[:, :, 2], out[:, :, 1]], axis=-1)
    # 90°-ish rotation like your original (flip x/y signs)
    Rz = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    out = out @ Rz.T
    # ground to Z=0 (global min)
    out[:, :, 2] -= out[:, :, 2].min()
    # translate
    out += np.array(translate, dtype=np.float32)
    return out

# ── DETECT HAND VERTICES (on frame 0, in original coords) ─────────────────────
hand_vertices = find_hand_vertices(mesh_data[0], all_joints[0])
hand_mask = np.zeros(num_vertices, dtype=bool)
hand_mask[list(hand_vertices)] = True
print(f"Hand vertices detected: {hand_mask.sum()}")

# ── TRANSFORM GEOMETRY ────────────────────────────────────────────────────────
mesh_data = transform_mesh_data(mesh_data, translate=translate)

# ── CLEANUP OLD OBJECT/MESH ───────────────────────────────────────────────────
if object_name in bpy.data.objects:
    old_obj = bpy.data.objects[object_name]
    old_mesh = old_obj.data
    bpy.data.objects.remove(old_obj, do_unlink=True)
    if old_mesh and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh, do_unlink=True)
if object_name in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes[object_name], do_unlink=True)

# ── CREATE MESH WITH MATERIAL INDICES (0 = body, 1 = hands) ───────────────────
mesh_datablock = bpy.data.meshes.new(object_name)
obj = bpy.data.objects.new(object_name, mesh_datablock)
bpy.context.collection.objects.link(obj)

verts_f0 = [tuple(v) for v in mesh_data[0]]
mesh_datablock.from_pydata(verts_f0, [], faces.tolist())
mesh_datablock.update()

# Smooth shading
for poly in mesh_datablock.polygons:
    poly.use_smooth = True

# Assign two materials (slot 0 = body, slot 1 = hands)
body_mat = bpy.data.materials.get("SMPLX_Body_White") or bpy.data.materials.new("SMPLX_Body_White")
hand_mat = bpy.data.materials.get("SMPLX_Hands_Green") or bpy.data.materials.new("SMPLX_Hands_Green")
body_mat.use_nodes = True
hand_mat.use_nodes = True

bsdf_body = body_mat.node_tree.nodes.get("Principled BSDF")
if bsdf_body:
    bsdf_body.inputs["Base Color"].default_value = (0.85, 0.85, 0.87, 1.0)  # light gray / white
    bsdf_body.inputs["Metallic"].default_value = 0.0
    bsdf_body.inputs["Roughness"].default_value = 0.45

bsdf_hand = hand_mat.node_tree.nodes.get("Principled BSDF")
if bsdf_hand:
    bsdf_hand.inputs["Base Color"].default_value = (0.05, 0.35, 0.05, 1.0)  # dark green
    bsdf_hand.inputs["Roughness"].default_value = 0.6

# Ensure material slots and order
obj.data.materials.clear()
obj.data.materials.append(body_mat)  # index 0
obj.data.materials.append(hand_mat)  # index 1

# Tag polygons: hands vs body based on vertex membership
# If require_all_vertices_in_face=True, only faces fully inside hand_mask get green.
for poly in mesh_datablock.polygons:
    a, b, c = [loop.vertex_index for loop in [mesh_datablock.loops[i] for i in range(poly.loop_start, poly.loop_start + poly.loop_total)]][0:3]
    if require_all_vertices_in_face:
        is_hand = hand_mask[a] and hand_mask[b] and hand_mask[c]
    else:
        is_hand = hand_mask[a] or hand_mask[b] or hand_mask[c]
    poly.material_index = 1 if is_hand else 0

# Optional subdivision
if subsurf_levels > 0:
    mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    mod.levels = subsurf_levels
    mod.render_levels = subsurf_levels

# (Optional) make it look green in Solid view too
hand_mat.diffuse_color = (0.05, 0.35, 0.05, 1.0)
body_mat.diffuse_color = (0.85, 0.85, 0.87, 1.0)

# ── SHAPE KEYS (animate like your original) ───────────────────────────────────
obj.shape_key_add(name="Basis")
for f in range(num_frames):
    sk = obj.shape_key_add(name=f"Frame_{f:04d}")
    frame_verts = mesh_data[f]
    # assign vertex positions
    sk_block = obj.data.shape_keys.key_blocks[f"Frame_{f:04d}"]
    for vi in range(num_vertices):
        sk_block.data[vi].co = tuple(frame_verts[vi])

print("Shape keys created.")

# ── ANIMATE: each frame toggles its corresponding shape key ───────────────────
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = num_frames
scene.frame_current = 1

# zero all keys initially
for kb in obj.data.shape_keys.key_blocks:
    kb.value = 0.0

for f in range(num_frames):
    scene.frame_set(f + 1)
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
        kb.keyframe_insert(data_path="value")
    key_name = f"Frame_{f:04d}"
    kb = obj.data.shape_keys.key_blocks.get(key_name)
    if kb:
        kb.value = 1.0
        kb.keyframe_insert(data_path="value")

print("Full-body animation with green hands ready.")
