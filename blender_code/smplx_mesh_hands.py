import numpy as np
import bpy
from pathlib import Path
import bmesh
from collections import deque
import time
from datetime import datetime
# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------
folder = 'test'
base_path = Path(bpy.context.blend_data.filepath).parent / "output_3d"
mesh_file = base_path / folder / "smoothed_all_meshes.npy"
faces_file = base_path / "smplx_faces.npy"
joints_file = base_path / folder / "smoothed_all_joints.npy"

vertex_radius = 0.06
neighborhood_hops = 2
min_vertices_for_hand = 50
require_all_vertices_in_face = False  # True = stricter

# joint indices for SMPL-X
left_wrist_idx = 20
right_wrist_idx = 21
fingertip_indices = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
mesh_data = np.load(mesh_file)      # (frames, num_vertices, 3)
faces = np.load(faces_file)         # (num_faces, 3)
all_joints = np.load(joints_file)   # (frames, n_joints, 3)

num_frames, num_vertices, _ = mesh_data.shape
num_faces = faces.shape[0]
print(f"Loaded {num_frames} frames, {num_vertices} vertices, {num_faces} faces.")

# ------------------------------------------------------------------
# BUILD ADJACENCY
# ------------------------------------------------------------------
def build_vertex_adjacency(num_vertices, faces):
    adj = [[] for _ in range(num_vertices)]
    for f in faces:
        a, b, c = map(int, f)
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])
    return [list(set(neis)) for neis in adj]

adj = build_vertex_adjacency(num_vertices, faces)

def expand_vertex_set(seed_set, adjacency, hops=3):
    visited = set(seed_set)
    queue = deque((v, 0) for v in seed_set)
    while queue:
        v, depth = queue.popleft()
        if depth >= hops:
            continue
        for nei in adjacency[v]:
            if nei not in visited:
                visited.add(nei)
                queue.append((nei, depth + 1))
    return visited

# ------------------------------------------------------------------
# FIND HAND VERTICES (frame 0)
# ------------------------------------------------------------------
def find_hand_vertices(vertices_frame, joints_frame):
    seed_vertex_indices = set()

    # joints-based seeds
    candidate_joint_positions = []
    if left_wrist_idx is not None:
        candidate_joint_positions.append(joints_frame[left_wrist_idx])
    if right_wrist_idx is not None:
        candidate_joint_positions.append(joints_frame[right_wrist_idx])
    for fi in fingertip_indices:
        if fi < joints_frame.shape[0]:
            candidate_joint_positions.append(joints_frame[fi])
    candidate_joint_positions = np.array(candidate_joint_positions)

    for jp in candidate_joint_positions:
        dists = np.linalg.norm(vertices_frame - jp.reshape(1, 3), axis=1)
        near_idx = np.where(dists <= vertex_radius)[0]
        seed_vertex_indices.update(near_idx.tolist())

    if len(seed_vertex_indices) < min_vertices_for_hand:
        # fallback heuristic
        xs = vertices_frame[:, 0]
        ys = vertices_frame[:, 1]
        pct = 0.05
        left_cut = np.percentile(xs, pct * 100)
        right_cut = np.percentile(xs, 100 - pct * 100)
        left_candidates = np.where(xs <= left_cut)[0]
        right_candidates = np.where(xs >= right_cut)[0]
        center = vertices_frame.mean(axis=0)
        dists_center = np.linalg.norm(vertices_frame - center.reshape(1, 3), axis=1)
        far_candidates = np.where(dists_center >= np.percentile(dists_center, 95))[0]
        seed_vertex_indices.update(left_candidates.tolist())
        seed_vertex_indices.update(right_candidates.tolist())
        seed_vertex_indices.update(far_candidates.tolist())

    expanded_vertices = expand_vertex_set(seed_vertex_indices, adj, neighborhood_hops)
    return expanded_vertices

hand_vertices = find_hand_vertices(mesh_data[0], all_joints[0])
print(f"Hand vertices detected: {len(hand_vertices)}")

# ------------------------------------------------------------------
# CREATE FACE MASK
# ------------------------------------------------------------------
hand_mask = np.zeros(num_vertices, dtype=bool)
hand_mask[list(hand_vertices)] = True

kept_face_indices = []
for fi in range(num_faces):
    a, b, c = faces[fi]
    if require_all_vertices_in_face:
        if hand_mask[a] and hand_mask[b] and hand_mask[c]:
            kept_face_indices.append(fi)
    else:
        if hand_mask[a] or hand_mask[b] or hand_mask[c]:
            kept_face_indices.append(fi)

print(f"Kept {len(kept_face_indices)} faces for hands.")

# ------------------------------------------------------------------
# CREATE MESH OBJECT
# ------------------------------------------------------------------
mesh_name = "SMPLX_Hands_Animated"
if mesh_name in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes[mesh_name], do_unlink=True)

hand_mesh = bpy.data.meshes.new(mesh_name)
hand_obj = bpy.data.objects.new("SMPLX_Hands_Animated", hand_mesh)
bpy.context.collection.objects.link(hand_obj)

verts_list = [tuple(v) for v in mesh_data[0].tolist()]
faces_list = faces[kept_face_indices].tolist()
hand_mesh.from_pydata(verts_list, [], faces_list)
hand_mesh.update()


for poly in hand_mesh.polygons:
    poly.use_smooth = True

subsurf = hand_obj.modifiers.new(name="Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 2

# ------------------------------------------------------------------
# MATERIAL: dark green for hands
# ------------------------------------------------------------------
#mat_name = "Hands_DarkGreen"
#mat = bpy.data.materials.get(mat_name)
#if mat is None:
#    mat = bpy.data.materials.new(name=mat_name)
#    mat.use_nodes = True
#    # Set Principled BSDF base color to dark green
#    bsdf = mat.node_tree.nodes.get("Principled BSDF")
#    if bsdf:
#        bsdf.inputs["Base Color"].default_value = (0.05, 0.35, 0.05, 1.0)  # RGBA
#        bsdf.inputs["Roughness"].default_value = 0.6

## Ensure the object has the material in slot 0
#if hand_obj.data.materials:
#    hand_obj.data.materials[0] = mat
#else:
#    hand_obj.data.materials.append(mat)

## (Optional) viewport color so it looks green in Solid view too
#mat.diffuse_color = (0.05, 0.35, 0.05, 1.0)

## Assign the material to all polygons (all are hands here)
#for poly in hand_mesh.polygons:
#    poly.material_index = 0

# ------------------------------------------------------------------
# TRANSFORMATION FUNCTION
# ------------------------------------------------------------------
def transform_mesh_data(mesh_data, translate=(0.0, 0.0, 0.0)):
    transformed_data = np.copy(mesh_data)
    rotation_matrix_z = np.array([
        [0, -1, 0],
        [-1,  0, 0],
        [0,  0, 1]
    ])
    for frame in range(mesh_data.shape[0]):
        swapped_data = np.column_stack([
            mesh_data[frame][:, 0],
            mesh_data[frame][:, 2],
            mesh_data[frame][:, 1]
        ])
        rotated_data = swapped_data @ rotation_matrix_z.T
        transformed_data[frame] = rotated_data
    global_min_z = np.min(transformed_data[:, :, 2])
    transformed_data[:, :, 2] -= global_min_z
    transformed_data += np.array(translate)
    return transformed_data

trans = (0,0,0)
base = (0.49, -0.03, 0.06)
if folder=='cha1':
    trans = (0,-0.04,-0.02)
if folder=='cha2':
    trans = (0.23,0.23,-0.09)
if folder=='cha3':
    trans = (0.25,0.06,-0.05)
if folder=='cha4':
    trans = (0.69,-0.68,-0.05)
if folder=='cha5':
    trans = (0.05,-0.03,-0.02)
if folder=='cha6':
    trans = (0,0.01,-0.03)
if folder=='cha7':
    trans = (0,0.04,-0.03)
if folder=='anna1':
    trans = (0.09,0,0)
if folder=='anna2':
    trans = (0.09,0,0)
if folder=='anna3':
    trans = (0.09,0,0)
    
translation = tuple(x + y for x, y in zip(base, trans))
mesh_data = transform_mesh_data(mesh_data, translate=translation) 

# ------------------------------------------------------------------
# SHAPE KEYS
# ------------------------------------------------------------------
hand_obj.shape_key_add(name="Basis")
for frame_idx in range(num_frames):
    sk = hand_obj.shape_key_add(name=f"Frame_{frame_idx:04d}")
    sk_data = mesh_data[frame_idx]
    for vidx in range(num_vertices):
        if vidx < len(sk.data):
            sk.data[vidx].co = sk_data[vidx]
    sk.value = 0.0

print("Shape keys created for all frames.")

# ------------------------------------------------------------------
# ANIMATE
# ------------------------------------------------------------------
for frame_idx in range(num_frames):
    bpy.context.scene.frame_set(frame_idx + 1)
    for sk in hand_obj.data.shape_keys.key_blocks:
        sk.value = 0.0
        sk.keyframe_insert(data_path="value")
    key_name = f"Frame_{frame_idx:04d}"
    if key_name in hand_obj.data.shape_keys.key_blocks:
        hand_obj.data.shape_keys.key_blocks[key_name].value = 1.0
        hand_obj.data.shape_keys.key_blocks[key_name].keyframe_insert(data_path="value")

print("Hand-only animation with dark green material ready.")

scene = bpy.context.scene
# 1) Set the global playback/render range (what the UI Start/End show)
scene.frame_start = 1
scene.frame_end   = num_frames
scene.frame_current = 1