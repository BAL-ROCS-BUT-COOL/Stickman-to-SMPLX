import json
import bpy
from mathutils import Matrix

# =================== EDIT THESE IF NEEDED ===================
JSON_RT_PATH = r"C:\Users\maga\SyntheticDataORX\code\aligned_poses.json"      # RT (world→camera)
RT_KEY       = "blender2gopro9"                          # key in aligned_poses.json
JSON_K_PATH  = r"C:\Users\maga\SyntheticDataORX\code\gopro9_synced_intrinsics.json" # intrinsics + distortion
CAMERA_NAME = "GoPro9"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Image resolution used when K was estimated (adjust to your footage)
IMG_W = 3840  # pixels
IMG_H = 2160  # pixels

# GoPro HERO9-ish sensor width in mm (tweak if you have exact)
SENSOR_WIDTH_MM  = 6.3
SENSOR_HEIGHT_MM = 5.5  # keep aspect consistent
# ===========================================================

# --- Load extrinsics (world→camera in CV convention) ---
with open(JSON_RT_PATH, "r") as f:
    extr = json.load(f)
if RT_KEY not in extr:
    raise KeyError(f"'{RT_KEY}' not found in {JSON_RT_PATH}. Keys: {list(extr.keys())}")

M = Matrix([list(map(float, row)) for row in extr[RT_KEY]])  # 4x4 world→cam (CV)

# Invert to get camera pose (cam→world) in CV coords
cam_to_world_cv = M.inverted()

# Constant basis change CV(+X,+Y↓,+Z→) → Blender(+X,+Y↑,−Z→)
S_h = Matrix((
    ( 1.0,  0.0,  0.0, 0.0),
    ( 0.0, -1.0,  0.0, 0.0),
    ( 0.0,  0.0, -1.0, 0.0),
    ( 0.0,  0.0,  0.0, 1.0),
))
cam_to_world_blender = cam_to_world_cv @ S_h

# --- Create or fetch camera ---
cam = bpy.data.objects.get(CAMERA_NAME)
if cam is None:
    cam_data = bpy.data.cameras.new(CAMERA_NAME)
    cam = bpy.data.objects.new(CAMERA_NAME, cam_data)
    bpy.context.scene.collection.objects.link(cam)

# Apply world transform
cam.matrix_world = cam_to_world_blender

# --- Load intrinsics (K and distortion) ---
with open(JSON_K_PATH, "r") as f:
    kin = json.load(f)

K = kin["sensors"]["RGB"]["intrinsics"]["data"]  # row-major 3x3
k1,k2,p1,p2,k3 = kin["sensors"]["RGB"]["distortionCoefficients"]["data"]
# K = [[fx, 0, cx],
#      [0, fy, cy],
#      [0,  0,  1]]
fx = K[0]; cx = K[2]
fy = K[4]; cy = K[5]

# --- Push intrinsics to Blender camera ---
cam.data.type = 'PERSP'
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width  = SENSOR_WIDTH_MM
cam.data.sensor_height = SENSOR_HEIGHT_MM

# Focal length in mm (for HORIZONTAL fit): f_mm = fx * sensor_width / image_width
f_mm = fx * cam.data.sensor_width / IMG_W
cam.data.lens = float(f_mm)
print(f_mm)
# Principal point → shift_x, shift_y (normalized in filmback units)
# Positive shift_x moves principal point RIGHT in render.
# Y note: CV uses Y down; Blender uses Y up, so we flip the sign.
cam.data.shift_x = (cx - (IMG_W * 0.5)) / IMG_W
cam.data.shift_y = - (cy - (IMG_H * 0.5)) / IMG_H

# --- Scene render resolution (to match K’s image size) ---
scene = bpy.context.scene
scene.render.resolution_x = IMG_W
scene.render.resolution_y = IMG_H
scene.render.pixel_aspect_x = 1
scene.render.pixel_aspect_y = 1

# --- Optional: lightweight radial distortion in compositor (k1 only) ---
# Blender's 'Lens Distortion' node approximates first-order radial distortion.
use_compositor_distortion = True
if use_compositor_distortion:
    scene.use_nodes = True
    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    n_in   = nt.nodes.new("CompositorNodeRLayers")
    n_ld   = nt.nodes.new("CompositorNodeLensdist")
    n_ld.inputs["Distortion"].default_value = float(k1)  # approximate with k1
    n_ld.use_projector = False
    n_ld.inputs["Dispersion"].default_value = 0.0
    n_ld.use_fit = True
    n_out  = nt.nodes.new("CompositorNodeComposite")

    nt.links.new(n_in.outputs["Image"], n_ld.inputs["Image"])
    nt.links.new(n_ld.outputs["Image"], n_out.inputs["Image"])

# --- Convenience: select camera and make it active ---
for o in bpy.context.selected_objects:
    o.select_set(False)
cam.select_set(True)
bpy.context.view_layer.objects.active = cam

print(f"Camera '{CAMERA_NAME}' posed from {RT_KEY} and calibrated: ")