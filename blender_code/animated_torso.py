
import bpy
import numpy as np
from pathlib import Path
from mathutils import Vector
import time
from datetime import datetime

# --- Ensure separate collections for Body and Hands, clearing old objects ---
for col_name in ("Body", "Hands"):
    if col_name in bpy.data.collections:
        col = bpy.data.collections[col_name]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        new_col = bpy.data.collections.new(col_name)
        bpy.context.scene.collection.children.link(new_col)

body_collection = bpy.data.collections["Body"]
hands_collection = bpy.data.collections["Hands"]

# --- Finger and Hand connectivity definitions ---
def FINGER(wrist, finger):
    return [
        (wrist, wrist + finger*4 + 1),
        (wrist + finger*4 + 1, wrist + finger*4 + 2),
        (wrist + finger*4 + 2, wrist + finger*4 + 3),
        (wrist + finger*4 + 3, wrist + finger*4 + 4)
    ]

def HAND(start):
    return (
        FINGER(start, 0)
        + FINGER(start, 1)
        + FINGER(start, 2)
        + FINGER(start, 3)
        + FINGER(start, 4)
    )

# --- Indices and connections ---
BODY_IDX       = 0
BODY_LEN       = 17
RIGHT_HAND_IDX = BODY_IDX + BODY_LEN
RIGHT_HAND_LEN = 21
LEFT_HAND_IDX  = RIGHT_HAND_IDX + RIGHT_HAND_LEN
LEFT_HAND_LEN  = 21

CONNECTIONS = [
    (0,1),(1,2),(1,5),(1,6),(5,7),(6,8),
    (7, RIGHT_HAND_IDX),(8, LEFT_HAND_IDX),
    (2,11),(2,12),(11,13),(12,14),(13,15),(14,16)
]
CONNECTIONS += HAND(RIGHT_HAND_IDX)
CONNECTIONS += HAND(LEFT_HAND_IDX)

HIDDEN = [9, 10, 3, 4, 13, 14, 15, 16]  # hide head, eyes, and leg keypoints

# Radii
BODY_RADIUS  = 0.03
WRIST_RADIUS = 0.015
HAND_RADIUS  = 0.007
FINGER_COLOR = (0.0,    0.392, 0.0,   1.0)   # dark green RGBA
GREY = (0.2,    0.2,   0.2,   1.0)


# --- Material helper ---
def create_material(name, color):
    # grab or make the material
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    
    # find the Principled BSDF node (assumes it’s there)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
    else:
        # fallback: create a new Principled node
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        # link it to the Material Output
        out = mat.node_tree.nodes.get('Material Output')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

# --- Connection class ---
class Connection:
    def __init__(self, start_kp, end_kp, parent=None):
        self.start = start_kp
        self.end   = end_kp
        # reuse if exists
        if parent:
            for ch in parent.children:
                if ch.name == f"Connection_{self.start.idx}_{self.end.idx}":
                    self.cyl = ch
                    self._assign_collection(self.cyl)
                    return
        # create new cylinder
        is_hand_conn = start_kp.idx >= RIGHT_HAND_IDX and end_kp.idx >= RIGHT_HAND_IDX
        conn_color   = GREY if is_hand_conn else (0.5, 0.5, 0.5, 1.0)
        self.mat = create_material(f"Connection_{self.start.idx}_{self.end.idx}_Material", conn_color)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.get_radius(), depth=1.0,
            location=self.get_position()
        )
        cyl = bpy.context.object
        cyl.name = f"Connection_{self.start.idx}_{self.end.idx}"
        cyl.data.materials.append(self.mat)
        if parent:
            cyl.parent = parent
        self._assign_collection(cyl)
        self.cyl = cyl

    def _assign_collection(self, obj):
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        a, b = self.start.idx, self.end.idx
        if a < BODY_LEN and b < BODY_LEN:
            body_collection.objects.link(obj)
        elif a >= BODY_LEN and b >= BODY_LEN:
            hands_collection.objects.link(obj)
        else:
            body_collection.objects.link(obj)
            hands_collection.objects.link(obj)

    def get_radius(self):
        return min(self.start.get_radius(), self.end.get_radius()) * 0.5

    def get_position(self):
        return (self.start.get_position() + self.end.get_position()) / 2

    def get_rotation(self):
        vec = self.end.get_position() - self.start.get_position()
        return vec.to_track_quat('Z','Y').to_euler()

    def get_depth(self):
        return (self.end.get_position() - self.start.get_position()).length

    def update(self):
        self.cyl.location       = self.get_position()
        self.cyl.rotation_euler = self.get_rotation()
        self.cyl.scale          = (1,1,self.get_depth())

    def keyframe(self, f):
        self.cyl.keyframe_insert(data_path='location', frame=f)
        self.cyl.keyframe_insert(data_path='rotation_euler', frame=f)
        self.cyl.keyframe_insert(data_path='scale', frame=f)

# --- Keypoint class ---
class Keypoint:
    def __init__(self, data, idx, parent=None):
        self.data  = data
        self.idx   = idx
        self.frame = 0
        # reuse if exists
        if parent:
            for ch in parent.children:
                if ch.name == f"Keypoint_{idx}":
                    self.sph = ch
                    self._assign_collection(self.sph)
                    return
        # create new sphere
        mat_color = FINGER_COLOR if idx >= RIGHT_HAND_IDX else (0.2, 0.2, 0.2, 1.0)
        self.mat = create_material(f"Keypoint_{idx}_Material",mat_color)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=self.get_radius(), location=self.get_position()
        )
        sph = bpy.context.object
        sph.name = f"Keypoint_{idx}"
        sph.data.materials.append(self.mat)
        if parent:
            sph.parent = parent
        self._assign_collection(sph)
        self.sph = sph

    def _assign_collection(self, obj):
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        if self.idx < BODY_LEN:
            body_collection.objects.link(obj)
        else:
            hands_collection.objects.link(obj)

    def get_radius(self):
        if self.idx in (RIGHT_HAND_IDX, LEFT_HAND_IDX):
            return WRIST_RADIUS
        if BODY_IDX <= self.idx < BODY_IDX + BODY_LEN:
            return BODY_RADIUS
        return HAND_RADIUS

    def get_position(self):
        return Vector(self.data.get(self.idx, self.frame))

    def goto_frame(self, f):
        self.frame = f
        self.sph.location = self.get_position()

    def keyframe(self, f):
        self.goto_frame(f)
        self.sph.keyframe_insert(data_path='location', frame=f)

# --- Keypoints container ---
class Keypoints:
    def __init__(self, data, conns, name="Keypoints"):
        empty = bpy.data.objects.get(name)
        if empty and empty.type=='EMPTY':
            self.parent = empty
        else:
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            self.parent = bpy.context.object
            self.parent.name = name
        self.kps = {}
        for i in range(data.len(0)):
            if i in HIDDEN: continue
            self.kps[i] = Keypoint(data, i, self.parent)
        self.conns = []
        for a,b in conns:
            if a in HIDDEN or b in HIDDEN: continue
            self.conns.append(Connection(self.kps[a], self.kps[b], self.parent))
        self.goto_frame(0)

    def goto_frame(self, f):
        for kp in self.kps.values(): kp.goto_frame(f)
        for c in self.conns:     c.update()

    def keyframe(self, f=None):
        if f is not None: self.goto_frame(f)
        for kp in self.kps.values(): kp.keyframe(f)
        for c  in self.conns:      c.keyframe(f)

# --- PoseData loader ---
class PoseData:
    def __init__(self, ds):
        root = Path(bpy.context.blend_data.filepath).parent/"output_3d"/ds
        self.body_file = root/"body_poses_3d.npz"
        self.hand_file = root/"hand_poses_3d.npz"
        self.load()

    def load(self):
        self.body = np.load(self.body_file)['poses_3d']
        self.hand = np.load(self.hand_file)['poses_3d']

    def len(self, f):
        return len(self.body[f]) + len(self.hand[f])

    def get(self, idx, f):
        body = self.body[f]
        hand = self.hand[f]
        # special midpoints for neck (1) and pelvis (2)
        if idx == 1:
            pos = (body[5] + body[6]) / 2
        elif idx == 2:
            pos = (body[11] + body[12]) / 2
        else:
            n_body = len(body)
            if 0 <= idx < n_body:
                pos = body[idx]
            elif idx < n_body + len(hand):
                pos = hand[idx - n_body]
            else:
                raise IndexError(f"Invalid keypoint index {idx}")
        
        return Vector((pos[0] + 0.27, pos[1] - 0.1, pos[2]+0.03))


# --- Main execution ---
folder = "test"
data = PoseData(folder)
stick = Keypoints(data, CONNECTIONS)
FRAMES = data.body.shape[0]
for frame in range(FRAMES):
    stick.goto_frame(frame)
    stick.keyframe(frame)
    bpy.context.scene.frame_set(frame)
    bpy.ops.wm.redraw_timer(type='DRAW_SWAP', iterations=1)
  
scene = bpy.context.scene
# 1) Set the global playback/render range (what the UI Start/End show)
scene.frame_start = 0
scene.frame_end   = FRAMES-1
scene.frame_current = scene.frame_start