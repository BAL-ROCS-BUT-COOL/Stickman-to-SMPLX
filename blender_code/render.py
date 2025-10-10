import numpy as np
import bpy
from pathlib import Path
import bmesh
from collections import deque
import time
from datetime import datetime




# ---- Force GPU usage (for RTX 3060) ----
prefs = bpy.context.preferences.addons['cycles'].preferences
# Use OPTIX if available, else fallback to CUDA
prefs.compute_device_type = 'OPTIX' if 'OPTIX' in prefs.get_device_types(bpy.context) else 'CUDA'
# Force Blender to refresh the list of devices
prefs.get_devices()
# Enable all devices (e.g., RTX 3060)
for device in prefs.devices:
    if not device.use:
        device.use = True

# Make sure the scene is set to use GPU
bpy.context.scene.cycles.device = 'GPU'
print("Enabled Cycles devices:")
for d in prefs.devices:
    print(f"  - {d.name} | Type: {d.type} | Use: {d.use}")


#------ rendering ---------------

def enable_still_settings():
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format        = 'MPEG4'
    scene.render.ffmpeg.codec         = 'H264'
    scene.render.ffmpeg.audio_codec   = 'NONE'
    

def enable_animation_settings():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.audio_codec = 'NONE'
    bpy.context.scene.eevee.taa_render_samples = 64


scene = bpy.context.scene

#enable_animation_settings()    # <----fast
enable_still_settings()         # <---- good


# 1) Timeline & output settings
scene.render.fps  = 30



## Optional: static camera list
#camera_names = ['Camera','Camera.1','Camera.2','Camera.3','Camera.4']
camera_names = ['GoPro9']

type = "hands_smplx" # "stickman"
folder = 'cha4'
output_dir = Path(bpy.path.abspath(f"//renders//{folder}"))
output_dir.mkdir(exist_ok=True)


# Choose either animation or single‐frame mode:
RENDER_ONE_FRAME = True
FRAME_TO_RENDER   = 1   # pick whichever frame you’d like


if RENDER_ONE_FRAME:
    for camera in camera_names:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cam = bpy.data.objects.get(camera)
        scene.camera = cam
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode  = 'RGB'  # or 'RGB'
        
        # Jump to your desired frame
        bpy.context.scene.frame_set(FRAME_TO_RENDER)
        
        # Build a filepath
        png_path = output_dir / f"{folder}_{camera}_{type}_{timestamp}.png"
        bpy.context.scene.render.filepath = str(png_path)
        
        # Render just that one still
        bpy.ops.render.render(write_still=True)
        print(f"Rendered single frame → {png_path}")
else:
    # your existing camera loop for animations
    for cam_name in camera_names:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cam = bpy.data.objects.get(cam_name)
        if cam is None:
            print(f"Camera '{cam_name}' not found, skipping.")
            continue
        bpy.context.scene.camera = cam
        bpy.context.scene.render.filepath = str(output_dir / f"{folder}_{cam_name}_{type}_{timestamp}.mp4")
        bpy.ops.render.render(animation=True)
        print(f"Rendered torso animation from {cam_name} → {bpy.context.scene.render.filepath}")
