# Stickman to smplx

This repository uses the conventional 3d points of a person and creates the smplx mesh out of them. Using the mesh, it can then be imported in Blender.
This repo uses python 3.11.9
![Stickman 3D visualization](assets/both_plots_demo.png)
**Figure 1.** Stickman 3D visualization.

 
## Setup Instructions

### Setup Environment
Create a new Python virtual environment (recommended to keep dependencies isolated) and download requirements.

- **Windows (PowerShell or CMD):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt

Download the smplx models from the [project website](https://smpl-x.is.tue.mpg.de/) and put the models folder in the same directory.

## Generate SMPL-X Joint Points

This script maps your custom 3D body and hand keypoints into the **SMPL-X joint format**.

### 1. Map stickman data to SMPL-X
Run the mapping script with your input body and hand joint files:

```
python mapping_stickman_to_smplx.py --body BODY_FILE --hand HAND_FILE [--output OUTPUT_FILE]
```

Now visualize the data using the **`visualize_joints.py`** file. 
- Make sure that the body stands along the positive Y-axis. 
- The pelvis should be close to the origin (0,0,0). 
- The Person should be facing the positive Z-direction 
- The left shoulder should be in the positive X-direction and right shoulder in negative X-direction (exactly like the right person in [Figure 1](#figure-1-stickman-3d-visualization).)

If this is not the case, use the permute_axes() function to adjust this and try again. Otherwise the SMPL-X mesh will not look good.

### 2. Get the SMPL-X Mesh
Run the script to approximate the best fitting smpl-x mesh to the given joints:

```
python get_mesh_from_3dpoints.py
```
This outputs 4 files:
- **`all_meshes.npy`** 
- **`all_joints.npy`** 
- **`all_meshes_smoothed.npy`** (applied low-pass filter)
- **`all_joints_smoothed.npy`**  (applied low-pass filter)

I recomend using the smoothed files for renders.

### 3. Visualize the mesh
After generating the meshes and joints, you can visualize any frame with:

```
python visualize.py --frame FRAME_NUMBER
```
This will open an interactive 3D plot (saved as 3d_smplx_plot.html) showing the SMPL-X body mesh along with joint markers and axes.


## Blender Implementation

### Setup
1. Place the output files `all_meshes_smoothed.npy` and `all_joints_smoothed.npy` inside the `output/` folder.  
2. Open your Blender project.  
3. Switch to the **Scripting** tab.  
4. Click **Open** → select the `output/` folder → choose `smplx_mesh_body.py`.  
5. Run the script.  

### File Overview
- **`smplx_mesh_body.py`**  
  Displays the SMPL-X body for all frames.  

- **`smplx_mesh_hands.py`**  
  Displays only the hands for all frames.  

- **`smplx_mesh_body_and_hands.py`**  
  Displays the SMPL-X body for all frames and highlights the hands in color.  

### Rendering
To preview the animation in Blender:
1. Click the **Python Console** icon (left-hand side).  
2. Switch to the **Timeline** view.  
3. Press **Play**.  

---

![Blender SMPL-X Visualization](assets/blender.png)
