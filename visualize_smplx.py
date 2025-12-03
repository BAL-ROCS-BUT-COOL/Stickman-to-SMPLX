import numpy as np
import plotly.graph_objects as go
import os
import smplx
import argparse

# ─── Arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualize SMPL-X mesh and joints")
parser.add_argument(
    "--frame",
    type=int,
    default=0,
    help="Frame index to visualize (0-based)"
)
args = parser.parse_args()

 

# ─── Load Mesh and Joints ─────────────────────────────────────
mesh = np.load('data/smoothed_all_meshes.npy')[args.frame]    # (N_vertices, 3)
joints = np.load('data/smoothed_all_joints.npy')[args.frame]  # (N_joints, 3)
print(mesh.shape)
print(joints.shape)
# Load faces from the SMPL-X model

model = smplx.create("models", model_type='smplx')
faces = model.faces  # (F, 3)

# ─── Plotly Mesh for the SMPL-X Body ───────────────────────────
mesh_plot = go.Mesh3d(
    x=mesh[:, 0],
    y=mesh[:, 1],
    z=mesh[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    color='#76c7c0',
    opacity=0.5,
    name='SMPL-X Mesh'
)

# ─── Scatter Plot for Joints with Labels ───────────────────────
scatter = go.Scatter3d(
    x=joints[:, 0],
    y=joints[:, 1],
    z=joints[:, 2],
    mode='markers+text',
    marker=dict(size=4, color='#ff6f61', opacity=0.9),
    text=[str(i) for i in range(len(joints))],
    textposition="top center",
    name='Joints'
)

# ─── Coordinate System Axes ────────────────────────────────────
axis_len = 0.1
origin = np.mean(mesh, axis=0)

axes = []
colors = ['#e07a5f', '#81b29a', '#3d405b']
labels = ['X', 'Y', 'Z']
vectors = np.eye(3) * axis_len

for i in range(3):
    axis = go.Scatter3d(
        x=[origin[0], origin[0] + vectors[i][0]],
        y=[origin[1], origin[1] + vectors[i][1]],
        z=[origin[2], origin[2] + vectors[i][2]],
        mode='lines+text',
        line=dict(width=6, color=colors[i]),
        text=[None, labels[i]],
        textposition="middle right",
        name=f"{labels[i]} axis"
    )
    axes.append(axis)

# ─── Layout and Display ────────────────────────────────────────
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectmode='data'
    ),
    title="SMPL-X Mesh and Joints",
    showlegend=True
)

fig = go.Figure(data=[mesh_plot, scatter] + axes, layout=layout)


import plotly.io as pio
pio.renderers.default = "browser"
fig.show()
