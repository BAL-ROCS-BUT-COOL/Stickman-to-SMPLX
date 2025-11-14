import numpy as np
import argparse
import plotly.graph_objects as go
from joints import joint_mapping, apply_joint_mapping

parser = argparse.ArgumentParser(description="Visualize SMPL-X mesh and joints")
parser.add_argument(
    "--frame",
    type=int,
    default=0,
    help="Frame index to visualize (0-based)"
)
args = parser.parse_args()

unmapped_joints = np.load('smplx_joints.npy')[args.frame]  # (N_joints, 3)
print(unmapped_joints.shape)

joints, mask = apply_joint_mapping(unmapped_joints, joint_mapping)
print(joints[mask])

RIGHT_HAND_IDX = 15
LEFT_HAND_IDX = RIGHT_HAND_IDX + 21

connections = [
    (0,1),(1,2),(1,5),(1,6),(6,8),(5,7),(2,10),(2,9),
    (7, RIGHT_HAND_IDX),(8, LEFT_HAND_IDX),
    (10,12),(12,14),(9,11),(11,13)
]

def finger_edges(wrist_idx):
    edges = []
    for f in range(5):
        base = wrist_idx + f*4 + 1
        edges.extend([
            (wrist_idx, base), (base, base+1),
            (base+1, base+2), (base+2, base+3)
        ])
    return edges

connections += finger_edges(RIGHT_HAND_IDX)
connections += finger_edges(LEFT_HAND_IDX)

# ─── Create Plotly traces ─────────────────────────────────────────────────────
# Scatter + labels
scatter = go.Scatter3d(
    x=joints[:,0], y=joints[:,1], z=joints[:,2],
    mode='markers+text',
    text=[str(i) for i in range(len(joints))],
    textposition='top center',
    marker=dict(size=5)
)
# Line segments
line_traces = []
for i, j in connections:
    line_traces.append(go.Scatter3d(
        x=[joints[i,0], joints[j,0], None],
        y=[joints[i,1], joints[j,1], None],
        z=[joints[i,2], joints[j,2], None],
        mode='lines',
        line=dict(width=2,color='blue')
    ))
# ─── Figure ───────────────────────────────────────────────────────────────────
fig = go.Figure(data=[scatter]+ line_traces)
fig.update_layout(
    title="Interactive Stickman (Frame 0)",
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    )
)

import plotly.io as pio
pio.renderers.default = "browser"
fig.show()