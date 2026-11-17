import torch
import smplx
import numpy as np
import argparse
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

from weights import load_fit_weights

JOINT_WEIGHTS = None


def infer_full_mesh_from_partial_joints(
    partial_joints_np,
    smplx_model_path,
    missing_threshold=1e-6,
    device=None,
):
    """partial_joints_np: (N_joints, 3) numpy array (N_joints should match model joint ordering, e.g. 76)"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    partial_joints = torch.tensor(partial_joints_np, dtype=torch.float32, device=device)

    joint_norms = torch.norm(partial_joints, dim=1)
    valid_mask = joint_norms > missing_threshold
    num_valid = valid_mask.sum().item()
    print(f"Valid joints: {num_valid}/{len(valid_mask)}")

    model = smplx.create(
        model_path=smplx_model_path,
        model_type="smplx",
        gender="male",
        use_pca=False,
        num_pca_comps=12,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_face_contour=False,
    ).to(device)

    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, 63), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=False)
    left_hand_pose = torch.zeros((1, 45), device=device, requires_grad=True)
    right_hand_pose = torch.zeros((1, 45), device=device, requires_grad=True)
    transl = torch.zeros((1, 3), device=device, requires_grad=True)


    joint_weights = JOINT_WEIGHTS if JOINT_WEIGHTS is not None else load_fit_weights()
    weights = torch.as_tensor(joint_weights, dtype=torch.float32, device=device)[: len(valid_mask)]

    def weighted_mse_loss(pred_joints, target_joints, valid_mask, weights):
        diff = (pred_joints - target_joints) ** 2
        weighted = diff * weights.unsqueeze(1)
        valid_rows = weighted[valid_mask]
        if valid_rows.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return valid_rows.mean()

    optimizer_stage1 = torch.optim.Adam(
        [global_orient, body_pose, transl], lr=0.02
    )
    n_iter_stage1 = 200
    for i in range(n_iter_stage1):
        optimizer_stage1.zero_grad()
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        loss.backward()
        optimizer_stage1.step()
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Stage1] Iter {i+1}/{n_iter_stage1}  loss={loss.item():.8f}")

    params_stage2 = [
        global_orient,
        body_pose,
        transl,
        left_hand_pose,
        right_hand_pose,
    ]
    optimizer_stage2 = torch.optim.Adam(params_stage2, lr=0.01)
    n_iter_stage2 = 400
    for i in range(n_iter_stage2):
        optimizer_stage2.zero_grad()
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
        )
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        reg = 1e-6 * (
            global_orient.pow(2).sum()
            + body_pose.pow(2).sum()
            + left_hand_pose.pow(2).sum()
            + right_hand_pose.pow(2).sum()
        )
        (loss + reg).backward()
        optimizer_stage2.step()
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Stage2] Iter {i+1}/{n_iter_stage2}  loss={loss.item():.8f}")

    params_refine = [global_orient, body_pose, transl, left_hand_pose, right_hand_pose]
    for p in params_refine:
        p.requires_grad = True

    optimizer_refine = torch.optim.LBFGS(params_refine, max_iter=50, line_search_fn="strong_wolfe", lr=1.0)

    print("Starting L-BFGS refinement (this may take a little while)...")

    def closure():
        optimizer_refine.zero_grad()
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
        )
        predicted_joints = output.joints[0, : partial_joints.shape[0]]
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        reg = 1e-8 * (
            global_orient.pow(2).sum()
            + body_pose.pow(2).sum()
            + left_hand_pose.pow(2).sum()
            + right_hand_pose.pow(2).sum()
        )
        (loss + reg).backward()
        return loss + reg

    try:
        optimizer_refine.step(closure)
    except Exception as e:
        print("LBFGS failed or terminated early:", e)

    final_output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=transl,
    )
    mesh = final_output.vertices[0].detach().cpu().numpy()
    joints = final_output.joints[0].detach().cpu().numpy()

    final_pred = joints[: partial_joints.shape[0]]
    residual = np.linalg.norm(final_pred[valid_mask.cpu().numpy()] - partial_joints.cpu().numpy()[valid_mask.cpu().numpy()], axis=1)
    print("Final per-joint residual (valid joints): min {:.6f}, mean {:.6f}, max {:.6f}".format(residual.min(), residual.mean(), residual.max()))

    return mesh, joints, valid_mask.cpu().numpy()


parser = argparse.ArgumentParser(description="Fit SMPL-X meshes from partial joints")
parser.add_argument(
    "--joints",
    type=str,
    default="data/smplx_joints.npy",
    help="Path to .npy file containing partial joints (default: smplx_joints.npy)"
)
parser.add_argument(
    "--model",
    type=str,
    default="models",
    help="Path to SMPL-X model folder (default: ./models)"
)
parser.add_argument(
    "--out_meshes",
    type=str,
    default="data/meshed_smplx/all_meshes.npy",
    help="Output .npy file for meshes (default: data/meshed_smplx/all_meshes.npy; smoothed_<name> is written next to it)"
)
parser.add_argument(
    "--out_joints",
    type=str,
    default="data/meshed_smplx/all_joints.npy",
    help="Output .npy file for joints (default: data/meshed_smplx/all_joints.npy; smoothed_<name> is written next to it)"
)
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Joint weight config JSON (default: config/fit_weights.json)"
)
args = parser.parse_args()
JOINT_WEIGHTS = load_fit_weights(args.weights)
 
if not os.path.exists(args.joints):
    raise FileNotFoundError(f"Joints file not found: {args.joints}")
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model path not found: {args.model}")

partial_joints = np.load(args.joints)

print(f"Loaded joints from {args.joints}, shape = {partial_joints.shape}")

print(args.joints, args.model)
all_meshes, all_joints = [], []

for i in range(len(partial_joints)):
    print(f"\n=== Processing frame {i+1}/{len(partial_joints)} ===")
    mesh, joints, valid_joints_mask = infer_full_mesh_from_partial_joints(partial_joints[i], args.model)
    all_meshes.append(mesh)
    all_joints.append(joints)

all_meshes = np.stack(all_meshes)
all_joints = np.stack(all_joints)

for _out in (args.out_meshes, args.out_joints):
    os.makedirs(os.path.dirname(_out) or ".", exist_ok=True)
np.save(args.out_meshes, all_meshes)
np.save(args.out_joints, all_joints)

print(f"\n Saved {len(all_meshes)} meshes → {args.out_meshes}")
print(f"Saved {len(all_joints)} joints → {args.out_joints}")

fps = 30.0
cutoff_hz_mesh = 3.0
cutoff_hz_joints = 4.0
order = 3

def lowpass_sos(cutoff_hz, fs, order=3):
    nyq = fs * 0.5
    wn = cutoff_hz / nyq
    return butter(order, wn, btype='low', output='sos')

def smooth_time(data_TxNx3, cutoff_hz):
    """data: shape (T, N, 3). Returns same shape, filtered along T with zero-phase."""
    T, N, C = data_TxNx3.shape
    flat = data_TxNx3.reshape(T, N * C)
    sos = lowpass_sos(cutoff_hz, fps, order)
    flat_sm = sosfiltfilt(sos, flat, axis=0)
    return flat_sm.reshape(T, N, C)


meshes_sm = smooth_time(all_meshes, cutoff_hz_mesh)
joints_sm = smooth_time(all_joints, cutoff_hz_joints)

folder_m, fname_m = os.path.split(args.out_meshes)
np.save(os.path.join(folder_m, f"smoothed_{fname_m}"), meshes_sm)

folder_j, fname_j = os.path.split(args.out_joints)
np.save(os.path.join(folder_j, f"smoothed_{fname_j}"), joints_sm)
