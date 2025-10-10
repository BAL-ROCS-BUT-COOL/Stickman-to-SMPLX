import torch
import smplx
import numpy as np
import argparse
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

finger_indices = [
        25, 26, 27, 67,  # left index
        28, 29, 30, 68,  # left middle
        31, 32, 33, 70,  # left pinky
        34, 35, 36, 69, # left ring
        37, 38, 39, 66, # left thumb
        40, 41, 42, 72, # right index
        43, 44, 45, 73, # right middle
        46, 47, 48, 75,  # right pinky
        49, 50, 51, 74, # right ring
        52, 53, 54, 71, # right thumb
        # 20, 21, # both wrists
        # 18,19 #include elbows
    ]


def infer_full_mesh_from_partial_joints(
    partial_joints_np,
    smplx_model_path,
    missing_threshold=1e-6,
    device=None,
):
    """
    partial_joints_np: (N_joints, 3) numpy array (N_joints should match model joint ordering, e.g. 76)
    finger_indices: list/iterable of int indices (indices within the joint array) to upweight
    missing_threshold: threshold to treat joint as missing (norm near zero)
    """
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

    # Initial parameters (require_grad=True for optimization)
    global_orient = torch.zeros((1, 3), device=device, requires_grad=True)
    body_pose = torch.zeros((1, 63), device=device, requires_grad=True)
    betas = torch.zeros((1, 10), device=device, requires_grad=False)
    left_hand_pose = torch.zeros((1, 45), device=device, requires_grad=True)
    right_hand_pose = torch.zeros((1, 45), device=device, requires_grad=True)
    transl = torch.zeros((1, 3), device=device, requires_grad=True)


    # create weight vector and ensure device type
    weights = torch.ones(len(valid_mask), device=device)
    for idx in finger_indices:
        if 0 <= idx < len(weights):
            weights[idx] = 5.0  # upweight fingers (tune this if needed)

    # helper: compute weighted MSE only over valid joints
    def weighted_mse_loss(pred_joints, target_joints, valid_mask, weights):
        # pred_joints: (num_joints, 3)
        # weights: (num_joints,)
        diff = (pred_joints - target_joints) ** 2  # (J, 3)
        weighted = diff * weights.unsqueeze(1)  # (J, 3)
        # select valid rows and average over elements
        valid_rows = weighted[valid_mask]
        if valid_rows.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return valid_rows.mean()

    # ---------------- Stage 1: Fit body (no hands) ----------------
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
        predicted_joints = output.joints[0, : partial_joints.shape[0]]  # (J,3)
        loss = weighted_mse_loss(predicted_joints, partial_joints, valid_mask, weights)
        loss.backward()
        optimizer_stage1.step()
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Stage1] Iter {i+1}/{n_iter_stage1}  loss={loss.item():.8f}")

    # ---------------- Stage 2: Fit hands (Adam to get close) ----------------
    # optimize hands (and allow slight body changes)
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
        # small regularizer to keep parameters numerically stable (tiny)
        reg = 1e-6 * (
            global_orient.pow(2).sum()
            + body_pose.pow(2).sum()
            + left_hand_pose.pow(2).sum()
            + right_hand_pose.pow(2).sum()
            #+ betas.pow(2).sum()
        )
        (loss + reg).backward()
        optimizer_stage2.step()
        if (i + 1) % 100 == 0 or i == 0:
            print(f"[Stage2] Iter {i+1}/{n_iter_stage2}  loss={loss.item():.8f}")

    # ---------------- Final refinement: L-BFGS (fine convergence) ----------------
    # LBFGS works better when parameters are small in number; we include all pose params here.
    # Note: LBFGS needs a closure that recomputes loss and gradients.
    params_refine = [global_orient, body_pose, transl, left_hand_pose, right_hand_pose]
    # Make sure all require grad
    for p in params_refine:
        p.requires_grad = True

    # LBFGS optimizer
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
        # very tiny regularizer to stabilize
        reg = 1e-8 * (
            global_orient.pow(2).sum()
            + body_pose.pow(2).sum()
            + left_hand_pose.pow(2).sum()
            + right_hand_pose.pow(2).sum()
            #+ betas.pow(2).sum()
        )
        (loss + reg).backward()
        return loss + reg

    try:
        optimizer_refine.step(closure)
    except Exception as e:
        print("LBFGS failed or terminated early:", e)

    # Final output
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

    # final residual on valid joints
    final_pred = joints[: partial_joints.shape[0]]
    residual = np.linalg.norm(final_pred[valid_mask.cpu().numpy()] - partial_joints.cpu().numpy()[valid_mask.cpu().numpy()], axis=1)
    print("Final per-joint residual (valid joints): min {:.6f}, mean {:.6f}, max {:.6f}".format(residual.min(), residual.mean(), residual.max()))

    return mesh, joints, valid_mask.cpu().numpy()



# ─── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fit SMPL-X meshes from partial joints")
parser.add_argument(
    "--joints",
    type=str,
    default="smplx_joints.npy",   # default file
    help="Path to .npy file containing partial joints (default: smplx_joints.npy)"
)
parser.add_argument(
    "--model",
    type=str,
    default="models",   # default folder
    help="Path to SMPL-X model folder (default: ./models)"
)
parser.add_argument(
    "--out_meshes",
    type=str,
    default="all_meshes.npy",
    help="Output .npy file for meshes (default: all_meshes.npy)"
)
parser.add_argument(
    "--out_joints",
    type=str,
    default="all_joints.npy",
    help="Output .npy file for joints (default: all_joints.npy)"
)
args = parser.parse_args()
 
# ─── Load Input ───────────────────────────────────────────────────────────────
if not os.path.exists(args.joints):
    raise FileNotFoundError(f"Joints file not found: {args.joints}")
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model path not found: {args.model}")

partial_joints = np.load(args.joints)

print(f"Loaded joints from {args.joints}, shape = {partial_joints.shape}")

print(args.joints, args.model)
# ─── Processing ───────────────────────────────────────────────────────────────
all_meshes, all_joints = [], []

for i in range(len(partial_joints)):
    print(f"\n=== Processing frame {i+1}/{len(partial_joints)} ===")
    mesh, joints, valid_joints_mask = infer_full_mesh_from_partial_joints(partial_joints[i], args.model)
    all_meshes.append(mesh)
    all_joints.append(joints)


# ─── Save Outputs ─────────────────────────────────────────────────────────────
np.save(args.out_meshes, np.stack(all_meshes))
np.save(args.out_joints, np.stack(all_joints))

print(f"\n Saved {len(all_meshes)} meshes → {args.out_meshes}")
print(f"Saved {len(all_joints)} joints → {args.out_joints}")

# ─── Smooth Outputs with Low-pass-filter─────────────────────────────────────────────────────────────
fps = 30.0                      # your sequence is 30 fps
cutoff_hz_mesh = 3.0            # keep motions slower than ~3 Hz (tweak!)
cutoff_hz_joints = 4.0          # joints can tolerate slightly higher cutoff
order = 3

def lowpass_sos(cutoff_hz, fs, order=3):
    nyq = fs * 0.5
    wn = cutoff_hz / nyq
    return butter(order, wn, btype='low', output='sos')

def smooth_time(data_TxNx3, cutoff_hz):
    """data: shape (T, N, 3). Returns same shape, filtered along T with zero-phase."""
    T, N, C = data_TxNx3.shape
    flat = data_TxNx3.reshape(T, N * C)           # (T, N*3)
    sos = lowpass_sos(cutoff_hz, fps, order)
    # sosfiltfilt applies per column when axis=0
    flat_sm = sosfiltfilt(sos, flat, axis=0)
    return flat_sm.reshape(T, N, C)


# ---- smooth ----
meshes_sm = smooth_time(all_meshes, cutoff_hz_mesh)
joints_sm = smooth_time(all_joints, cutoff_hz_joints)

# ---- save ----
np.save("all_meshes_smoothed.npy", meshes_sm)
np.save("all_joints_smoothed.npy", joints_sm)
