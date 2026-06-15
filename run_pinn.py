"""
run_pinn.py — M2 PINN Wrapper
==============================
Thin wrapper around the PINN-Glioblastoma- repo.
Clones the repo once, then exposes two functions:

    result = run_pinn(t1_path, t1ce_path, t2_path, flair_path, out_dir)
    delta  = compute_delta(scan1_result, scan2_result)

Output dict (per scan):
    {
        "mu_D":       float,       # diffusion ratio      ∈ [0.75, 1.25]
        "mu_R":       float,       # proliferation ratio  ∈ [0.75, 1.25]
        "gamma":      float,       # go-grow index = mu_R / mu_D
        "u_pred":     np.ndarray,  # 3D cell density map (64³) float32
        "converged":  bool,        # False → agent uses radiomics-only M3 fallback
        "loss_final": float,
        "out_dir":    str,
    }
"""

import os
import sys
import json
import subprocess
import numpy as np

# ── Repo config ───────────────────────────────────────────────────────────────
REPO_URL   = "https://github.com/arnavmishra4/PINN-Glioblastoma-"
REPO_NAME  = "PINN-Glioblastoma-"
CLONE_ROOT = "/kaggle/working"
REPO_DIR   = os.path.join(CLONE_ROOT, REPO_NAME)
MU_VALID   = (0.75, 1.25)


def _ensure_repo():
    if not os.path.isdir(REPO_DIR):
        print(f"[pinn] Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def run_pinn(
    t1_path:    str,
    t1ce_path:  str,
    t2_path:    str,   # accepted for interface consistency; PINN only needs T1/T1ce/FLAIR
    flair_path: str,
    out_dir:    str = "pinn_output",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    _ensure_repo()

    from preprocess_glioma import preprocess
    from train_glioma import train

    # ── Preprocess → .mat ─────────────────────────────────────────────────────
    mat_path = os.path.join(out_dir, "data.mat")
    if not os.path.exists(mat_path):
        print("[pinn] Preprocessing NIfTI → .mat ...")
        preprocess(
            t1_path    = t1_path,
            t1ce_path  = t1ce_path,
            flair_path = flair_path,
            output_mat = mat_path,
        )

    # ── Train PINN ────────────────────────────────────────────────────────────
    finetune_dir = os.path.join(out_dir, "model", "finetune")
    if os.path.exists(os.path.join(finetune_dir, "options.json")):
        print("[pinn] Checkpoint found — skipping training.")
        from train_glioma import predict_cell_density
        u_pred, raw_params = predict_cell_density(
            mat_file     = mat_path,
            finetune_dir = finetune_dir,
            output_dir   = os.path.join(out_dir, "model"),
        )
    else:
        print("[pinn] Starting two-stage PINN training ...")
        u_pred, raw_params = train(
            mat_file      = mat_path,
            output_dir    = os.path.join(out_dir, "model"),
            predict_after = True,
            predict_res   = 64,
        )

    # ── Package output ────────────────────────────────────────────────────────
    mu_D  = float(raw_params.get("rD",   raw_params.get("mu_D",  1.0)))
    mu_R  = float(raw_params.get("rRHO", raw_params.get("mu_R",  1.0)))
    gamma = mu_R / (mu_D + 1e-8)
    converged = (MU_VALID[0] <= mu_D <= MU_VALID[1] and
                 MU_VALID[0] <= mu_R <= MU_VALID[1])

    if not converged:
        print(f"[pinn] WARNING: μ_D={mu_D:.4f}, μ_R={mu_R:.4f} out of range — "
              f"agent should fall back to radiomics-only M3.")
    else:
        print(f"[pinn] ✓ μ_D={mu_D:.4f}  μ_R={mu_R:.4f}  γ={gamma:.4f}")

    result = {
        "mu_D":       mu_D,
        "mu_R":       mu_R,
        "gamma":      gamma,
        "u_pred":     u_pred.astype(np.float32) if u_pred is not None else None,
        "converged":  converged,
        "loss_final": float("nan"),   # populated below if log exists
        "out_dir":    out_dir,
    }

    # Try to read final loss from solverinfo.json (saved by the repo's PINNSolver)
    solverinfo_path = os.path.join(finetune_dir, "solverinfo.json")
    if os.path.exists(solverinfo_path):
        with open(solverinfo_path) as f:
            info = json.load(f)
        losses = info.get("scipylbfgslosses") or info.get("tfadamloss") or {}
        result["loss_final"] = float(losses.get("total", float("nan")))

    # Persist summary (without the large u_pred array) next to the checkpoint
    summary = {k: v for k, v in result.items() if k != "u_pred"}
    with open(os.path.join(out_dir, "pinn_result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return result


def compute_delta(scan1: dict, scan2: dict) -> dict:
    """
    Compute Δμ_D, Δμ_R, Δγ between two scans.
    This is the primary signal fed into M3 (XGBoost + MLP).

    Clinical interpretation:
        μ_R ↓  μ_D ↓  → Both suppressed. True treatment response.
        μ_R ↓  μ_D ↑  → Dispersing not dying. Pseudoregression risk.
        μ_R ↑  μ_D ~  → Proliferation-dominant true progression.
        μ_R ~  μ_D ↑  → Invasion-dominant spread along WM tracts.
    """
    delta = {
        "delta_mu_D":  scan2["mu_D"]  - scan1["mu_D"],
        "delta_mu_R":  scan2["mu_R"]  - scan1["mu_R"],
        "delta_gamma": scan2["gamma"] - scan1["gamma"],
    }
    print(f"\n[pinn] Δμ_D={delta['delta_mu_D']:+.4f}  "
          f"Δμ_R={delta['delta_mu_R']:+.4f}  "
          f"Δγ={delta['delta_gamma']:+.4f}")
    return delta
