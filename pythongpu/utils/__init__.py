from pathlib import Path

import torch


def get_laplacian(A, norm=None, device="cpu"):
    A = torch.as_tensor(A, dtype=torch.float32, device=device)
    D = torch.diag(A.sum(dim=1))
    L = D - A

    if norm is None:
        return L

    elif norm == "sym":
        d_inv_sqrt = torch.pow(A.sum(dim=1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.eye(A.shape[0], device=device) - D_inv_sqrt @ A @ D_inv_sqrt

    elif norm == "rw":
        d_inv = torch.pow(A.sum(dim=1), -1.0)
        d_inv[torch.isinf(d_inv)] = 0.0
        D_inv = torch.diag(d_inv)
        return torch.eye(A.shape[0], device=device) - D_inv @ A


_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_plot_path(script_name, filename, outdir=None):
    """
    Unified figure-output path. Routes plots to <outdir>/derivatives/
    (default: <repo_root>/data/derivatives/), tagging each filename with
    its originating script for provenance. Creates the directory if needed.
    """
    base = Path(outdir) / "derivatives" if outdir is not None else _REPO_ROOT / "data" / "derivatives"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{script_name}__{filename}"


def get_clean_path(system, data_type, source, metric, thresh_or_seed,
                   coupling_type, k_clusters, coupling, base_dir="data"):
    """
    Ultimate dynamic path generator for fMRI-coupled network dynamics.
    Handles variables evident from code execution, literature, and controls.
    """
    # 1. Standardize parameter strings
    coupling_str = f"coup_{str(coupling).replace('.', 'p')}"
    cluster_str = f"k_{k_clusters}"
    
    if data_type == "empirical":
        src_str = f"sub-{source:02d}" if isinstance(source, int) else f"sub-{source}"
        tos_str = f"thresh_{str(thresh_or_seed).replace('.', 'p')}"
    else:
        src_str = str(source)
        tos_str = f"seed_{thresh_or_seed}"

    # 2. Construct the comprehensive nested path structure
    target_path = (Path(base_dir) / system / data_type / src_str / 
                   metric / tos_str / coupling_type / cluster_str / coupling_str)
    
    # 3. Create folders seamlessly if they don't exist yet
    plot_dir = target_path / "plots"
    data_dir = target_path / "derivatives"
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return plot_dir, data_dir
