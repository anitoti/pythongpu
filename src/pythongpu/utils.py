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
