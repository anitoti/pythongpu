import torch
from .utils import get_laplacian

def rossler(state, L, a=0.2, b=0.2, c=5.7, coupling=0.1):
    # Extract x, y, z for each node
    x, y, z = state[0], state[1], state[2]
    
    # Rössler equations with Laplacian coupling
    dx = -y - z - coupling * (L @ x)
    dy = x + a * y
    dz = b + z * (x - c)
    
    return torch.stack([dx, dy, dz])

def simulate(L, steps=2000, dt=0.01):
    n = L.shape[0]
    
    # Initialize random state on the same device as the Laplacian (L)
    state = torch.randn((3, n), device=L.device)
    
    for _ in range(steps):
        # Euler integration step
        state = state + rossler(state, L) * dt
        
    return state

if __name__ == "__main__":
    # Detect and set target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with a 100-node identity matrix directly on device
    L = torch.eye(100, device=device)
    
    print("running simulation on:", device)
    
    # Execute simulation using default step parameters
    final = simulate(L)
    
    print("done! final state shape:", final.shape)
