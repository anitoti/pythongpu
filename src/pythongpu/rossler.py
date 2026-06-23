import torch

class RosslerNetwork:
    def __init__(self, L, a=0.2, b=0.2, c=5.7, coupling=0.1, device='cpu'):
        """
        Initializes the Rössler Network.
        
        Parameters:
        - L (Tensor): Laplacian or adjacency matrix representing network connectivity (N x N).
        - a, b, c (float): Internal parameters of the chaotic Rössler oscillator.
        - coupling (float): Global coupling strength between nodes.
        - device (str): 'cpu' or 'cuda' for GPU acceleration.
        """
        self.L = torch.as_tensor(L, dtype=torch.float32, device=device)
        self.N = self.L.shape[0]
        self.a = a
        self.b = b
        self.c = c
        self.coupling = coupling
        self.device = device

    def forward(self, state):
        """
        Calculates the derivatives for the network.
        
        Parameters:
        - state (Tensor): Current state of the network with shape (3, N) 
                          where row 0 is X, row 1 is Y, and row 2 is Z.
                          
        Returns:
        - dstate (Tensor): The derivatives (dX, dY, dZ) with shape (3, N).
        """
        X = state[0, :]
        Y = state[1, :]
        Z = state[2, :]

        # Rössler internal dynamics
        dX = -Y - Z
        dY = X + self.a * Y
        dZ = self.b + Z * (X - self.c)

        # Network coupling (diffusive coupling applied to the X component)
        # Assuming self.L is a Laplacian matrix where (L @ X) computes the network influence
        network_influence = torch.matmul(self.L, X)
        dX = dX - self.coupling * network_influence

        # Combine derivatives back into the same shape as state
        return torch.stack([dX, dY, dZ], dim=0)

    def integrate(self, initial_state, dt, steps):
        """
        Integrates the system over time using the 4th-order Runge-Kutta (RK4) method.
        """
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=self.device)
        history = torch.zeros((steps, 3, self.N), device=self.device)
        
        for t in range(steps):
            history[t] = state
            
            # RK4 Integration steps
            k1 = self.forward(state)
            k2 = self.forward(state + 0.5 * dt * k1)
            k3 = self.forward(state + 0.5 * dt * k2)
            k4 = self.forward(state + dt * k3)
            
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        return history

# --- Example Usage ---
if __name__ == "__main__":
    # Create a simple 3-node ring network Laplacian matrix
    # L = Degree Matrix - Adjacency Matrix
    L_matrix = torch.tensor([
        [ 2., -1., -1.],
        [-1.,  2., -1.],
        [-1., -1.,  2.]
    ])

    # 3 nodes, each needs an initial (x, y, z) coordinate close to the attractor
    init_state = torch.tensor([
        [0.1, 0.2, 0.3],  # X coordinates for nodes 1, 2, 3
        [0.1, 0.2, 0.3],  # Y coordinates for nodes 1, 2, 3
        [0.1, 0.2, 0.3]   # Z coordinates for nodes 1, 2, 3
    ])

    net = RosslerNetwork(L=L_matrix, coupling=0.05)
    traj = net.integrate(init_state, dt=0.01, steps=1000)
    print("Simulation finished. Output shape (steps, variables, nodes):", traj.shape)
