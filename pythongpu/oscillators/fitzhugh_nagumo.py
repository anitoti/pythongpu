from pythongpu.oscillators.base_oscillator import BaseOscillator


class FitzHughNagumoNetwork(BaseOscillator):
    """
    Placeholder for a network of coupled FitzHugh-Nagumo neuronal spiking
    oscillators. Not yet implemented.
    """

    def __init__(self, L, a=0.7, b=0.8, tau=12.5, coupling=0.1, device='cpu'):
        super().__init__(L, device=device, a=a, b=b, tau=tau, coupling=coupling)

    def rhs(self, state):
        raise NotImplementedError("FitzHugh-Nagumo dynamics are not implemented yet.")
