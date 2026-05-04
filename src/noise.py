import torch
import numpy as np

class ForwardNoising:
    """
    Forward Noising process in Diffusion.
    x0 ---> xT 
    """
    def __init__(self):
        """
        Constructor
        """
        self.T = 1000
        self.diffusion_times = torch.tensor([x / self.T for x in range(self.T)])
        self.min_signal_rate = 0.02
        self.max_signal_rate = 0.95
        self.start_angle = np.acos(self.max_signal_rate)
        self.end_angle = np.acos(self.min_signal_rate)
        self._offset_cosine_scheduler()

    def _offset_cosine_scheduler(self):
        """
        Noise Schedule function
        """
        diffusion_angles = self.start_angle + self.diffusion_times * (self.end_angle - self.start_angle)
        self.signal_rates = torch.cos(diffusion_angles)
        self.noise_rates = torch.sin(diffusion_angles)

    def noise_image(self, x0, t):
        """
        Apply noising process to the image

        Args
        ----
        x0 : Torch Tensor
            Input Image
        t : int
            Timestep
        """
        epsilon = torch.randn_like(x0)
        noisy_image = self.signal_rates[t] * x0 + self.noise_rates[t] * epsilon
        return noisy_image, epsilon
