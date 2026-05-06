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
        self.start_angle = np.arccos(self.max_signal_rate)
        self.end_angle = np.arccos(self.min_signal_rate)
        self._offset_cosine_scheduler()

    def _offset_cosine_scheduler(self):
        """
        Noise Schedule function
        """
        diffusion_angles = self.start_angle + self.diffusion_times * (self.end_angle - self.start_angle)
        self.signal_rates = torch.cos(diffusion_angles)
        self.noise_rates = torch.sin(diffusion_angles)
    
    def get_signal_noise_rates(self):
        """
        Getter function
        """
        return self.signal_rates, self.noise_rates

    def noise_image(self, x0, t):
        """
        Apply noising process to the image

        Args
        ----
        x0 : Torch Tensor
            Batch of input images
        t : Torch Tensor
            Batch of timesteps
        """
        epsilon = torch.randn_like(x0)
        t = t[:,None, None, None]
        noisy_image = self.signal_rates.to(x0.device)[t] * x0 + self.noise_rates.to(x0.device)[t] * epsilon
        return noisy_image, epsilon

if __name__ == '__main__':
    forward_noising = ForwardNoising()
    test_x0 = torch.rand(1, 3, 128, 128)
    test_t = torch.randint(0,1000, (1,))

    noisy_image , epsilon = forward_noising.noise_image(test_x0, test_t)
    print('Shape of noisy_image is ', noisy_image.shape)
    print('Shape of epsilon is ', epsilon.shape)
