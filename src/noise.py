import torch
import numpy as np

class ForwardNoising:
    """
    Forward Noising process in Diffusion.
    x0 ---> xT 
    """
    def __init__(self):
        """
        Constructor for ForwardNoising.
        Precomputes signal and noise rates for all T timesteps using an
        offset cosine schedule bounded between min and max signal rates.
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
        Computes signal and noise rates for all timesteps using an offset cosine schedule.
        Angles are linearly interpolated between start_angle and end_angle so that
        signal_rate = cos(angle) and noise_rate = sin(angle), satisfying signal² + noise² = 1.
        """
        diffusion_angles = self.start_angle + self.diffusion_times * (self.end_angle - self.start_angle)
        self.signal_rates = torch.cos(diffusion_angles)
        self.noise_rates = torch.sin(diffusion_angles)
    
    def get_signal_noise_rates(self):
        """
        Returns the precomputed signal and noise rate schedules.

        Returns
        -------
        signal_rates : tensor
            Cosine signal rates for all T timesteps, shape (T,)
        noise_rates : tensor
            Sine noise rates for all T timesteps, shape (T,)
        """
        return self.signal_rates, self.noise_rates

    def noise_image(self, x0, t):
        """
        Applies the forward noising process to a batch of images at timesteps t.
        Samples Gaussian noise and blends it with x0 using the precomputed rates:
        x_t = signal_rate[t] * x0 + noise_rate[t] * epsilon

        Args
        ----
        x0 : tensor
            Clean input images, shape (B, C, H, W)
        t : tensor
            Batch of timestep indices, shape (B,)

        Returns
        -------
        noisy_image : tensor
            Noised image at timestep t, shape (B, C, H, W)
        epsilon : tensor
            Gaussian noise that was added, shape (B, C, H, W)
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
