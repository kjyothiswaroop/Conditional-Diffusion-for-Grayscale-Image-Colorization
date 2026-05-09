import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from noise import ForwardNoising
from unet import UNet
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import os
import wandb
import hydra
from omegaconf import DictConfig

# Done so that checkpoints, wandb, samples folders save at root and not src/
PROJECT_ROOT = Path(__file__).parent.parent
class Diffusion:
    """
    Diffusion Model Class
    """
    def __init__(self, lr=0.0002, batch_size=32, epochs=1000,
                 dataset_name="kjswaroopNU/celebahq-128-gray", inference=False):
        """
        Constructor for the Diffusion class.
        Builds the UNet and its EMA copy. In training mode also initialises
        the optimiser, loss, dataloader, sample directory, and wandb run.

        Args
        ----
        lr : float
            Learning rate for the Adam optimiser
        batch_size : int
            Number of samples per training batch
        epochs : int
            Total number of training epochs
        dataset_name : str
            HuggingFace dataset repo to load colour/gray image pairs from
        inference : bool
            When True, skips optimiser, dataloader, and wandb initialisation
        """
        self.device = 'cuda:0'
        self.forward_noising= ForwardNoising()

        self.unet = UNet(4,3).to(self.device)
        self.ema_network = copy.deepcopy(self.unet)
        self.ema_network.requires_grad_(False)

        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_name = dataset_name

        if not inference:
            self.optim = torch.optim.Adam(self.unet.parameters(), lr=lr)
            self.loss = nn.MSELoss()
            self.dataset = self._build_dataloader()
            os.makedirs(PROJECT_ROOT / 'samples', exist_ok=True)
            wandb.init(
                project="conditional-diffusion",
                config={
                    "lr": lr,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "ema_decay": 0.999,
                    "dataset_name": self.dataset_name,
                }
            )

    def _build_dataloader(self):
        """
        Builds the training DataLoader from the HuggingFace dataset.

        Returns
        -------
        dataset : DataLoader
            Shuffled DataLoader over the train split
        """
        print("Pulling Dataset from HuggingFace .......")
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.with_format("torch")
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataset

    def _normalize_images(self, images):
        """
        Converts image tensors from [0, 255] to the diffusion model range [-1, 1].

        Args
        ----
        images : tensor
            Uint8 image tensor of shape (B, C, H, W)

        Returns
        -------
        images : tensor
            Float tensor scaled to [-1, 1]
        """
        return images.float().div(127.5).sub(1.0)

    def _denormalize_images(self, images):
        """
        Converts image tensors from [-1, 1] back to [0, 1] for display/logging.

        Args
        ----
        images : tensor
            Float tensor in [-1, 1] of shape (B, C, H, W)

        Returns
        -------
        images : tensor
            Float tensor clamped to [0, 1]
        """
        return images.add(1.0).div(2.0).clamp(0.0, 1.0)
    
    def _train_step(self, batch):
        """
        Runs a single training step on one batch.
        Samples random timesteps, adds noise via the forward process,
        predicts the noise with the UNet, computes MSE loss, and updates weights.

        Args
        ----
        batch : dict
            Dict with keys 'color' and 'gray', each a tensor of shape (B, C, H, W)

        Returns
        -------
        loss : float
            Scalar MSE loss for this batch
        """
        color_images = self._normalize_images(batch['color']).to(self.device)
        gray_images = self._normalize_images(batch['gray']).to(self.device)

        t = torch.randint(0, self.forward_noising.T, (self.batch_size,)).to(self.device)
        signal_rates, noise_rates = self.forward_noising.get_signal_noise_rates()
        nr = noise_rates.to(self.device)[t]
        sr = signal_rates.to(self.device)[t]

        x_T, eps = self.forward_noising.noise_image(color_images, t)

        eps_pred, _ = self.denoise(x_T, gray_images, nr, sr, training=True)
        loss = self.loss(eps, eps_pred)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        wandb.log({"train_loss" : loss.item()})

        self._update_ema()
        
        return loss.item()

    @torch.no_grad()
    def _update_ema(self):
        """
        Updates EMA weights and copies non-trainable BatchNorm buffers from the
        live UNet to the EMA network after each training step.
        """
        ema_decay = 0.999
        for weight, ema_weight in zip(self.unet.parameters(), self.ema_network.parameters()):
            ema_weight.mul_(ema_decay).add_(weight, alpha=1.0 - ema_decay)

        for buffer, ema_buffer in zip(self.unet.buffers(), self.ema_network.buffers()):
            ema_buffer.copy_(buffer)
    
    def train(self, resume=False, checkpoint_path=None):
        """
        Runs the full training loop for the configured number of epochs.
        Every 10 epochs saves a sample grid to samples/ and logs to wandb.
        Every 50 epochs saves a named checkpoint to checkpoints/.

        Args
        ----
        resume : bool
            When True, loads weights and optimiser state from checkpoint_path
            before starting
        checkpoint_path : str or None
            Path to the checkpoint file to resume from (relative to project root
            or absolute)
        """
        start_epoch = 0
        if resume:
            resolved = Path(checkpoint_path)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            checkpoint = torch.load(resolved, map_location=self.device)
            self.unet.load_state_dict(checkpoint['unet'])
            self.ema_network.load_state_dict(checkpoint['ema_network'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resumed from epoch {checkpoint["epoch"]}')

        os.makedirs(PROJECT_ROOT / 'checkpoints', exist_ok=True)

        for epoch in range(start_epoch, self.epochs):
            total_loss = 0.0
            for batch in self.dataset:
                loss = self._train_step(batch)
                total_loss+= loss

            if epoch % 10 == 0:
                # Take 10 images from the batch (both color and gray scale)
                test_color = self._normalize_images(batch['color']).to(self.device)[:10]
                test_gray = self._normalize_images(batch['gray']).to(self.device)[:10]

                #Pass through the reverse diffusion 
                generated = self.reverse_diffusion(test_gray, diffusion_steps=100)
                display_gray = self._denormalize_images(test_gray)
                display_generated = self._denormalize_images(generated)
                display_color = self._denormalize_images(test_color)
                
                #Save the image to samples directory
                fig, axes = plt.subplots(3, 10, figsize=(12, 9))
                for col in range(10):
                    axes[0, col].imshow(display_gray[col].squeeze().cpu(), cmap='gray')
                    axes[1, col].imshow(display_generated[col].permute(1, 2, 0).cpu())
                    axes[2, col].imshow(display_color[col].permute(1, 2, 0).cpu())
                    for row in range(3):
                        axes[row, col].axis('off')

                axes[0, 0].set_title('Input Gray')
                axes[1, 0].set_title('Generated')
                axes[2, 0].set_title('Ground Truth')

                plt.savefig(PROJECT_ROOT / 'samples' / f'output_epoch_{epoch}.png', bbox_inches='tight')
                plt.close(fig)

                wandb.log({
                    "samples": [
                        wandb.Image(
                            display_generated[i].permute(1, 2, 0).cpu().numpy(),
                            caption=f"img_{i}"
                        )
                        for i in range(10)
                    ]
                })

            avg_loss = total_loss / len(self.dataset)
            wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch})
            print(f'Average Loss for epoch {epoch} is {avg_loss}')

            if epoch % 50 == 0 and epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'unet': self.unet.state_dict(),
                    'ema_network': self.ema_network.state_dict(),
                    'optimizer': self.optim.state_dict(),
                }, PROJECT_ROOT / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt')

        wandb.finish()
    
    def denoise(self, noisy_images, gray_images, noise_rates, signal_rates, training):
        """
        Predicts the noise in noisy_images conditioned on gray_images,
        then computes the denoised image estimate.

        Args
        ----
        noisy_images : tensor
            Noisy RGB tensor of shape (B, 3, H, W)
        gray_images : tensor
            Grayscale conditioning tensor of shape (B, 1, H, W)
        noise_rates : tensor
            Noise rate at timestep t, shape (B,)
        signal_rates : tensor
            Signal rate at timestep t, shape (B,)
        training : bool
            When True uses the live UNet; when False uses the EMA network

        Returns
        -------
        eps_pred : tensor
            Predicted noise, shape (B, 3, H, W)
        pred_images : tensor
            Denoised image estimate, shape (B, 3, H, W)
        """
        if training:
            network = self.unet
        else:
            network = self.ema_network

        nr = noise_rates[:, None, None, None]
        sr = signal_rates[:, None, None, None]

        x_cond = torch.cat([noisy_images, gray_images], dim=1)
        eps_pred = network(x_cond, noise_rates ** 2)
        pred_images = (noisy_images - nr * eps_pred) / sr

        return eps_pred, pred_images
    
    @torch.no_grad()
    def reverse_diffusion(self, gray_images, diffusion_steps):
        """
        Runs the reverse diffusion process to colorize a batch of grayscale images.
        Iteratively denoises from pure noise to a clean RGB image using the EMA network.

        Args
        ----
        gray_images : tensor
            Grayscale conditioning tensor of shape (B, 1, H, W) in [-1, 1]
        diffusion_steps : int
            Number of denoising steps (fewer steps = faster, lower quality)

        Returns
        -------
        x : tensor
            Colorized RGB tensor of shape (B, 3, H, W) in [-1, 1]
        """
        batch_size = gray_images.shape[0]
        self.ema_network.eval()

        x = torch.randn(batch_size, 3, 128, 128, device=self.device)
       
        # Pass through forward noising process
        signal_rates, noise_rates = self.forward_noising.get_signal_noise_rates()


        timesteps = torch.linspace(self.forward_noising.T - 1, 0, diffusion_steps).long()

        for i, t in enumerate(timesteps):
            nr = noise_rates[t].to(self.device).expand(batch_size)
            sr = signal_rates[t].to(self.device).expand(batch_size)

            eps_pred, pred_images = self.denoise(x, gray_images, nr, sr, training=False)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                nr_prev = noise_rates[t_prev].to(self.device).expand(batch_size)[:, None, None, None]
                sr_prev = signal_rates[t_prev].to(self.device).expand(batch_size)[:, None, None, None]
                x = sr_prev * pred_images + nr_prev * eps_pred
            else:
                x = pred_images

        return x
    
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume and cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required when resume=true")

    model = Diffusion(
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        dataset_name=cfg.dataset_name,
        inference=cfg.inference
    )
    model.train(resume=cfg.resume, checkpoint_path=cfg.checkpoint_path)


if __name__ == '__main__':
    main()
