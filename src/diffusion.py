import torch
from datasets import load_dataset, Image
from torch.utils.data import DataLoader
from noise import ForwardNoising
from unet import UNet
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import os
import wandb
class Diffusion:
    """
    Diffusion Model Class
    """
    def __init__(self):
        """
        Constructor
        """
        self.device = 'cuda:0'
        self.forward_noising= ForwardNoising()

        self.unet = UNet(4,3).to(self.device)
        self.ema_network = copy.deepcopy(self.unet)
        self.ema_network.requires_grad_(False)

        self.optim = torch.optim.Adam(self.unet.parameters(), lr=0.0002)
        self.loss = nn.MSELoss()
        self.batch_size = 32
        self.epochs = 1000

        self.dataset = self._build_dataloader()

        os.makedirs('samples', exist_ok=True)

        wandb.init(
            project="conditional-diffusion",
            config = {
                "lr" : 0.0002,
                "batch_size" : self.batch_size,
                "epochs" : self.epochs,
                "ema_decay" : 0.999
            }
        )

    def _build_dataloader(self):
        """
        Function to build the dataset
        """
        dataset = load_dataset("kjswaroopNU/celebahq-128-gray", split="train")
        dataset = dataset.with_format("torch")
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataset
    
    def _train_step(self, batch):
        """
        Train step per batch
        """
        color_images = batch['color'].float().div(255).to(self.device)
        gray_images = batch['gray'].float().div(255).to(self.device)

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

        for weight, ema_weight in zip(self.unet.parameters(), self.ema_network.parameters()):
            ema_weight.data = 0.999 * ema_weight.data + 0.001 * weight.data
        
        return loss.item()
    
    def train(self):
        """
        Train the U-Net model
        """
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in self.dataset:
                loss = self._train_step(batch)
                total_loss+= loss

            if epoch % 10 == 0:
                # Take 10 images from the batch (both color and gray scale)
                test_color = batch['color'].float().div(255).to(self.device)[:10]
                test_gray = batch['gray'].float().div(255).to(self.device)[:10]

                #Pass through the reverse diffusion 
                generated = self.reverse_diffusion(test_gray, diffusion_steps=20)
                
                #Save the image to samples directory
                fig, axes = plt.subplots(3, 10, figsize=(12, 9))
                for col in range(10):
                    axes[0, col].imshow(test_gray[col].squeeze().cpu(), cmap='gray')
                    axes[1, col].imshow(generated[col].permute(1, 2, 0).cpu())
                    axes[2, col].imshow(test_color[col].permute(1, 2, 0).cpu())
                    for row in range(3):
                        axes[row, col].axis('off')

                axes[0, 0].set_title('Input Gray')
                axes[1, 0].set_title('Generated')
                axes[2, 0].set_title('Ground Truth')

                plt.savefig(f'samples/output_epoch_{epoch}.png', bbox_inches='tight')
                plt.close(fig)

                wandb.log({
                    "samples": [
                        wandb.Image(
                            generated[i].permute(1, 2, 0).cpu().numpy(),
                            caption=f"img_{i}"
                        )
                        for i in range(10)
                    ]
                }, step=epoch)

            avg_loss = total_loss / len(self.dataset)
            wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch}, step=epoch)
            print(f'Average Loss for epoch {epoch} is {avg_loss}')

        wandb.finish()
    
    def denoise(self, noisy_images, gray_images, noise_rates, signal_rates, training):
        """
        Denoise method
        """
        if training:
            network = self.unet
        else:
            network = self.ema_network

        nr = noise_rates[:, None, None, None]
        sr = signal_rates[:, None, None, None]

        x_cond = torch.cat([noisy_images, gray_images], dim=1)
        eps_pred = network(x_cond, noise_rates ** 2)
        pred_images = torch.clamp((noisy_images - nr * eps_pred) / sr, 0.0, 1.0)

        return eps_pred, pred_images
    
    @torch.no_grad()
    def reverse_diffusion(self, gray_images, diffusion_steps):
        """
        Inference function to reconstruct the image
        """
        batch_size = gray_images.shape[0]
        
        #Start with random noise
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
    
if __name__ == '__main__':
    dataset = load_dataset("kjswaroopNU/celebahq-128-gray", split="train")
    dataset = dataset.with_format("torch")

    sample_gray = dataset[0]['gray']
    sample_color = dataset[0]['color']
    print(f'Shape of Color image is {sample_color.shape}')
    print(f'Shape of Smaple_gray is {sample_gray.shape}')
